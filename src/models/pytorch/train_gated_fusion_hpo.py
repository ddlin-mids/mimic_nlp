import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from transformers import get_cosine_schedule_with_warmup
import mlflow
import mlflow.pytorch
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Gated Fusion Layer
# -----------------------------------------------------------------------------
class GatedMultimodalUnit(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.linear_z = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_a, x_b):
        combined = torch.cat([x_a, x_b], dim=1)
        z = self.sigmoid(self.linear_z(combined))
        h = z * x_a + (1 - z) * x_b
        return self.dropout(h), z

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class GatedFusionDataset(Dataset):
    def __init__(self, ehr_data, txt_data, y):
        self.ehr = torch.FloatTensor(ehr_data)
        self.txt = torch.FloatTensor(txt_data)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.ehr[idx], self.txt[idx], self.y[idx]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class GatedFusionModel(nn.Module):
    def __init__(self, ehr_dim, txt_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.ehr_proj = nn.Sequential(
            nn.Linear(ehr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gated_fusion = GatedMultimodalUnit(hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_ehr, x_txt):
        h_ehr = self.ehr_proj(x_ehr)
        h_txt = self.txt_proj(x_txt)
        fused, gate_vals = self.gated_fusion(h_ehr, h_txt)
        logits = self.classifier(fused)
        return logits, gate_vals

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
class FusionDataLoader:
    def __init__(self, base_dir=".", embedding_dir=None):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        if embedding_dir:
            embed_base = Path(embedding_dir)
            self.structured_path = embed_base / "structured_ehr_embeddings.npz"
            self.structured_mapping_path = embed_base / "structured_ehr_mapping.csv"
        else:
            self.structured_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
            self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"

    def load_data(self, pca_components=0):
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        y = cohort['readmitted_within_window'].astype(int).values
        splits = cohort['split'].values
        ids = cohort['hadm_id'].values
        
        ehr_data = self._load_aligned_embeddings(self.structured_path, ids, mapping_file=self.structured_mapping_path)
        
        disch = self._load_aligned_embeddings(self.discharge_path, ids)
        rad = self._load_radiology_embeddings(ids)
        txt_data = np.hstack([disch, rad])
        
        # Simple Mean Imputation for Text NaN if any (though usually zeros)
        # Assuming dense embeddings, zeros is fine for missingness
        
        return ehr_data, txt_data, y, splits

    def _load_aligned_embeddings(self, filepath, target_ids, mapping_file=None):
        if not filepath.exists(): return np.zeros((len(target_ids), 0))
        data = np.load(filepath)
        if mapping_file:
            mapping = pd.read_csv(mapping_file)
            embeddings = data['embeddings']
            hid_to_idx = {}
            for _, row in mapping.iterrows():
                parts = str(row['node_name']).split('_')
                if len(parts) == 2: hid_to_idx[int(parts[1])] = row['row_idx']
            dim = embeddings.shape[1]
            aligned = []
            for tid in target_ids:
                if tid in hid_to_idx: aligned.append(embeddings[hid_to_idx[tid]])
                else: aligned.append(np.zeros(dim))
            return np.array(aligned)
        else:
            id_key = 'hadm_ids' if 'hadm_ids' in data else 'ids'
            source_ids = data[id_key]
            source_embeds = data['embeddings']
            id_map = {int(k) if isinstance(k, (float, np.floating)) else k: v for k, v in zip(source_ids, source_embeds)}
            dim = source_embeds.shape[1]
            return np.array([id_map.get(tid, np.zeros(dim)) for tid in target_ids])

    def _load_radiology_embeddings(self, target_ids):
        if not self.radiology_path.exists(): return np.zeros((len(target_ids), 0))
        data = np.load(self.radiology_path)
        from collections import defaultdict
        grouped = defaultdict(list)
        for hid, emb in zip(data['hadm_ids'], data['embeddings']):
            grouped[int(hid)].append(emb)
        dim = data['embeddings'].shape[1]
        aligned = []
        for tid in target_ids:
            if tid in grouped: aligned.append(np.mean(grouped[tid], axis=0))
            else: aligned.append(np.zeros(dim))
        return np.array(aligned)

# -----------------------------------------------------------------------------
# HPO Logic
# -----------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for x_ehr, x_txt, y in loader:
        x_ehr, x_txt, y = x_ehr.to(device), x_txt.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits, _ = model(x_ehr, x_txt)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_targets = []
    for x_ehr, x_txt, y in loader:
        x_ehr, x_txt, y = x_ehr.to(device), x_txt.to(device), y.to(device).unsqueeze(1)
        logits, _ = model(x_ehr, x_txt)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(y.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    auc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    f1 = f1_score(all_targets, (all_preds > 0.5).astype(int))
    return auc, auprc, f1

def objective(trial, ehr_data, txt_data, y, splits, pca_components=64, device="cuda"):
    # Hyperparameters - expanded search space
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": 30  # Slightly longer for better convergence
    }
    
    # PCA inside objective (or cached outside)
    # Caching outside is better for speed, but inside is cleaner for HPO if PCA dim is tuned.
    # Here we fix PCA dim to 64 for speed as per previous experiments.
    # txt_data is already loaded (full raw)
    
    # Apply PCA on the fly (it's fast for 11k rows)
    from sklearn.decomposition import PCA
    train_mask = splits == 'train'
    pca = PCA(n_components=pca_components)
    # Fit PCA on train only to avoid leakage, then transform all
    pca.fit(txt_data[train_mask])
    txt_reduced = pca.transform(txt_data)
    
    train_ds = GatedFusionDataset(ehr_data[splits=='train'], txt_reduced[splits=='train'], y[splits=='train'])
    val_ds = GatedFusionDataset(ehr_data[splits=='val'], txt_reduced[splits=='val'], y[splits=='val'])
    
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"])
    
    model = GatedFusionModel(
        ehr_dim=ehr_data.shape[1],
        txt_dim=txt_reduced.shape[1],
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"]
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, len(train_loader) * params["epochs"])
    
    best_val_auc = 0
    
    # Pruning
    for epoch in range(params["epochs"]):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        
        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_val_auc = max(best_val_auc, val_auc)
        
    return best_val_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default=None,
        help="Optional directory containing structured_ehr_embeddings.npz and structured_ehr_mapping.csv. "
        "If not set, defaults to GRU-based embeddings under data/interim/ehr_long_los/embeddings/.",
    )
    args = parser.parse_args()
    
    mlflow.set_experiment("mimic_cardiorenal_gated_hpo")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loader = FusionDataLoader(embedding_dir=args.embedding_dir)
    ehr_data, txt_data, y, splits = loader.load_data(pca_components=0) # Load raw, PCA inside
    
    study = optuna.create_study(direction="maximize")
    
    def obj_wrapper(trial):
        return objective(trial, ehr_data, txt_data, y, splits, args.pca, device)
    
    study.optimize(obj_wrapper, n_trials=args.trials)
    
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best AUC: {study.best_value}")
    
    # Train best model on train+val, evaluate on test
    best_params = study.best_params
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    train_mask = splits == 'train'
    val_mask = splits == 'val'
    test_mask = splits == 'test'

    # Fit PCA on train+val, then transform all
    pca = PCA(n_components=args.pca)
    pca.fit(txt_data[train_mask | val_mask])
    txt_reduced = pca.transform(txt_data)

    # Build train/val_internal/test splits for final training
    ehr_train_full = ehr_data[train_mask | val_mask]
    txt_train_full = txt_reduced[train_mask | val_mask]
    y_train_full = y[train_mask | val_mask]

    ehr_test = ehr_data[test_mask]
    txt_test = txt_reduced[test_mask]
    y_test = y[test_mask]

    ehr_tr, ehr_val_int, txt_tr, txt_val_int, y_tr, y_val_int = train_test_split(
        ehr_train_full,
        txt_train_full,
        y_train_full,
        test_size=0.1,
        random_state=42,
    )

    train_ds = GatedFusionDataset(ehr_tr, txt_tr, y_tr)
    val_ds = GatedFusionDataset(ehr_val_int, txt_val_int, y_val_int)
    test_ds = GatedFusionDataset(ehr_test, txt_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = GatedFusionModel(
        ehr_dim=ehr_train_full.shape[1],
        txt_dim=txt_train_full.shape[1],
        hidden_dim=best_params["hidden_dim"],
        dropout=best_params["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 0, len(train_loader) * 30
    )

    best_state = None
    best_val_auc = 0.0

    # Log best to MLflow, including final test metrics
    with mlflow.start_run(run_name="Best_HPO_Gated"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_auc", study.best_value)

        for epoch in range(30):
            train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_auc, _, _ = evaluate(model, val_loader, criterion, device)
            mlflow.log_metric("val_auc", val_auc, step=epoch)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = model.state_dict()

        if best_state is not None:
            model.load_state_dict(best_state)

        test_auc, test_auprc, test_f1 = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics(
            {
                "test_auc": test_auc,
                "test_auprc": test_auprc,
                "test_f1": test_f1,
            }
        )

        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()
