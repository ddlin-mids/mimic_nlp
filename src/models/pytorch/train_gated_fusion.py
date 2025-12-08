import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_recall_curve, precision_score, recall_score
from transformers import get_cosine_schedule_with_warmup
import mlflow
import mlflow.pytorch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Find threshold that maximizes F1 on validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )
    # precision_recall_curve returns n+1 precision/recall but n thresholds
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

# -----------------------------------------------------------------------------
# Gated Fusion Layer
# -----------------------------------------------------------------------------
class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit (GMU) for fusing two modalities.
    It learns a gate z to weight the contribution of each modality.
    h = z * h_A + (1-z) * h_B
    """
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.linear_z = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_a, x_b):
        # x_a, x_b: (Batch, dim)
        
        # Concat to predict gate
        combined = torch.cat([x_a, x_b], dim=1)
        z = self.sigmoid(self.linear_z(combined))
        
        # Weighted sum
        # Using z for A and (1-z) for B
        # Often A is tanh(W_a x_a) but here we assume inputs are already projected/embedded
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
        
        # Project both to same hidden_dim
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
        
        # Gating Mechanism
        self.gated_fusion = GatedMultimodalUnit(hidden_dim, dropout=dropout)
        
        # Classifier
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
            self.structured_path = Path(embedding_dir) / "structured_ehr_embeddings.npz"
            self.structured_mapping_path = Path(embedding_dir) / "structured_ehr_mapping.csv"
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
        
        # 1. Structured Data (2D Summary)
        ehr_data = self._load_aligned_embeddings(self.structured_path, ids, mapping_file=self.structured_mapping_path)
        
        # 2. Text Group
        disch = self._load_aligned_embeddings(self.discharge_path, ids)
        rad = self._load_radiology_embeddings(ids)
        txt_data = np.hstack([disch, rad])
        
        pca_explained = None
        if pca_components > 0:
            logger.info(f"Applying PCA (n={pca_components}) to text...")
            train_idx = splits == 'train'
            pca = PCA(n_components=pca_components)
            pca.fit(txt_data[train_idx])
            pca_explained = np.sum(pca.explained_variance_ratio_)
            txt_data = pca.transform(txt_data)
            logger.info(f"PCA Variance: {pca_explained:.4f}")
        
        return ehr_data, txt_data, y, splits, pca_explained

    def _load_aligned_embeddings(self, filepath, target_ids, mapping_file=None):
        if not filepath.exists(): 
            logger.warning(f"File not found: {filepath}")
            return np.zeros((len(target_ids), 0))
            
        data = np.load(filepath)
        
        if mapping_file:
            # Structured EHR case
            mapping = pd.read_csv(mapping_file)
            embeddings = data['embeddings']
            
            # Create ID map
            hid_to_idx = {}
            for _, row in mapping.iterrows():
                parts = str(row['node_name']).split('_')
                if len(parts) == 2: hid_to_idx[int(parts[1])] = row['row_idx']
            
            dim = embeddings.shape[1]
            aligned = []
            for tid in target_ids:
                if tid in hid_to_idx:
                    aligned.append(embeddings[hid_to_idx[tid]])
                else:
                    aligned.append(np.zeros(dim))
            return np.array(aligned)
            
        else:
            # Notes case
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
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for x_ehr, x_txt, y in loader:
        x_ehr, x_txt, y = x_ehr.to(device), x_txt.to(device), y.to(device).unsqueeze(1)
        
        logits, _ = model(x_ehr, x_txt)
        loss = criterion(logits, y)
        
        probs = torch.sigmoid(logits).cpu().numpy()
        
        total_loss += loss.item()
        all_preds.extend(probs)
        all_targets.extend(y.cpu().numpy())
        
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    auc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    preds_binary = (all_preds > threshold).astype(int)
    f1 = f1_score(all_targets, preds_binary, zero_division=0)
    prec = precision_score(all_targets, preds_binary, zero_division=0)
    rec = recall_score(all_targets, preds_binary, zero_division=0)
    acc = accuracy_score(all_targets, preds_binary)
    
    return total_loss / len(loader), auc, auprc, f1, prec, rec, acc, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_components", type=int, default=64)
    parser.add_argument("--embedding_dir", type=str, default=None)
    parser.add_argument("--embedding_type", type=str, default="GRU", choices=["GRU", "Transformer"],
                        help="Type of embeddings being used (GRU or Transformer)")
    args = parser.parse_args()

    if "MLFLOW_RUN_ID" in os.environ: del os.environ["MLFLOW_RUN_ID"]
    if mlflow.active_run(): mlflow.end_run()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mlflow.set_experiment(f"mimic_cardiorenal_gated_fusion_{args.embedding_type.lower()}")
    
    loader = FusionDataLoader(embedding_dir=args.embedding_dir)
    ehr_data, txt_data, y, splits, pca_explained = loader.load_data(pca_components=args.pca_components)
    
    train_idx = splits == 'train'
    val_idx = splits == 'val'
    test_idx = splits == 'test'
    
    train_ds = GatedFusionDataset(ehr_data[train_idx], txt_data[train_idx], y[train_idx])
    val_ds = GatedFusionDataset(ehr_data[val_idx], txt_data[val_idx], y[val_idx])
    test_ds = GatedFusionDataset(ehr_data[test_idx], txt_data[test_idx], y[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    model = GatedFusionModel(
        ehr_dim=ehr_data.shape[1],
        txt_dim=txt_data.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(len(train_loader) * args.epochs * 0.1),
        num_training_steps=len(train_loader) * args.epochs
    )
    
    best_auc = 0.0
    best_state = None
    best_threshold = 0.5
    patience = 8
    no_improve = 0
    
    with mlflow.start_run(nested=True):
        mlflow.set_tag("embedding_type", args.embedding_type)
        mlflow.set_tag("model_type", "gated_fusion")
        mlflow.log_params(vars(args))
        if pca_explained: mlflow.log_metric("pca_explained", pca_explained)
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_auc, val_auprc, val_f1, _, _, _, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
            
            logger.info(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
            mlflow.log_metrics({"val_auc": val_auc, "train_loss": train_loss}, step=epoch)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
                # Find optimal threshold on validation set
                best_threshold, val_f1_at_thresh = find_optimal_threshold(val_targets, val_preds)
                logger.info(f"  New best! Optimal threshold: {best_threshold:.3f} (Val F1: {val_f1_at_thresh:.3f})")
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.info("Early stopping")
                break
                
        if best_state: model.load_state_dict(best_state)
        
        # Evaluate test set with optimal threshold
        _, test_auc, test_auprc, test_f1, test_prec, test_rec, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device, threshold=best_threshold
        )
        
        logger.info(f"Final Test AUC: {test_auc:.4f} | AUPRC: {test_auprc:.4f} | F1: {test_f1:.4f}")
        logger.info(f"Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | Threshold: {best_threshold:.3f}")
        
        mlflow.log_metrics({
            "test_auc": test_auc, 
            "test_auprc": test_auprc, 
            "test_f1": test_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_acc": test_acc,
            "optimal_threshold": best_threshold
        })

if __name__ == "__main__":
    main()
