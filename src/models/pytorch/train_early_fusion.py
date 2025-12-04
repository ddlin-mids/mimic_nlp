import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReadmissionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class FusionDataLoader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        self.static_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
        self.structured_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
        self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"

    def load_and_prep_data(self, pca_components=0):
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        cohort['target'] = cohort['readmitted_within_window'].astype(int)
        splits = cohort['split'].values
        
        logger.info("Processing modalities...")
        demo_features = self._process_demographics(cohort)
        static_embeds = self._load_aligned_embeddings(self.static_ehr_path, cohort['hadm_id'].values)
        struct_embeds = self._load_structured_ehr(cohort['hadm_id'].values)
        discharge_embeds = self._load_aligned_embeddings(self.discharge_path, cohort['hadm_id'].values)
        rad_embeds = self._load_radiology_embeddings(cohort['hadm_id'].values)
        
        pca_explained = None
        
        # PCA for Text
        if pca_components > 0:
            logger.info(f"Applying PCA (n={pca_components}) to text embeddings...")
            text_combined = np.hstack([discharge_embeds, rad_embeds])
            train_mask = (splits == 'train')
            
            pca = PCA(n_components=pca_components)
            pca.fit(text_combined[train_mask])
            
            pca_explained = np.sum(pca.explained_variance_ratio_)
            logger.info(f"PCA Explained Variance: {pca_explained:.4f}")
            # Do NOT log to mlflow here to avoid starting a run prematurely
            
            text_features = pca.transform(text_combined)
            X = np.hstack([demo_features, static_embeds, struct_embeds, text_features])
        else:
            X = np.hstack([demo_features, static_embeds, struct_embeds, discharge_embeds, rad_embeds])
            
        y = cohort['target'].values
        
        return X, y, splits, pca_explained

    def _process_demographics(self, df):
        demo_df = df[['age_at_admit', 'gender', 'race']].copy()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['age_at_admit']),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['gender', 'race'])
            ]
        )
        return preprocessor.fit_transform(demo_df)

    def _load_aligned_embeddings(self, filepath, target_ids):
        if not filepath.exists(): return np.zeros((len(target_ids), 0))
        data = np.load(filepath)
        id_key = 'hadm_ids' if 'hadm_ids' in data else 'ids'
        source_ids = data[id_key]
        source_embeds = data['embeddings']
        
        id_map = {int(k) if isinstance(k, (float, np.floating)) else k: v 
                  for k, v in zip(source_ids, source_embeds)}
        
        dim = source_embeds.shape[1]
        aligned = [id_map.get(tid, np.zeros(dim)) for tid in target_ids]
        return np.array(aligned)

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
            if tid in grouped:
                aligned.append(np.mean(grouped[tid], axis=0))
            else:
                aligned.append(np.zeros(dim))
        return np.array(aligned)

    def _load_structured_ehr(self, target_ids):
        mapping = pd.read_csv(self.structured_mapping_path)
        embeddings = np.load(self.structured_ehr_path)['embeddings']
        
        hid_to_idx = {}
        for _, row in mapping.iterrows():
            parts = str(row['node_name']).split('_')
            if len(parts) == 2:
                hid_to_idx[int(parts[1])] = row['row_idx']
                
        dim = embeddings.shape[1]
        aligned = [embeddings[hid_to_idx[tid]] if tid in hid_to_idx else np.zeros(dim) 
                   for tid in target_ids]
        return np.array(aligned)

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        probs = torch.sigmoid(logits).cpu().numpy()
        
        total_loss += loss.item()
        all_preds.extend(probs)
        all_targets.extend(y_batch.cpu().numpy())
        
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    auc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    
    return total_loss / len(loader), auc, auprc, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[512, 256])
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_components", type=int, default=0, help="PCA components for text (0=disable)")
    args = parser.parse_args()

    # --- MLflow Safety Check ---
    if "MLFLOW_RUN_ID" in os.environ:
        logger.warning(f"Found MLFLOW_RUN_ID in env: {os.environ['MLFLOW_RUN_ID']}. Clearing it.")
        del os.environ["MLFLOW_RUN_ID"]
    
    if mlflow.active_run():
        logger.warning(f"Ending active run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # MLflow
    mlflow.set_experiment("mimic_cardiorenal_readmission_pytorch")
    
    # Load Data
    loader = FusionDataLoader()
    X, y, splits, pca_explained = loader.load_and_prep_data(pca_components=args.pca_components)
    
    X_train = X[splits == 'train']
    y_train = y[splits == 'train']
    X_val = X[splits == 'val']
    y_val = y[splits == 'val']
    X_test = X[splits == 'test']
    y_test = y[splits == 'test']
    
    logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Datasets & Loaders
    train_ds = ReadmissionDataset(X_train, y_train)
    val_ds = ReadmissionDataset(X_val, y_val)
    test_ds = ReadmissionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model
    model = MLP(
        input_dim=X_train.shape[1], 
        hidden_dims=args.hidden_dims, 
        dropout_rate=args.dropout
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training Loop
    best_val_auc = 0.0
    best_model_state = None
    patience = 10
    no_improve = 0
    
    # Use nested=True to prevent crashes if a run is already active in the environment
    with mlflow.start_run(nested=True):
        mlflow.log_params(vars(args))
        
        if pca_explained is not None:
            mlflow.log_metric("pca_explained_variance", pca_explained)
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_auc, val_auprc, _, _ = evaluate(model, val_loader, criterion, device)
            
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_auprc": val_auprc
            }, step=epoch)
            
            # Early Stopping Check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load Best Model & Test
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        test_loss, test_auc, test_auprc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
        test_acc = accuracy_score(test_targets, (test_preds > 0.5).astype(int))
        test_f1 = f1_score(test_targets, (test_preds > 0.5).astype(int))
        
        logger.info(f"--- Final Test Results ---")
        logger.info(f"AUC: {test_auc:.4f} | AUPRC: {test_auprc:.4f} | Acc: {test_acc:.4f}")
        
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_auc": test_auc,
            "test_auprc": test_auprc,
            "test_accuracy": test_acc,
            "test_f1": test_f1
        })
        
        # Save Model
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()
