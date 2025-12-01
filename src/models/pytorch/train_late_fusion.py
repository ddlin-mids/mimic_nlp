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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LateFusionDataset(Dataset):
    def __init__(self, ehr_data, txt_data, y):
        self.ehr = torch.FloatTensor(ehr_data)
        self.txt = torch.FloatTensor(txt_data)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            'ehr': self.ehr[idx],
            'txt': self.txt[idx]
        }, self.y[idx]

class LateFusionModel(nn.Module):
    def __init__(self, ehr_dim, txt_dim, hidden_dim=128, dropout=0.5):
        super().__init__()
        
        # Tower A: Structured EHR (Demo + Static + GRU)
        # Process dense, high-signal features
        self.ehr_tower = nn.Sequential(
            nn.Linear(ehr_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2) # Less dropout for structured data
        )
        
        # Tower B: Text Embeddings (Notes)
        # Bottleneck for high-dimensional, noisy text features
        self.txt_tower = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) # Higher dropout for text
        )
        
        # Gating / Fusion Head
        # Concatenate processed features
        fusion_dim = hidden_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_ehr, x_txt):
        ehr_emb = self.ehr_tower(x_ehr)
        txt_emb = self.txt_tower(x_txt)
        
        # Concatenate
        combined = torch.cat([ehr_emb, txt_emb], dim=1)
        
        return self.classifier(combined)

class SeparateDataLoader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        self.static_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
        self.structured_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
        self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"

    def load_data(self):
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        y = cohort['readmitted_within_window'].astype(int).values
        splits = cohort['split'].values
        ids = cohort['hadm_id'].values
        
        # 1. EHR Group (Demo + Static + GRU)
        demo = self._process_demographics(cohort)
        static = self._load_aligned_embeddings(self.static_ehr_path, ids)
        struct = self._load_structured_ehr(ids)
        
        ehr_data = np.hstack([demo, static, struct])
        
        # 2. Text Group (Discharge + Radiology)
        disch = self._load_aligned_embeddings(self.discharge_path, ids)
        rad = self._load_radiology_embeddings(ids)
        
        txt_data = np.hstack([disch, rad])
        
        logger.info(f"EHR Dim: {ehr_data.shape[1]}, Text Dim: {txt_data.shape[1]}")
        
        return ehr_data, txt_data, y, splits

    def _process_demographics(self, df):
        demo_df = df[['age_at_admit', 'gender', 'race']].copy()
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), ['age_at_admit']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['gender', 'race'])
        ])
        return preprocessor.fit_transform(demo_df)

    def _load_aligned_embeddings(self, filepath, target_ids):
        if not filepath.exists(): return np.zeros((len(target_ids), 0))
        data = np.load(filepath)
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

    def _load_structured_ehr(self, target_ids):
        mapping = pd.read_csv(self.structured_mapping_path)
        embeddings = np.load(self.structured_ehr_path)['embeddings']
        hid_to_idx = {}
        for _, row in mapping.iterrows():
            parts = str(row['node_name']).split('_')
            if len(parts) == 2: hid_to_idx[int(parts[1])] = row['row_idx']
        dim = embeddings.shape[1]
        return np.array([embeddings[hid_to_idx[tid]] if tid in hid_to_idx else np.zeros(dim) for tid in target_ids])

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for inputs, y_batch in loader:
        x_ehr = inputs['ehr'].to(device)
        x_txt = inputs['txt'].to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(x_ehr, x_txt)
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
    
    for inputs, y_batch in loader:
        x_ehr = inputs['ehr'].to(device)
        x_txt = inputs['txt'].to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        
        logits = model(x_ehr, x_txt)
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
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mlflow.set_experiment("mimic_cardiorenal_late_fusion")
    
    loader = SeparateDataLoader()
    ehr_data, txt_data, y, splits = loader.load_data()
    
    train_idx = splits == 'train'
    val_idx = splits == 'val'
    test_idx = splits == 'test'
    
    train_ds = LateFusionDataset(ehr_data[train_idx], txt_data[train_idx], y[train_idx])
    val_ds = LateFusionDataset(ehr_data[val_idx], txt_data[val_idx], y[val_idx])
    test_ds = LateFusionDataset(ehr_data[test_idx], txt_data[test_idx], y[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    model = LateFusionModel(
        ehr_dim=ehr_data.shape[1],
        txt_dim=txt_data.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, 200, len(train_loader)*args.epochs)
    
    best_auc = 0.0
    best_state = None
    patience = 12
    no_improve = 0
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_auc, val_auprc, _, _ = evaluate(model, val_loader, criterion, device)
            
            logger.info(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
            mlflow.log_metrics({"val_auc": val_auc, "train_loss": train_loss}, step=epoch)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.info("Early stopping")
                break
        
        if best_state: model.load_state_dict(best_state)
        
        _, test_auc, test_auprc, preds, targets = evaluate(model, test_loader, criterion, device)
        acc = accuracy_score(targets, (preds > 0.5).astype(int))
        
        logger.info(f"Final Test AUC: {test_auc:.4f} | AUPRC: {test_auprc:.4f}")
        mlflow.log_metrics({"test_auc": test_auc, "test_auprc": test_auprc, "test_acc": acc})
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()
