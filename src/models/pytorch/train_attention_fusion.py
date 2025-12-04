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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from transformers import get_cosine_schedule_with_warmup
import mlflow
import mlflow.pytorch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttentionFusionDataset(Dataset):
    def __init__(self, ehr_data, txt_data, y):
        self.ehr = torch.FloatTensor(ehr_data)
        self.txt = torch.FloatTensor(txt_data)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.ehr[idx], self.txt[idx], self.y[idx]

class AttentionFusionModel(nn.Module):
    def __init__(self, ehr_dim, txt_dim, embed_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Projectors to common dimension
        self.ehr_proj = nn.Sequential(
            nn.Linear(ehr_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cross-Modal Attention
        # embed_dim must be divisible by num_heads
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), # Concatenate attended features
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, x_ehr, x_txt):
        # Project to latent space
        # (Batch, Dim) -> (Batch, 1, Dim)
        h_ehr = self.ehr_proj(x_ehr).unsqueeze(1)
        h_txt = self.txt_proj(x_txt).unsqueeze(1)
        
        # Create sequence: [EHR, Text]
        # (Batch, 2, Dim)
        sequence = torch.cat([h_ehr, h_txt], dim=1)
        
        # Self-Attention
        # attn_output: (Batch, 2, Dim)
        attn_output, attn_weights = self.attention(sequence, sequence, sequence)
        
        # Flatten and Classify
        # (Batch, 2*Dim)
        flat = attn_output.reshape(attn_output.size(0), -1)
        logits = self.classifier(flat)
        
        return logits, attn_weights

class FusionDataLoader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        self.static_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
        self.structured_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
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
        
        # 1. EHR Group (Demo + Static + GRU)
        demo = self._process_demographics(cohort)
        static = self._load_aligned_embeddings(self.static_ehr_path, ids)
        struct = self._load_structured_ehr(ids)
        ehr_data = np.hstack([demo, static, struct])
        
        # 2. Text Group (Discharge + Radiology)
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
        
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    auc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    
    return total_loss / len(loader), auc, auprc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4) # Lower LR for Attention
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_components", type=int, default=0, help="PCA for text (0=disable)")
    args = parser.parse_args()

    # MLflow Robustness
    if "MLFLOW_RUN_ID" in os.environ: del os.environ["MLFLOW_RUN_ID"]
    if mlflow.active_run(): mlflow.end_run()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mlflow.set_experiment("mimic_cardiorenal_attention_fusion")
    
    loader = FusionDataLoader()
    ehr_data, txt_data, y, splits, pca_explained = loader.load_data(pca_components=args.pca_components)
    
    train_idx = splits == 'train'
    val_idx = splits == 'val'
    test_idx = splits == 'test'
    
    train_ds = AttentionFusionDataset(ehr_data[train_idx], txt_data[train_idx], y[train_idx])
    val_ds = AttentionFusionDataset(ehr_data[val_idx], txt_data[val_idx], y[val_idx])
    test_ds = AttentionFusionDataset(ehr_data[test_idx], txt_data[test_idx], y[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    model = AttentionFusionModel(
        ehr_dim=ehr_data.shape[1],
        txt_dim=txt_data.shape[1],
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine Schedule with Warmup (better for attention)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(len(train_loader) * args.epochs * 0.1), # 10% warmup
        num_training_steps=len(train_loader) * args.epochs
    )
    
    best_auc = 0.0
    best_state = None
    patience = 8
    no_improve = 0
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(vars(args))
        if pca_explained: mlflow.log_metric("pca_explained", pca_explained)
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_auc, val_auprc = evaluate(model, val_loader, criterion, device)
            
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
        _, test_auc, test_auprc = evaluate(model, test_loader, criterion, device)
        logger.info(f"Final Test AUC: {test_auc:.4f} | AUPRC: {test_auprc:.4f}")
        mlflow.log_metrics({"test_auc": test_auc, "test_auprc": test_auprc})

if __name__ == "__main__":
    main()
