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

class TemporalAttentionFusionDataset(Dataset):
    def __init__(self, ehr_seq, txt_data, y):
        # ehr_seq: (Batch, SeqLen, FeatDim)
        # txt_data: (Batch, TxtDim)
        self.ehr_seq = torch.FloatTensor(ehr_seq)
        self.txt = torch.FloatTensor(txt_data)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.ehr_seq[idx], self.txt[idx], self.y[idx]

class TemporalAttentionFusionModel(nn.Module):
    def __init__(self, ehr_dim, txt_dim, embed_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Projectors
        # EHR Sequence is already (Batch, SeqLen, ehr_dim)
        # We project it to (Batch, SeqLen, embed_dim)
        self.ehr_proj = nn.Linear(ehr_dim, embed_dim)
        
        # Text is (Batch, txt_dim) -> (Batch, 1, embed_dim)
        self.txt_proj = nn.Linear(txt_dim, embed_dim)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Cross-Modal Attention
        # Query: Text (we want to know which parts of EHR sequence explain the text context? Or vice versa?)
        # Actually, we want to predict readmission.
        # Hypothesis: "Look at the timeline (Key/Value) and find events relevant to the global context (Query)"
        # But global context (Text) is static.
        # Let's try: Query = Text, Key/Value = EHR Sequence.
        # Output: "Summarized EHR timeline weighted by relevance to Text"
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        # We also need a self-attended representation of the EHR sequence itself (standard RNN/Attention aggregation)
        # But we already have the GRU sequence.
        # Let's add a simple attention pooling for EHR as well?
        # Or just use the Cross-Attn output + Text.
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), # Concat [Text, Attended_EHR]
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, x_ehr_seq, x_txt):
        # x_ehr_seq: (Batch, SeqLen, ehr_dim)
        # x_txt: (Batch, txt_dim)
        
        # Project
        h_ehr = self.ehr_proj(x_ehr_seq) # (Batch, SeqLen, embed_dim)
        h_ehr = self.ln1(h_ehr)
        
        h_txt = self.txt_proj(x_txt).unsqueeze(1) # (Batch, 1, embed_dim)
        h_txt = self.ln2(h_txt)
        
        # Cross Attention
        # Query: h_txt, Key: h_ehr, Value: h_ehr
        # "For this patient context (Text), what matters in the timeline (EHR)?"
        attn_out, attn_weights = self.cross_attn(h_txt, h_ehr, h_ehr) 
        # attn_out: (Batch, 1, embed_dim)
        
        # Fusion: Concat Text + Attended EHR context
        # (Batch, 2*embed_dim)
        fused = torch.cat([h_txt.squeeze(1), attn_out.squeeze(1)], dim=1)
        
        logits = self.classifier(fused)
        
        return logits, attn_weights

class FusionDataLoader:
    def __init__(self, base_dir=".", embedding_dir=None):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"

        # Use custom embedding directory if provided (e.g., for transformer embeddings)
        if embedding_dir:
            embeddings_path = Path(embedding_dir)
        else:
            # Default to old GRU embeddings location
            embeddings_path = self.base_dir / "data/interim/ehr_long_los/embeddings"

        # Note the _seq file
        self.structured_seq_path = embeddings_path / "structured_ehr_embeddings_seq.npz"
        self.structured_mapping_path = embeddings_path / "structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"

    def load_data(self, pca_components=0):
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        y = cohort['readmitted_within_window'].astype(int).values
        splits = cohort['split'].values
        ids = cohort['hadm_id'].values
        
        # 1. EHR Sequence
        # This function needs to load the 3D tensor
        ehr_seq = self._load_structured_seq(ids)
        
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
        
        logger.info(f"EHR Seq Shape: {ehr_seq.shape}, Text Dim: {txt_data.shape[1]}")
        
        return ehr_seq, txt_data, y, splits, pca_explained

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

    def _load_structured_seq(self, target_ids):
        if not self.structured_seq_path.exists():
            raise FileNotFoundError(f"Missing seq data: {self.structured_seq_path}")
            
        mapping = pd.read_csv(self.structured_mapping_path)
        data = np.load(self.structured_seq_path)
        embeddings_seq = data['embeddings'] # (TotalPatients, SeqLen, Dim)
        
        hid_to_idx = {}
        for _, row in mapping.iterrows():
            parts = str(row['node_name']).split('_')
            if len(parts) == 2: hid_to_idx[int(parts[1])] = row['row_idx']
            
        seq_len = embeddings_seq.shape[1]
        dim = embeddings_seq.shape[2]
        
        aligned = []
        for tid in target_ids:
            if tid in hid_to_idx:
                aligned.append(embeddings_seq[hid_to_idx[tid]])
            else:
                aligned.append(np.zeros((seq_len, dim)))
                
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
    f1 = f1_score(all_targets, (all_preds > 0.5).astype(int))
    
    return total_loss / len(loader), auc, auprc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_components", type=int, default=64, help="PCA for text (0=disable)")
    parser.add_argument("--embedding_dir", type=str, default=None, help="Path to structured embeddings directory (e.g., embeddings_transformer)")
    args = parser.parse_args()

    # MLflow Robustness
    if "MLFLOW_RUN_ID" in os.environ: del os.environ["MLFLOW_RUN_ID"]
    if mlflow.active_run(): mlflow.end_run()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mlflow.set_experiment("mimic_cardiorenal_temporal_attention")
    
    loader = FusionDataLoader(embedding_dir=args.embedding_dir)
    ehr_seq, txt_data, y, splits, pca_explained = loader.load_data(pca_components=args.pca_components)
    
    train_idx = splits == 'train'
    val_idx = splits == 'val'
    test_idx = splits == 'test'
    
    train_ds = TemporalAttentionFusionDataset(ehr_seq[train_idx], txt_data[train_idx], y[train_idx])
    val_ds = TemporalAttentionFusionDataset(ehr_seq[val_idx], txt_data[val_idx], y[val_idx])
    test_ds = TemporalAttentionFusionDataset(ehr_seq[test_idx], txt_data[test_idx], y[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    model = TemporalAttentionFusionModel(
        ehr_dim=ehr_seq.shape[2],
        txt_dim=txt_data.shape[1],
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
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
    patience = 8
    no_improve = 0
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(vars(args))
        if pca_explained: mlflow.log_metric("pca_explained", pca_explained)
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_auc, val_auprc, val_f1 = evaluate(model, val_loader, criterion, device)
            
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
        _, test_auc, test_auprc, test_f1 = evaluate(model, test_loader, criterion, device)
        logger.info(f"Final Test AUC: {test_auc:.4f} | AUPRC: {test_auprc:.4f} | F1: {test_f1:.4f}")
        mlflow.log_metrics({"test_auc": test_auc, "test_auprc": test_auprc, "test_f1": test_f1})

if __name__ == "__main__":
    main()
