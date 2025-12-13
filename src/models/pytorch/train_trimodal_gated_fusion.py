"""
Trimodal Gated Fusion Model for 30-day Readmission Prediction

This model fuses three modalities:
1. Structured EHR embeddings (from pre-trained GRU encoder)
2. Clinical text embeddings (discharge + radiology notes)
3. Lab trajectory features (135 extracted features)

Uses a hierarchical gated fusion approach:
- First GMU: EHR + Labs → EHR-Labs fusion
- Second GMU: EHR-Labs + Text → Final fusion
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    accuracy_score, precision_recall_curve, precision_score, recall_score
)
from transformers import get_cosine_schedule_with_warmup
import mlflow
import mlflow.pytorch

# Configure logging with flush to stdout
import sys
# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[FlushHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Find threshold that maximizes F1 on validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# -----------------------------------------------------------------------------
# Focal Loss for Class Imbalance
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# -----------------------------------------------------------------------------
# Gated Multimodal Unit
# -----------------------------------------------------------------------------
class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit for fusing two modalities.
    h = z * h_A + (1-z) * h_B
    """
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
class TrimodalDataset(Dataset):
    def __init__(self, ehr_data, txt_data, lab_data, y):
        self.ehr = torch.FloatTensor(ehr_data)
        self.txt = torch.FloatTensor(txt_data)
        self.lab = torch.FloatTensor(lab_data)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.ehr[idx], self.txt[idx], self.lab[idx], self.y[idx]


# -----------------------------------------------------------------------------
# Trimodal Gated Fusion Model
# -----------------------------------------------------------------------------
class TrimodalGatedFusion(nn.Module):
    """
    Hierarchical gated fusion for three modalities.
    
    Architecture:
    - EHR Projection → h_ehr
    - Lab Projection → h_lab
    - Text Projection → h_txt
    - GMU1: h_ehr ⊕ h_lab → h_ehr_lab
    - GMU2: h_ehr_lab ⊕ h_txt → h_fused
    - Classifier: h_fused → prediction
    """
    def __init__(self, ehr_dim, txt_dim, lab_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        
        # Projection layers
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
        
        self.lab_proj = nn.Sequential(
            nn.Linear(lab_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Hierarchical Gated Fusion
        self.gmu_ehr_lab = GatedMultimodalUnit(hidden_dim, dropout=dropout)
        self.gmu_final = GatedMultimodalUnit(hidden_dim, dropout=dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_ehr, x_txt, x_lab):
        # Project all modalities
        h_ehr = self.ehr_proj(x_ehr)
        h_txt = self.txt_proj(x_txt)
        h_lab = self.lab_proj(x_lab)
        
        # Hierarchical fusion
        h_ehr_lab, gate1 = self.gmu_ehr_lab(h_ehr, h_lab)
        h_fused, gate2 = self.gmu_final(h_ehr_lab, h_txt)
        
        # Classification
        logits = self.classifier(h_fused)
        
        return logits, (gate1, gate2)


# -----------------------------------------------------------------------------
# Alternative: Attention-based Trimodal Fusion
# -----------------------------------------------------------------------------
class AttentionTrimodalFusion(nn.Module):
    """
    Attention-based fusion for three modalities.
    Uses self-attention across the three modality embeddings.
    """
    def __init__(self, ehr_dim, txt_dim, lab_dim, hidden_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Projection layers
        self.ehr_proj = nn.Linear(ehr_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.lab_proj = nn.Linear(lab_dim, hidden_dim)
        
        # Modality embeddings (learnable)
        self.modality_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_ehr, x_txt, x_lab):
        batch_size = x_ehr.size(0)
        
        # Project all modalities
        h_ehr = self.ehr_proj(x_ehr)
        h_txt = self.txt_proj(x_txt)
        h_lab = self.lab_proj(x_lab)
        
        # Stack as sequence: (batch, 3, hidden_dim)
        x = torch.stack([h_ehr, h_txt, h_lab], dim=1)
        
        # Add modality embeddings
        x = x + self.modality_embed.unsqueeze(0)
        
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        # Pool across modalities (mean)
        h_fused = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(h_fused)
        
        return logits, attn_weights


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
class TrimodalDataLoader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        self.structured_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
        self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"
        self.lab_path = self.base_dir / "data/interim/ehr_long_los/lab_features/lab_features.npz"

    def load_data(self, pca_text=64, pca_lab=32):
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        y = cohort['readmitted_within_window'].astype(int).values
        splits = cohort['split'].values
        ids = cohort['hadm_id'].values
        
        # 1. Structured EHR Embeddings
        ehr_data = self._load_aligned_embeddings(self.structured_path, ids, mapping_file=self.structured_mapping_path)
        logger.info(f"Loaded EHR embeddings: {ehr_data.shape}")
        
        # 2. Text Embeddings
        disch = self._load_aligned_embeddings(self.discharge_path, ids)
        rad = self._load_radiology_embeddings(ids)
        txt_data = np.hstack([disch, rad])
        logger.info(f"Loaded text embeddings: {txt_data.shape}")
        
        # 3. Lab Features
        lab_data = self._load_lab_features(ids)
        logger.info(f"Loaded lab features: {lab_data.shape}")
        
        # Apply PCA to text if requested
        train_idx = splits == 'train'
        pca_text_explained = None
        if pca_text > 0 and txt_data.shape[1] > pca_text:
            logger.info(f"Applying PCA (n={pca_text}) to text...")
            pca = PCA(n_components=pca_text)
            pca.fit(txt_data[train_idx])
            pca_text_explained = np.sum(pca.explained_variance_ratio_)
            txt_data = pca.transform(txt_data)
            logger.info(f"Text PCA Variance: {pca_text_explained:.4f}")
            
        # Scale lab features
        logger.info("Scaling lab features...")
        scaler = StandardScaler()
        scaler.fit(lab_data[train_idx])
        lab_data = scaler.transform(lab_data)
        
        # Apply PCA to labs if requested (optional dimensionality reduction)
        pca_lab_explained = None
        if pca_lab > 0 and lab_data.shape[1] > pca_lab:
            logger.info(f"Applying PCA (n={pca_lab}) to lab features...")
            pca = PCA(n_components=pca_lab)
            pca.fit(lab_data[train_idx])
            pca_lab_explained = np.sum(pca.explained_variance_ratio_)
            lab_data = pca.transform(lab_data)
            logger.info(f"Lab PCA Variance: {pca_lab_explained:.4f}")
        
        return ehr_data, txt_data, lab_data, y, splits, {
            'pca_text_explained': pca_text_explained,
            'pca_lab_explained': pca_lab_explained
        }

    def _load_aligned_embeddings(self, filepath, target_ids, mapping_file=None):
        if not filepath.exists(): 
            logger.warning(f"File not found: {filepath}")
            return np.zeros((len(target_ids), 0))
            
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
                if tid in hid_to_idx:
                    aligned.append(embeddings[hid_to_idx[tid]])
                else:
                    aligned.append(np.zeros(dim))
            return np.array(aligned)
        else:
            id_key = 'hadm_ids' if 'hadm_ids' in data else 'ids'
            source_ids = data[id_key]
            source_embeds = data['embeddings']
            id_map = {int(k) if isinstance(k, (float, np.floating)) else k: v for k, v in zip(source_ids, source_embeds)}
            dim = source_embeds.shape[1]
            return np.array([id_map.get(tid, np.zeros(dim)) for tid in target_ids])

    def _load_radiology_embeddings(self, target_ids):
        if not self.radiology_path.exists(): 
            return np.zeros((len(target_ids), 0))
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
    
    def _load_lab_features(self, target_ids):
        if not self.lab_path.exists():
            logger.warning(f"Lab features not found: {self.lab_path}")
            return np.zeros((len(target_ids), 135))  # Default 135 features
            
        data = np.load(self.lab_path)
        features = data['embeddings']  # Key is 'embeddings' not 'features'
        hadm_ids = data['hadm_ids']
        
        # Create mapping
        id_to_features = {int(hid): feat for hid, feat in zip(hadm_ids, features)}
        dim = features.shape[1]
        
        aligned = []
        for tid in target_ids:
            if tid in id_to_features:
                aligned.append(id_to_features[tid])
            else:
                aligned.append(np.zeros(dim))
        return np.array(aligned)


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for x_ehr, x_txt, x_lab, y in loader:
        x_ehr = x_ehr.to(device)
        x_txt = x_txt.to(device)
        x_lab = x_lab.to(device)
        y = y.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits, _ = model(x_ehr, x_txt, x_lab)
        loss = criterion(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    
    for x_ehr, x_txt, x_lab, y in loader:
        x_ehr = x_ehr.to(device)
        x_txt = x_txt.to(device)
        x_lab = x_lab.to(device)
        y = y.to(device).unsqueeze(1)
        
        logits, _ = model(x_ehr, x_txt, x_lab)
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
    print(">>> Starting Trimodal Gated Fusion Training...", flush=True)
    parser = argparse.ArgumentParser(description="Trimodal Gated Fusion Training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_text", type=int, default=64, help="PCA components for text (0=no PCA)")
    parser.add_argument("--pca_lab", type=int, default=0, help="PCA components for labs (0=no PCA)")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use focal loss")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--model_type", type=str, default="gated", choices=["gated", "attention"])
    args = parser.parse_args()

    # Clean MLflow state
    if "MLFLOW_RUN_ID" in os.environ: 
        del os.environ["MLFLOW_RUN_ID"]
    if mlflow.active_run(): 
        mlflow.end_run()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    mlflow.set_experiment("mimic_cardiorenal_trimodal_fusion")
    print(">>> MLflow experiment set, loading data...", flush=True)
    
    # Load data
    loader = TrimodalDataLoader()
    ehr_data, txt_data, lab_data, y, splits, pca_info = loader.load_data(
        pca_text=args.pca_text, 
        pca_lab=args.pca_lab
    )
    
    train_idx = splits == 'train'
    val_idx = splits == 'val'
    test_idx = splits == 'test'
    
    # Create datasets
    train_ds = TrimodalDataset(
        ehr_data[train_idx], txt_data[train_idx], lab_data[train_idx], y[train_idx]
    )
    val_ds = TrimodalDataset(
        ehr_data[val_idx], txt_data[val_idx], lab_data[val_idx], y[val_idx]
    )
    test_ds = TrimodalDataset(
        ehr_data[test_idx], txt_data[test_idx], lab_data[test_idx], y[test_idx]
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"EHR dim: {ehr_data.shape[1]}, Text dim: {txt_data.shape[1]}, Lab dim: {lab_data.shape[1]}")
    
    # Create model
    if args.model_type == "gated":
        model = TrimodalGatedFusion(
            ehr_dim=ehr_data.shape[1],
            txt_dim=txt_data.shape[1],
            lab_dim=lab_data.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        ).to(device)
    else:
        model = AttentionTrimodalFusion(
            ehr_dim=ehr_data.shape[1],
            txt_dim=txt_data.shape[1],
            lab_dim=lab_data.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        ).to(device)
    
    # Loss function
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        logger.info(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        # Weighted BCE to handle class imbalance
        pos_weight = torch.tensor([(1 - y[train_idx].mean()) / y[train_idx].mean()]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"Using Weighted BCE (pos_weight={pos_weight.item():.2f})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(len(train_loader) * args.epochs * 0.1),
        num_training_steps=len(train_loader) * args.epochs
    )
    
    best_auc = 0.0
    best_state = None
    best_threshold = 0.5
    patience = 10
    no_improve = 0
    
    with mlflow.start_run(nested=True):
        mlflow.set_tag("model_type", f"trimodal_{args.model_type}")
        mlflow.set_tag("loss_type", "focal" if args.use_focal_loss else "weighted_bce")
        mlflow.log_params(vars(args))
        
        if pca_info['pca_text_explained']: 
            mlflow.log_metric("pca_text_explained", pca_info['pca_text_explained'])
        if pca_info['pca_lab_explained']: 
            mlflow.log_metric("pca_lab_explained", pca_info['pca_lab_explained'])
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_auc, val_auprc, val_f1, _, _, _, val_preds, val_targets = evaluate(
                model, val_loader, criterion, device
            )
            
            logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f}")
            mlflow.log_metrics({
                "val_auc": val_auc, 
                "val_auprc": val_auprc,
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=epoch)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_threshold, val_f1_at_thresh = find_optimal_threshold(val_targets, val_preds)
                logger.info(f"  ★ New best! Threshold: {best_threshold:.3f} (Val F1: {val_f1_at_thresh:.3f})")
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        # Load best model and evaluate on test set
        if best_state: 
            model.load_state_dict(best_state)
            model.to(device)
        
        _, test_auc, test_auprc, test_f1, test_prec, test_rec, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device, threshold=best_threshold
        )
        
        logger.info("=" * 60)
        logger.info(f"Final Test Results:")
        logger.info(f"  AUC:       {test_auc:.4f}")
        logger.info(f"  AUPRC:     {test_auprc:.4f}")
        logger.info(f"  F1:        {test_f1:.4f}")
        logger.info(f"  Precision: {test_prec:.4f}")
        logger.info(f"  Recall:    {test_rec:.4f}")
        logger.info(f"  Accuracy:  {test_acc:.4f}")
        logger.info(f"  Threshold: {best_threshold:.3f}")
        logger.info("=" * 60)
        
        mlflow.log_metrics({
            "test_auc": test_auc, 
            "test_auprc": test_auprc, 
            "test_f1": test_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_acc": test_acc,
            "optimal_threshold": best_threshold,
            "best_val_auc": best_auc
        })


if __name__ == "__main__":
    main()
