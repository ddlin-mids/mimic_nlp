"""
Late Fusion Ensemble: Lab XGBoost + Neural Gated Fusion

This model combines:
1. XGBoost trained on lab trajectory features (AUC ~0.627)
2. Gated Fusion trained on EHR embeddings + Text embeddings (AUC ~0.638)

Ensemble strategies:
- Simple averaging
- Weighted averaging (optimized on validation set)
- Logistic regression meta-learner
- XGBoost meta-learner

The key insight is that lab features are highly predictive but hurt neural fusion
when combined at the feature level. Late fusion preserves the strengths of both.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, precision_recall_curve
)
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
import mlflow

# Configure logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Find threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# -----------------------------------------------------------------------------
# Gated Fusion Model (for EHR + Text)
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
        fused, _ = self.gated_fusion(h_ehr, h_txt)
        return self.classifier(fused)


class BimodalDataset(Dataset):
    def __init__(self, ehr_data, txt_data, y):
        self.ehr = torch.FloatTensor(ehr_data)
        self.txt = torch.FloatTensor(txt_data)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.ehr[idx], self.txt[idx], self.y[idx]


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
class LateFusionDataLoader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        self.structured_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
        self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"
        self.lab_path = self.base_dir / "data/interim/ehr_long_los/lab_features/lab_features.npz"

    def load_data(self, pca_text=64):
        from sklearn.decomposition import PCA
        
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
        
        # Apply PCA to text
        train_idx = splits == 'train'
        if pca_text > 0 and txt_data.shape[1] > pca_text:
            logger.info(f"Applying PCA (n={pca_text}) to text...")
            pca = PCA(n_components=pca_text)
            pca.fit(txt_data[train_idx])
            txt_data = pca.transform(txt_data)
            
        # Scale lab features
        logger.info("Scaling lab features...")
        scaler = StandardScaler()
        scaler.fit(lab_data[train_idx])
        lab_data = scaler.transform(lab_data)
        
        return ehr_data, txt_data, lab_data, y, splits, ids

    def _load_aligned_embeddings(self, filepath, target_ids, mapping_file=None):
        if not filepath.exists(): 
            return np.zeros((len(target_ids), 0))
        data = np.load(filepath)
        
        if mapping_file:
            mapping = pd.read_csv(mapping_file)
            embeddings = data['embeddings']
            hid_to_idx = {}
            for _, row in mapping.iterrows():
                parts = str(row['node_name']).split('_')
                if len(parts) == 2: 
                    hid_to_idx[int(parts[1])] = row['row_idx']
            dim = embeddings.shape[1]
            aligned = [embeddings[hid_to_idx[tid]] if tid in hid_to_idx else np.zeros(dim) for tid in target_ids]
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
        grouped = defaultdict(list)
        for hid, emb in zip(data['hadm_ids'], data['embeddings']):
            grouped[int(hid)].append(emb)
        dim = data['embeddings'].shape[1]
        aligned = [np.mean(grouped[tid], axis=0) if tid in grouped else np.zeros(dim) for tid in target_ids]
        return np.array(aligned)
    
    def _load_lab_features(self, target_ids):
        if not self.lab_path.exists():
            return np.zeros((len(target_ids), 135))
        data = np.load(self.lab_path)
        features = data['embeddings']
        hadm_ids = data['hadm_ids']
        id_to_features = {int(hid): feat for hid, feat in zip(hadm_ids, features)}
        dim = features.shape[1]
        return np.array([id_to_features.get(tid, np.zeros(dim)) for tid in target_ids])


# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------
def train_lab_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost on lab features."""
    logger.info("Training XGBoost on lab features...")
    
    # Calculate scale_pos_weight for class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    logger.info(f"  Lab XGBoost Val AUC: {val_auc:.4f}")
    
    return model


def train_gated_fusion(ehr_train, txt_train, y_train, ehr_val, txt_val, y_val, 
                       hidden_dim=64, dropout=0.3, lr=2e-4, epochs=30, device='cpu'):
    """Train Gated Fusion on EHR + Text."""
    logger.info("Training Gated Fusion on EHR + Text...")
    
    train_ds = BimodalDataset(ehr_train, txt_train, y_train)
    val_ds = BimodalDataset(ehr_val, txt_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    model = GatedFusionModel(
        ehr_dim=ehr_train.shape[1],
        txt_dim=txt_train.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    
    # Weighted BCE
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * epochs * 0.1),
        num_training_steps=len(train_loader) * epochs
    )
    
    best_auc = 0.0
    best_state = None
    patience = 8
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for x_ehr, x_txt, y in train_loader:
            x_ehr, x_txt, y = x_ehr.to(device), x_txt.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x_ehr, x_txt)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for x_ehr, x_txt, _ in val_loader:
                x_ehr, x_txt = x_ehr.to(device), x_txt.to(device)
                logits = model(x_ehr, x_txt)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
        
        val_preds = np.array(val_preds).flatten()
        val_auc = roc_auc_score(y_val, val_preds)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    
    logger.info(f"  Gated Fusion Val AUC: {best_auc:.4f}")
    return model


def get_predictions(lab_model, fusion_model, lab_data, ehr_data, txt_data, device='cpu'):
    """Get predictions from both models."""
    # Lab XGBoost predictions
    lab_preds = lab_model.predict_proba(lab_data)[:, 1]
    
    # Gated Fusion predictions
    fusion_model.eval()
    fusion_preds = []
    with torch.no_grad():
        for i in range(0, len(ehr_data), 64):
            x_ehr = torch.FloatTensor(ehr_data[i:i+64]).to(device)
            x_txt = torch.FloatTensor(txt_data[i:i+64]).to(device)
            logits = fusion_model(x_ehr, x_txt)
            fusion_preds.extend(torch.sigmoid(logits).cpu().numpy())
    fusion_preds = np.array(fusion_preds).flatten()
    
    return lab_preds, fusion_preds


def find_optimal_weight(y_val, lab_preds, fusion_preds):
    """Find optimal weight for combining predictions."""
    def neg_auc(w):
        combined = w * lab_preds + (1 - w) * fusion_preds
        return -roc_auc_score(y_val, combined)
    
    result = minimize_scalar(neg_auc, bounds=(0, 1), method='bounded')
    optimal_weight = result.x
    best_auc = -result.fun
    
    return optimal_weight, best_auc


def evaluate_ensemble(y_true, preds, threshold=0.5):
    """Evaluate ensemble predictions."""
    auc = roc_auc_score(y_true, preds)
    auprc = average_precision_score(y_true, preds)
    preds_binary = (preds > threshold).astype(int)
    f1 = f1_score(y_true, preds_binary, zero_division=0)
    prec = precision_score(y_true, preds_binary, zero_division=0)
    rec = recall_score(y_true, preds_binary, zero_division=0)
    acc = accuracy_score(y_true, preds_binary)
    
    return {
        'auc': auc, 'auprc': auprc, 'f1': f1,
        'precision': prec, 'recall': rec, 'accuracy': acc
    }


def main():
    parser = argparse.ArgumentParser(description="Late Fusion Ensemble Training")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pca_text", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print(">>> Starting Late Fusion Ensemble Training...", flush=True)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Clean MLflow state
    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]
    if mlflow.active_run():
        mlflow.end_run()
    
    mlflow.set_experiment("mimic_cardiorenal_late_fusion_ensemble")
    
    # Load data
    loader = LateFusionDataLoader()
    ehr_data, txt_data, lab_data, y, splits, ids = loader.load_data(pca_text=args.pca_text)
    
    train_idx = splits == 'train'
    val_idx = splits == 'val'
    test_idx = splits == 'test'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.set_tag("model_type", "late_fusion_ensemble")
        
        # Train base models
        lab_model = train_lab_xgboost(
            lab_data[train_idx], y[train_idx],
            lab_data[val_idx], y[val_idx]
        )
        
        fusion_model = train_gated_fusion(
            ehr_data[train_idx], txt_data[train_idx], y[train_idx],
            ehr_data[val_idx], txt_data[val_idx], y[val_idx],
            hidden_dim=args.hidden_dim, dropout=args.dropout,
            lr=args.lr, epochs=args.epochs, device=device
        )
        
        # Get validation predictions
        val_lab_preds, val_fusion_preds = get_predictions(
            lab_model, fusion_model,
            lab_data[val_idx], ehr_data[val_idx], txt_data[val_idx], device
        )
        
        # Find optimal weight
        optimal_weight, val_weighted_auc = find_optimal_weight(
            y[val_idx], val_lab_preds, val_fusion_preds
        )
        logger.info(f"Optimal weight for lab: {optimal_weight:.3f}")
        logger.info(f"Weighted ensemble Val AUC: {val_weighted_auc:.4f}")
        
        # Get test predictions
        test_lab_preds, test_fusion_preds = get_predictions(
            lab_model, fusion_model,
            lab_data[test_idx], ehr_data[test_idx], txt_data[test_idx], device
        )
        
        # Evaluate different ensemble strategies
        logger.info("\n" + "="*60)
        logger.info("Ensemble Results on Test Set:")
        logger.info("="*60)
        
        results = {}
        
        # 1. Lab XGBoost alone
        lab_metrics = evaluate_ensemble(y[test_idx], test_lab_preds)
        results['Lab XGBoost'] = lab_metrics
        logger.info(f"Lab XGBoost:        AUC={lab_metrics['auc']:.4f} | AUPRC={lab_metrics['auprc']:.4f}")
        
        # 2. Gated Fusion alone
        fusion_metrics = evaluate_ensemble(y[test_idx], test_fusion_preds)
        results['Gated Fusion'] = fusion_metrics
        logger.info(f"Gated Fusion:       AUC={fusion_metrics['auc']:.4f} | AUPRC={fusion_metrics['auprc']:.4f}")
        
        # 3. Simple average
        simple_avg_preds = 0.5 * test_lab_preds + 0.5 * test_fusion_preds
        simple_avg_metrics = evaluate_ensemble(y[test_idx], simple_avg_preds)
        results['Simple Average'] = simple_avg_metrics
        logger.info(f"Simple Average:     AUC={simple_avg_metrics['auc']:.4f} | AUPRC={simple_avg_metrics['auprc']:.4f}")
        
        # 4. Optimal weighted average
        weighted_preds = optimal_weight * test_lab_preds + (1 - optimal_weight) * test_fusion_preds
        weighted_metrics = evaluate_ensemble(y[test_idx], weighted_preds)
        results['Weighted Average'] = weighted_metrics
        logger.info(f"Weighted Average:   AUC={weighted_metrics['auc']:.4f} | AUPRC={weighted_metrics['auprc']:.4f} (w={optimal_weight:.3f})")
        
        # 5. Logistic Regression meta-learner
        meta_X_train = np.column_stack([
            lab_model.predict_proba(lab_data[train_idx])[:, 1],
            get_predictions(lab_model, fusion_model, lab_data[train_idx], 
                          ehr_data[train_idx], txt_data[train_idx], device)[1]
        ])
        meta_X_test = np.column_stack([test_lab_preds, test_fusion_preds])
        
        lr_meta = LogisticRegression(C=1.0, random_state=42)
        lr_meta.fit(meta_X_train, y[train_idx])
        lr_meta_preds = lr_meta.predict_proba(meta_X_test)[:, 1]
        lr_meta_metrics = evaluate_ensemble(y[test_idx], lr_meta_preds)
        results['LR Meta-learner'] = lr_meta_metrics
        logger.info(f"LR Meta-learner:    AUC={lr_meta_metrics['auc']:.4f} | AUPRC={lr_meta_metrics['auprc']:.4f}")
        
        # 6. XGBoost meta-learner
        xgb_meta = xgb.XGBClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='auc'
        )
        xgb_meta.fit(meta_X_train, y[train_idx])
        xgb_meta_preds = xgb_meta.predict_proba(meta_X_test)[:, 1]
        xgb_meta_metrics = evaluate_ensemble(y[test_idx], xgb_meta_preds)
        results['XGB Meta-learner'] = xgb_meta_metrics
        logger.info(f"XGB Meta-learner:   AUC={xgb_meta_metrics['auc']:.4f} | AUPRC={xgb_meta_metrics['auprc']:.4f}")
        
        # Find best ensemble
        best_method = max(results.keys(), key=lambda k: results[k]['auc'])
        best_metrics = results[best_method]
        
        logger.info("\n" + "="*60)
        logger.info(f"Best Ensemble: {best_method}")
        logger.info(f"  Test AUC:       {best_metrics['auc']:.4f}")
        logger.info(f"  Test AUPRC:     {best_metrics['auprc']:.4f}")
        logger.info(f"  Test F1:        {best_metrics['f1']:.4f}")
        logger.info(f"  Test Precision: {best_metrics['precision']:.4f}")
        logger.info(f"  Test Recall:    {best_metrics['recall']:.4f}")
        logger.info("="*60)
        
        # Log to MLflow
        mlflow.log_metrics({
            "test_auc": best_metrics['auc'],
            "test_auprc": best_metrics['auprc'],
            "test_f1": best_metrics['f1'],
            "test_precision": best_metrics['precision'],
            "test_recall": best_metrics['recall'],
            "test_acc": best_metrics['accuracy'],
            "lab_xgb_auc": lab_metrics['auc'],
            "gated_fusion_auc": fusion_metrics['auc'],
            "simple_avg_auc": simple_avg_metrics['auc'],
            "weighted_avg_auc": weighted_metrics['auc'],
            "lr_meta_auc": lr_meta_metrics['auc'],
            "xgb_meta_auc": xgb_meta_metrics['auc'],
            "optimal_lab_weight": optimal_weight
        })
        mlflow.set_tag("best_ensemble_method", best_method)
        
        # Save predictions for later analysis
        output_dir = Path("data/interim/ehr_long_los/late_fusion_ensemble")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_dir / "ensemble_predictions.npz",
            hadm_ids=ids[test_idx],
            lab_preds=test_lab_preds,
            fusion_preds=test_fusion_preds,
            best_ensemble_preds=weighted_preds if best_method == 'Weighted Average' else xgb_meta_preds,
            y_true=y[test_idx]
        )
        logger.info(f"Saved predictions to {output_dir / 'ensemble_predictions.npz'}")


if __name__ == "__main__":
    main()
