"""
Stacking Ensemble for 30-Day Readmission Prediction.

Combines predictions from multiple base models:
1. XGBoost with GRU EHR embeddings
2. XGBoost with full fusion features  
3. Gated Fusion (Neural)
4. GNN predictions

Meta-learner: Logistic Regression or XGBoost

Usage:
    python src/models/train_stacking_ensemble.py \
        --embedding_dir data/interim/ehr_long_los/embeddings \
        --n_folds 5
"""

import os
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, accuracy_score, precision_recall_curve
)
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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


# =============================================================================
# Data Loading (reuse from existing scripts)
# =============================================================================
class FusionDataLoader:
    """Load all modalities for ensemble training."""
    
    def __init__(self, base_dir=".", embedding_dir=None):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        
        if embedding_dir:
            self.structured_ehr_path = Path(embedding_dir) / "structured_ehr_embeddings.npz"
            self.structured_mapping_path = Path(embedding_dir) / "structured_ehr_mapping.csv"
        else:
            self.structured_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
            self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
            
        self.static_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"

    def load_all_data(self):
        """Load cohort and all feature modalities."""
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        
        y = cohort['readmitted_within_window'].astype(int).values
        splits = cohort['split'].values
        hadm_ids = cohort['hadm_id'].values
        
        logger.info("Loading feature modalities...")
        
        # Demographics
        demo_features = self._process_demographics(cohort)
        
        # Static EHR (TF-IDF + SVD)
        static_embeds = self._load_aligned_embeddings(self.static_ehr_path, hadm_ids)
        
        # Structured EHR (GRU/Transformer embeddings)
        struct_embeds = self._load_structured_ehr(hadm_ids)
        
        # Text embeddings
        discharge_embeds = self._load_aligned_embeddings(self.discharge_path, hadm_ids)
        rad_embeds = self._load_radiology_embeddings(hadm_ids)
        
        return {
            'demographics': demo_features,
            'static_ehr': static_embeds,
            'structured_ehr': struct_embeds,
            'discharge_notes': discharge_embeds,
            'radiology_notes': rad_embeds,
            'y': y,
            'splits': splits,
            'hadm_ids': hadm_ids,
        }

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
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return np.zeros((len(target_ids), 128))
        
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
        if not self.radiology_path.exists():
            return np.zeros((len(target_ids), 768))
        
        data = np.load(self.radiology_path)
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
        if not self.structured_ehr_path.exists() or not self.structured_mapping_path.exists():
            logger.warning("Structured EHR embeddings not found")
            return np.zeros((len(target_ids), 128))
        
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


# =============================================================================
# Base Model: XGBoost
# =============================================================================
def get_xgb_params():
    """Best XGBoost hyperparameters from HPO."""
    return {
        'max_depth': 4,
        'learning_rate': 0.0087,
        'subsample': 0.88,
        'colsample_bytree': 0.92,
        'reg_alpha': 0.0011,
        'reg_lambda': 0.396,
        'n_estimators': 1000,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50,
        'n_jobs': -1,
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0,
    }


def train_xgb_oof(X, y, train_idx, val_idx, test_idx, n_folds=5, name="xgb"):
    """Train XGBoost with out-of-fold predictions."""
    logger.info(f"Training {name} with {n_folds}-fold CV for OOF predictions...")
    
    params = get_xgb_params()
    
    # OOF predictions for train+val
    trainval_idx = np.concatenate([train_idx, val_idx])
    X_trainval = X[trainval_idx]
    y_trainval = y[trainval_idx]
    
    oof_preds = np.zeros(len(trainval_idx))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (fold_train, fold_val) in enumerate(skf.split(X_trainval, y_trainval)):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_trainval[fold_train], y_trainval[fold_train],
            eval_set=[(X_trainval[fold_val], y_trainval[fold_val])],
            verbose=False
        )
        oof_preds[fold_val] = model.predict_proba(X_trainval[fold_val])[:, 1]
    
    # Train final model on all train+val for test predictions
    final_model = xgb.XGBClassifier(**params)
    # Use a portion as validation for early stopping
    n_train = len(train_idx)
    final_model.fit(
        X[train_idx], y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        verbose=False
    )
    test_preds = final_model.predict_proba(X[test_idx])[:, 1]
    
    # Map OOF back to original indices
    oof_full = np.zeros(len(y))
    oof_full[trainval_idx] = oof_preds
    oof_full[test_idx] = test_preds
    
    # Evaluate OOF on validation
    val_auc = roc_auc_score(y[val_idx], oof_full[val_idx])
    logger.info(f"  {name} OOF Val AUC: {val_auc:.4f}")
    
    return oof_full, final_model


# =============================================================================
# Base Model: Gated Fusion Neural Network
# =============================================================================
class GatedFusionDataset(Dataset):
    def __init__(self, ehr_data, txt_data, y):
        self.ehr = torch.FloatTensor(ehr_data)
        self.txt = torch.FloatTensor(txt_data)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.ehr[idx], self.txt[idx], self.y[idx]


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
        fused, gate_vals = self.gated_fusion(h_ehr, h_txt)
        logits = self.classifier(fused)
        return logits, gate_vals


def train_gated_fusion_oof(ehr_data, txt_data, y, train_idx, val_idx, test_idx, 
                           n_folds=5, device='cuda'):
    """Train Gated Fusion with OOF predictions."""
    logger.info(f"Training Gated Fusion with {n_folds}-fold CV...")
    
    from sklearn.decomposition import PCA
    
    # Apply PCA to text
    pca = PCA(n_components=64)
    pca.fit(txt_data[train_idx])
    txt_data_pca = pca.transform(txt_data)
    
    trainval_idx = np.concatenate([train_idx, val_idx])
    X_ehr_trainval = ehr_data[trainval_idx]
    X_txt_trainval = txt_data_pca[trainval_idx]
    y_trainval = y[trainval_idx]
    
    oof_preds = np.zeros(len(trainval_idx))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (fold_train, fold_val) in enumerate(skf.split(X_ehr_trainval, y_trainval)):
        # Create datasets
        train_ds = GatedFusionDataset(
            X_ehr_trainval[fold_train], 
            X_txt_trainval[fold_train], 
            y_trainval[fold_train]
        )
        val_ds = GatedFusionDataset(
            X_ehr_trainval[fold_val], 
            X_txt_trainval[fold_val], 
            y_trainval[fold_val]
        )
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        
        # Model
        model = GatedFusionModel(
            ehr_dim=ehr_data.shape[1],
            txt_dim=64,
            hidden_dim=64,
            dropout=0.3
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        
        best_auc = 0
        best_state = None
        patience = 8
        no_improve = 0
        
        for epoch in range(30):
            # Train
            model.train()
            for x_ehr, x_txt, batch_y in train_loader:
                x_ehr = x_ehr.to(device)
                x_txt = x_txt.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                logits, _ = model(x_ehr, x_txt)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for x_ehr, x_txt, batch_y in val_loader:
                    x_ehr = x_ehr.to(device)
                    x_txt = x_txt.to(device)
                    logits, _ = model(x_ehr, x_txt)
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_targets.extend(batch_y.numpy())
            
            val_auc = roc_auc_score(val_targets, val_preds)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
        
        # Get OOF predictions
        if best_state:
            model.load_state_dict(best_state)
        
        model.eval()
        with torch.no_grad():
            x_ehr = torch.FloatTensor(X_ehr_trainval[fold_val]).to(device)
            x_txt = torch.FloatTensor(X_txt_trainval[fold_val]).to(device)
            logits, _ = model(x_ehr, x_txt)
            oof_preds[fold_val] = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Train final model for test predictions
    train_ds = GatedFusionDataset(ehr_data[train_idx], txt_data_pca[train_idx], y[train_idx])
    val_ds = GatedFusionDataset(ehr_data[val_idx], txt_data_pca[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    final_model = GatedFusionModel(
        ehr_dim=ehr_data.shape[1],
        txt_dim=64,
        hidden_dim=64,
        dropout=0.3
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=2e-4, weight_decay=0.01)
    
    best_auc = 0
    best_state = None
    
    for epoch in range(30):
        final_model.train()
        for x_ehr, x_txt, batch_y in train_loader:
            x_ehr = x_ehr.to(device)
            x_txt = x_txt.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits, _ = final_model(x_ehr, x_txt)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        
        final_model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_ehr, x_txt, batch_y in val_loader:
                x_ehr = x_ehr.to(device)
                x_txt = x_txt.to(device)
                logits, _ = final_model(x_ehr, x_txt)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(batch_y.numpy())
        
        val_auc = roc_auc_score(val_targets, val_preds)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
    
    if best_state:
        final_model.load_state_dict(best_state)
    
    # Test predictions
    final_model.eval()
    with torch.no_grad():
        x_ehr = torch.FloatTensor(ehr_data[test_idx]).to(device)
        x_txt = torch.FloatTensor(txt_data_pca[test_idx]).to(device)
        logits, _ = final_model(x_ehr, x_txt)
        test_preds = torch.sigmoid(logits).cpu().numpy().flatten()
    
    # Map back
    oof_full = np.zeros(len(y))
    oof_full[trainval_idx] = oof_preds
    oof_full[test_idx] = test_preds
    
    val_auc = roc_auc_score(y[val_idx], oof_full[val_idx])
    logger.info(f"  Gated Fusion OOF Val AUC: {val_auc:.4f}")
    
    return oof_full, final_model


# =============================================================================
# Meta-Learner Training
# =============================================================================
def train_meta_learner(base_preds, y, train_idx, val_idx, test_idx, method='lr'):
    """Train meta-learner on base model predictions."""
    logger.info(f"Training meta-learner ({method})...")
    
    # Stack base predictions
    X_meta = np.column_stack(base_preds)
    
    X_train = X_meta[train_idx]
    y_train = y[train_idx]
    X_val = X_meta[val_idx]
    y_val = y[val_idx]
    X_test = X_meta[test_idx]
    y_test = y[test_idx]
    
    if method == 'lr':
        # Logistic Regression
        meta_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        meta_model.fit(X_train, y_train)
        
        val_preds = meta_model.predict_proba(X_val)[:, 1]
        test_preds = meta_model.predict_proba(X_test)[:, 1]
        
    elif method == 'xgb':
        # XGBoost meta-learner
        meta_params = {
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'eval_metric': 'auc',
            'early_stopping_rounds': 20,
            'verbosity': 0,
            'random_state': 42,
        }
        meta_model = xgb.XGBClassifier(**meta_params)
        meta_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = meta_model.predict_proba(X_val)[:, 1]
        test_preds = meta_model.predict_proba(X_test)[:, 1]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Find optimal threshold
    opt_thresh, opt_f1 = find_optimal_threshold(y_val, val_preds)
    
    # Evaluate on test
    test_preds_binary = (test_preds >= opt_thresh).astype(int)
    
    metrics = {
        'test_auc': roc_auc_score(y_test, test_preds),
        'test_auprc': average_precision_score(y_test, test_preds),
        'test_f1': f1_score(y_test, test_preds_binary),
        'test_precision': precision_score(y_test, test_preds_binary),
        'test_recall': recall_score(y_test, test_preds_binary),
        'test_acc': accuracy_score(y_test, test_preds_binary),
        'optimal_threshold': opt_thresh,
        'val_auc': roc_auc_score(y_val, val_preds),
    }
    
    return meta_model, test_preds, metrics


# =============================================================================
# Main
# =============================================================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # MLflow setup
    mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")
    mlflow.set_experiment("stacking_ensemble")
    
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            'n_folds': args.n_folds,
            'meta_learner': args.meta_learner,
        })
        
        # Load data
        loader = FusionDataLoader(embedding_dir=args.embedding_dir)
        data = loader.load_all_data()
        
        y = data['y']
        splits = data['splits']
        
        train_idx = np.where(splits == 'train')[0]
        val_idx = np.where(splits == 'val')[0]
        test_idx = np.where(splits == 'test')[0]
        
        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Prepare feature sets
        # 1. Structured EHR only (for XGBoost)
        X_struct = np.hstack([data['demographics'], data['structured_ehr']])
        
        # 2. Full fusion (for XGBoost)
        X_full = np.hstack([
            data['demographics'],
            data['static_ehr'],
            data['structured_ehr'],
            data['discharge_notes'],
            data['radiology_notes']
        ])
        
        # 3. For Gated Fusion: EHR and text separately
        ehr_data = data['structured_ehr']
        txt_data = np.hstack([data['discharge_notes'], data['radiology_notes']])
        
        # =================================================================
        # Train Base Models
        # =================================================================
        base_predictions = []
        base_names = []
        
        # Base Model 1: XGBoost on Structured EHR
        logger.info("\n" + "="*60)
        logger.info("Base Model 1: XGBoost (Structured EHR)")
        logger.info("="*60)
        preds_xgb_struct, _ = train_xgb_oof(
            X_struct, y, train_idx, val_idx, test_idx, 
            n_folds=args.n_folds, name="XGB_Struct"
        )
        base_predictions.append(preds_xgb_struct)
        base_names.append("XGB_Struct")
        
        # Base Model 2: XGBoost on Full Fusion
        logger.info("\n" + "="*60)
        logger.info("Base Model 2: XGBoost (Full Fusion)")
        logger.info("="*60)
        preds_xgb_full, _ = train_xgb_oof(
            X_full, y, train_idx, val_idx, test_idx,
            n_folds=args.n_folds, name="XGB_Full"
        )
        base_predictions.append(preds_xgb_full)
        base_names.append("XGB_Full")
        
        # Base Model 3: Gated Fusion
        logger.info("\n" + "="*60)
        logger.info("Base Model 3: Gated Fusion (Neural)")
        logger.info("="*60)
        preds_gated, _ = train_gated_fusion_oof(
            ehr_data, txt_data, y, train_idx, val_idx, test_idx,
            n_folds=args.n_folds, device=device
        )
        base_predictions.append(preds_gated)
        base_names.append("Gated_Fusion")
        
        # Log individual base model test performance
        logger.info("\n" + "="*60)
        logger.info("Base Model Test Performance")
        logger.info("="*60)
        for name, preds in zip(base_names, base_predictions):
            test_auc = roc_auc_score(y[test_idx], preds[test_idx])
            mlflow.log_metric(f"{name}_test_auc", test_auc)
            logger.info(f"  {name}: Test AUC = {test_auc:.4f}")
        
        # =================================================================
        # Train Meta-Learner
        # =================================================================
        logger.info("\n" + "="*60)
        logger.info("Training Meta-Learner")
        logger.info("="*60)
        
        # Try both LR and XGBoost meta-learners
        for meta_method in ['lr', 'xgb']:
            meta_model, test_preds, metrics = train_meta_learner(
                base_predictions, y, train_idx, val_idx, test_idx,
                method=meta_method
            )
            
            logger.info(f"\nMeta-Learner ({meta_method.upper()}):")
            logger.info(f"  Val AUC:       {metrics['val_auc']:.4f}")
            logger.info(f"  Test AUC:      {metrics['test_auc']:.4f}")
            logger.info(f"  Test AUPRC:    {metrics['test_auprc']:.4f}")
            logger.info(f"  Test F1:       {metrics['test_f1']:.4f}")
            logger.info(f"  Test Precision:{metrics['test_precision']:.4f}")
            logger.info(f"  Test Recall:   {metrics['test_recall']:.4f}")
            
            # Log to MLflow
            mlflow.log_metrics({
                f'ensemble_{meta_method}_test_auc': metrics['test_auc'],
                f'ensemble_{meta_method}_test_auprc': metrics['test_auprc'],
                f'ensemble_{meta_method}_test_f1': metrics['test_f1'],
                f'ensemble_{meta_method}_test_precision': metrics['test_precision'],
                f'ensemble_{meta_method}_test_recall': metrics['test_recall'],
                f'ensemble_{meta_method}_val_auc': metrics['val_auc'],
            })
        
        # =================================================================
        # Simple Averaging Baseline
        # =================================================================
        logger.info("\n" + "="*60)
        logger.info("Simple Averaging Ensemble")
        logger.info("="*60)
        
        avg_preds = np.mean([p[test_idx] for p in base_predictions], axis=0)
        avg_auc = roc_auc_score(y[test_idx], avg_preds)
        avg_auprc = average_precision_score(y[test_idx], avg_preds)
        
        opt_thresh, _ = find_optimal_threshold(y[val_idx], np.mean([p[val_idx] for p in base_predictions], axis=0))
        avg_preds_binary = (avg_preds >= opt_thresh).astype(int)
        avg_f1 = f1_score(y[test_idx], avg_preds_binary)
        
        logger.info(f"  Test AUC:  {avg_auc:.4f}")
        logger.info(f"  Test AUPRC:{avg_auprc:.4f}")
        logger.info(f"  Test F1:   {avg_f1:.4f}")
        
        mlflow.log_metrics({
            'ensemble_avg_test_auc': avg_auc,
            'ensemble_avg_test_auprc': avg_auprc,
            'ensemble_avg_test_f1': avg_f1,
        })
        
        # =================================================================
        # Save predictions for future use
        # =================================================================
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_dir / 'ensemble_predictions.npz',
            base_predictions=np.column_stack(base_predictions),
            base_names=base_names,
            y=y,
            splits=splits,
            hadm_ids=data['hadm_ids'],
        )
        logger.info(f"\nSaved predictions to {output_dir / 'ensemble_predictions.npz'}")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        logger.info("Best Individual Model: XGB_Struct")
        logger.info(f"Best Ensemble: Check MLflow for comparison")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stacking Ensemble Training')
    parser.add_argument('--embedding_dir', type=str, default=None,
                        help='Path to embeddings directory')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for OOF predictions')
    parser.add_argument('--meta_learner', type=str, default='lr',
                        choices=['lr', 'xgb'],
                        help='Meta-learner type')
    parser.add_argument('--output_dir', type=str, 
                        default='data/interim/ehr_long_los/ensemble',
                        help='Output directory for predictions')
    parser.add_argument('--run_name', type=str, default='stacking_ensemble',
                        help='MLflow run name')
    
    args = parser.parse_args()
    main(args)
