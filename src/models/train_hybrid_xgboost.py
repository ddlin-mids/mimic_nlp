#!/usr/bin/env python3
"""
Hybrid XGBoost with Neural Embeddings

This approach combines:
1. Neural embeddings from temporal/fusion models as features
2. XGBoost for final prediction

Goal: Get the best of both worlds - neural representation learning + gradient boosting
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    f1_score, precision_score, recall_score, precision_recall_curve
)
import xgboost as xgb
import mlflow

# Reconfigure stdout for Unicode
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
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


class HybridDataLoader:
    """Load data for hybrid XGBoost with neural embeddings."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        
        # Embedding paths
        self.ehr_temporal_path = self.base_dir / "data/interim/ehr_long_los/temporal/ehr.npz"
        self.rad_temporal_path = self.base_dir / "data/interim/ehr_long_los/temporal/radiology.npz"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.structured_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
        self.static_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
        self.clinical_events_path = self.base_dir / "data/interim/embeddings/notes/clinical_events.npz"
        
        # PCA-reduced EHR sequences
        self.ehr_pca256_path = self.base_dir / "data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl"
        self.ehr_pca512_path = self.base_dir / "data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca512.pkl"
        
    def load_cohort(self) -> pd.DataFrame:
        """Load cardiorenal long-LOS cohort."""
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        return cohort
    
    def _process_demographics(self, df: pd.DataFrame) -> np.ndarray:
        """Process demographic features."""
        demo_df = df[['age_at_admit', 'gender', 'race']].copy()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['age_at_admit']),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                 ['gender', 'race'])
            ]
        )
        return preprocessor.fit_transform(demo_df)
    
    def _load_aligned_embeddings(
        self, 
        path: Path, 
        hadm_ids: np.ndarray,
        key: str = 'embeddings',
        id_key: str = 'hadm_ids'
    ) -> np.ndarray:
        """Load embeddings aligned to cohort hadm_ids."""
        if not path.exists():
            logger.warning(f"Embeddings not found: {path}")
            return np.zeros((len(hadm_ids), 1))
        
        data = np.load(path)
        embeddings = data[key]
        embed_ids = data[id_key]
        
        # Create lookup
        id_to_idx = {int(hid): i for i, hid in enumerate(embed_ids)}
        
        # Align to cohort
        dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 1
        aligned = np.zeros((len(hadm_ids), dim))
        
        for i, hid in enumerate(hadm_ids):
            if int(hid) in id_to_idx:
                aligned[i] = embeddings[id_to_idx[int(hid)]]
        
        return aligned
    
    def _load_temporal_ehr_pooled(self, hadm_ids: np.ndarray, pca_dim: int = 64) -> np.ndarray:
        """
        Load temporal EHR sequences and pool to fixed-size representation.
        Uses mean pooling over time + PCA for dimensionality reduction.
        """
        if not self.ehr_temporal_path.exists():
            logger.warning("Temporal EHR not found")
            return np.zeros((len(hadm_ids), pca_dim))
        
        logger.info(f"Loading temporal EHR from {self.ehr_temporal_path}")
        data = np.load(self.ehr_temporal_path)
        
        sequences = data['raw_sequences']  # (N, 100, 6219)
        masks = data['mask']  # (N, 100)
        ehr_hadm_ids = data['hadm_ids']
        
        # Mean pooling over time (masked)
        logger.info("Computing mean-pooled EHR features...")
        mask_exp = masks[:, :, np.newaxis]  # (N, 100, 1)
        pooled = (sequences * mask_exp).sum(axis=1) / mask_exp.sum(axis=1).clip(min=1)  # (N, 6219)
        
        # PCA reduction
        logger.info(f"Applying PCA to reduce from {pooled.shape[1]} to {pca_dim} dims...")
        pca = PCA(n_components=pca_dim, random_state=42)
        pooled_pca = pca.fit_transform(pooled)
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Align to cohort
        id_to_idx = {int(hid): i for i, hid in enumerate(ehr_hadm_ids)}
        aligned = np.zeros((len(hadm_ids), pca_dim))
        for i, hid in enumerate(hadm_ids):
            if int(hid) in id_to_idx:
                aligned[i] = pooled_pca[id_to_idx[int(hid)]]
        
        return aligned
    
    def _load_temporal_rad_pooled(self, hadm_ids: np.ndarray) -> np.ndarray:
        """
        Load temporal radiology embeddings and pool to fixed-size.
        Uses attention-weighted mean pooling.
        """
        if not self.rad_temporal_path.exists():
            logger.warning("Temporal radiology not found")
            return np.zeros((len(hadm_ids), 768))
        
        logger.info(f"Loading temporal radiology from {self.rad_temporal_path}")
        data = np.load(self.rad_temporal_path)
        
        embeddings = data['embeddings']  # (N, 20, 768)
        masks = data['mask']  # (N, 20)
        hours = data['hours_to_discharge']  # (N, 20)
        rad_hadm_ids = data['hadm_ids']
        
        # Weighted mean pooling - weight by recency (closer to discharge = higher weight)
        # Convert hours to weights: w = exp(-hours/24) for exponential decay over days
        weights = np.exp(-hours / 24.0) * masks  # (N, 20)
        weights = weights / weights.sum(axis=1, keepdims=True).clip(min=1e-8)  # Normalize
        
        # Weighted sum
        pooled = (embeddings * weights[:, :, np.newaxis]).sum(axis=1)  # (N, 768)
        
        # Align to cohort
        id_to_idx = {int(hid): i for i, hid in enumerate(rad_hadm_ids)}
        aligned = np.zeros((len(hadm_ids), 768))
        for i, hid in enumerate(hadm_ids):
            if int(hid) in id_to_idx:
                aligned[i] = pooled[id_to_idx[int(hid)]]
        
        return aligned
    
    def _load_radiology_mean(self, hadm_ids: np.ndarray) -> np.ndarray:
        """Load radiology embeddings with simple mean pooling per admission."""
        rad_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"
        if not rad_path.exists():
            return np.zeros((len(hadm_ids), 768))
        
        data = np.load(rad_path)
        embeddings = data['embeddings']
        rad_hadm_ids = data['hadm_ids']
        
        # Group by hadm_id and mean
        from collections import defaultdict
        hadm_embeds = defaultdict(list)
        for i, hid in enumerate(rad_hadm_ids):
            hadm_embeds[int(hid)].append(embeddings[i])
        
        aligned = np.zeros((len(hadm_ids), 768))
        for i, hid in enumerate(hadm_ids):
            if int(hid) in hadm_embeds:
                aligned[i] = np.mean(hadm_embeds[int(hid)], axis=0)
        
        return aligned
    
    def load_all_features(self, use_temporal_ehr: bool = True, ehr_pca_dim: int = 64) -> dict:
        """
        Load all features for hybrid XGBoost.
        
        Returns dict with:
            - demographics: demographic features
            - ehr_temporal: mean-pooled temporal EHR (PCA reduced)
            - ehr_structured: GRU/Transformer encoded EHR
            - rad_temporal: weighted-mean pooled radiology
            - discharge: discharge summary embeddings
            - clinical_events: clinical event embeddings
            - y: labels
            - splits: train/val/test splits
        """
        cohort = self.load_cohort()
        hadm_ids = cohort['hadm_id'].values
        y = cohort['readmitted_within_window'].astype(int).values
        splits = cohort['split'].values
        
        logger.info(f"Cohort: {len(cohort)} admissions, {y.mean():.1%} readmission rate")
        
        features = {
            'hadm_ids': hadm_ids,
            'y': y,
            'splits': splits,
        }
        
        # Demographics
        logger.info("Loading demographics...")
        features['demographics'] = self._process_demographics(cohort)
        logger.info(f"  Demographics: {features['demographics'].shape}")
        
        # Temporal EHR (mean-pooled + PCA)
        if use_temporal_ehr and self.ehr_temporal_path.exists():
            logger.info("Loading temporal EHR (mean-pooled)...")
            features['ehr_temporal'] = self._load_temporal_ehr_pooled(hadm_ids, pca_dim=ehr_pca_dim)
            logger.info(f"  Temporal EHR: {features['ehr_temporal'].shape}")
        
        # Structured EHR (GRU/Transformer encoded)
        if self.structured_ehr_path.exists():
            logger.info("Loading structured EHR embeddings...")
            struct_data = np.load(self.structured_ehr_path)
            struct_embeds = struct_data['embeddings']
            struct_names = struct_data['node_names']
            
            # Align to cohort
            cohort_node = cohort['subject_id'].astype(str) + '_' + cohort['hadm_id'].astype(str)
            name_to_idx = {str(n): i for i, n in enumerate(struct_names)}
            
            aligned = np.zeros((len(hadm_ids), struct_embeds.shape[1]))
            for i, node in enumerate(cohort_node):
                if node in name_to_idx:
                    aligned[i] = struct_embeds[name_to_idx[node]]
            
            features['ehr_structured'] = aligned
            logger.info(f"  Structured EHR: {features['ehr_structured'].shape}")
        
        # Static EHR (bag-of-events)
        if self.static_ehr_path.exists():
            logger.info("Loading static EHR embeddings...")
            features['ehr_static'] = self._load_aligned_embeddings(
                self.static_ehr_path, hadm_ids
            )
            logger.info(f"  Static EHR: {features['ehr_static'].shape}")
        
        # Temporal radiology (weighted-mean pooled)
        if self.rad_temporal_path.exists():
            logger.info("Loading temporal radiology (weighted-mean pooled)...")
            features['rad_temporal'] = self._load_temporal_rad_pooled(hadm_ids)
            logger.info(f"  Temporal radiology: {features['rad_temporal'].shape}")
        else:
            # Fall back to simple mean
            logger.info("Loading radiology (simple mean)...")
            features['rad_mean'] = self._load_radiology_mean(hadm_ids)
            logger.info(f"  Radiology mean: {features['rad_mean'].shape}")
        
        # Discharge embeddings
        logger.info("Loading discharge embeddings...")
        features['discharge'] = self._load_aligned_embeddings(
            self.discharge_path, hadm_ids
        )
        logger.info(f"  Discharge: {features['discharge'].shape}")
        
        # Clinical events
        if self.clinical_events_path.exists():
            logger.info("Loading clinical events embeddings...")
            features['clinical_events'] = self._load_aligned_embeddings(
                self.clinical_events_path, hadm_ids
            )
            logger.info(f"  Clinical events: {features['clinical_events'].shape}")
        
        return features


def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    params: dict, run_name: str
) -> dict:
    """Train XGBoost and evaluate."""
    
    # Handle class imbalance
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=pos_weight,
        use_label_encoder=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predictions
    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold on validation
    optimal_threshold, val_f1 = find_optimal_threshold(y_val, val_preds)
    
    # Apply threshold
    test_preds_class = (test_preds >= optimal_threshold).astype(int)
    
    metrics = {
        "test_auc": roc_auc_score(y_test, test_preds),
        "test_auprc": average_precision_score(y_test, test_preds),
        "test_f1": f1_score(y_test, test_preds_class, zero_division=0),
        "test_acc": accuracy_score(y_test, test_preds_class),
        "test_precision": precision_score(y_test, test_preds_class, zero_division=0),
        "test_recall": recall_score(y_test, test_preds_class, zero_division=0),
        "optimal_threshold": optimal_threshold,
        "n_features": X_train.shape[1]
    }
    
    return metrics, model


def main():
    parser = argparse.ArgumentParser(description="Hybrid XGBoost with Neural Embeddings")
    parser.add_argument("--ehr_pca_dim", type=int, default=64, 
                        help="PCA dimensions for temporal EHR")
    parser.add_argument("--no_temporal_ehr", action="store_true",
                        help="Skip temporal EHR features")
    args = parser.parse_args()
    
    mlflow.set_experiment("mimic_cardiorenal_hybrid_xgboost")
    
    # Load all features
    loader = HybridDataLoader()
    features = loader.load_all_features(
        use_temporal_ehr=not args.no_temporal_ehr,
        ehr_pca_dim=args.ehr_pca_dim
    )
    
    splits = features['splits']
    y = features['y']
    
    # Best XGBoost params (from previous HPO)
    best_params = {
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
        'random_state': 42
    }
    
    # Define experiments
    experiments = {}
    
    # Baseline
    experiments["Baseline (Demo)"] = ["demographics"]
    
    # Single modalities
    if 'ehr_structured' in features:
        experiments["Structured EHR (GRU)"] = ["demographics", "ehr_structured"]
    
    if 'ehr_temporal' in features:
        experiments["Temporal EHR (Pooled)"] = ["demographics", "ehr_temporal"]
    
    if 'ehr_static' in features:
        experiments["Static EHR (BoE)"] = ["demographics", "ehr_static"]
    
    experiments["Discharge Only"] = ["demographics", "discharge"]
    
    if 'rad_temporal' in features:
        experiments["Radiology (Temporal)"] = ["demographics", "rad_temporal"]
    elif 'rad_mean' in features:
        experiments["Radiology (Mean)"] = ["demographics", "rad_mean"]
    
    if 'clinical_events' in features:
        experiments["Clinical Events"] = ["demographics", "clinical_events"]
    
    # Combined modalities
    if 'ehr_structured' in features and 'ehr_temporal' in features:
        experiments["EHR Combined (Struct+Temporal)"] = [
            "demographics", "ehr_structured", "ehr_temporal"
        ]
    
    if 'ehr_structured' in features:
        rad_key = 'rad_temporal' if 'rad_temporal' in features else 'rad_mean'
        if rad_key in features:
            experiments["Struct EHR + Radiology"] = [
                "demographics", "ehr_structured", rad_key
            ]
        experiments["Struct EHR + Discharge"] = [
            "demographics", "ehr_structured", "discharge"
        ]
        experiments["Struct EHR + All Text"] = [
            "demographics", "ehr_structured", "discharge", rad_key
        ]
    
    # Full fusion variants
    full_modalities = ["demographics"]
    if 'ehr_structured' in features:
        full_modalities.append("ehr_structured")
    if 'ehr_temporal' in features:
        full_modalities.append("ehr_temporal")
    full_modalities.append("discharge")
    rad_key = 'rad_temporal' if 'rad_temporal' in features else 'rad_mean'
    if rad_key in features:
        full_modalities.append(rad_key)
    
    experiments["Full Hybrid Fusion"] = full_modalities.copy()
    
    if 'clinical_events' in features:
        experiments["Full Hybrid + Events"] = full_modalities + ["clinical_events"]
    
    if 'ehr_static' in features:
        experiments["Full Hybrid + Static"] = full_modalities + ["ehr_static"]
    
    # Run experiments
    results = []
    
    for exp_name, modality_list in experiments.items():
        # Check all modalities exist
        missing = [m for m in modality_list if m not in features]
        if missing:
            logger.warning(f"Skipping {exp_name}: missing {missing}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {exp_name}")
        logger.info(f"Modalities: {modality_list}")
        
        # Concatenate features
        X = np.hstack([features[m] for m in modality_list])
        
        # Split
        X_train = X[splits == 'train']
        y_train = y[splits == 'train']
        X_val = X[splits == 'val']
        y_val = y[splits == 'val']
        X_test = X[splits == 'test']
        y_test = y[splits == 'test']
        
        logger.info(f"Features: {X_train.shape[1]} dims")
        
        with mlflow.start_run(run_name=exp_name):
            mlflow.log_param("experiment", exp_name)
            mlflow.log_param("modalities", ",".join(modality_list))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("ehr_pca_dim", args.ehr_pca_dim)
            
            metrics, model = train_and_evaluate(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                best_params, exp_name
            )
            
            mlflow.log_metrics(metrics)
            
            logger.info(f"Results: AUC={metrics['test_auc']:.4f}, "
                       f"AUPRC={metrics['test_auprc']:.4f}, "
                       f"F1={metrics['test_f1']:.4f}")
            
            results.append({
                'experiment': exp_name,
                'test_auc': metrics['test_auc'],
                'test_auprc': metrics['test_auprc'],
                'test_f1': metrics['test_f1'],
                'n_features': metrics['n_features']
            })
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY - Hybrid XGBoost Results")
    logger.info(f"{'='*60}")
    
    results_df = pd.DataFrame(results).sort_values('test_auc', ascending=False)
    for _, row in results_df.iterrows():
        logger.info(f"{row['experiment']:40s} | AUC: {row['test_auc']:.4f} | "
                   f"AUPRC: {row['test_auprc']:.4f} | F1: {row['test_f1']:.4f}")
    
    # Save results
    results_path = Path("results/hybrid_xgboost_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
