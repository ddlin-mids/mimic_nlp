import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded Best Params from HPO Job 46810022
BEST_PARAMS = {
    'max_depth': 6, 
    'learning_rate': 0.2408, 
    'subsample': 0.8462, 
    'colsample_bytree': 0.7631, 
    'reg_alpha': 0.0022, 
    'reg_lambda': 0.0938,
    'n_estimators': 1000,
    'eval_metric': 'auc',
    'n_jobs': -1,
    'tree_method': 'hist',
    'early_stopping_rounds': 50,
    'random_state': 42
}

class SimpleDataLoader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        self.static_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
        self.structured_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
        self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"

    def load_data(self):
        logger.info("Loading cohort...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        y = cohort['readmitted_within_window'].astype(int).values
        splits = cohort['split'].values
        ids = cohort['hadm_id'].values
        
        logger.info("Loading features...")
        # 1. Demo
        demo_df = cohort[['age_at_admit', 'gender', 'race']].copy()
        ct = ColumnTransformer([
            ('num', StandardScaler(), ['age_at_admit']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['gender', 'race'])
        ])
        demo_feat = ct.fit_transform(demo_df)
        demo_names = ct.get_feature_names_out()
        
        # 2. Static EHR
        static_feat, static_names = self._load_npz(self.static_ehr_path, ids, "static")
        
        # 3. Structured EHR (GRU)
        struct_feat, _ = self._load_struct(ids)
        struct_names = [f"struct_gru_{i}" for i in range(struct_feat.shape[1])]
        
        # 4. Text (Discharge + Rad)
        disch_feat, _ = self._load_npz(self.discharge_path, ids, "disch")
        disch_names = [f"disch_emb_{i}" for i in range(disch_feat.shape[1])]
        
        rad_feat, _ = self._load_radiology(ids)
        rad_names = [f"rad_emb_{i}" for i in range(rad_feat.shape[1])]
        
        # Combine
        X = np.hstack([demo_feat, static_feat, struct_feat, disch_feat, rad_feat])
        feature_names = np.concatenate([demo_names, static_names, struct_names, disch_names, rad_names])
        
        return X, y, splits, feature_names

    def _load_npz(self, path, target_ids, prefix):
        if not path.exists(): return np.zeros((len(target_ids), 0)), []
        data = np.load(path)
        id_key = 'hadm_ids' if 'hadm_ids' in data else 'ids'
        s_ids = data[id_key]
        s_emb = data['embeddings']
        dim = s_emb.shape[1]
        id_map = {int(k): v for k, v in zip(s_ids, s_emb)}
        
        aligned = np.array([id_map.get(tid, np.zeros(dim)) for tid in target_ids])
        
        # Safe name generation
        if 'node_names' in data:
            raw_names = data['node_names']
            if len(raw_names) == dim:
                names = raw_names
            else:
                names = [f"{prefix}_{i}" for i in range(dim)]
        else:
            names = [f"{prefix}_{i}" for i in range(dim)]
            
        return aligned, names

    def _load_radiology(self, target_ids):
        if not self.radiology_path.exists(): return np.zeros((len(target_ids), 0)), []
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
        return np.array(aligned), [f"rad_{i}" for i in range(dim)]

    def _load_struct(self, target_ids):
        mapping = pd.read_csv(self.structured_mapping_path)
        embeddings = np.load(self.structured_ehr_path)['embeddings']
        hid_to_idx = {}
        for _, row in mapping.iterrows():
            parts = str(row['node_name']).split('_')
            if len(parts) == 2: hid_to_idx[int(parts[1])] = row['row_idx']
        dim = embeddings.shape[1]
        aligned = np.array([embeddings[hid_to_idx[tid]] if tid in hid_to_idx else np.zeros(dim) for tid in target_ids])
        return aligned, []

def main():
    loader = SimpleDataLoader()
    X, y, splits, feat_names = loader.load_data()
    
    train_idx = (splits == 'train')
    # Use small validation set for early stopping
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X[train_idx], y[train_idx], test_size=0.1, random_state=42
    )
    
    logger.info(f"Training XGBoost (Input Dim: {X.shape[1]})...")
    model = xgb.XGBClassifier(**BEST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    logger.info("Computing SHAP values...")
    # Use TreeExplainer (optimized for trees)
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP on a subset of validation data (for speed, e.g. 500 samples)
    # Full dataset might be slow
    subset_size = min(1000, X_val.shape[0])
    X_sample = X_val[:subset_size]
    shap_values = explainer.shap_values(X_sample)
    
    logger.info("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 10))
    # Use beeswarm plot
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names, show=False, max_display=20)
    plt.title("SHAP Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig("feature_importance_shap.png")
    
    logger.info("Saved feature_importance_shap.png")

if __name__ == "__main__":
    main()
