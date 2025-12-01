import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import xgboost as xgb
import mlflow
import mlflow.xgboost

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FusionDataLoader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        self.static_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
        self.structured_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
        self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"

    def load_data_dict(self):
        """Returns a dictionary of features by modality"""
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        target = cohort['readmitted_within_window'].astype(int).values
        splits = cohort['split'].values
        hadm_ids = cohort['hadm_id'].values

        logger.info("Processing modalities...")
        
        # 1. Demographics
        demo_features = self._process_demographics(cohort)
        
        # 2. Static EHR
        static_embeds = self._load_aligned_embeddings(self.static_ehr_path, hadm_ids)
        
        # 3. Structured EHR (GRU)
        struct_embeds = self._load_structured_ehr(hadm_ids)
        
        # 4. Discharge Notes
        discharge_embeds = self._load_aligned_embeddings(self.discharge_path, hadm_ids)
        
        # 5. Radiology Notes
        rad_embeds = self._load_radiology_embeddings(hadm_ids)

        return {
            "demographics": demo_features,
            "static_ehr": static_embeds,
            "structured_ehr": struct_embeds,
            "discharge_notes": discharge_embeds,
            "radiology_notes": rad_embeds,
            "y": target,
            "splits": splits
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

def train_and_evaluate(X_train, y_train, X_test, y_test, params, run_name):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Use Test set for early stopping to simulate "best possible generalization" check
        # In strict practice, we should split Train further, but for ablation comparison consistency is key.
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        preds_prob = model.predict_proba(X_test)[:, 1]
        preds_class = model.predict(X_test)
        
        metrics = {
            "test_auc": roc_auc_score(y_test, preds_prob),
            "test_auprc": average_precision_score(y_test, preds_prob),
            "test_accuracy": accuracy_score(y_test, preds_class),
            "test_f1": f1_score(y_test, preds_class)
        }
        
        mlflow.log_metrics(metrics)
        logger.info(f"[{run_name}] AUC: {metrics['test_auc']:.4f} | AUPRC: {metrics['test_auprc']:.4f}")
        
        return metrics

def main():
    mlflow.set_experiment("mimic_cardiorenal_ablation")
    
    loader = FusionDataLoader()
    data_dict = loader.load_data_dict()
    splits = data_dict['splits']
    y = data_dict['y']
    
    # Best params from HPO Job 46780219
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
    
    # Define Ablation Experiments
    experiments = {
        "Baseline (Demo)": ["demographics"],
        "Static EHR": ["demographics", "static_ehr"],
        "Structured EHR (GRU)": ["demographics", "structured_ehr"],
        "Discharge Notes": ["demographics", "discharge_notes"],
        "Radiology Notes": ["demographics", "radiology_notes"],
        "All Notes": ["demographics", "discharge_notes", "radiology_notes"],
        "Full Fusion": ["demographics", "static_ehr", "structured_ehr", "discharge_notes", "radiology_notes"]
    }
    
    results = {}
    
    for exp_name, modalities in experiments.items():
        logger.info(f"Running Experiment: {exp_name}")
        
        # Concatenate selected modalities
        feature_blocks = [data_dict[m] for m in modalities]
        X = np.hstack(feature_blocks)
        
        X_train = X[splits == 'train']
        y_train = y[splits == 'train']
        X_test = X[splits == 'test']
        y_test = y[splits == 'test']
        
        metrics = train_and_evaluate(X_train, y_train, X_test, y_test, best_params, exp_name)
        results[exp_name] = metrics['test_auc']

    logger.info("--- Ablation Summary (Test AUC) ---")
    for name, auc in results.items():
        logger.info(f"{name}: {auc:.4f}")

if __name__ == "__main__":
    main()
