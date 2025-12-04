import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
from optuna.integration.mlflow import MLflowCallback

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
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        target = cohort['readmitted_within_window'].astype(int).values
        splits = cohort['split'].values
        hadm_ids = cohort['hadm_id'].values

        logger.info("Processing modalities...")
        demo = self._process_demographics(cohort)
        static = self._load_aligned_embeddings(self.static_ehr_path, hadm_ids)
        struct = self._load_structured_ehr(hadm_ids)
        disch = self._load_aligned_embeddings(self.discharge_path, hadm_ids)
        rad = self._load_radiology_embeddings(hadm_ids)
        
        return {
            "demo": demo,
            "static": static,
            "struct": struct,
            "text": np.hstack([disch, rad]),
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
        # Handle key variations
        id_key = 'hadm_ids' if 'hadm_ids' in data else 'ids'
        embed_key = 'embeddings'
        
        source_ids = data[id_key]
        source_embeds = data[embed_key]
        
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

from sklearn.decomposition import PCA

def objective_xgboost(trial, data_dict):
    splits = data_dict['splits']
    train_idx = splits == 'train'
    val_idx = splits == 'val'
    
    # 1. Feature Engineering (PCA)
    pca_n = trial.suggest_categorical('pca_components', [0, 32, 64, 128])
    text_data = data_dict['text']
    
    if pca_n > 0:
        pca = PCA(n_components=pca_n)
        # Fit on train only
        pca.fit(text_data[train_idx])
        text_feat = pca.transform(text_data)
    else:
        text_feat = text_data
        
    # 2. Concat
    X = np.hstack([data_dict['demo'], data_dict['static'], data_dict['struct'], text_feat])
    y = data_dict['y']
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    params = {
        'n_estimators': 1000, 
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'eval_metric': 'auc',
        'n_jobs': -1,
        'random_state': 42,
        'tree_method': 'hist', 
        'early_stopping_rounds': 50
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_param("pca_components", pca_n)
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        
        mlflow.log_metric("val_auc", auc)
        return auc

def objective_mlp(trial, data_dict):
    splits = data_dict['splits']
    train_idx = splits == 'train'
    val_idx = splits == 'val'
    
    # PCA
    pca_n = trial.suggest_categorical('pca_components', [0, 64, 128])
    text_data = data_dict['text']
    
    if pca_n > 0:
        pca = PCA(n_components=pca_n)
        pca.fit(text_data[train_idx])
        text_feat = pca.transform(text_data)
    else:
        text_feat = text_data
        
    X = np.hstack([data_dict['demo'], data_dict['static'], data_dict['struct'], text_feat])
    y = data_dict['y']
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    
    # Layer configs
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        units = trial.suggest_int(f'n_units_l{i}', 32, 512, log=True)
        layers.append(units)
    
    params = {
        'hidden_layer_sizes': tuple(layers),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': 'adam',
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'batch_size': 128,
        'max_iter': 200,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'random_state': 42
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_param("pca_components", pca_n)
        
        model = MLPClassifier(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        
        mlflow.log_metric("val_auc", auc)
        return auc

def train_best_model(model_type, best_params, data_dict):
    logger.info(f"Training best {model_type} model...")
    
    splits = data_dict['splits']
    train_idx = splits == 'train'
    test_idx = splits == 'test'
    
    # Apply PCA if in params
    pca_n = best_params.get('pca_components', 0)
    text_data = data_dict['text']
    
    if pca_n > 0:
        pca = PCA(n_components=pca_n)
        pca.fit(text_data[train_idx])
        text_feat = pca.transform(text_data)
    else:
        text_feat = text_data
        
    X = np.hstack([data_dict['demo'], data_dict['static'], data_dict['struct'], text_feat])
    y = data_dict['y']
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    if model_type == "xgboost":
        # Re-add fixed params not optimized
        final_params = best_params.copy()
        # Remove non-XGB params
        final_params.pop('pca_components', None)
        
        final_params.update({
            'n_estimators': 1000, 
            'eval_metric': 'auc', 
            'n_jobs': -1,
            'random_state': 42,
            'tree_method': 'hist',
            'early_stopping_rounds': 50
        })
        
        # Internal split for early stopping
        from sklearn.model_selection import train_test_split
        X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        
        model = xgb.XGBClassifier(**final_params)
        model.fit(
            X_tr_final, y_tr_final,
            eval_set=[(X_val_final, y_val_final)],
            verbose=False
        )
        
    elif model_type == "mlp":
        # Construct layers tuple from numbered params
        layers = []
        i = 0
        while f'n_units_l{i}' in best_params:
            layers.append(best_params[f'n_units_l{i}'])
            i += 1
        
        # Filter out helper keys
        final_params = {
            k: v for k, v in best_params.items() 
            if not k.startswith('n_units_l') and k != 'n_layers' and k != 'pca_components'
        }
        final_params['hidden_layer_sizes'] = tuple(layers)
        final_params.update({'max_iter': 500, 'random_state': 42, 'solver': 'adam', 'batch_size': 128})
        
        model = MLPClassifier(**final_params)
        model.fit(X_train, y_train)

    # Evaluation
    preds_prob = model.predict_proba(X_test)[:, 1]
    preds_class = model.predict(X_test)
    
    metrics = {
        "test_auc": roc_auc_score(y_test, preds_prob),
        "test_auprc": average_precision_score(y_test, preds_prob),
        "test_accuracy": accuracy_score(y_test, preds_class),
        "test_f1": f1_score(y_test, preds_class)
    }
    
    # Artifacts
    try:
        if model_type == "xgboost":
            import matplotlib.pyplot as plt
            xgb.plot_importance(model, max_num_features=20)
            plt.title(f"XGBoost Feature Importance")
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
    except OSError as e:
        logger.warning(f"Failed to log artifacts due to filesystem error: {e}")
    except Exception as e:
        logger.warning(f"Failed to log artifacts: {e}")

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--model", type=str, required=True, choices=["xgboost", "mlp"])
    args = parser.parse_args()

    mlflow.set_experiment("mimic_cardiorenal_readmission")
    
    loader = FusionDataLoader()
    data_dict = loader.load_data_dict()
    
    logger.info(f"Loaded data. Splits: {pd.Series(data_dict['splits']).value_counts().to_dict()}")
    
    with mlflow.start_run(run_name=f"HPO_{args.model.upper()}"):
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("n_trials", args.trials)
        
        study = optuna.create_study(direction="maximize")
        
        if args.model == "xgboost":
            objective = lambda trial: objective_xgboost(trial, data_dict)
        else:
            objective = lambda trial: objective_mlp(trial, data_dict)
            
        study.optimize(objective, n_trials=args.trials)
        
        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Best val AUC: {study.best_value}")
        
        # Log best params to parent run
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_auc", study.best_value)
        
        test_metrics = train_best_model(args.model, study.best_params, data_dict)
        
        mlflow.log_metrics(test_metrics)
        logger.info(f"Test Metrics: {test_metrics}")

if __name__ == "__main__":
    main()
