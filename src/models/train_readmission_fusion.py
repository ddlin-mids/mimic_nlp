import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, classification_report
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FusionDataLoader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        
        # Paths to embeddings
        self.static_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
        self.structured_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
        self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"

    def load_and_prep_data(self):
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        
        # Filter for Cardiorenal Long
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        logger.info(f"Filtered cohort size (Cardiorenal Long): {len(cohort)}")
        
        # Ensure target is integer
        cohort['target'] = cohort['readmitted_within_window'].astype(int)
        
        # 1. Process Demographics
        logger.info("Processing demographics...")
        demo_features = self._process_demographics(cohort)
        
        # 2. Process Static EHR
        logger.info("Processing Static EHR embeddings...")
        static_embeds = self._load_aligned_embeddings(
            self.static_ehr_path, 
            cohort['hadm_id'].values,
            id_key='hadm_ids',
            embed_key='embeddings'
        )
        
        # 3. Process Structured EHR
        logger.info("Processing Structured EHR embeddings...")
        struct_embeds = self._load_structured_ehr(cohort['hadm_id'].values)
        
        # 4. Process Discharge Notes
        logger.info("Processing Discharge Note embeddings...")
        discharge_embeds = self._load_aligned_embeddings(
            self.discharge_path,
            cohort['hadm_id'].values,
            id_key='hadm_ids',
            embed_key='embeddings'
        )
        
        # 5. Process Radiology Reports (with aggregation)
        logger.info("Processing Radiology Report embeddings...")
        rad_embeds = self._load_radiology_embeddings(cohort['hadm_id'].values)
        
        # Concatenate all features
        logger.info("Concatenating all modalities...")
        # Check shapes
        logger.info(f"Shapes: Demo={demo_features.shape}, Static={static_embeds.shape}, "
                    f"Struct={struct_embeds.shape}, Discharge={discharge_embeds.shape}, Rad={rad_embeds.shape}")
        
        X = np.hstack([
            demo_features,
            static_embeds,
            struct_embeds,
            discharge_embeds,
            rad_embeds
        ])
        
        y = cohort['target'].values
        splits = cohort['split'].values
        
        return X, y, splits

    def _process_demographics(self, df):
        # Select raw cols
        demo_df = df[['age_at_admit', 'gender', 'race']].copy()
        
        # Pipeline for preprocessing
        numeric_features = ['age_at_admit']
        categorical_features = ['gender', 'race']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        
        return preprocessor.fit_transform(demo_df)

    def _load_aligned_embeddings(self, filepath, target_hadm_ids, id_key='hadm_ids', embed_key='embeddings'):
        """Generic loader for aligned .npz files with direct 1:1 or 1:N mapping"""
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}. Returning zeros.")
            return np.zeros((len(target_hadm_ids), 0)) # Dimension issue if missing? handled by actual load logic
            
        data = np.load(filepath)
        source_ids = data[id_key]
        source_embeds = data[embed_key]
        
        # Create map
        id_to_embed = {}
        for idx, hid in enumerate(source_ids):
            # Handle float ids if present
            hid = int(hid) if isinstance(hid, (float, np.floating)) else hid
            id_to_embed[hid] = source_embeds[idx]
            
        # Align
        dim = source_embeds.shape[1]
        aligned = []
        missing_count = 0
        
        for hid in target_hadm_ids:
            if hid in id_to_embed:
                aligned.append(id_to_embed[hid])
            else:
                aligned.append(np.zeros(dim))
                missing_count += 1
                
        if missing_count > 0:
            logger.info(f"Missing {missing_count}/{len(target_hadm_ids)} entries for {filepath.name}")
            
        return np.array(aligned)

    def _load_radiology_embeddings(self, target_hadm_ids):
        """Special handler for radiology to aggregate multiple reports per admission"""
        data = np.load(self.radiology_path)
        source_ids = data['hadm_ids']
        source_embeds = data['embeddings']
        
        # Group by hadm_id
        from collections import defaultdict
        id_to_embeds = defaultdict(list)
        
        for idx, hid in enumerate(source_ids):
            hid = int(hid) if isinstance(hid, (float, np.floating)) else hid
            id_to_embeds[hid].append(source_embeds[idx])
            
        # Align and Mean Pool
        dim = source_embeds.shape[1]
        aligned = []
        missing_count = 0
        
        for hid in target_hadm_ids:
            if hid in id_to_embeds:
                # Mean pool
                stacked = np.stack(id_to_embeds[hid])
                aligned.append(np.mean(stacked, axis=0))
            else:
                aligned.append(np.zeros(dim))
                missing_count += 1
                
        logger.info(f"Missing Radiology for {missing_count}/{len(target_hadm_ids)} patients")
        return np.array(aligned)

    def _load_structured_ehr(self, target_hadm_ids):
        """Load GRU embeddings using the mapping CSV"""
        mapping_df = pd.read_csv(self.structured_mapping_path)
        data = np.load(self.structured_ehr_path)
        embeddings = data['embeddings']
        
        # Mapping: node_name (subject_hadm) -> row_idx
        # We need hadm_id -> row_idx
        
        hadm_to_idx = {}
        for _, row in mapping_df.iterrows():
            # node_name format: 10001338_22119639
            parts = str(row['node_name']).split('_')
            if len(parts) == 2:
                hadm_id = int(parts[1])
                hadm_to_idx[hadm_id] = row['row_idx']
        
        dim = embeddings.shape[1]
        aligned = []
        missing_count = 0
        
        for hid in target_hadm_ids:
            if hid in hadm_to_idx:
                idx = hadm_to_idx[hid]
                aligned.append(embeddings[idx])
            else:
                aligned.append(np.zeros(dim))
                missing_count += 1
                
        if missing_count > 0:
            logger.info(f"Missing Structured EHR for {missing_count}/{len(target_hadm_ids)} patients")
        
        return np.array(aligned)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"--- {model_name} Results ---")
    logger.info(f"AUROC: {auroc:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": acc,
        "F1": f1
    }

def main():
    loader = FusionDataLoader()
    X, y, splits = loader.load_and_prep_data()
    
    # Split Data
    train_mask = splits == 'train'
    val_mask = splits == 'val'
    test_mask = splits == 'test'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"Train size: {X_train.shape}, Val size: {X_val.shape}, Test size: {X_test.shape}")
    
    results = {}
    
    # --- Model A: XGBoost ---
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # --- Model B: MLP ---
    logger.info("Training MLP...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1, # Uses internal split, but we have explicit val set. 
                                 # Standard MLP doesn't take explicit val set easily for stopping without custom loop.
                                 # We will just rely on its internal split or standard convergence for now.
        random_state=42
    )
    
    # Ideally we concatenate Train+Val for MLP if using internal split, or just train on Train.
    # To be comparable to XGBoost (which stopped on Val), let's train on Train and maybe check Val score manually if we were tuning.
    # For this script, we'll train on Train.
    mlp_model.fit(X_train, y_train)
    
    results['MLP'] = evaluate_model(mlp_model, X_test, y_test, "MLP")
    
    # Save Results
    os.makedirs("results/cohort_results", exist_ok=True)
    with open("results/cohort_results/cardiorenal_fusion_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Done. Results saved to results/cohort_results/cardiorenal_fusion_metrics.json")

if __name__ == "__main__":
    main()
