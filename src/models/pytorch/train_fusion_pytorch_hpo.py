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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch
import optuna
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReadmissionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class FusionDataLoader:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        self.static_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/static_ehr_embeddings.npz"
        self.structured_ehr_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
        self.structured_mapping_path = self.base_dir / "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
        self.discharge_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.radiology_path = self.base_dir / "data/interim/embeddings/notes/radiology_report.npz"

    def load_and_prep_data(self):
        logger.info("Loading cohort data...")
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        cohort['target'] = cohort['readmitted_within_window'].astype(int)
        
        logger.info("Processing modalities...")
        demo_features = self._process_demographics(cohort)
        static_embeds = self._load_aligned_embeddings(self.static_ehr_path, cohort['hadm_id'].values)
        struct_embeds = self._load_structured_ehr(cohort['hadm_id'].values)
        discharge_embeds = self._load_aligned_embeddings(self.discharge_path, cohort['hadm_id'].values)
        rad_embeds = self._load_radiology_embeddings(cohort['hadm_id'].values)
        
        X = np.hstack([demo_features, static_embeds, struct_embeds, discharge_embeds, rad_embeds])
        y = cohort['target'].values
        splits = cohort['split'].values
        return X, y, splits

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
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        total_loss += loss.item()
        all_preds.extend(probs)
        all_targets.extend(y_batch.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    auc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    return total_loss / len(loader), auc, auprc, all_preds, all_targets

def objective(trial, X_train, y_train, X_val, y_val, device):
    # Hyperparams
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.3, 0.7),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "epochs": 50
    }
    
    hidden_dims = []
    for i in range(params["n_layers"]):
        dim = trial.suggest_int(f"n_units_l{i}", 64, 512, log=True)
        hidden_dims.append(dim)
    params["hidden_dims"] = hidden_dims

    # Data Loaders
    train_ds = ReadmissionDataset(X_train, y_train)
    val_ds = ReadmissionDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"])

    # Model Setup
    model = MLP(X_train.shape[1], hidden_dims, params["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader)*params["epochs"])

    best_val_auc = 0.0
    patience = 8
    no_improve = 0

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        for epoch in range(params["epochs"]):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_auc, val_auprc, _, _ = evaluate(model, val_loader, criterion, device)
            
            # Pruning
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
        
        mlflow.log_metric("val_auc", best_val_auc)
    
    return best_val_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    mlflow.set_experiment("mimic_cardiorenal_readmission_pytorch_hpo")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    loader = FusionDataLoader()
    X, y, splits = loader.load_and_prep_data()
    X_train, y_train = X[splits=='train'], y[splits=='train']
    X_val, y_val = X[splits=='val'], y[splits=='val']
    X_test, y_test = X[splits=='test'], y[splits=='test']

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, device), n_trials=args.trials)

    logger.info(f"Best params: {study.best_params}")
    
    # Train Best Model
    best_params = study.best_params
    hidden_dims = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]
    
    model = MLP(X_train.shape[1], hidden_dims, best_params["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    train_loader = DataLoader(ReadmissionDataset(X_train, y_train), batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(ReadmissionDataset(X_val, y_val), batch_size=best_params["batch_size"])
    test_loader = DataLoader(ReadmissionDataset(X_test, y_test), batch_size=best_params["batch_size"])
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_loader)*50)
    criterion = nn.BCEWithLogitsLoss()
    
    best_state = None
    best_auc = 0.0
    
    with mlflow.start_run(run_name="Best_PyTorch_Model"):
        mlflow.log_params(best_params)
        
        for epoch in range(50):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_auc, val_auprc, _, _ = evaluate(model, val_loader, criterion, device)
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc
            }, step=epoch)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
        
        if best_state: model.load_state_dict(best_state)
        
        # Final Test
        test_loss, test_auc, test_auprc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
        test_acc = accuracy_score(test_targets, (test_preds > 0.5).astype(int))
        test_f1 = f1_score(test_targets, (test_preds > 0.5).astype(int))
        
        mlflow.log_metrics({
            "test_auc": test_auc, 
            "test_auprc": test_auprc,
            "test_accuracy": test_acc, 
            "test_f1": test_f1
        })
        
        mlflow.pytorch.log_model(model, "model")
        
        logger.info(f"Final Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()
