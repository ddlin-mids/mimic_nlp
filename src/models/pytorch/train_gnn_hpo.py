import os
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import mlflow
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find threshold that maximizes F1 on validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

# Reusing the GNNClassifier from train_gnn.py
class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3, use_residual=True):
        super().__init__()
        self.dropout = dropout
        self.use_residual = use_residual
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x_in = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual and x_in.shape == x.shape:
                x = x + x_in
        
        x = self.convs[-1](x, edge_index)
        return x

def objective(trial, args, device, x, edge_index, y, train_mask, val_mask):
    # Hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    use_residual = True # Fixed for now as it usually helps deep GNNs

    model = GNNClassifier(x.size(1), hidden_dim, 1, num_layers, dropout, use_residual).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_auc = 0
    patience = 15
    no_improve = 0
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out[train_mask].squeeze(), y[train_mask])
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                probs = torch.sigmoid(out).squeeze()
                val_auc = float(roc_auc_score(y[val_mask].cpu(), probs[val_mask].cpu()))
                
                # Report to Optuna
                trial.report(val_auc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                mlflow.log_metrics({"val_auc": val_auc, "train_loss": loss.item()}, step=epoch)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
                        
        mlflow.log_metric("best_val_auc", best_val_auc)
        
    return best_val_auc

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data (One-time load)
    logger.info(f"Loading graph from {args.graph_path}...")
    graph_data = torch.load(args.graph_path, weights_only=False)
    edge_index = graph_data.edge_index.to(device)
    
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        x = graph_data.x.float().to(device)
    elif args.feature_path:
        logger.info(f"Loading embeddings from {args.feature_path}...")
        emb_data = np.load(args.feature_path)
        x = torch.from_numpy(emb_data['embeddings']).float().to(device)
    else:
        raise ValueError("No features found")

    # Load mapping/cohort for labels
    logger.info(f"Loading cohort from {args.cohort_path}...")
    cohort = pd.read_csv(args.cohort_path)
    cohort["node_name"] = cohort["subject_id"].astype(str) + "_" + cohort["hadm_id"].astype(str)
    
    # Try to find mapping file
    graph_dir = os.path.dirname(args.graph_path)
    mapping_path = os.path.join(graph_dir, "graph_node_map.csv")
    if os.path.exists(mapping_path):
        map_df = pd.read_csv(mapping_path)
        node_names = map_df["node_name"].tolist()
    else:
        # Fallback logic if needed, but for HPO we assume stable environment
        # If external features used, they usually have node_names
        if args.feature_path:
             node_names = np.load(args.feature_path)['node_names']
        else:
             raise ValueError("Cannot determine node names")

    node_to_idx = {name: i for i, name in enumerate(node_names)}
    
    y = torch.zeros(len(node_names), dtype=torch.float).to(device)
    train_mask = torch.zeros(len(node_names), dtype=torch.bool).to(device)
    val_mask = torch.zeros(len(node_names), dtype=torch.bool).to(device)
    
    for _, row in cohort.iterrows():
        name = row["node_name"]
        if name in node_to_idx:
            idx = node_to_idx[name]
            y[idx] = float(row["readmitted_within_window"])
            split = row.get("split", "train")
            if split == "train": train_mask[idx] = True
            elif split in ["val", "validation"]: val_mask[idx] = True
            
    # Optuna Study
    mlflow.set_experiment("mimic_cardiorenal_gnn_hpo")
    
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, args, device, x, edge_index, y, train_mask, val_mask), n_trials=args.n_trials)
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
        
    # Save best params
    with open(os.path.join(args.save_dir, "best_params.json"), "w") as f:
        json.dump(trial.params, f, indent=4)

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--feature_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="data/interim/ehr_long_los/models_hpo")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_trials", type=int, default=20)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
