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
    # Hyperparameters - expanded search space
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    lr = trial.suggest_float("lr", 5e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    use_residual = trial.suggest_categorical("use_residual", [True, False])

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
    
    # Load cohort first to determine valid nodes
    logger.info(f"Loading cohort from {args.cohort_path}...")
    cohort = pd.read_csv(args.cohort_path)
    cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
    cohort["node_name"] = cohort["subject_id"].astype(str) + "_" + cohort["hadm_id"].astype(str)
    cohort_nodes = set(cohort["node_name"])
    
    # Load Data (One-time load)
    logger.info(f"Loading graph from {args.graph_path}...")
    graph_data = torch.load(args.graph_path, weights_only=False)
    
    # Load mapping for re-indexing
    graph_dir = os.path.dirname(args.graph_path)
    mapping_path = os.path.join(graph_dir, "graph_node_map.csv")
    
    if os.path.exists(mapping_path):
        map_df = pd.read_csv(mapping_path)
        all_node_names = map_df["node_name"].tolist()
        name_to_orig_idx = dict(zip(map_df['node_name'], map_df['node_idx']))
    else:
        raise ValueError("graph_node_map.csv not found")
    
    # Get valid nodes (in both graph and cohort)
    valid_nodes = [n for n in all_node_names if n in cohort_nodes]
    logger.info(f"Valid nodes: {len(valid_nodes)} (from {len(all_node_names)} graph nodes)")
    
    # Create orig -> new index mapping
    valid_orig_indices = np.array([name_to_orig_idx[n] for n in valid_nodes])
    max_orig_idx = int(graph_data.edge_index.max().item()) + 1
    orig_to_new_arr = np.full(max_orig_idx, -1, dtype=np.int64)
    for new_idx, orig_idx in enumerate(valid_orig_indices):
        orig_to_new_arr[orig_idx] = new_idx
    
    # Re-index edges
    edge_index_np = graph_data.edge_index.numpy()
    src_new = orig_to_new_arr[edge_index_np[0]]
    dst_new = orig_to_new_arr[edge_index_np[1]]
    valid_edges_mask = (src_new >= 0) & (dst_new >= 0)
    
    edge_index = torch.tensor(
        np.stack([src_new[valid_edges_mask], dst_new[valid_edges_mask]]),
        dtype=torch.long, device=device
    )
    logger.info(f"Re-indexed edges: {edge_index.shape[1]} (from {graph_data.edge_index.shape[1]})")
    
    # Load features - need to select only valid nodes
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        x_full = graph_data.x.float()
        x = x_full[valid_orig_indices].to(device)
    elif args.feature_path:
        logger.info(f"Loading embeddings from {args.feature_path}...")
        emb_data = np.load(args.feature_path)
        x_full = torch.from_numpy(emb_data['embeddings']).float()
        x = x_full[valid_orig_indices].to(device)
    else:
        raise ValueError("No features found")
    
    logger.info(f"Feature matrix shape: {x.shape}")
    
    # Create node_name -> new_idx mapping for labels
    node_to_idx = {name: i for i, name in enumerate(valid_nodes)}
    
    y = torch.zeros(len(valid_nodes), dtype=torch.float).to(device)
    train_mask = torch.zeros(len(valid_nodes), dtype=torch.bool).to(device)
    val_mask = torch.zeros(len(valid_nodes), dtype=torch.bool).to(device)
    test_mask = torch.zeros(len(valid_nodes), dtype=torch.bool).to(device)
    
    for _, row in cohort.iterrows():
        name = row["node_name"]
        if name in node_to_idx:
            idx = node_to_idx[name]
            y[idx] = float(row["readmitted_within_window"])
            split = row.get("split", "train")
            if split == "train": train_mask[idx] = True
            elif split in ["val", "validation"]: val_mask[idx] = True
            elif split == "test": test_mask[idx] = True
    
    logger.info(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
            
    # Optuna Study
    mlflow.set_experiment("mimic_cardiorenal_gnn_hpo")
    
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, args, device, x, edge_index, y, train_mask, val_mask), n_trials=args.n_trials)
    
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info(f"  Value: {best_trial.value}")
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")
        
    # Save best params
    with open(os.path.join(args.save_dir, "gnn_best_params.json"), "w") as f:
        json.dump(best_trial.params, f, indent=4)
    
    # Train final model with best params and evaluate on test
    logger.info("Training final model with best params...")
    best_params = best_trial.params
    
    model = GNNClassifier(
        x.size(1), 
        best_params["hidden_dim"], 
        1, 
        best_params["num_layers"], 
        best_params["dropout"], 
        best_params["use_residual"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_auc = 0
    best_state = None
    
    with mlflow.start_run(run_name="GNN_HPO_Best"):
        mlflow.log_params(best_params)
        
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
                mlflow.log_metrics({"val_auc": val_auc, "train_loss": loss.item()}, step=epoch)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = model.state_dict().copy()
        
        # Load best model and evaluate on test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            probs = torch.sigmoid(out).squeeze()
            
            test_probs = probs[test_mask].cpu().numpy()
            test_y = y[test_mask].cpu().numpy()
            
            test_auc = float(roc_auc_score(test_y, test_probs))
            test_auprc = float(average_precision_score(test_y, test_probs))
            opt_thresh, _ = find_optimal_threshold(y[val_mask].cpu().numpy(), probs[val_mask].cpu().numpy())
            test_preds = (test_probs >= opt_thresh).astype(int)
            test_f1 = float(f1_score(test_y, test_preds))
            
            mlflow.log_metrics({
                "best_val_auc": best_val_auc,
                "test_auc": test_auc,
                "test_auprc": test_auprc,
                "test_f1": test_f1,
                "optimal_threshold": opt_thresh,
            })
            
            logger.info("=" * 60)
            logger.info("Final Test Results:")
            logger.info(f"  Test AUC:   {test_auc:.4f}")
            logger.info(f"  Test AUPRC: {test_auprc:.4f}")
            logger.info(f"  Test F1:    {test_f1:.4f}")
            logger.info("=" * 60)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(args.save_dir, "gnn_best_model.pt"))
        mlflow.pytorch.log_model(model, "model")

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
