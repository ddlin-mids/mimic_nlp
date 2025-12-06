import os
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index) # SAGEConv doesn't strictly need edge_weight for mean agg
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load Graph
    logger.info(f"Loading graph from {args.graph_path}...")
    graph_data = torch.load(args.graph_path, weights_only=False)
    edge_index = graph_data.edge_index.to(device)
    # SAGEConv (mean) usually doesn't use edge weights, but we could extend if needed.
    
    # 2. Load Features
    logger.info(f"Loading embeddings from {args.feature_path}...")
    emb_data = np.load(args.feature_path)
    x = torch.from_numpy(emb_data['embeddings']).float().to(device)
    node_names = emb_data['node_names']
    
    # 3. Load Labels & Masks
    logger.info(f"Loading cohort from {args.cohort_path}...")
    cohort = pd.read_csv(args.cohort_path)
    cohort["node_name"] = cohort["subject_id"].astype(str) + "_" + cohort["hadm_id"].astype(str)
    
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    
    y = torch.zeros(len(node_names), dtype=torch.float).to(device)
    train_mask = torch.zeros(len(node_names), dtype=torch.bool).to(device)
    val_mask = torch.zeros(len(node_names), dtype=torch.bool).to(device)
    test_mask = torch.zeros(len(node_names), dtype=torch.bool).to(device)
    
    for _, row in cohort.iterrows():
        name = row["node_name"]
        if name in node_to_idx:
            idx = node_to_idx[name]
            y[idx] = float(row["readmitted_within_window"])
            split = row.get("split", "train")
            if split == "train": train_mask[idx] = True
            elif split in ["val", "validation"]: val_mask[idx] = True
            elif split == "test": test_mask[idx] = True
            
    model = GNNClassifier(x.size(1), args.hidden_dim, 1, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_auc = 0
    patience = 20
    no_improve = 0
    
    mlflow.set_experiment("mimic_cardiorenal_gnn")
    with mlflow.start_run():
        mlflow.set_tag("embedding_source", "transformer")
        mlflow.set_tag("framework", "pyg")
        mlflow.log_params(vars(args))
        
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
                
                val_auc = roc_auc_score(y[val_mask].cpu(), probs[val_mask].cpu())
                
                logger.info(f"Epoch {epoch+1}: Loss {loss.item():.4f}, Val AUC {val_auc:.4f}")
                mlflow.log_metrics({"train_loss": loss.item(), "val_auc": val_auc}, step=epoch)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    no_improve = 0
                    
                    test_auc = roc_auc_score(y[test_mask].cpu(), probs[test_mask].cpu())
                    test_ap = average_precision_score(y[test_mask].cpu(), probs[test_mask].cpu())
                    test_f1 = f1_score(y[test_mask].cpu(), (probs[test_mask].cpu() > 0.5).int())
                    
                    mlflow.log_metrics({"test_auc": test_auc, "test_auprc": test_ap, "test_f1": test_f1})
                else:
                    no_improve += 1
                    if no_improve >= patience: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="data/interim/ehr_long_los/models")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train_model(args)
