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

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers - 1):
            x_in = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual and x_in.shape == x.shape:
                x = x + x_in
        
        # Final layer
        x = self.convs[-1](x, edge_index)
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
    
    # 2. Load Features
    # If args.feature_path is provided, try to load it. But graph_data might already have x.
    # The previous script loaded separate embeddings. 
    # With preprocess_graph_multimodal, node features are saved in graph_pyg.pt (data.x)
    # BUT we might want to load external embeddings if we didn't save them in graph.
    
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        logger.info("Using node features from graph file.")
        x = graph_data.x.float().to(device)
        # We need node names to align labels. They should be in mapping file.
        # The previous version loaded them from feature .npz. 
        # Now we must rely on args.feature_path usually containing the node_names if we use that, 
        # OR we load the mapping file.
    elif args.feature_path:
        logger.info(f"Loading embeddings from {args.feature_path}...")
        emb_data = np.load(args.feature_path)
        x = torch.from_numpy(emb_data['embeddings']).float().to(device)
        # Assuming node_names are consistent if we use external features
    else:
        raise ValueError("No features found in graph and no feature_path provided.")

    # We need node_names to map labels. 
    # Let's try to load them from a mapping file if it exists in the graph directory
    graph_dir = os.path.dirname(args.graph_path)
    mapping_path = os.path.join(graph_dir, "graph_node_map.csv")
    if os.path.exists(mapping_path):
        logger.info(f"Loading node mapping from {mapping_path}")
        map_df = pd.read_csv(mapping_path)
        node_names = map_df["node_name"].tolist()
    elif args.feature_path and os.path.exists(args.feature_path):
         emb_data = np.load(args.feature_path)
         if 'node_names' in emb_data:
             node_names = emb_data['node_names']
         else:
             raise ValueError("No node_names found in feature file.")
    else:
        raise ValueError("Could not determine node names for label alignment.")

    # 3. Load Labels & Masks
    logger.info(f"Loading cohort from {args.cohort_path}...")
    cohort = pd.read_csv(args.cohort_path)
    cohort["node_name"] = cohort["subject_id"].astype(str) + "_" + cohort["hadm_id"].astype(str)
    
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    
    y = torch.zeros(len(node_names), dtype=torch.float).to(device)
    train_mask = torch.zeros(len(node_names), dtype=torch.bool).to(device)
    val_mask = torch.zeros(len(node_names), dtype=torch.bool).to(device)
    test_mask = torch.zeros(len(node_names), dtype=torch.bool).to(device)
    
    match_count = 0
    for _, row in cohort.iterrows():
        name = row["node_name"]
        if name in node_to_idx:
            idx = node_to_idx[name]
            y[idx] = float(row["readmitted_within_window"])
            split = row.get("split", "train")
            if split == "train": train_mask[idx] = True
            elif split in ["val", "validation"]: val_mask[idx] = True
            elif split == "test": test_mask[idx] = True
            match_count += 1
            
    logger.info(f"Total Nodes: {len(node_names)}")
    logger.info(f"Matched Cohort Entries: {match_count}")
    logger.info(f"Train Size: {train_mask.sum().item()}")
    logger.info(f"Val Size: {val_mask.sum().item()}")
    logger.info(f"Test Size: {test_mask.sum().item()}")
    
    if train_mask.sum().item() == 0:
        logger.error("Train mask is empty! Check cohort node_name matching.")
        return

    model = GNNClassifier(x.size(1), args.hidden_dim, 1, args.num_layers, args.dropout, args.use_residual).to(device)
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
                
                val_auc = float(roc_auc_score(y[val_mask].cpu(), probs[val_mask].cpu()))
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch+1}: Loss {loss.item():.4f}, Val AUC {val_auc:.4f}")
                mlflow.log_metrics({"train_loss": loss.item(), "val_auc": val_auc}, step=epoch)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    no_improve = 0
                    
                    # Compute all metrics
                    test_probs = probs[test_mask].cpu().numpy()
                    test_y = y[test_mask].cpu().numpy()
                    test_preds = (test_probs > 0.5).astype(int)
                    
                    test_auc = float(roc_auc_score(test_y, test_probs))
                    test_ap = float(average_precision_score(test_y, test_probs))
                    test_f1 = float(f1_score(test_y, test_preds))
                    
                    from sklearn.metrics import precision_score, recall_score, accuracy_score
                    test_precision = float(precision_score(test_y, test_preds, zero_division=0))
                    test_recall = float(recall_score(test_y, test_preds))
                    test_acc = float(accuracy_score(test_y, test_preds))
                    
                    mlflow.log_metrics({
                        "test_auc": test_auc, 
                        "test_auprc": test_ap, 
                        "test_f1": test_f1,
                        "test_precision": test_precision,
                        "test_recall": test_recall,
                        "test_acc": test_acc
                    })
                else:
                    no_improve += 1
                    if no_improve >= patience: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, required=False, help="Path to embeddings (optional if in graph)")
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="data/interim/ehr_long_los/models")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--use_residual", action="store_true", default=True)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train_model(args)
