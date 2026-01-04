"""
Training script for STGNN on Integrated Features.

Since we have static (non-temporal) features, this uses a simplified
GraphSAGE-based approach rather than the full GConvGRU.

Usage:
    python src/models/pytorch/train_stgnn_integrated.py \
        --graph_path data/interim/ehr_long_los/patient_graph.pkl \
        --output_dir data/interim/ehr_long_los/stgnn_integrated
"""

import argparse
import logging
import pickle
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from torch_geometric.nn import SAGEConv, GATConv
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphSAGEClassifier(nn.Module):
    """
    Simple GraphSAGE classifier for node classification.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_classes: int = 1,
        dropout: float = 0.3,
        aggregator: str = "mean"
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggregator))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        logits = self.classifier(x)
        return logits.squeeze(-1)


def load_graph_data(graph_path: str, device: torch.device):
    """Load graph data and prepare for training."""
    logger.info(f"Loading graph from {graph_path}")
    
    with open(graph_path, "rb") as f:
        graph_data = pickle.load(f)
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(
        np.array([graph_data["src_nodes"], graph_data["dst_nodes"]]),
        dtype=torch.long,
        device=device
    )
    
    edge_weight = torch.tensor(
        graph_data["weights"],
        dtype=torch.float32,
        device=device
    )
    
    node_features = torch.tensor(
        graph_data["node_features"],
        dtype=torch.float32,
        device=device
    )
    
    labels = torch.tensor(
        graph_data["labels"],
        dtype=torch.float32,
        device=device
    )
    
    train_mask = torch.tensor(graph_data["train_mask"], device=device)
    val_mask = torch.tensor(graph_data["val_mask"], device=device)
    test_mask = torch.tensor(graph_data["test_mask"], device=device)
    
    logger.info(f"Graph: {node_features.shape[0]} nodes, {edge_index.shape[1]} edges")
    logger.info(f"Features: {node_features.shape[1]}")
    
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "x": node_features,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def evaluate(model, data, mask, threshold=0.5):
    """Evaluate model on a specific split."""
    model.eval()
    
    with torch.no_grad():
        logits = model(data["x"], data["edge_index"], data["edge_weight"])
        probs = torch.sigmoid(logits)
    
    y_true = data["labels"][mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "auc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    
    return metrics


def train_epoch(model, data, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    logits = model(data["x"], data["edge_index"], data["edge_weight"])
    loss = criterion(logits[data["train_mask"]], data["labels"][data["train_mask"]])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train STGNN on integrated features")
    parser.add_argument(
        "--graph_path",
        type=str,
        default="data/interim/ehr_long_los/patient_graph.pkl",
        help="Path to patient graph"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/interim/ehr_long_los/stgnn_integrated",
        help="Output directory"
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_graph_data(args.graph_path, device)
    
    # Create model
    in_dim = data["x"].shape[1]
    model = GraphSAGEClassifier(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_classes=1,
        dropout=args.dropout,
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weight for imbalanced data
    pos_weight = (data["labels"][data["train_mask"]] == 0).sum() / \
                 (data["labels"][data["train_mask"]] == 1).sum()
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("readmission_prediction")
    
    with mlflow.start_run(run_name="stgnn_integrated_features"):
        # Log parameters
        mlflow.log_params({
            "model_type": "stgnn_graphsage",
            "feature_set": "integrated",
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "n_features": in_dim,
        })
        
        # Training loop
        best_val_auc = 0
        best_metrics = None
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Train
            train_loss = train_epoch(model, data, optimizer, criterion)
            
            # Evaluate
            val_metrics = evaluate(model, data, data["val_mask"])
            scheduler.step(val_metrics["auc"])
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )
            
            # Early stopping
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_metrics = val_metrics.copy()
                patience_counter = 0
                
                # Save best model
                torch.save(model.state_dict(), output_dir / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and evaluate on test
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))
        test_metrics = evaluate(model, data, data["test_mask"])
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "best_val_auc": best_val_auc,
            "test_auc": test_metrics["auc"],
            "test_auprc": test_metrics["auprc"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_acc": test_metrics["accuracy"],
        })
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Best Val AUC: {best_val_auc:.4f}")
        logger.info(f"Test AUC:     {test_metrics['auc']:.4f}")
        logger.info(f"Test AUPRC:   {test_metrics['auprc']:.4f}")
        logger.info(f"Test F1:      {test_metrics['f1']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
        
        # Compare to baselines
        logger.info("\nComparison to previous results:")
        logger.info("  - Lab Features Only XGBoost: 0.627 AUC")
        logger.info("  - Integrated Features XGB:   0.624 AUC")
        logger.info("  - Gated Fusion (Neural):     0.638 AUC")
        logger.info(f"  - STGNN (This model):        {test_metrics['auc']:.3f} AUC")
        
        # Save results
        results = {
            "val": best_metrics,
            "test": test_metrics,
            "args": vars(args),
        }
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
