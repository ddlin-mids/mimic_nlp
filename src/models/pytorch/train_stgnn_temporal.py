"""
Temporal STGNN training script using daily EHR sequences.

Matches reference MM-STGNN architecture with:
- 9-day temporal sequences (last 9 days of stay)
- Top 1% edges by demographic similarity (Gaussian kernel)
- Weighted GraphSAGE in GConvGRU
- Inductive learning setup

Reference: Tang et al. (2022) - Multimodal spatiotemporal graph neural 
networks for improved prediction of 30-day all-cause hospital readmission.

Usage:
    python src/models/pytorch/train_stgnn_temporal.py \
        --seq_file data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl \
        --features_file data/interim/ehr_long_los/integrated_features.csv \
        --output_dir data/interim/ehr_long_los/stgnn_temporal
"""

import argparse
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import mlflow

# Handle both direct execution and module import
try:
    from stgnn import STGNN
except ImportError:
    from src.models.pytorch.stgnn import STGNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Features to use for edge construction (demographics only, like reference)
# Reference uses: age, gender, ethnicity
EDGE_FEATURES_DEMO = [
    "age_at_admit",
    "is_male",
    "charlson_comorbidity_index",  # Additional clinical similarity
]


def load_temporal_sequences(
    seq_file: str,
    max_seq_len: int = 9,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], int]:
    """
    Load and truncate/pad temporal sequences to fixed length.
    
    Like reference, we take the LAST max_seq_len days of stay.
    Shorter sequences are front-padded with zeros.
    
    Args:
        seq_file: Path to temporal sequence pickle file
        max_seq_len: Maximum sequence length (reference uses 9)
        
    Returns:
        feat_dict: {node_name: (max_seq_len, feat_dim)}
        lengths: {node_name: actual_length}
        feat_dim: Feature dimension
    """
    logger.info(f"Loading temporal sequences from {seq_file}")
    
    with open(seq_file, 'rb') as f:
        data = pickle.load(f)
    
    raw_feat_dict = data['feat_dict']
    
    # Get feature dimension from first sample
    sample_key = list(raw_feat_dict.keys())[0]
    feat_dim = raw_feat_dict[sample_key].shape[1]
    
    logger.info(f"Found {len(raw_feat_dict)} sequences, feature dim: {feat_dim}")
    
    # Process sequences
    processed = {}
    lengths = {}
    
    for node_name, seq in raw_feat_dict.items():
        seq_len = seq.shape[0]
        
        if seq_len > max_seq_len:
            # Take LAST max_seq_len days (most recent, like reference)
            seq = seq[-max_seq_len:]
            actual_len = max_seq_len
        elif seq_len < max_seq_len:
            # Front-pad with zeros (reference: pad_front=False means front-pad)
            pad = np.zeros((max_seq_len - seq_len, feat_dim))
            seq = np.vstack([pad, seq])
            actual_len = seq_len
        else:
            actual_len = seq_len
        
        processed[node_name] = seq.astype(np.float32)
        lengths[node_name] = actual_len
    
    # Statistics
    all_lengths = list(lengths.values())
    logger.info(f"Sequence lengths - min: {min(all_lengths)}, max: {max(all_lengths)}, "
                f"mean: {np.mean(all_lengths):.1f}")
    
    return processed, lengths, feat_dim


def compute_top_percent_edges_train(
    features: np.ndarray,
    top_perc: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute edges for train nodes using top X% most similar pairs.
    
    Matches reference: refs/readmit-stgnn/data/readmission_utils.py
    
    Args:
        features: Edge features for train nodes, shape (n_train, feat_dim)
        top_perc: Top percentage of edges to keep
        
    Returns:
        src, dst, weights, sigma (for consistent scaling)
    """
    n = len(features)
    logger.info(f"Computing top {top_perc*100}% edges for {n} train nodes...")
    
    # Compute pairwise Euclidean distances
    dist_matrix = cdist(features, features, metric='euclidean')
    
    # Apply Gaussian kernel (like reference)
    mask = ~np.eye(n, dtype=bool)
    sigma = np.std(dist_matrix[mask])
    if sigma == 0:
        sigma = 1.0
    sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    
    # Zero diagonal (no self-loops)
    np.fill_diagonal(sim_matrix, 0)
    
    # Find threshold for top X% edges
    flat_sim = sim_matrix.flatten()
    nonzero = flat_sim[flat_sim > 0]
    threshold = np.percentile(nonzero, 100 - top_perc * 100)
    
    # Get edges above threshold
    src, dst = np.where(sim_matrix >= threshold)
    weights = sim_matrix[src, dst]
    
    avg_neighbors = len(src) / n
    logger.info(f"Train edges: {len(src):,}, avg neighbors: {avg_neighbors:.1f}")
    
    return src, dst, weights, sigma


def compute_inference_edges(
    source_features: np.ndarray,
    target_features: np.ndarray,
    top_perc: float,
    sigma: float,
    source_offset: int,
    target_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute edges from source (val/test) to target (train) nodes.
    
    Args:
        source_features: Features of val/test nodes
        target_features: Features of train nodes
        top_perc: Top percentage of edges per source node
        sigma: Sigma from train data (for consistent scaling)
        source_offset: Index offset for source nodes
        target_offset: Index offset for target nodes
        
    Returns:
        src, dst, weights
    """
    n_src = len(source_features)
    n_tgt = len(target_features)
    
    # Compute distances
    dist_matrix = cdist(source_features, target_features, metric='euclidean')
    
    # Apply Gaussian kernel with train sigma
    sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    
    # For each source, keep top X% connections to targets
    k = max(1, int(n_tgt * top_perc))
    
    src_list = []
    dst_list = []
    weight_list = []
    
    for i in range(n_src):
        sims = sim_matrix[i, :]
        top_k_idx = np.argsort(sims)[-k:]
        
        for j in top_k_idx:
            if sims[j] > 0:
                src_list.append(i + source_offset)
                dst_list.append(j + target_offset)
                weight_list.append(sims[j])
    
    return np.array(src_list), np.array(dst_list), np.array(weight_list)


def prepare_data(
    seq_file: str,
    features_file: str,
    max_seq_len: int = 9,
    top_perc: float = 0.01,
    device: torch.device = None,
) -> Dict:
    """
    Prepare data for temporal STGNN training.
    
    Args:
        seq_file: Path to temporal sequence pickle
        features_file: Path to integrated features CSV (for splits and edge features)
        max_seq_len: Maximum sequence length
        top_perc: Top percentage of edges to keep
        device: Torch device
        
    Returns:
        Dictionary with all training data
    """
    # Load temporal sequences
    seq_dict, lengths_dict, feat_dim = load_temporal_sequences(seq_file, max_seq_len)
    
    # Load features for splits and edge construction
    logger.info(f"Loading features from {features_file}")
    df = pd.read_csv(features_file)
    
    # Create node_name column (like reference uses subject_id_hadm_id)
    df["node_name"] = df["subject_id"].astype(str) + "_" + df["hadm_id"].astype(str)
    
    # Filter to nodes that have sequences
    available_nodes = set(seq_dict.keys())
    df = df[df["node_name"].isin(available_nodes)].copy()
    logger.info(f"Matched {len(df)} nodes with sequences")
    
    # Get splits
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    logger.info(f"Splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    # Reorder data: train, val, test
    reorder_idx = np.concatenate([train_idx, val_idx, test_idx])
    df_reordered = df.iloc[reorder_idx].reset_index(drop=True)
    
    # Get edge features (demographics for edge construction)
    available_edge_features = [f for f in EDGE_FEATURES_DEMO if f in df.columns]
    logger.info(f"Using {len(available_edge_features)} edge features: {available_edge_features}")
    
    X_edge = df_reordered[available_edge_features].fillna(0).values
    edge_scaler = StandardScaler()
    edge_scaler.fit(X_edge[:len(train_idx)])  # Fit on train only
    X_edge_scaled = edge_scaler.transform(X_edge)
    
    # Build train graph (top X% edges)
    train_edge_feat = X_edge_scaled[:len(train_idx)]
    train_src, train_dst, train_weights, sigma = compute_top_percent_edges_train(
        train_edge_feat, top_perc=top_perc
    )
    
    # Build inference edges (val/test -> train)
    n_train = len(train_idx)
    n_val = len(val_idx)
    
    val_edge_feat = X_edge_scaled[n_train:n_train+n_val]
    val_src, val_dst, val_weights = compute_inference_edges(
        val_edge_feat, train_edge_feat, top_perc, sigma,
        source_offset=n_train, target_offset=0
    )
    logger.info(f"Val->Train edges: {len(val_weights)}")
    
    test_edge_feat = X_edge_scaled[n_train+n_val:]
    test_src, test_dst, test_weights = compute_inference_edges(
        test_edge_feat, train_edge_feat, top_perc, sigma,
        source_offset=n_train+n_val, target_offset=0
    )
    logger.info(f"Test->Train edges: {len(test_weights)}")
    
    # Prepare sequences in order
    node_names = df_reordered["node_name"].values
    sequences = np.stack([seq_dict[name] for name in node_names])  # (n_nodes, seq_len, feat_dim)
    seq_lengths = np.array([lengths_dict[name] for name in node_names])
    
    # Get labels
    labels = df_reordered["readmitted_within_window"].values
    
    # Create masks
    n_total = len(df_reordered)
    new_train_mask = np.zeros(n_total, dtype=bool)
    new_train_mask[:n_train] = True
    
    new_val_mask = np.zeros(n_total, dtype=bool)
    new_val_mask[n_train:n_train+n_val] = True
    
    new_test_mask = np.zeros(n_total, dtype=bool)
    new_test_mask[n_train+n_val:] = True
    
    # Convert to tensors
    train_edge_index = torch.tensor(
        np.array([train_src, train_dst]), dtype=torch.long, device=device
    )
    train_edge_weight = torch.tensor(train_weights, dtype=torch.float32, device=device)
    
    inference_src = np.concatenate([train_src, val_src, test_src])
    inference_dst = np.concatenate([train_dst, val_dst, test_dst])
    inference_weights = np.concatenate([train_weights, val_weights, test_weights])
    
    inference_edge_index = torch.tensor(
        np.array([inference_src, inference_dst]), dtype=torch.long, device=device
    )
    inference_edge_weight = torch.tensor(inference_weights, dtype=torch.float32, device=device)
    
    # Create val-only and test-only inference graphs
    val_inference_src = np.concatenate([train_src, val_src])
    val_inference_dst = np.concatenate([train_dst, val_dst])
    val_inference_weights = np.concatenate([train_weights, val_weights])
    val_edge_index = torch.tensor(
        np.array([val_inference_src, val_inference_dst]), dtype=torch.long, device=device
    )
    val_edge_weight = torch.tensor(val_inference_weights, dtype=torch.float32, device=device)
    
    return {
        # Sequences
        "sequences": torch.tensor(sequences, dtype=torch.float32, device=device),
        "lengths": torch.tensor(seq_lengths, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.float32, device=device),
        
        # Train graph
        "train_edge_index": train_edge_index,
        "train_edge_weight": train_edge_weight,
        
        # Val inference graph  
        "val_edge_index": val_edge_index,
        "val_edge_weight": val_edge_weight,
        
        # Full inference graph
        "inference_edge_index": inference_edge_index,
        "inference_edge_weight": inference_edge_weight,
        
        # Masks
        "train_mask": torch.tensor(new_train_mask, device=device),
        "val_mask": torch.tensor(new_val_mask, device=device),
        "test_mask": torch.tensor(new_test_mask, device=device),
        
        # Sizes
        "n_train": n_train,
        "n_val": n_val,
        "n_test": len(test_idx),
        "feat_dim": feat_dim,
        "max_seq_len": max_seq_len,
    }


def train_epoch(model, data, optimizer, criterion, activation='elu'):
    """
    Train for one epoch using train subgraph.
    
    Uses only train sequences and train-train edges.
    """
    model.train()
    optimizer.zero_grad()
    
    n_train = data["n_train"]
    
    # Get train data only
    train_seq = data["sequences"][:n_train]  # (n_train, seq_len, feat_dim)
    train_lengths = data["lengths"][:n_train]
    train_labels = data["labels"][:n_train]
    
    # Forward pass with train graph
    logits, _ = model(
        train_seq,
        data["train_edge_index"],
        data["train_edge_weight"],
        lengths=train_lengths,
    )
    
    loss = criterion(logits, train_labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, data, mask, edge_index, edge_weight, threshold=0.5):
    """Evaluate model on a specific split."""
    model.eval()
    
    with torch.no_grad():
        # Need to run on all nodes up to the mask
        # For val: use all nodes up to n_train + n_val
        # For test: use all nodes
        
        logits, _ = model(
            data["sequences"],
            edge_index,
            edge_weight,
            lengths=data["lengths"],
        )
        probs = torch.sigmoid(logits)
    
    y_true = data["labels"][mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "auc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train temporal STGNN (matching reference)")
    parser.add_argument(
        "--seq_file",
        type=str,
        default="data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl",
        help="Path to temporal sequence pickle"
    )
    parser.add_argument(
        "--features_file",
        type=str,
        default="data/interim/ehr_long_los/integrated_features.csv",
        help="Path to integrated features CSV (for splits and edge features)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/interim/ehr_long_los/stgnn_temporal",
        help="Output directory"
    )
    
    # Architecture (matching reference)
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension (reference: 256)")
    parser.add_argument("--num_gru_layers", type=int, default=1,
                        help="Number of GRU layers (reference: 1)")
    parser.add_argument("--max_seq_len", type=int, default=9,
                        help="Maximum sequence length (reference: 9)")
    parser.add_argument("--conv_type", type=str, default="weighted_sage",
                        choices=["weighted_sage", "sage", "gcn", "gat"],
                        help="Graph convolution type")
    
    # Training (matching reference)
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout (reference: 0.2)")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate (reference: 3e-3)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (reference: 5e-4)")
    parser.add_argument("--pos_weight", type=float, default=4.0,
                        help="Positive class weight (reference: 4)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs (reference: 100)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (reference: 10)")
    
    # Graph construction (matching reference)
    parser.add_argument("--top_perc", type=float, default=0.01,
                        help="Top percentage of edges (reference: 0.01 = 1%%)")
    
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
    
    # Prepare data
    data = prepare_data(
        args.seq_file,
        args.features_file,
        max_seq_len=args.max_seq_len,
        top_perc=args.top_perc,
        device=device,
    )
    
    # Create model
    model = STGNN(
        ehr_input_dim=data["feat_dim"],
        hidden_dim=args.hidden_dim,
        num_gru_layers=args.num_gru_layers,
        num_classes=1,
        conv_type=args.conv_type,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Loss with class weight
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("readmission_prediction")
    
    with mlflow.start_run(run_name="stgnn_temporal"):
        # Log parameters
        mlflow.log_params({
            "model_type": "stgnn_temporal_gconvgru",
            "conv_type": args.conv_type,
            "hidden_dim": args.hidden_dim,
            "num_gru_layers": args.num_gru_layers,
            "max_seq_len": args.max_seq_len,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pos_weight": args.pos_weight,
            "top_perc": args.top_perc,
            "feat_dim": data["feat_dim"],
            "n_train": data["n_train"],
            "n_val": data["n_val"],
            "n_test": data["n_test"],
        })
        
        # Training loop
        best_val_auc = 0
        best_metrics = None
        patience_counter = 0
        
        logger.info("\n" + "="*60)
        logger.info("TEMPORAL STGNN - GConvGRU with weighted GraphSAGE")
        logger.info(f"Architecture: {args.hidden_dim}d hidden, {args.num_gru_layers} GRU layers")
        logger.info(f"Sequences: {args.max_seq_len} timesteps, {data['feat_dim']} features")
        logger.info(f"Graph: top {args.top_perc*100}% edges")
        logger.info("="*60 + "\n")
        
        for epoch in range(args.epochs):
            # Train
            train_loss = train_epoch(model, data, optimizer, criterion)
            
            # Evaluate on val
            val_metrics = evaluate(
                model, data, data["val_mask"],
                data["val_edge_index"], data["val_edge_weight"]
            )
            scheduler.step(val_metrics["auc"])
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                    f"Val AUC: {val_metrics['auc']:.4f} | "
                    f"Val AUPRC: {val_metrics['auprc']:.4f}"
                )
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_auc": val_metrics["auc"],
                }, step=epoch)
            
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
        model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
        test_metrics = evaluate(
            model, data, data["test_mask"],
            data["inference_edge_index"], data["inference_edge_weight"]
        )
        
        # Log final metrics
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
        logger.info("FINAL RESULTS - TEMPORAL STGNN")
        logger.info("="*60)
        logger.info(f"Best Val AUC:   {best_val_auc:.4f}")
        logger.info(f"Test AUC:       {test_metrics['auc']:.4f}")
        logger.info(f"Test AUPRC:     {test_metrics['auprc']:.4f}")
        logger.info(f"Test F1:        {test_metrics['f1']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
        
        logger.info("\nComparison to baselines:")
        logger.info("  - Lab Features XGBoost:      0.627 AUC")
        logger.info("  - Gated Fusion (Neural):     0.638 AUC")
        logger.info("  - STGNN Static (Inductive):  0.585 AUC")
        logger.info(f"  - STGNN Temporal (This):     {test_metrics['auc']:.3f} AUC")
        logger.info("\nReference paper (MM-STGNN):    0.79 AUC (different cohort)")
        
        # Save results
        results = {
            "val": best_metrics,
            "test": test_metrics,
            "args": vars(args),
            "architecture": {
                "type": "temporal_stgnn_gconvgru",
                "conv_type": args.conv_type,
                "hidden_dim": args.hidden_dim,
                "num_gru_layers": args.num_gru_layers,
                "max_seq_len": args.max_seq_len,
            },
            "note": "Temporal GConvGRU matching reference architecture"
        }
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
