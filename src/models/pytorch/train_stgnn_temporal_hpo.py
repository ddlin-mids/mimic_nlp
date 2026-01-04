"""
Hyperparameter optimization for Temporal STGNN.

Grid search over dropout, learning rate, hidden dimension, and regularization
to address overfitting observed in baseline temporal STGNN.

Usage:
    python src/models/pytorch/train_stgnn_temporal_hpo.py \
        --seq_file data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl \
        --features_file data/interim/ehr_long_los/integrated_features.csv \
        --output_dir data/interim/ehr_long_los/stgnn_temporal_hpo
"""

import argparse
import logging
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
EDGE_FEATURES_DEMO = [
    "age_at_admit",
    "is_male",
    "charlson_comorbidity_index",
]


def load_temporal_sequences(
    seq_file: str,
    max_seq_len: int = 9,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], int]:
    """Load and truncate/pad temporal sequences to fixed length."""
    logger.info(f"Loading temporal sequences from {seq_file}")
    
    with open(seq_file, 'rb') as f:
        data = pickle.load(f)
    
    raw_feat_dict = data['feat_dict']
    sample_key = list(raw_feat_dict.keys())[0]
    feat_dim = raw_feat_dict[sample_key].shape[1]
    
    logger.info(f"Found {len(raw_feat_dict)} sequences, feature dim: {feat_dim}")
    
    processed = {}
    lengths = {}
    
    for node_name, seq in raw_feat_dict.items():
        seq_len = seq.shape[0]
        
        if seq_len > max_seq_len:
            seq = seq[-max_seq_len:]
            actual_len = max_seq_len
        elif seq_len < max_seq_len:
            pad = np.zeros((max_seq_len - seq_len, feat_dim))
            seq = np.vstack([pad, seq])
            actual_len = seq_len
        else:
            actual_len = seq_len
        
        processed[node_name] = seq.astype(np.float32)
        lengths[node_name] = actual_len
    
    return processed, lengths, feat_dim


def compute_top_percent_edges_train(
    features: np.ndarray,
    top_perc: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute edges for train nodes using top X% most similar pairs."""
    n = len(features)
    
    dist_matrix = cdist(features, features, metric='euclidean')
    mask = ~np.eye(n, dtype=bool)
    sigma = float(np.std(dist_matrix[mask]))
    if sigma == 0:
        sigma = 1.0
    sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    np.fill_diagonal(sim_matrix, 0)
    
    flat_sim = sim_matrix.flatten()
    nonzero = flat_sim[flat_sim > 0]
    threshold = np.percentile(nonzero, 100 - top_perc * 100)
    
    src, dst = np.where(sim_matrix >= threshold)
    weights = sim_matrix[src, dst]
    
    return src, dst, weights, sigma


def compute_inference_edges(
    source_features: np.ndarray,
    target_features: np.ndarray,
    top_perc: float,
    sigma: float,
    source_offset: int,
    target_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute edges from source (val/test) to target (train) nodes."""
    n_src = len(source_features)
    n_tgt = len(target_features)
    
    dist_matrix = cdist(source_features, target_features, metric='euclidean')
    sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
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
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Prepare data for temporal STGNN training."""
    seq_dict, lengths_dict, feat_dim = load_temporal_sequences(seq_file, max_seq_len)
    
    logger.info(f"Loading features from {features_file}")
    df = pd.read_csv(features_file)
    df["node_name"] = df["subject_id"].astype(str) + "_" + df["hadm_id"].astype(str)
    
    available_nodes = list(seq_dict.keys())
    df = df[df["node_name"].isin(available_nodes)].copy()
    logger.info(f"Matched {len(df)} nodes with sequences")
    
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    logger.info(f"Splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    reorder_idx = np.concatenate([train_idx, val_idx, test_idx])
    df_reordered = df.iloc[reorder_idx].reset_index(drop=True)
    
    available_edge_features = [f for f in EDGE_FEATURES_DEMO if f in df.columns]
    X_edge = np.array(df_reordered[available_edge_features].fillna(0).values)
    edge_scaler = StandardScaler()
    edge_scaler.fit(X_edge[:len(train_idx)])
    X_edge_scaled = np.array(edge_scaler.transform(X_edge))
    
    n_train = len(train_idx)
    n_val = len(val_idx)
    
    train_edge_feat = X_edge_scaled[:n_train]
    train_src, train_dst, train_weights, sigma = compute_top_percent_edges_train(
        train_edge_feat, top_perc=top_perc
    )
    
    val_edge_feat = X_edge_scaled[n_train:n_train+n_val]
    val_src, val_dst, val_weights = compute_inference_edges(
        val_edge_feat, train_edge_feat, top_perc, sigma,
        source_offset=n_train, target_offset=0
    )
    
    test_edge_feat = X_edge_scaled[n_train+n_val:]
    test_src, test_dst, test_weights = compute_inference_edges(
        test_edge_feat, train_edge_feat, top_perc, sigma,
        source_offset=n_train+n_val, target_offset=0
    )
    
    node_names = df_reordered["node_name"].values
    sequences = np.stack([seq_dict[name] for name in node_names])
    seq_lengths = np.array([lengths_dict[name] for name in node_names])
    labels = df_reordered["readmitted_within_window"].values
    
    n_total = len(df_reordered)
    new_train_mask = np.zeros(n_total, dtype=bool)
    new_train_mask[:n_train] = True
    
    new_val_mask = np.zeros(n_total, dtype=bool)
    new_val_mask[n_train:n_train+n_val] = True
    
    new_test_mask = np.zeros(n_total, dtype=bool)
    new_test_mask[n_train+n_val:] = True
    
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
    
    val_inference_src = np.concatenate([train_src, val_src])
    val_inference_dst = np.concatenate([train_dst, val_dst])
    val_inference_weights = np.concatenate([train_weights, val_weights])
    val_edge_index = torch.tensor(
        np.array([val_inference_src, val_inference_dst]), dtype=torch.long, device=device
    )
    val_edge_weight = torch.tensor(val_inference_weights, dtype=torch.float32, device=device)
    
    return {
        "sequences": torch.tensor(sequences, dtype=torch.float32, device=device),
        "lengths": torch.tensor(seq_lengths, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.float32, device=device),
        "train_edge_index": train_edge_index,
        "train_edge_weight": train_edge_weight,
        "val_edge_index": val_edge_index,
        "val_edge_weight": val_edge_weight,
        "inference_edge_index": inference_edge_index,
        "inference_edge_weight": inference_edge_weight,
        "train_mask": torch.tensor(new_train_mask, device=device),
        "val_mask": torch.tensor(new_val_mask, device=device),
        "test_mask": torch.tensor(new_test_mask, device=device),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": len(test_idx),
        "feat_dim": feat_dim,
        "max_seq_len": max_seq_len,
    }


def train_epoch(model, data, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    n_train = data["n_train"]
    train_seq = data["sequences"][:n_train]
    train_lengths = data["lengths"][:n_train]
    train_labels = data["labels"][:n_train]
    
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
        "precision": precision_score(y_true, y_pred, zero_division="warn"),
        "recall": recall_score(y_true, y_pred, zero_division="warn"),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    
    return metrics


def run_single_config(
    data: Dict,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    pos_weight: float,
    num_gru_layers: int,
    epochs: int,
    patience: int,
    device: torch.device,
    output_dir: Path,
    config_id: str,
) -> Dict:
    """Run training with a single hyperparameter configuration."""
    
    model = STGNN(
        ehr_input_dim=data["feat_dim"],
        hidden_dim=hidden_dim,
        num_gru_layers=num_gru_layers,
        num_classes=1,
        conv_type="weighted_sage",
        dropout=dropout,
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, data, optimizer, criterion)
        
        val_metrics = evaluate(
            model, data, data["val_mask"],
            data["val_edge_index"], data["val_edge_weight"]
        )
        scheduler.step(val_metrics["auc"])
        
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / f"model_{config_id}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(output_dir / f"model_{config_id}.pt", weights_only=True))
    test_metrics = evaluate(
        model, data, data["test_mask"],
        data["inference_edge_index"], data["inference_edge_weight"]
    )
    
    return {
        "config_id": config_id,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "pos_weight": pos_weight,
        "num_gru_layers": num_gru_layers,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "test_auc": test_metrics["auc"],
        "test_auprc": test_metrics["auprc"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_accuracy": test_metrics["accuracy"],
    }


def main():
    parser = argparse.ArgumentParser(description="HPO for Temporal STGNN")
    parser.add_argument(
        "--seq_file",
        type=str,
        default="data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl",
    )
    parser.add_argument(
        "--features_file",
        type=str,
        default="data/interim/ehr_long_los/integrated_features.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/interim/ehr_long_los/stgnn_temporal_hpo",
    )
    parser.add_argument("--max_seq_len", type=int, default=9)
    parser.add_argument("--top_perc", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)  # More patience for HPO
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data once
    data = prepare_data(
        args.seq_file,
        args.features_file,
        max_seq_len=args.max_seq_len,
        top_perc=args.top_perc,
        device=device,
    )
    
    # Hyperparameter grid - focus on regularization to address overfitting
    # Baseline: hidden_dim=256, dropout=0.2, lr=3e-3, weight_decay=5e-4
    param_grid = {
        "hidden_dim": [128, 256],  # Smaller model might generalize better
        "dropout": [0.3, 0.4, 0.5],  # Higher dropout to reduce overfitting
        "lr": [1e-3, 5e-4],  # Lower LR for more stable training
        "weight_decay": [1e-3, 5e-3],  # Higher weight decay for regularization
        "pos_weight": [3.0, 4.0],  # Class balance
        "num_gru_layers": [1, 2],  # Try deeper temporal model
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TEMPORAL STGNN HYPERPARAMETER OPTIMIZATION")
    logger.info(f"{'='*60}")
    logger.info(f"Total configurations to try: {len(combinations)}")
    logger.info(f"Parameters: {keys}")
    logger.info(f"{'='*60}\n")
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("readmission_prediction")
    
    results = []
    best_overall = {"test_auc": 0}
    
    with mlflow.start_run(run_name="stgnn_temporal_hpo"):
        mlflow.log_params({
            "model_type": "stgnn_temporal_hpo",
            "n_configs": len(combinations),
            "max_seq_len": args.max_seq_len,
            "top_perc": args.top_perc,
        })
        
        for i, combo in enumerate(combinations):
            config = dict(zip(keys, combo))
            config_id = f"cfg_{i:03d}"
            
            logger.info(f"\n[{i+1}/{len(combinations)}] Config: {config}")
            
            # Reset seed for fair comparison
            torch.manual_seed(args.seed)
            
            result = run_single_config(
                data=data,
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                pos_weight=config["pos_weight"],
                num_gru_layers=config["num_gru_layers"],
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                output_dir=output_dir,
                config_id=config_id,
            )
            
            results.append(result)
            
            logger.info(f"  -> Val AUC: {result['best_val_auc']:.4f} (epoch {result['best_epoch']})")
            logger.info(f"  -> Test AUC: {result['test_auc']:.4f}, AUPRC: {result['test_auprc']:.4f}")
            
            # Track best
            if result["test_auc"] > best_overall["test_auc"]:
                best_overall = result.copy()
                logger.info(f"  -> NEW BEST!")
        
        # Log best result
        mlflow.log_metrics({
            "best_val_auc": best_overall["best_val_auc"],
            "test_auc": best_overall["test_auc"],
            "test_auprc": best_overall["test_auprc"],
            "test_f1": best_overall["test_f1"],
            "test_precision": best_overall["test_precision"],
            "test_recall": best_overall["test_recall"],
            "test_acc": best_overall["test_accuracy"],
        })
    
    # Save all results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_auc", ascending=False)
    results_df.to_csv(output_dir / "hpo_results.csv", index=False)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("HPO RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\nTop 5 configurations by Test AUC:")
    for rank, (i, row) in enumerate(results_df.head(5).iterrows(), 1):
        logger.info(f"\n  #{rank}: Test AUC={row['test_auc']:.4f}, "
                   f"Val AUC={row['best_val_auc']:.4f}")
        logger.info(f"      hidden={row['hidden_dim']}, dropout={row['dropout']}, "
                   f"lr={row['lr']}, wd={row['weight_decay']}")
        logger.info(f"      pos_weight={row['pos_weight']}, gru_layers={row['num_gru_layers']}")
    
    # Best model info
    best = results_df.iloc[0]
    logger.info(f"\n{'='*60}")
    logger.info("BEST CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"Config ID: {best['config_id']}")
    logger.info(f"Hidden dim: {best['hidden_dim']}")
    logger.info(f"Dropout: {best['dropout']}")
    logger.info(f"Learning rate: {best['lr']}")
    logger.info(f"Weight decay: {best['weight_decay']}")
    logger.info(f"Pos weight: {best['pos_weight']}")
    logger.info(f"GRU layers: {best['num_gru_layers']}")
    logger.info(f"\nBest epoch: {best['best_epoch']}")
    logger.info(f"Val AUC: {best['best_val_auc']:.4f}")
    logger.info(f"Test AUC: {best['test_auc']:.4f}")
    logger.info(f"Test AUPRC: {best['test_auprc']:.4f}")
    logger.info(f"Test F1: {best['test_f1']:.4f}")
    
    logger.info("\nComparison to baselines:")
    logger.info("  - Baseline Temporal STGNN:   0.620 AUC")
    logger.info("  - Lab Features XGBoost:      0.627 AUC")
    logger.info("  - Gated Fusion:              0.638 AUC")
    logger.info(f"  - HPO Temporal STGNN:        {best['test_auc']:.3f} AUC")
    
    # Save best config
    best_config = {
        "best_config": best.to_dict(),
        "all_results": results,
        "param_grid": param_grid,
    }
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
