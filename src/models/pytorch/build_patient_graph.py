"""
Build Patient Similarity Graph for STGNN.

Constructs a graph where:
- Nodes: Hospital admissions
- Edges: Weighted by patient similarity based on demographics, comorbidities, and labs

Edge construction strategy:
1. Compute pairwise similarities using selected features
2. Keep top-k% of edges to create sparse graph
3. Apply Gaussian kernel for edge weights

Usage:
    python src/models/pytorch/build_patient_graph.py \
        --features_path data/interim/ehr_long_los/integrated_features.csv \
        --output_path data/interim/ehr_long_los/patient_graph.pkl
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Features to use for edge construction (demographics + key clinical markers)
EDGE_FEATURES = [
    # Demographics
    "age_at_admit",
    "is_male",
    "length_of_stay_days",
    "is_emergency_admission",
    # Admission history
    "num_prior_admissions",
    "num_admissions_past_6mo",
    "has_prior_30d_readmit",
    # Risk scores
    "charlson_comorbidity_index",
    "lace_score",
    # Key comorbidities
    "cci_congestive_heart_failure",
    "cci_renal_disease",
    "cci_diabetes_without_complications",
    "cci_chronic_pulmonary_disease",
    "cci_malignancy",
    # ICU markers
    "had_icu_stay",
    "had_dialysis",
    "had_mechanical_vent",
    # Key medications
    "received_loop_diuretic",
    "received_ace_inhibitor",
    "received_beta_blocker",
    "received_insulin",
]


def load_features(features_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load integrated features and identify columns for edge construction."""
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    
    # Find available edge features
    available_edge_features = [f for f in EDGE_FEATURES if f in df.columns]
    logger.info(f"Using {len(available_edge_features)} features for edge construction")
    
    # If we don't have enough edge features, add some lab features
    if len(available_edge_features) < 15:
        lab_features = [c for c in df.columns if any(x in c for x in 
                       ["creatinine_last", "bun_last", "sodium_last", "hemoglobin_last"])]
        available_edge_features.extend(lab_features[:5])
        logger.info(f"Added lab features, now using {len(available_edge_features)} features")
    
    return df, available_edge_features


def compute_similarity_matrix(
    features: np.ndarray,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute pairwise similarity/distance matrix.
    
    Args:
        features: (n_samples, n_features) array
        metric: 'euclidean' or 'cosine'
    
    Returns:
        similarity_matrix: (n_samples, n_samples) array
    """
    logger.info(f"Computing {metric} distances for {features.shape[0]} nodes...")
    
    if metric == "cosine":
        # Cosine similarity (1 = identical, 0 = orthogonal)
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(features)
    else:
        # Euclidean distance -> similarity via Gaussian kernel
        dist_matrix = cdist(features, features, metric="euclidean")
        # Normalize distances
        dist_matrix = dist_matrix / (dist_matrix.max() + 1e-8)
        # Convert to similarity using Gaussian kernel
        sigma = np.median(dist_matrix[dist_matrix > 0])
        sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    
    # Set diagonal to 0 (no self-loops initially)
    np.fill_diagonal(sim_matrix, 0)
    
    return sim_matrix


def construct_sparse_graph(
    sim_matrix: np.ndarray,
    top_k_percent: float = 0.05,
    min_edges_per_node: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct sparse graph by keeping top-k% of edges.
    
    Args:
        sim_matrix: (n, n) similarity matrix
        top_k_percent: Percentage of edges to keep
        min_edges_per_node: Minimum edges per node
    
    Returns:
        src_nodes: Source node indices
        dst_nodes: Destination node indices
        weights: Edge weights
    """
    n = sim_matrix.shape[0]
    logger.info(f"Constructing sparse graph with top {top_k_percent*100:.1f}% edges...")
    
    # For each node, keep at least min_edges_per_node connections
    src_list = []
    dst_list = []
    weight_list = []
    
    for i in range(n):
        # Get similarities to all other nodes
        sims = sim_matrix[i, :]
        
        # Get top-k neighbors
        k = max(min_edges_per_node, int(n * top_k_percent))
        top_k_idx = np.argsort(sims)[-k:]
        
        for j in top_k_idx:
            if j != i and sims[j] > 0:
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(sims[j])
    
    src_nodes = np.array(src_list)
    dst_nodes = np.array(dst_list)
    weights = np.array(weight_list)
    
    logger.info(f"Created {len(weights)} directed edges ({len(weights)/n:.1f} per node avg)")
    
    return src_nodes, dst_nodes, weights


def build_patient_graph(
    df: pd.DataFrame,
    edge_features: List[str],
    top_k_percent: float = 0.05,
    metric: str = "euclidean"
) -> Dict:
    """
    Build patient similarity graph.
    
    Args:
        df: DataFrame with features
        edge_features: Features to use for similarity
        top_k_percent: Percentage of edges to keep
        metric: Similarity metric
    
    Returns:
        Dictionary with graph data
    """
    # Extract and standardize edge features
    X_edge = df[edge_features].fillna(0).values
    scaler = StandardScaler()
    X_edge_scaled = scaler.fit_transform(X_edge)
    
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(X_edge_scaled, metric=metric)
    
    # Construct sparse graph
    src_nodes, dst_nodes, weights = construct_sparse_graph(
        sim_matrix, 
        top_k_percent=top_k_percent
    )
    
    # Create graph data structure
    graph_data = {
        "src_nodes": src_nodes,
        "dst_nodes": dst_nodes,
        "weights": weights,
        "n_nodes": len(df),
        "n_edges": len(weights),
        "hadm_ids": df["hadm_id"].values,
        "edge_features": edge_features,
        "similarity_metric": metric,
        "top_k_percent": top_k_percent,
    }
    
    # Add node features (all features except ID columns)
    exclude_cols = ["hadm_id", "subject_id", "split", "gender", 
                    "admission_type", "discharge_location"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Standardize node features
    X_node = df[feature_cols].fillna(0).values
    node_scaler = StandardScaler()
    
    # Fit on train only
    train_mask = df["split"] == "train"
    node_scaler.fit(X_node[train_mask])
    X_node_scaled = node_scaler.transform(X_node)
    
    graph_data["node_features"] = X_node_scaled.astype(np.float32)
    graph_data["feature_cols"] = feature_cols
    graph_data["node_scaler"] = node_scaler
    
    # Add labels and splits
    graph_data["labels"] = df["readmitted_within_window"].values.astype(np.float32)
    graph_data["train_mask"] = (df["split"] == "train").values
    graph_data["val_mask"] = (df["split"] == "val").values
    graph_data["test_mask"] = (df["split"] == "test").values
    
    return graph_data


def main():
    parser = argparse.ArgumentParser(description="Build patient similarity graph")
    parser.add_argument(
        "--features_path",
        type=str,
        default="data/interim/ehr_long_los/integrated_features.csv",
        help="Path to integrated features CSV"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/interim/ehr_long_los/patient_graph.pkl",
        help="Output path for graph data"
    )
    parser.add_argument(
        "--top_k_percent",
        type=float,
        default=0.05,
        help="Percentage of edges to keep (0.05 = 5%)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Similarity metric"
    )
    args = parser.parse_args()
    
    # Load features
    df, edge_features = load_features(args.features_path)
    
    # Build graph
    graph_data = build_patient_graph(
        df, 
        edge_features,
        top_k_percent=args.top_k_percent,
        metric=args.metric
    )
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(graph_data, f)
    
    logger.info(f"\nSaved graph to {output_path}")
    
    # Print summary
    logger.info("\n=== Graph Summary ===")
    logger.info(f"Nodes: {graph_data['n_nodes']:,}")
    logger.info(f"Edges: {graph_data['n_edges']:,}")
    logger.info(f"Avg edges per node: {graph_data['n_edges']/graph_data['n_nodes']:.1f}")
    logger.info(f"Node features: {graph_data['node_features'].shape[1]}")
    logger.info(f"Edge features used: {len(graph_data['edge_features'])}")
    
    # Split stats
    logger.info(f"\nSplit distribution:")
    logger.info(f"  Train: {graph_data['train_mask'].sum():,}")
    logger.info(f"  Val: {graph_data['val_mask'].sum():,}")
    logger.info(f"  Test: {graph_data['test_mask'].sum():,}")


if __name__ == "__main__":
    main()
