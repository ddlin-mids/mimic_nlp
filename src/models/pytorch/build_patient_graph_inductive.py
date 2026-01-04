"""
Build Patient Similarity Graph for STGNN - INDUCTIVE LEARNING VERSION.

This version prevents data leakage by:
1. Building the graph ONLY on train nodes during training
2. Test/val nodes are completely unseen during training
3. At inference, test nodes are added with edges only to train nodes

This ensures the model cannot exploit test-to-test or test-to-train
similarity patterns during training.

Usage:
    python src/models/pytorch/build_patient_graph_inductive.py \
        --features_path data/interim/ehr_long_los/integrated_features.csv \
        --output_path data/interim/ehr_long_los/patient_graph_inductive.pkl
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Features to use for edge construction
# NOTE: Removed admission history features (num_prior_admissions, has_prior_30d_readmit, etc.)
# and LACE score as they are highly predictive of readmission and create label homophily
# in the graph, leading to artificially high performance.
EDGE_FEATURES_SAFE = [
    # Demographics only
    "age_at_admit",
    "is_male",
    "is_emergency_admission",
    # Key comorbidities (clinical similarity, not readmission history)
    "cci_congestive_heart_failure",
    "cci_renal_disease",
    "cci_diabetes_without_complications",
    "cci_chronic_pulmonary_disease",
    "cci_malignancy",
    "charlson_comorbidity_index",
    # ICU markers (severity, not outcome)
    "had_icu_stay",
    "had_dialysis",
    "had_mechanical_vent",
    # Key medications (treatment similarity)
    "received_loop_diuretic",
    "received_ace_inhibitor",
    "received_beta_blocker",
    "received_insulin",
]

# Original features (including potentially leaky ones) - kept for reference
EDGE_FEATURES = [
    # Demographics
    "age_at_admit",
    "is_male",
    "length_of_stay_days",
    "is_emergency_admission",
    # Admission history - POTENTIALLY LEAKY
    "num_prior_admissions",
    "num_admissions_past_6mo",
    "has_prior_30d_readmit",
    # Risk scores - LACE includes readmission history
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


def load_features(features_path: str, use_safe_features: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Load integrated features and identify columns for edge construction.
    
    Args:
        features_path: Path to integrated features CSV
        use_safe_features: If True, use EDGE_FEATURES_SAFE which excludes
                          readmission history features that cause label homophily.
                          If False, use original EDGE_FEATURES.
    """
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    
    # Choose feature set
    feature_set = EDGE_FEATURES_SAFE if use_safe_features else EDGE_FEATURES
    feature_set_name = "SAFE (no readmission history)" if use_safe_features else "FULL (with readmission history)"
    logger.info(f"Using {feature_set_name} feature set for edge construction")
    
    # Find available edge features
    available_edge_features = [f for f in feature_set if f in df.columns]
    logger.info(f"Using {len(available_edge_features)} features for edge construction")
    
    # If we don't have enough edge features, add some lab features
    if len(available_edge_features) < 10:
        lab_features = [c for c in df.columns if any(x in c for x in 
                       ["creatinine_last", "bun_last", "sodium_last", "hemoglobin_last"])]
        available_edge_features.extend(lab_features[:5])
        logger.info(f"Added lab features, now using {len(available_edge_features)} features")
    
    return df, available_edge_features


def compute_knn_edges(
    source_features: np.ndarray,
    target_features: np.ndarray,
    k: int = 10,
    source_offset: int = 0,
    target_offset: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbor edges from source to target.
    
    Args:
        source_features: Features of source nodes
        target_features: Features of target nodes  
        k: Number of neighbors
        source_offset: Index offset for source nodes
        target_offset: Index offset for target nodes
    
    Returns:
        src_indices, dst_indices, weights
    """
    # Compute distances
    dist_matrix = cdist(source_features, target_features, metric="euclidean")
    
    # Convert to similarity
    dist_matrix = dist_matrix / (dist_matrix.max() + 1e-8)
    sigma = np.median(dist_matrix[dist_matrix > 0])
    sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    
    src_list = []
    dst_list = []
    weight_list = []
    
    for i in range(source_features.shape[0]):
        # Get k nearest neighbors in target
        sims = sim_matrix[i, :]
        top_k_idx = np.argsort(sims)[-k:]
        
        for j in top_k_idx:
            if sims[j] > 0:
                src_list.append(i + source_offset)
                dst_list.append(j + target_offset)
                weight_list.append(sims[j])
    
    return np.array(src_list), np.array(dst_list), np.array(weight_list)


def compute_top_percent_edges(
    features: np.ndarray,
    top_perc: float = 0.01,
    use_gauss_kernel: bool = True,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute edges keeping top X% most similar pairs.
    Matches reference implementation in refs/readmit-stgnn/data/readmission_utils.py
    
    Args:
        features: Node features for similarity, shape (num_nodes, feat_dim)
        top_perc: Percentage of edges to keep (0.01 = top 1%)
        use_gauss_kernel: Apply Gaussian kernel to distances (like reference)
        metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        src_indices, dst_indices, weights
    """
    n = len(features)
    logger.info(f"Computing top {top_perc*100}% edges for {n} nodes...")
    
    # Compute pairwise distances
    if metric == 'cosine':
        # Normalize features for cosine
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features_norm = features / norms
        dist_matrix = cdist(features_norm, features_norm, metric='cosine')
    else:
        dist_matrix = cdist(features, features, metric='euclidean')
    
    # Apply Gaussian kernel: similarity = exp(-dist^2 / (2*sigma^2))
    # Reference: readmission_utils.py lines 270-275
    if use_gauss_kernel:
        # Exclude diagonal for sigma calculation
        mask = ~np.eye(n, dtype=bool)
        sigma = np.std(dist_matrix[mask])
        if sigma == 0:
            sigma = 1.0  # Fallback if all same
        sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    else:
        # Simple distance to similarity conversion
        max_dist = dist_matrix.max()
        if max_dist > 0:
            sim_matrix = 1 - (dist_matrix / max_dist)
        else:
            sim_matrix = np.ones_like(dist_matrix)
    
    # Zero out diagonal (no self-loops)
    np.fill_diagonal(sim_matrix, 0)
    
    # Find threshold for top X% edges
    # Reference: readmission_utils.py lines 278-290
    flat_sim = sim_matrix.flatten()
    # Remove zeros from consideration
    nonzero_sim = flat_sim[flat_sim > 0]
    if len(nonzero_sim) == 0:
        logger.warning("No non-zero similarities found!")
        return np.array([]), np.array([]), np.array([])
    
    # Compute threshold for top X%
    threshold = np.percentile(nonzero_sim, 100 - top_perc * 100)
    
    # Get edges above threshold
    src, dst = np.where(sim_matrix >= threshold)
    weights = sim_matrix[src, dst]
    
    # Statistics
    avg_neighbors = len(src) / n
    logger.info(f"Top {top_perc*100}% edges: {len(src):,} edges")
    logger.info(f"  Threshold: {threshold:.4f}")
    logger.info(f"  Avg neighbors per node: {avg_neighbors:.1f}")
    logger.info(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    return src, dst, weights


def compute_top_percent_edges_cross(
    source_features: np.ndarray,
    target_features: np.ndarray,
    top_perc: float = 0.01,
    use_gauss_kernel: bool = True,
    source_offset: int = 0,
    target_offset: int = 0,
    reference_sigma: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute top X% edges from source to target nodes.
    For inductive setting: val/test nodes connect to train nodes.
    
    Args:
        source_features: Features of source nodes
        target_features: Features of target nodes
        top_perc: Percentage of edges to keep per source node
        use_gauss_kernel: Apply Gaussian kernel
        source_offset: Index offset for source nodes
        target_offset: Index offset for target nodes
        reference_sigma: Sigma from training data (for consistent scaling)
        
    Returns:
        src_indices, dst_indices, weights, sigma
    """
    n_src = len(source_features)
    n_tgt = len(target_features)
    
    # Compute distances
    dist_matrix = cdist(source_features, target_features, metric='euclidean')
    
    # Use reference sigma for consistent scaling, or compute from this data
    if reference_sigma is not None:
        sigma = reference_sigma
    else:
        mask = dist_matrix > 0
        if mask.sum() > 0:
            sigma = np.std(dist_matrix[mask])
        else:
            sigma = 1.0
    
    # Apply Gaussian kernel
    if use_gauss_kernel:
        sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))
    else:
        max_dist = dist_matrix.max()
        if max_dist > 0:
            sim_matrix = 1 - (dist_matrix / max_dist)
        else:
            sim_matrix = np.ones_like(dist_matrix)
    
    # For each source, keep top X% connections to targets
    k = max(1, int(n_tgt * top_perc))  # At least 1 edge per source
    
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
    
    return np.array(src_list), np.array(dst_list), np.array(weight_list), sigma


def build_inductive_graph(
    df: pd.DataFrame,
    edge_features: List[str],
    k_neighbors: int = 10,
    edge_method: str = "knn",
    top_perc: float = 0.01,
    use_gauss_kernel: bool = True,
) -> Dict:
    """
    Build patient similarity graph with proper train/val/test isolation.
    
    INDUCTIVE LEARNING:
    - Train graph: Only edges between train nodes
    - Val inference: Val nodes connect to train nodes only
    - Test inference: Test nodes connect to train nodes only
    
    Args:
        df: DataFrame with features
        edge_features: Features to use for similarity
        k_neighbors: Number of neighbors per node (for knn method)
        edge_method: Edge construction method - 'knn' or 'top_percent'
        top_perc: Top percentage of edges to keep (for top_percent method)
        use_gauss_kernel: Apply Gaussian kernel (for top_percent method)
    
    Returns:
        Dictionary with graph data for train/val/test
    """
    # Split data
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Extract and standardize edge features
    X_edge = df[edge_features].fillna(0).values
    scaler = StandardScaler()
    scaler.fit(X_edge[train_mask])  # Fit on train only
    X_edge_scaled = scaler.transform(X_edge)
    
    # =========================================
    # 1. Build TRAIN-ONLY graph (for training)
    # =========================================
    logger.info(f"Building train-only graph using method: {edge_method}...")
    X_train_edge = X_edge_scaled[train_mask].astype(np.float64)
    
    if edge_method == "top_percent":
        # Use top X% edges (like reference paper)
        train_src, train_dst, train_weights = compute_top_percent_edges(
            X_train_edge,
            top_perc=top_perc,
            use_gauss_kernel=use_gauss_kernel,
        )
        # Store sigma for consistent scaling of val/test edges
        mask = np.eye(len(X_train_edge), dtype=bool)
        dist_matrix = cdist(X_train_edge, X_train_edge, metric='euclidean')
        train_sigma = float(np.std(dist_matrix[~mask]))
    else:
        # Use k-NN (original method)
        train_src, train_dst, train_weights = compute_knn_edges(
            X_train_edge, X_train_edge, 
            k=k_neighbors,
            source_offset=0, target_offset=0
        )
        train_sigma = None
    
    # Remove self-loops
    non_self = train_src != train_dst
    train_src = train_src[non_self]
    train_dst = train_dst[non_self]
    train_weights = train_weights[non_self]
    
    logger.info(f"Train graph: {len(train_idx)} nodes, {len(train_weights)} edges")
    avg_train_neighbors = len(train_weights) / len(train_idx)
    logger.info(f"Avg neighbors per train node: {avg_train_neighbors:.1f}")
    
    # =========================================
    # 2. Build inference edges (val/test -> train only)
    # =========================================
    logger.info("Building inference edges...")
    
    X_val_edge = X_edge_scaled[val_mask]
    X_test_edge = X_edge_scaled[test_mask]
    
    if edge_method == "top_percent":
        # For val/test, use same percentage to connect to train nodes
        val_src, val_dst, val_weights, _ = compute_top_percent_edges_cross(
            X_val_edge, X_train_edge,
            top_perc=top_perc,
            use_gauss_kernel=use_gauss_kernel,
            source_offset=len(train_idx),
            target_offset=0,
            reference_sigma=train_sigma,
        )
        
        test_src, test_dst, test_weights, _ = compute_top_percent_edges_cross(
            X_test_edge, X_train_edge,
            top_perc=top_perc,
            use_gauss_kernel=use_gauss_kernel,
            source_offset=len(train_idx) + len(val_idx),
            target_offset=0,
            reference_sigma=train_sigma,
        )
    else:
        # Use k-NN for inference edges
        val_src, val_dst, val_weights = compute_knn_edges(
            X_val_edge, X_train_edge,
            k=k_neighbors,
            source_offset=len(train_idx),
            target_offset=0
        )
        
        test_src, test_dst, test_weights = compute_knn_edges(
            X_test_edge, X_train_edge,
            k=k_neighbors,
            source_offset=len(train_idx) + len(val_idx),
            target_offset=0
        )
    
    logger.info(f"Val->Train edges: {len(val_weights)}")
    logger.info(f"Test->Train edges: {len(test_weights)}")
    
    # =========================================
    # 3. Prepare node features (reindexed)
    # =========================================
    # CRITICAL: Exclude the target variable and any features that directly encode
    # readmission history (which would leak label information through message passing)
    exclude_cols = [
        # ID columns
        "hadm_id", "subject_id", "split", 
        # Categorical columns
        "gender", "admission_type", "discharge_location",
        # TARGET VARIABLE - MUST EXCLUDE!
        "readmitted_within_window",
        # Readmission history features (label-leaky through message passing)
        "num_prior_admissions", "num_prior_30d_readmissions",
        "num_admissions_past_6mo", "num_admissions_past_1yr",
        "is_first_admission", "has_prior_30d_readmit",
        # LACE score (includes readmission history)
        "lace_score", "lace_l_score", "lace_a_score", "lace_c_score", "lace_e_score",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} node features (excluded target and history features)")
    
    X_node = df[feature_cols].fillna(0).values
    node_scaler = StandardScaler()
    node_scaler.fit(X_node[train_mask])
    X_node_scaled = node_scaler.transform(X_node)
    
    # Reorder: train, val, test
    reorder_idx = np.concatenate([train_idx, val_idx, test_idx])
    X_node_reordered = X_node_scaled[reorder_idx]
    labels_reordered = df["readmitted_within_window"].values[reorder_idx]
    hadm_ids_reordered = df["hadm_id"].values[reorder_idx]
    
    # Create masks for reordered data
    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_idx)
    n_total = n_train + n_val + n_test
    
    new_train_mask = np.zeros(n_total, dtype=bool)
    new_train_mask[:n_train] = True
    
    new_val_mask = np.zeros(n_total, dtype=bool)
    new_val_mask[n_train:n_train+n_val] = True
    
    new_test_mask = np.zeros(n_total, dtype=bool)
    new_test_mask[n_train+n_val:] = True
    
    # =========================================
    # 4. Package graph data
    # =========================================
    graph_data = {
        # Train-only graph (used during training)
        "train_src_nodes": train_src,
        "train_dst_nodes": train_dst,
        "train_weights": train_weights,
        
        # Val inference edges
        "val_src_nodes": val_src,
        "val_dst_nodes": val_dst,
        "val_weights": val_weights,
        
        # Test inference edges
        "test_src_nodes": test_src,
        "test_dst_nodes": test_dst,
        "test_weights": test_weights,
        
        # Full graph for inference (train + val/test edges, no val-val or test-test)
        # This is used at inference time
        "inference_src_nodes": np.concatenate([train_src, val_src, test_src]),
        "inference_dst_nodes": np.concatenate([train_dst, val_dst, test_dst]),
        "inference_weights": np.concatenate([train_weights, val_weights, test_weights]),
        
        # Node data (reordered: train, val, test)
        "node_features": X_node_reordered.astype(np.float32),
        "labels": labels_reordered.astype(np.float32),
        "hadm_ids": hadm_ids_reordered,
        "feature_cols": feature_cols,
        
        # Masks (for reordered data)
        "train_mask": new_train_mask,
        "val_mask": new_val_mask,
        "test_mask": new_test_mask,
        
        # Metadata
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_nodes": n_total,
        "k_neighbors": k_neighbors,
        "edge_method": edge_method,
        "top_perc": top_perc if edge_method == "top_percent" else None,
        "use_gauss_kernel": use_gauss_kernel,
        "edge_features": edge_features,
        
        # Original indices for reference
        "original_train_idx": train_idx,
        "original_val_idx": val_idx,
        "original_test_idx": test_idx,
    }
    
    return graph_data


def main():
    parser = argparse.ArgumentParser(description="Build inductive patient graph")
    parser.add_argument(
        "--features_path",
        type=str,
        default="data/interim/ehr_long_los/integrated_features.csv",
        help="Path to integrated features CSV"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/interim/ehr_long_los/patient_graph_inductive.pkl",
        help="Output path for graph data"
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=10,
        help="Number of neighbors per node (for knn method)"
    )
    parser.add_argument(
        "--edge_method",
        type=str,
        default="knn",
        choices=["knn", "top_percent"],
        help="Edge construction method: 'knn' (k nearest neighbors) or 'top_percent' (top X%% most similar pairs, like reference paper)"
    )
    parser.add_argument(
        "--top_perc",
        type=float,
        default=0.01,
        help="Top percentage of edges to keep (for top_percent method, default 0.01 = 1%%)"
    )
    parser.add_argument(
        "--use_gauss_kernel",
        action="store_true",
        default=True,
        help="Apply Gaussian kernel to convert distances to similarities (default: True)"
    )
    parser.add_argument(
        "--use_full_features",
        action="store_true",
        help="Use full feature set including readmission history (may cause label homophily)"
    )
    args = parser.parse_args()
    
    # Load features (use safe by default, full if requested)
    df, edge_features = load_features(
        args.features_path, 
        use_safe_features=not args.use_full_features
    )
    
    # Build graph
    graph_data = build_inductive_graph(
        df, 
        edge_features,
        k_neighbors=args.k_neighbors,
        edge_method=args.edge_method,
        top_perc=args.top_perc,
        use_gauss_kernel=args.use_gauss_kernel,
    )
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(graph_data, f)
    
    logger.info(f"\nSaved graph to {output_path}")
    
    # Print summary
    logger.info("\n=== Inductive Graph Summary ===")
    logger.info(f"Total nodes: {graph_data['n_nodes']:,}")
    logger.info(f"Train nodes: {graph_data['n_train']:,}")
    logger.info(f"Val nodes: {graph_data['n_val']:,}")
    logger.info(f"Test nodes: {graph_data['n_test']:,}")
    logger.info(f"\nTrain graph edges: {len(graph_data['train_weights']):,}")
    logger.info(f"Val->Train edges: {len(graph_data['val_weights']):,}")
    logger.info(f"Test->Train edges: {len(graph_data['test_weights']):,}")
    logger.info(f"Node features: {graph_data['node_features'].shape[1]}")
    
    # Verify no leakage
    logger.info("\n=== Leakage Check ===")
    train_nodes_set = set(range(graph_data['n_train']))
    val_start = graph_data['n_train']
    val_end = val_start + graph_data['n_val']
    test_start = val_end
    
    # Check val edges only go to train
    val_targets = set(graph_data['val_dst_nodes'])
    val_leak = val_targets - train_nodes_set
    logger.info(f"Val targets outside train: {len(val_leak)} (should be 0)")
    
    # Check test edges only go to train
    test_targets = set(graph_data['test_dst_nodes'])
    test_leak = test_targets - train_nodes_set
    logger.info(f"Test targets outside train: {len(test_leak)} (should be 0)")
    
    if len(val_leak) == 0 and len(test_leak) == 0:
        logger.info("✓ No data leakage detected!")
    else:
        logger.error("✗ DATA LEAKAGE DETECTED!")


if __name__ == "__main__":
    main()
