"""
PCA dimensionality reduction for EHR sequences.

Reduces the high-dimensional EHR feature space (6,219 features) to a compact
representation (default 256 dimensions) to enable STGNN training on GPU.

Key design decisions:
- StandardScaler + PCA fit on TRAIN split only (avoid data leakage)
- Transform applied to all splits
- Preserves sequence structure (num_patients, seq_len, pca_dim)

Usage:
    python ehr/reduce_ehr_features_pca.py \
        --input_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
        --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
        --output_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl \
        --n_components 256
"""

import argparse
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cohort_splits(cohort_path: str) -> dict:
    """Load cohort and return node_name -> split mapping."""
    cohort = pd.read_csv(cohort_path)
    cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
    cohort['node_name'] = cohort['subject_id'].astype(str) + "_" + cohort['hadm_id'].astype(str)
    return dict(zip(cohort['node_name'], cohort['split']))


def flatten_sequences(feat_dict: dict, node_names: list) -> np.ndarray:
    """
    Flatten sequences for PCA fitting.
    
    Args:
        feat_dict: Dict mapping node_name -> (seq_len, feat_dim) array
        node_names: List of node names to include
        
    Returns:
        Flattened array of shape (total_timesteps, feat_dim)
    """
    arrays = []
    for node in node_names:
        if node in feat_dict:
            seq = feat_dict[node]
            # Handle object dtype
            if seq.dtype == object:
                seq = np.array(seq.tolist(), dtype=np.float32)
            else:
                seq = seq.astype(np.float32)
            arrays.append(seq)
    
    return np.vstack(arrays)


def transform_sequences(feat_dict: dict, scaler: StandardScaler, pca: PCA) -> dict:
    """
    Apply StandardScaler + PCA to all sequences.
    
    Args:
        feat_dict: Dict mapping node_name -> (seq_len, feat_dim) array
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        
    Returns:
        Dict mapping node_name -> (seq_len, pca_dim) array
    """
    transformed = {}
    
    for node, seq in tqdm(feat_dict.items(), desc="Transforming sequences"):
        # Handle object dtype
        if seq.dtype == object:
            seq = np.array(seq.tolist(), dtype=np.float32)
        else:
            seq = seq.astype(np.float32)
        
        # Replace NaN/Inf with 0 before transformation
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply scaler and PCA
        seq_scaled = scaler.transform(seq)
        seq_pca = pca.transform(seq_scaled)
        
        transformed[node] = seq_pca.astype(np.float32)
    
    return transformed


def main(args):
    logger.info(f"Loading EHR sequences from {args.input_path}...")
    with open(args.input_path, 'rb') as f:
        data = pickle.load(f)
    
    feat_dict = data['feat_dict']
    cat_idxs = data.get('cat_idxs', [])
    cat_dims = data.get('cat_dims', [])
    
    # Get sample info
    sample_key = next(iter(feat_dict))
    sample_seq = feat_dict[sample_key]
    if sample_seq.dtype == object:
        sample_seq = np.array(sample_seq.tolist(), dtype=np.float32)
    
    original_dim = sample_seq.shape[1]
    logger.info(f"Original feature dimension: {original_dim}")
    logger.info(f"Number of sequences: {len(feat_dict)}")
    
    # Load cohort splits
    logger.info(f"Loading cohort splits from {args.cohort_path}...")
    split_map = load_cohort_splits(args.cohort_path)
    
    # Get train nodes
    train_nodes = [node for node in feat_dict.keys() if split_map.get(node) == 'train']
    logger.info(f"Train nodes for PCA fitting: {len(train_nodes)}")
    
    # Flatten train sequences
    logger.info("Flattening train sequences for PCA fitting...")
    train_flat = flatten_sequences(feat_dict, train_nodes)
    logger.info(f"Flattened train shape: {train_flat.shape}")
    
    # Replace NaN/Inf
    train_flat = np.nan_to_num(train_flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fit StandardScaler on train
    logger.info("Fitting StandardScaler on train data...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat)
    
    # Fit PCA on train
    n_components = min(args.n_components, original_dim, train_scaled.shape[0])
    logger.info(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_scaled)
    
    # Log explained variance
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA explained variance ratio: {explained_var:.4f} ({explained_var*100:.2f}%)")
    logger.info(f"Top 10 component variances: {pca.explained_variance_ratio_[:10]}")
    
    # Transform all sequences
    logger.info("Transforming all sequences...")
    transformed_dict = transform_sequences(feat_dict, scaler, pca)
    
    # Verify transformed shape
    sample_transformed = transformed_dict[sample_key]
    logger.info(f"Transformed feature dimension: {sample_transformed.shape[1]}")
    
    # Calculate memory savings
    old_memory = len(feat_dict) * 30 * original_dim * 4 / (1024**3)
    new_memory = len(feat_dict) * 30 * n_components * 4 / (1024**3)
    logger.info(f"Memory estimate: {old_memory:.2f} GB -> {new_memory:.2f} GB ({(1-new_memory/old_memory)*100:.1f}% reduction)")
    
    # Save output
    output_data = {
        'feat_dict': transformed_dict,
        'cat_idxs': [],  # PCA removes categorical structure
        'cat_dims': [],
        'pca_components': n_components,
        'explained_variance_ratio': float(explained_var),
        'original_dim': original_dim,
    }
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving transformed data to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    # Save scaler and PCA for reproducibility
    models_path = output_path.parent / f"pca_models_{n_components}.pkl"
    logger.info(f"Saving scaler and PCA models to {models_path}...")
    with open(models_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'pca': pca}, f)
    
    logger.info("Done!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PCA Dimensionality Reduction Summary")
    print("=" * 60)
    print(f"Input:  {args.input_path}")
    print(f"Output: {output_path}")
    print(f"Original dimension: {original_dim}")
    print(f"Reduced dimension:  {n_components}")
    print(f"Explained variance: {explained_var*100:.2f}%")
    print(f"Memory reduction:   {(1-new_memory/old_memory)*100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCA dimensionality reduction for EHR sequences')
    
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input EHR sequence pickle')
    parser.add_argument('--cohort_path', type=str, required=True,
                        help='Path to cohort CSV with split information')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save PCA-reduced sequences')
    parser.add_argument('--n_components', type=int, default=256,
                        help='Number of PCA components (default: 256)')
    
    args = parser.parse_args()
    main(args)
