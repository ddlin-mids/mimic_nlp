"""
Prepare Temporal EHR Sequences for Multimodal Fusion

This script prepares EHR sequences with temporal (day-level) information
for use in temporal attention models. It creates aligned sequences with
the cohort and adds temporal position information.

Input:
    - data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl 
      (raw features by day)
    - data/interim/ehr_long_los/embeddings/ehr_encoder_best.pt (pre-trained GRU)
    - data/interim/readmit_analysis/long_los_cohort.csv (cohort)

Output:
    - data/interim/ehr_long_los/temporal/ehr.npz
        - hadm_ids: (N,) admission IDs
        - raw_sequences: (N, max_days, feat_dim) - raw daily features
        - lengths: (N,) - actual number of days per admission
        - mask: (N, max_days) - 1=real day, 0=padding
"""

import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_temporal_ehr(
    sequence_pkl_path: str,
    cohort_path: str,
    output_path: str,
    max_days: int = 100,
):
    """
    Prepare temporal EHR sequences.
    
    Args:
        sequence_pkl_path: Path to preprocessed sequence pickle
        cohort_path: Path to cohort CSV
        output_path: Path to save output npz
        max_days: Maximum number of days per admission (truncate from start)
    """
    logger.info("Loading EHR sequence data...")
    with open(sequence_pkl_path, "rb") as f:
        seq_data = pickle.load(f)
    
    feat_dict = seq_data["feat_dict"]
    cat_idxs = seq_data.get("cat_idxs", [])
    cat_dims = seq_data.get("cat_dims", [])
    
    logger.info(f"  Loaded {len(feat_dict)} admissions")
    
    # Get feature dimension from a sample
    sample_key = next(iter(feat_dict))
    sample_seq = feat_dict[sample_key]
    feat_dim = sample_seq.shape[1]
    logger.info(f"  Feature dimension: {feat_dim}")
    logger.info(f"  Sample sequence length: {sample_seq.shape[0]} days")
    
    logger.info("Loading cohort data...")
    cohort = pd.read_csv(cohort_path)
    cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
    
    # Create node_name key (matching EHR encoder format)
    cohort['node_name'] = cohort['subject_id'].astype(str) + "_" + cohort['hadm_id'].astype(str)
    
    logger.info(f"  Cardiorenal cohort: {len(cohort)} admissions")
    
    # Filter feat_dict to only cohort admissions
    cohort_keys = set(cohort['node_name'])
    feat_dict_filtered = {k: v for k, v in feat_dict.items() if k in cohort_keys}
    
    logger.info(f"  EHR sequences in cohort: {len(feat_dict_filtered)}")
    
    # Check for missing EHR data
    missing_ehr = cohort_keys - set(feat_dict_filtered.keys())
    if missing_ehr:
        logger.warning(f"  Admissions missing EHR data: {len(missing_ehr)}")
    
    # Sort cohort by hadm_id for consistent ordering
    cohort = cohort.sort_values('hadm_id').reset_index(drop=True)
    
    # Prepare output arrays
    n_admissions = len(cohort)
    out_hadm_ids = cohort['hadm_id'].values.astype(np.int64)
    out_sequences = np.zeros((n_admissions, max_days, feat_dim), dtype=np.float32)
    out_lengths = np.zeros(n_admissions, dtype=np.int32)
    out_mask = np.zeros((n_admissions, max_days), dtype=np.int8)
    
    logger.info("Processing admissions...")
    n_with_data = 0
    seq_lengths = []
    
    for i, row in tqdm(cohort.iterrows(), total=len(cohort)):
        node_name = row['node_name']
        
        if node_name not in feat_dict_filtered:
            # No EHR data - leave as zeros
            out_lengths[i] = 0
            continue
        
        seq = feat_dict_filtered[node_name]
        
        # Ensure float32 and handle object arrays
        if seq.dtype == object:
            try:
                seq = seq.astype(np.float32)
            except ValueError:
                df_seq = pd.DataFrame(seq)
                df_seq = df_seq.apply(pd.to_numeric, errors='coerce').fillna(0)
                seq = df_seq.values.astype(np.float32)
        else:
            seq = seq.astype(np.float32)
        
        # Handle NaN/Inf
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_days = seq.shape[0]
        seq_lengths.append(n_days)
        n_with_data += 1
        
        # Truncate from start (keep most recent days before discharge)
        if n_days > max_days:
            seq = seq[-max_days:]
            n_to_store = max_days
        else:
            n_to_store = n_days
        
        out_sequences[i, :n_to_store] = seq
        out_lengths[i] = n_to_store
        out_mask[i, :n_to_store] = 1
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        hadm_ids=out_hadm_ids,
        raw_sequences=out_sequences,
        lengths=out_lengths,
        mask=out_mask,
        cat_idxs=np.array(cat_idxs, dtype=np.int32),
        cat_dims=np.array(cat_dims, dtype=np.int32),
        feat_dim=feat_dim,
    )
    
    logger.info(f"Saved to {output_path}")
    logger.info(f"  Shape: {out_sequences.shape}")
    logger.info(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    # Print statistics
    if seq_lengths:
        seq_lengths = np.array(seq_lengths)
        logger.info("\nStatistics:")
        logger.info(f"  Admissions with EHR data: {n_with_data}/{n_admissions}")
        logger.info(f"  Days per admission: mean={seq_lengths.mean():.1f}, "
                    f"median={np.median(seq_lengths):.1f}, "
                    f"max={seq_lengths.max()}")
        logger.info(f"  Admissions with < {max_days} days: "
                    f"{(seq_lengths < max_days).sum()} "
                    f"({(seq_lengths < max_days).mean()*100:.1f}%)")
    
    return {
        'n_admissions': n_admissions,
        'n_with_data': n_with_data,
        'mean_days': seq_lengths.mean() if len(seq_lengths) > 0 else 0,
        'max_days_actual': seq_lengths.max() if len(seq_lengths) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare temporal EHR data")
    parser.add_argument(
        "--sequence_pkl",
        type=str,
        default="data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl",
        help="Path to preprocessed sequence pickle"
    )
    parser.add_argument(
        "--cohort_path",
        type=str,
        default="data/interim/readmit_analysis/long_los_cohort.csv",
        help="Path to cohort CSV"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/interim/ehr_long_los/temporal/ehr.npz",
        help="Output path"
    )
    parser.add_argument(
        "--max_days",
        type=int,
        default=100,
        help="Maximum days per admission"
    )
    args = parser.parse_args()
    
    prepare_temporal_ehr(
        sequence_pkl_path=args.sequence_pkl,
        cohort_path=args.cohort_path,
        output_path=args.output_path,
        max_days=args.max_days,
    )


if __name__ == "__main__":
    main()
