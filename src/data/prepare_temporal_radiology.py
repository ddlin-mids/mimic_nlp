"""
Prepare Temporal Radiology Sequences for Multimodal Fusion

This script prepares radiology report embeddings with temporal information
for use in temporal attention models.

Input:
    - data/interim/embeddings/notes/radiology_report.npz (231K embeddings with charttime)
    - data/interim/readmit_analysis/long_los_cohort.csv (cohort with discharge times)

Output:
    - data/interim/ehr_long_los/temporal/radiology.npz
        - hadm_ids: (N,) admissions with radiology
        - embeddings: (N, max_reports, 768) - padded sequences
        - hours_to_discharge: (N, max_reports) - temporal info
        - mask: (N, max_reports) - 1=real, 0=padded
        - num_reports: (N,) - actual count per admission
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_temporal_radiology(
    radiology_path: str,
    cohort_path: str,
    output_path: str,
    max_reports: int = 20,
):
    """
    Prepare temporal radiology sequences.
    
    Args:
        radiology_path: Path to radiology_report.npz
        cohort_path: Path to cohort CSV with discharge times
        output_path: Path to save output npz
        max_reports: Maximum number of reports per admission (take last N)
    """
    logger.info("Loading radiology embeddings...")
    rad_data = np.load(radiology_path, allow_pickle=True)
    
    logger.info(f"  Embeddings shape: {rad_data['embeddings'].shape}")
    logger.info(f"  Keys: {list(rad_data.keys())}")
    
    # Create DataFrame for easier processing
    rad_df = pd.DataFrame({
        'hadm_id': rad_data['hadm_ids'].astype(int),
        'charttime': pd.to_datetime(rad_data['charttime']),
        'embedding_idx': np.arange(len(rad_data['hadm_ids']))
    })
    
    logger.info("Loading cohort data...")
    cohort = pd.read_csv(cohort_path)
    cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
    cohort['admittime'] = pd.to_datetime(cohort['admittime'])
    cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])
    
    logger.info(f"  Cardiorenal cohort: {len(cohort)} admissions")
    
    # Create time mappings
    admit_times = dict(zip(cohort['hadm_id'], cohort['admittime']))
    disch_times = dict(zip(cohort['hadm_id'], cohort['dischtime']))
    cohort_hadm_ids = set(cohort['hadm_id'])
    
    # Filter radiology to cohort admissions
    rad_df = rad_df[rad_df['hadm_id'].isin(cohort_hadm_ids)].copy()
    logger.info(f"  Radiology reports in cohort: {len(rad_df)}")
    
    # Add admission times
    rad_df['admittime'] = rad_df['hadm_id'].map(admit_times)
    rad_df['dischtime'] = rad_df['hadm_id'].map(disch_times)
    
    # Filter to reports during admission
    rad_df = rad_df[
        (rad_df['charttime'] >= rad_df['admittime']) & 
        (rad_df['charttime'] <= rad_df['dischtime'])
    ].copy()
    logger.info(f"  Reports during admission: {len(rad_df)}")
    
    # Compute hours to discharge
    rad_df['hours_to_discharge'] = (
        rad_df['dischtime'] - rad_df['charttime']
    ).dt.total_seconds() / 3600
    
    # Get unique admissions with radiology
    admissions_with_rad = sorted(rad_df['hadm_id'].unique())
    logger.info(f"  Unique admissions with radiology: {len(admissions_with_rad)}")
    
    # Check which cohort admissions are missing radiology
    missing_rad = cohort_hadm_ids - set(admissions_with_rad)
    logger.info(f"  Admissions without radiology (excluded): {len(missing_rad)}")
    
    # Prepare output arrays
    n_admissions = len(admissions_with_rad)
    embed_dim = rad_data['embeddings'].shape[1]
    
    out_hadm_ids = np.array(admissions_with_rad, dtype=np.int64)
    out_embeddings = np.zeros((n_admissions, max_reports, embed_dim), dtype=np.float16)
    out_hours = np.zeros((n_admissions, max_reports), dtype=np.float32)
    out_mask = np.zeros((n_admissions, max_reports), dtype=np.int8)
    out_num_reports = np.zeros(n_admissions, dtype=np.int32)
    
    # Load full embeddings into memory for indexing
    all_embeddings = rad_data['embeddings']
    
    logger.info("Processing admissions...")
    for i, hadm_id in enumerate(tqdm(admissions_with_rad)):
        # Get reports for this admission, sorted by time (ascending)
        adm_reports = rad_df[rad_df['hadm_id'] == hadm_id].sort_values('charttime')
        
        n_reports = len(adm_reports)
        out_num_reports[i] = n_reports
        
        # Take last max_reports (most recent to discharge)
        if n_reports > max_reports:
            adm_reports = adm_reports.tail(max_reports)
            n_to_store = max_reports
        else:
            n_to_store = n_reports
        
        # Store embeddings, hours, and mask
        embed_idxs = adm_reports['embedding_idx'].values
        hours = adm_reports['hours_to_discharge'].values
        
        out_embeddings[i, :n_to_store] = all_embeddings[embed_idxs]
        out_hours[i, :n_to_store] = hours
        out_mask[i, :n_to_store] = 1
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        hadm_ids=out_hadm_ids,
        embeddings=out_embeddings,
        hours_to_discharge=out_hours,
        mask=out_mask,
        num_reports=out_num_reports,
    )
    
    logger.info(f"Saved to {output_path}")
    logger.info(f"  Shape: {out_embeddings.shape}")
    logger.info(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    # Print statistics
    logger.info("\nStatistics:")
    logger.info(f"  Reports per admission: mean={out_num_reports.mean():.1f}, "
                f"median={np.median(out_num_reports):.1f}, "
                f"max={out_num_reports.max()}")
    logger.info(f"  Admissions with < {max_reports} reports: "
                f"{(out_num_reports < max_reports).sum()} "
                f"({(out_num_reports < max_reports).mean()*100:.1f}%)")
    
    # Return stats for verification
    return {
        'n_admissions': n_admissions,
        'n_excluded': len(missing_rad),
        'mean_reports': out_num_reports.mean(),
        'max_reports': out_num_reports.max(),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare temporal radiology data")
    parser.add_argument(
        "--radiology_path",
        type=str,
        default="data/interim/embeddings/notes/radiology_report.npz",
        help="Path to radiology embeddings"
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
        default="data/interim/ehr_long_los/temporal/radiology.npz",
        help="Output path"
    )
    parser.add_argument(
        "--max_reports",
        type=int,
        default=20,
        help="Maximum reports per admission"
    )
    args = parser.parse_args()
    
    prepare_temporal_radiology(
        radiology_path=args.radiology_path,
        cohort_path=args.cohort_path,
        output_path=args.output_path,
        max_reports=args.max_reports,
    )


if __name__ == "__main__":
    main()
