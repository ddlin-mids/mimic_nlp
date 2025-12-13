"""
Extract Lab Trajectory Features for Readmission Prediction Fusion Model.

Extracts summary statistics from lab values for cardiorenal patients:
- Key labs: Creatinine, BUN, Potassium, Sodium, Hemoglobin, NTproBNP, Troponin, WBC, Platelets, Glucose, Albumin
- Features per lab: last value, slope (last 72h), max, min, mean, std, count, missing indicator
- Time windows: last 24h, last 72h, entire admission

Usage:
    python src/data/extract_lab_trajectory_features.py \
        --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
        --output_dir data/interim/ehr_long_los/lab_features
"""

import os
import logging
import argparse
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Key lab item IDs for cardiorenal readmission prediction
LAB_ITEMS = {
    # Renal function
    50912: 'creatinine',
    51006: 'bun',
    50971: 'potassium',
    50983: 'sodium',
    
    # Cardiac markers  
    50963: 'ntprobnp',
    51002: 'troponin_i',
    51003: 'troponin_t',
    
    # Hematology
    51222: 'hemoglobin',
    51265: 'platelet',
    51300: 'wbc',
    
    # Metabolic
    50931: 'glucose',
    50862: 'albumin',
    50885: 'bilirubin',
    
    # Additional relevant labs
    50813: 'lactate',
    50820: 'ph',
}


def load_cohort(cohort_path: Path) -> pd.DataFrame:
    """Load cohort with admission times."""
    logger.info(f"Loading cohort from {cohort_path}")
    df = pd.read_csv(cohort_path)
    df = df[df['is_cardiorenal_long'] == True].copy()
    
    # Parse datetime columns
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    
    logger.info(f"Loaded {len(df)} cardiorenal long-LOS admissions")
    return df


def load_labevents_for_cohort(labevents_path: Path, cohort_hadm_ids: set, lab_item_ids: list) -> pd.DataFrame:
    """Load lab events for cohort admissions (streaming to handle large file)."""
    logger.info(f"Loading lab events from {labevents_path}")
    logger.info(f"Filtering for {len(cohort_hadm_ids)} admissions and {len(lab_item_ids)} lab items")
    
    chunks = []
    chunk_size = 1_000_000  # 1M rows per chunk
    
    with gzip.open(labevents_path, 'rt') as f:
        # Read in chunks
        for chunk in tqdm(pd.read_csv(f, chunksize=chunk_size, 
                                       usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'],
                                       dtype={'hadm_id': 'float64', 'itemid': 'int64', 'valuenum': 'float64'}),
                         desc="Reading labevents"):
            # Filter chunk
            chunk = chunk.dropna(subset=['hadm_id', 'valuenum'])
            chunk['hadm_id'] = chunk['hadm_id'].astype(int)
            
            filtered = chunk[
                (chunk['hadm_id'].isin(cohort_hadm_ids)) & 
                (chunk['itemid'].isin(lab_item_ids))
            ].copy()
            
            if len(filtered) > 0:
                chunks.append(filtered)
    
    if not chunks:
        logger.warning("No lab events found for cohort!")
        return pd.DataFrame()
    
    labs = pd.concat(chunks, ignore_index=True)
    labs['charttime'] = pd.to_datetime(labs['charttime'])
    
    # Map item IDs to lab names
    labs['lab_name'] = labs['itemid'].map(LAB_ITEMS)
    
    logger.info(f"Loaded {len(labs):,} lab events for {labs['hadm_id'].nunique()} admissions")
    return labs


def compute_lab_features_for_admission(
    labs_df: pd.DataFrame,
    hadm_id: int,
    dischtime: pd.Timestamp,
    admittime: pd.Timestamp
) -> dict:
    """Compute lab trajectory features for a single admission."""
    features = {}
    
    # Filter to this admission
    adm_labs = labs_df[labs_df['hadm_id'] == hadm_id].copy()
    
    if len(adm_labs) == 0:
        # Return all missing
        for lab_name in LAB_ITEMS.values():
            features[f'{lab_name}_last'] = np.nan
            features[f'{lab_name}_slope_72h'] = np.nan
            features[f'{lab_name}_max'] = np.nan
            features[f'{lab_name}_min'] = np.nan
            features[f'{lab_name}_mean'] = np.nan
            features[f'{lab_name}_std'] = np.nan
            features[f'{lab_name}_count'] = 0
            features[f'{lab_name}_missing'] = 1
            features[f'{lab_name}_first_last_diff'] = np.nan
        return features
    
    # Compute features per lab type
    for lab_name in LAB_ITEMS.values():
        lab_vals = adm_labs[adm_labs['lab_name'] == lab_name].copy()
        
        if len(lab_vals) == 0:
            features[f'{lab_name}_last'] = np.nan
            features[f'{lab_name}_slope_72h'] = np.nan
            features[f'{lab_name}_max'] = np.nan
            features[f'{lab_name}_min'] = np.nan
            features[f'{lab_name}_mean'] = np.nan
            features[f'{lab_name}_std'] = np.nan
            features[f'{lab_name}_count'] = 0
            features[f'{lab_name}_missing'] = 1
            features[f'{lab_name}_first_last_diff'] = np.nan
            continue
        
        lab_vals = lab_vals.sort_values('charttime')
        values = lab_vals['valuenum'].values
        times = lab_vals['charttime'].values
        
        # Basic statistics
        features[f'{lab_name}_last'] = values[-1]
        features[f'{lab_name}_max'] = np.max(values)
        features[f'{lab_name}_min'] = np.min(values)
        features[f'{lab_name}_mean'] = np.mean(values)
        features[f'{lab_name}_std'] = np.std(values) if len(values) > 1 else 0
        features[f'{lab_name}_count'] = len(values)
        features[f'{lab_name}_missing'] = 0
        
        # Slope in last 72 hours before discharge
        cutoff_72h = dischtime - timedelta(hours=72)
        recent_labs = lab_vals[lab_vals['charttime'] >= cutoff_72h]
        
        if len(recent_labs) >= 2:
            recent_vals = recent_labs['valuenum'].values
            recent_times = recent_labs['charttime'].values
            # Time in hours from first measurement
            hours = [(t - recent_times[0]) / np.timedelta64(1, 'h') for t in recent_times]
            if max(hours) > 0:  # Avoid division by zero
                # Simple linear regression slope
                slope = np.polyfit(hours, recent_vals, 1)[0]
                features[f'{lab_name}_slope_72h'] = slope
            else:
                features[f'{lab_name}_slope_72h'] = 0
        else:
            features[f'{lab_name}_slope_72h'] = 0
        
        # First vs last change
        if len(values) >= 2:
            features[f'{lab_name}_first_last_diff'] = values[-1] - values[0]
        else:
            features[f'{lab_name}_first_last_diff'] = 0
    
    return features


def extract_lab_features(cohort: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """Extract lab features for all admissions in cohort."""
    logger.info("Extracting lab trajectory features...")
    
    all_features = []
    
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Processing admissions"):
        hadm_id = row['hadm_id']
        dischtime = row['dischtime']
        admittime = row['admittime']
        
        features = compute_lab_features_for_admission(labs, hadm_id, dischtime, admittime)
        features['hadm_id'] = hadm_id
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Reorder columns
    cols = ['hadm_id'] + [c for c in features_df.columns if c != 'hadm_id']
    features_df = features_df[cols]
    
    return features_df


def normalize_features(features_df: pd.DataFrame, train_hadm_ids: set) -> pd.DataFrame:
    """Normalize features using training set statistics."""
    logger.info("Normalizing features...")
    
    # Get feature columns (exclude hadm_id and missing indicators)
    feature_cols = [c for c in features_df.columns 
                    if c != 'hadm_id' and not c.endswith('_missing') and not c.endswith('_count')]
    
    # Compute statistics on training set
    train_mask = features_df['hadm_id'].isin(train_hadm_ids)
    train_df = features_df[train_mask]
    
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std()
    stds = stds.replace(0, 1)  # Avoid division by zero
    
    # Normalize
    normalized = features_df.copy()
    for col in feature_cols:
        normalized[col] = (normalized[col] - means[col]) / stds[col]
        # Fill NaN with 0 (will be masked by missing indicator)
        normalized[col] = normalized[col].fillna(0)
    
    return normalized


def main(args):
    # Paths
    cohort_path = Path(args.cohort_path)
    labevents_path = Path(args.labevents_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cohort
    cohort = load_cohort(cohort_path)
    cohort_hadm_ids = set(cohort['hadm_id'].values)
    
    # Load lab events
    labs = load_labevents_for_cohort(labevents_path, cohort_hadm_ids, list(LAB_ITEMS.keys()))
    
    if len(labs) == 0:
        logger.error("No lab events loaded. Exiting.")
        return
    
    # Extract features
    features_df = extract_lab_features(cohort, labs)
    
    # Get train hadm_ids for normalization
    train_hadm_ids = set(cohort[cohort['split'] == 'train']['hadm_id'].values)
    
    # Normalize
    normalized_df = normalize_features(features_df, train_hadm_ids)
    
    # Save
    raw_path = output_dir / 'lab_features_raw.csv'
    norm_path = output_dir / 'lab_features_normalized.csv'
    
    features_df.to_csv(raw_path, index=False)
    normalized_df.to_csv(norm_path, index=False)
    
    logger.info(f"Saved raw features to {raw_path}")
    logger.info(f"Saved normalized features to {norm_path}")
    
    # Summary statistics
    logger.info("\n=== Lab Feature Summary ===")
    for lab_name in LAB_ITEMS.values():
        missing_rate = features_df[f'{lab_name}_missing'].mean() * 100
        logger.info(f"  {lab_name}: {missing_rate:.1f}% missing")
    
    # Save as numpy for easy loading
    feature_cols = [c for c in normalized_df.columns if c != 'hadm_id']
    np.savez(
        output_dir / 'lab_features.npz',
        embeddings=normalized_df[feature_cols].values.astype(np.float32),
        hadm_ids=normalized_df['hadm_id'].values,
        feature_names=np.array(feature_cols),
    )
    logger.info(f"Saved numpy arrays to {output_dir / 'lab_features.npz'}")
    logger.info(f"Feature shape: {normalized_df[feature_cols].shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract lab trajectory features')
    parser.add_argument('--cohort_path', type=str, 
                        default='data/interim/readmit_analysis/long_los_cohort.csv',
                        help='Path to cohort CSV')
    parser.add_argument('--labevents_path', type=str,
                        default='physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz',
                        help='Path to labevents.csv.gz')
    parser.add_argument('--output_dir', type=str,
                        default='data/interim/ehr_long_los/lab_features',
                        help='Output directory')
    
    args = parser.parse_args()
    main(args)
