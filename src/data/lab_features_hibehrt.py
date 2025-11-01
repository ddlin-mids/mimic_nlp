"""
Lab feature extraction for HiBEHRT - Notebook compatible version.
Processes MIMIC-IV lab events into HiBEHRT-compatible format.
Separate from text processing (discharge/radiology notes).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pickle
from pathlib import Path

# Priority lab items for readmission prediction
PRIORITY_LAB_ITEMS = [
    # Basic Metabolic Panel
    'Sodium', 'Potassium', 'Chloride', 'Carbon Dioxide', 'Glucose', 
    'BUN', 'Creatinine', 'Calcium',
    
    # Complete Blood Count  
    'Hemoglobin', 'Hematocrit', 'Platelet Count', 'WBC Count',
    'Neutrophils', 'Lymphocytes', 'Monocytes', 
    
    # Liver Function
    'ALT', 'AST', 'Alkaline Phosphatase', 'Total Bilirubin', 'Albumin',
    
    # Cardiac Markers
    'Troponin I', 'Troponin T',
    
    # Coagulation
    'PT', 'PTT', 'INR',
    
    # Blood Gas
    'pH', 'pCO2', 'pO2', 'Bicarbonate', 'Lactate'
]

# Reference ranges for outlier detection
REFERENCE_RANGES = {
    'Sodium': (135, 145), 'Potassium': (3.5, 5.0), 'Chloride': (98, 107),
    'Carbon Dioxide': (22, 29), 'Glucose': (70, 140), 'BUN': (7, 20),
    'Creatinine': (0.6, 1.3), 'Calcium': (8.5, 10.5), 'Hemoglobin': (12, 18),
    'Hematocrit': (36, 52), 'Platelet Count': (150, 400), 'WBC Count': (4, 11),
    'pH': (7.35, 7.45), 'pCO2': (35, 45), 'pO2': (80, 100), 'Lactate': (0.5, 2.0),
    'ALT': (7, 56), 'AST': (10, 40), 'Albumin': (3.5, 5.0), 'Total Bilirubin': (0.1, 1.2),
    'INR': (0.8, 1.2)
}


def process_lab_events_for_hibehrt(
    labevents_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    max_events_per_admission: int = 2048,
    time_window_hours: int = 6,
    min_frequency: int = 5
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Process lab events into HiBEHRT-compatible format.
    
    Args:
        labevents_df: Lab events from MIMIC (with item labels)
        admissions_df: Admissions dataframe
        max_events_per_admission: Max lab events per admission
        time_window_hours: Time window for grouping events
        min_frequency: Min frequency for vocabulary inclusion
        
    Returns:
        Processed lab events, code mappings, vocabulary info
    """
    print("Processing lab events for HiBEHRT...")
    
    # Convert datetime columns
    labs = labevents_df.copy()
    labs['charttime'] = pd.to_datetime(labs['charttime'])
    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
    admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])
    
    # Filter to priority labs only
    priority_mask = labs['label'].str.contains('|'.join(PRIORITY_LAB_ITEMS), case=False, na=False)
    labs = labs[priority_mask].copy()
    print(f"Filtered to {len(labs)} priority lab measurements")
    
    # Clean values
    labs['valuenum'] = pd.to_numeric(labs['valuenum'], errors='coerce')
    labs = labs[pd.notna(labs['valuenum'])]  # Remove missing values
    labs = labs[labs['valuenum'] != 0]  # Remove zero values (often missing/invalid)
    
    # Remove error-flagged values
    if 'flag' in labs.columns:
        labs = labs[labs['flag'] != 'E']
    
    # Remove outliers based on reference ranges
    labs = _remove_lab_outliers(labs)
    print(f"After cleaning: {len(labs)} lab measurements")
    
    # Build vocabulary
    code_counts = defaultdict(int)
    for _, lab in labs.iterrows():
        base_code = f"LAB|{lab['label']}"
        if lab.get('flag') in ['H', 'L', 'A', 'C']:  # Include abnormal flags
            composite_code = f"{base_code}|{lab['flag']}"
        else:
            composite_code = base_code
        code_counts[composite_code] += 1
    
    # Filter by frequency
    vocab_codes = [code for code, count in code_counts.items() if count >= min_frequency]
    vocab_codes = sorted(vocab_codes, key=lambda x: code_counts[x], reverse=True)
    
    # Create mappings
    code_to_id = {'[PAD]': 0, '[UNK]': 1}
    for i, code in enumerate(vocab_codes):
        code_to_id[code] = i + 2
    id_to_code = {v: k for k, v in code_to_id.items()}
    
    print(f"Built vocabulary with {len(code_to_id)} lab codes")
    
    # Add token IDs to dataframe
    labs['code_composite'] = labs.apply(lambda row: 
        f"LAB|{row['label']}|{row['flag']}" if row.get('flag') in ['H', 'L', 'A', 'C'] 
        else f"LAB|{row['label']}", axis=1)
    
    labs['token_id'] = labs['code_composite'].apply(
        lambda x: code_to_id.get(x, code_to_id['[UNK]']))
    
    return labs, code_to_id, id_to_code


def _remove_lab_outliers(labs: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers based on reference ranges."""
    valid_mask = pd.Series(True, index=labs.index)
    
    for lab_name, (low_normal, high_normal) in REFERENCE_RANGES.items():
        lab_mask = labs['label'].str.contains(lab_name, case=False, na=False)
        if lab_mask.sum() == 0:
            continue
            
        # Allow 5x normal range to accommodate pathology
        reasonable_low = low_normal * 0.2 if low_normal > 0 else low_normal - abs(low_normal)
        reasonable_high = high_normal * 5 if high_normal > 0 else high_normal + abs(high_normal)
        
        value_mask = (labs['valuenum'] >= reasonable_low) & (labs['valuenum'] <= reasonable_high)
        valid_mask = valid_mask & (~lab_mask | value_mask)
    
    return labs[valid_mask].copy()


def create_lab_sequences_by_admission(
    processed_labs: pd.DataFrame,
    admissions_df: pd.DataFrame,
    max_events_per_admission: int = 2048,
    time_window_hours: int = 6
) -> Dict[str, List[Dict]]:
    """
    Create time-sequenced lab events for each admission.
    
    Args:
        processed_labs: Lab events with token IDs
        admissions_df: Admissions info
        max_events_per_admission: Maximum events per admission
        time_window_hours: Time window for grouping
        
    Returns:
        Dictionary mapping hadm_id to sequences of lab events
    """
    print("Creating lab sequences by admission...")
    
    sequences = defaultdict(list)
    
    # Group labs by admission
    labs_by_admission = processed_labs.groupby('hadm_id')
    
    for hadm_id, admission_labs in labs_by_admission:
        if hadm_id not in admissions_df['hadm_id'].values:
            continue
            
        admission_info = admissions_df[admissions_df['hadm_id'] == hadm_id].iloc[0]
        admit_time = pd.to_datetime(admission_info['admittime'])
        discharge_time = pd.to_datetime(admission_info['dischtime'])
        
        # Filter to admission window only (prevent data leakage)
        admission_labs = admission_labs[
            (admission_labs['charttime'] >= admit_time) & 
            (admission_labs['charttime'] <= discharge_time)
        ].copy()
        
        # Calculate time from admission
        admission_labs['hours_from_admission'] = (
            admission_labs['charttime'] - admit_time
        ).dt.total_seconds() / 3600
        
        # Group by time windows
        admission_labs['time_window'] = (
            admission_labs['hours_from_admission'] // time_window_hours
        ).astype(int)
        
        # Create events for HiBEHRT
        for _, lab in admission_labs.iterrows():
            event = {
                # Required for HiBEHRT
                'token_id': int(lab['token_id']),
                'token_str': lab['code_composite'],
                'value': float(lab['valuenum']),  # Normalized value for embedding
                'time_window': int(lab['time_window']),
                'hours_from_admission': float(lab['hours_from_admission']),
                
                # Additional metadata
                'code': lab['code_composite'],
                'label': lab['label'],
                'value_original': float(lab['valuenum']),
                'unit': lab.get('valueuom', ''),
                'flag': lab.get('flag', ''),
                'category': 'lab',
                'loinc_code': lab.get('loinc_code', '')
            }
            sequences[hadm_id].append(event)
        
        # Sort by time and limit sequence length
        sequences[hadm_id].sort(key=lambda x: x['hours_from_admission'])
        if len(sequences[hadm_id]) > max_events_per_admission:
            sequences[hadm_id] = sequences[hadm_id][:max_events_per_admission]
    
    print(f"Created sequences for {len(sequences)} admissions")
    
    # Print statistics
    event_counts = [len(seq) for seq in sequences.values()]
    if event_counts:
        print(f"Event statistics - Mean: {np.mean(event_counts):.1f}, "
              f"Median: {np.median(event_counts):.1f}, "
              f"Max: {np.max(event_counts)}, Min: {np.min(event_counts)}")
    
    return dict(sequences)


def save_lab_sequences_hibehrt(
    sequences: Dict[str, List[Dict]],
    code_mappings: Tuple[Dict[str, int], Dict[int, str]],
    output_path: str,
    split_type: str = 'train'
):
    """
    Save lab sequences in HiBEHRT format.
    
    Args:
        sequences: Lab sequences by admission
        code_mappings: (code_to_id, id_to_code) mappings
        output_path: Output file path
        split_type: Dataset split type
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    features = {
        'sequences': sequences,
        'vocabulary': {
            'code_to_id': code_mappings[0],
            'id_to_code': code_mappings[1],
            'vocab_size': len(code_mappings[0])
        },
        'metadata': {
            'split_type': split_type,
            'admission_count': len(sequences),
            'total_events': sum(len(seq) for seq in sequences.values()),
            'feature_type': 'lab_sequences_hibehrt',
            'max_events_per_admission': 2048,  # Configurable
            'time_window_hours': 6  # Configurable
        }
    }
    
    # Add statistics
    event_counts = [len(seq) for seq in sequences.values()]
    if event_counts:
        features['statistics'] = {
            'mean_events_per_admission': float(np.mean(event_counts)),
            'median_events_per_admission': float(np.median(event_counts)),
            'max_events_per_admission': int(np.max(event_counts)),
            'min_events_per_admission': int(np.min(event_counts))
        }
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"Saved {split_type} lab features to {output_path}")
    print(f"Features: {len(sequences)} admissions, {features['metadata']['total_events']} total events")


def load_lab_hibehrt_features(feature_path: str) -> Dict:
    """Load saved lab features."""
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    return features


def get_lab_summary_stats(sequences: Dict[str, List[Dict]]) -> Dict:
    """Get summary statistics for lab sequences."""
    if not sequences:
        return {}
    
    event_counts = [len(seq) for seq in sequences.values()]
    
    # Count by lab type
    lab_type_counts = defaultdict(int)
    for seq in sequences.values():
        for event in seq:
            lab_type_counts[event['label']] += 1
    
    return {
        'total_admissions': len(sequences),
        'total_lab_events': sum(event_counts),
        'mean_events_per_admission': np.mean(event_counts),
        'median_events_per_admission': np.median(event_counts),
        'max_events_per_admission': np.max(event_counts),
        'min_events_per_admission': np.min(event_counts),
        'top_lab_types': dict(sorted(lab_type_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    }


# Integration function for existing pipeline
def process_labs_for_hibehrt_pipeline(
    labevents_path: str,
    admissions_df: pd.DataFrame,
    output_dir: str,
    admission_ids: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Process labs for HiBEHRT in the existing pipeline.
    
    Args:
        labevents_path: Path to labevents CSV
        admissions_df: Admissions dataframe with cohort selection
        output_dir: Directory to save features
        admission_ids: Optional list of specific admission IDs to process
        
    Returns:
        Dictionary with paths to saved feature files
    """
    print("Starting lab processing for HiBEHRT...")
    
    # Load lab events
    print(f"Loading lab events from {labevents_path}")
    labevents_df = pd.read_csv(labevents_path, compression='gzip')
    
    # Load lab items to get labels
    labitems_path = labevents_path.replace('labevents.csv.gz', 'd_labitems.csv.gz')
    if Path(labitems_path).exists():
        labitems_df = pd.read_csv(labitems_path, compression='gzip')
        labevents_df = labevents_df.merge(
            labitems_df[['itemid', 'label', 'fluid', 'category', 'loinc_code']], 
            on='itemid', how='left'
        )
    else:
        print(f"Warning: Lab items file not found at {labitems_path}")
        labevents_df['label'] = labevents_df['itemid'].astype(str)
    
    # Filter to specific admissions if provided
    if admission_ids:
        labevents_df = labevents_df[labevents_df['hadm_id'].isin(admission_ids)]
        admissions_df = admissions_df[admissions_df['hadm_id'].isin(admission_ids)]
    
    # Process labs
    processed_labs, code_to_id, id_to_code = process_lab_events_for_hibehrt(
        labevents_df, admissions_df
    )
    
    # Create sequences
    sequences = create_lab_sequences_by_admission(
        processed_labs, admissions_df
    )
    
    # Get summary stats
    stats = get_lab_summary_stats(sequences)
    print("Lab processing statistics:")
    print(f"  Admissions: {stats['total_admissions']}")
    print(f"  Total events: {stats['total_lab_events']}")
    print(f"  Mean events/admission: {stats['mean_events_per_admission']:.1f}")
    print(f"  Top lab types: {list(stats['top_lab_types'].keys())[:5]}")
    
    # Save features
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_path = output_dir / "lab_features_hibehrt.pkl"
    save_lab_sequences_hibehrt(
        sequences, (code_to_id, id_to_code), str(feature_path), split_type='all'
    )
    
    # Save vocabulary separately for inspection
    vocab_path = output_dir / "lab_vocabulary.txt"
    with open(vocab_path, 'w') as f:
        f.write(f"Lab Vocabulary (Size: {len(code_to_id)})\n")
        f.write("="*50 + "\n")
        for code, code_id in sorted(code_to_id.items(), key=lambda x: x[1]):
            f.write(f"{code_id:4d}: {code}\n")
    
    return {
        'sequences': sequences,
        'feature_path': str(feature_path),
        'vocab_path': str(vocab_path),
        'code_mappings': (code_to_id, id_to_code),
        'statistics': stats
    }


# Example usage in notebook
if __name__ == '__main__':
    # Example for testing
    print("Lab Encoder for HiBEHRT - Testing Mode")
    
    # Create sample data
    sample_labs = pd.DataFrame({
        'hadm_id': ['A001', 'A001', 'A001', 'A002', 'A002'],
        'charttime': ['2180-01-01 08:00', '2180-01-01 14:00', '2180-01-02 08:00', '2180-01-03 10:00', '2180-01-03 16:00'],
        'itemid': [50809, 50811, 50813, 50809, 50811],  # Sodium, Potassium, Creatinine
        'label': ['Sodium', 'Potassium', 'Creatinine', 'Sodium', 'Potassium'],
        'valuenum': [140, 4.2, 1.1, 138, 4.0],
        'valueuom': ['mmol/L', 'mmol/L', 'mg/dL', 'mmol/L', 'mmol/L'],
        'flag': ['', '', '', '', '']
    })
    
    sample_admissions = pd.DataFrame({
        'hadm_id': ['A001', 'A002'],
        'subject_id': ['P001', 'P002'],
        'admittime': ['2180-01-01 06:00', '2180-01-03 08:00'],
        'dischtime': ['2180-01-02 18:00', '2180-01-04 12:00']
    })
    
    # Process sample data
    processed_labs, code_to_id, id_to_code = process_lab_events_for_hibehrt(
        sample_labs, sample_admissions
    )
    
    sequences = create_lab_sequences_by_admission(
        processed_labs, sample_admissions
    )
    
    print(f"Created sequences for {len(sequences)} admissions")
    print(f"Vocabulary size: {len(code_to_id)}")
    print(f"Sample sequence for A001: {sequences['A001'][:2]}")
    
    stats = get_lab_summary_stats(sequences)
    print(f"Statistics: {stats}")