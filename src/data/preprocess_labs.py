#!/usr/bin/env python3
"""
Lab data preprocessing for MIMIC-IV readmission prediction.
Adapted for HiBEHRT hierarchical transformer architecture.
Processes laboratory values into time-sequenced events with value normalization.
"""

import pandas as pd
import numpy as np
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Key laboratory items from MIMIC-IV d_labitems that are clinically relevant
# Focus on common labs that appear frequently and have prognostic value
PRIORITY_LAB_ITEMS = [
    # Basic Metabolic Panel
    'Sodium', 'Potassium', 'Chloride', 'Carbon Dioxide', 
    'Glucose', 'BUN', 'Creatinine', 'Calcium',
    
    # Complete Blood Count  
    'Hemoglobin', 'Hematocrit', 'Platelet Count', 'WBC Count',
    'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophils',
    
    # Liver Function
    'ALT', 'AST', 'Alkaline Phosphatase', 'Total Bilirubin', 'Albumin',
    
    # Cardiac/Coagulation
    'Troponin I', 'Troponin T', 'PT', 'PTT', 'INR',
    
    # ABG/Ventilation
    'pH', 'pCO2', 'pO2', 'Lactate', 'Bicarbonate',
    
    # Renal Function
    'Creatinine', 'BUN', 'Glomerular Filtration Rate', 'Cystatin C',
    
    # Inflammatory
    'CRP', 'ESR', 'Procalcitonin', 'Ferritin'
]

# LOINC codes for standardization where available
LOINC_MAPPING = {
    'Sodium': '2951-2',
    'Potassium': '2823-3', 
    'Chloride': '2075-0',
    'Carbon Dioxide': '1963-8',
    'Glucose': '1558-6',
    'BUN': '3094-0',
    'Creatinine': '2160-0',
    'Calcium': '17861-6',
    'Hemoglobin': '718-7',
    'Hematocrit': '4544-3',
    'Platelet Count': '777-3',
    'WBC Count': '6690-2'
}


def load_mimic_labs(labevents_path: str, labitems_path: str) -> pd.DataFrame:
    """
    Load MIMIC-IV lab events and items tables.
    
    Args:
        labevents_path: Path to labevents.csv.gz
        labitems_path: Path to d_labitems.csv.gz
        
    Returns:
        DataFrame with lab events merged with item descriptions
    """
    logger.info(f"Loading lab events from {labevents_path}")
    labevents = pd.read_csv(labevents_path, compression='gzip')
    
    logger.info(f"Loading lab items from {labitems_path}")
    labitems = pd.read_csv(labitems_path, compression='gzip')
    
    # Merge to get item labels
    labs = labevents.merge(
        labitems[['itemid', 'label', 'fluid', 'category', 'loinc_code']], 
        on='itemid', 
        how='left'
    )
    
    logger.info(f"Loaded {len(labs)} lab measurements")
    return labs


def filter_priority_labs(labs: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to priority lab items that are clinically relevant for readmission prediction.
    
    Args:
        labs: DataFrame with lab events
        
    Returns:
        Filtered DataFrame with priority labs only
    """
    # Filter to priority items
    priority_labs = labs[
        labs['label'].str.contains('|'.join(PRIORITY_LAB_ITEMS), case=False, na=False)
    ].copy()
    
    logger.info(f"Filtered to {len(priority_labs)} priority lab measurements")
    logger.info(f"Priority lab distribution:\n{priority_labs['label'].value_counts().head(10)}")
    
    return priority_labs


def process_lab_values(labs: pd.DataFrame) -> pd.DataFrame:
    """
    Process lab values: handle missing, outliers, and normalization.
    
    Args:
        labs: DataFrame with lab events
        
    Returns:
        DataFrame with processed lab values
    """
    labs = labs.copy()
    
    # Convert value to numeric, handle missing
    labs['valuenum'] = pd.to_numeric(labs['valuenum'], errors='coerce')
    
    # Remove clearly erroneous values (flagged in MIMIC)
    labs = labs[labs['flag'] != 'E']  # Remove error values
    
    # Basic outlier removal based on reference ranges when available
    normal_ranges = {
        'Sodium': (135, 145),
        'Potassium': (3.5, 5.0),
        'Chloride': (98, 107),
        'Glucose': (70, 140),  # mg/dL
        'Creatinine': (0.6, 1.3),  # mg/dL
        'Hemoglobin': (12, 18),  # g/dL
        'Hematocrit': (36, 52),  # %
        'Platelet Count': (150, 400),  # K/uL
        'WBC Count': (4, 11),  # K/uL
    }
    
    for lab_name, (low, high) in normal_ranges.items():
        mask = labs['label'].str.contains(lab_name, case=False, na=False)
        # Keep values within 3x normal range (allow some pathology)
        reasonable_mask = (labs['valuenum'] >= low * 0.33) & (labs['valuenum'] <= high * 3)
        labs.loc[mask, 'valuenum'] = labs.loc[mask & reasonable_mask, 'valuenum']
    
    logger.info(f"Processed lab values, removed outliers")
    return labs


def create_lab_sequences(
    labs: pd.DataFrame, 
    admissions: pd.DataFrame,
    time_window_hours: int = 24
) -> Dict[str, List[Dict]]:
    """
    Create time-sequenced lab events for each admission.
    
    Args:
        labs: DataFrame with processed lab events
        admissions: DataFrame with admission info
        time_window_hours: Time window for sequencing (hours)
        
    Returns:
        Dictionary mapping hadm_id to list of lab events
    """
    # Convert datetime columns
    labs['charttime'] = pd.to_datetime(labs['charttime'])
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    
    lab_sequences = defaultdict(list)
    
    # Group labs by admission
    labs_by_admission = labs.groupby('hadm_id')
    
    for hadm_id, admission_labs in labs_by_admission:
        if hadm_id not in admissions['hadm_id'].values:
            continue
            
        admission_info = admissions[admissions['hadm_id'] == hadm_id].iloc[0]
        admit_time = admission_info['admittime']
        discharge_time = admission_info['dischtime']
        
        # Filter labs to admission window only (prevent data leakage)
        admission_labs = admission_labs[
            (admission_labs['charttime'] >= admit_time) & 
            (admission_labs['charttime'] <= discharge_time)
        ].copy()
        
        # Calculate relative time from admission (hours)
        admission_labs['hours_since_admission'] = (
            admission_labs['charttime'] - admit_time
        ).dt.total_seconds() / 3600
        
        # Group by time windows
        admission_labs['time_window'] = (
            admission_labs['hours_since_admission'] // time_window_hours
        ).astype(int)
        
        # Create time-sequenced events
        for _, lab in admission_labs.iterrows():
            if pd.notna(lab['valuenum']):  # Only include labs with numeric values
                event = {
                    'code': lab.get('loinc_code', lab['label']),  # Use LOINC if available
                    'label': lab['label'],
                    'value': lab['valuenum'],
                    'unit': lab.get('valueuom', ''),
                    'flag': lab.get('flag', ''),
                    'time_window': lab['time_window'],
                    'hours_since_admission': lab['hours_since_admission'],
                    'category': lab.get('category', 'lab')
                }
                lab_sequences[hadm_id].append(event)
    
    logger.info(f"Created lab sequences for {len(lab_sequences)} admissions")
    return dict(lab_sequences)


def normalize_lab_values(
    lab_sequences: Dict[str, List[Dict]], 
    method: str = 'zscore'
) -> Dict[str, List[Dict]]:
    """
    Normalize lab values across all admissions.
    
    Args:
        lab_sequences: Dictionary of lab events by admission
        method: Normalization method ('zscore', 'minmax', 'quantile')
        
    Returns:
        Normalized lab sequences
    """
    # Collect all values by lab type for normalization
    lab_values_by_type = defaultdict(list)
    
    for hadm_id, events in lab_sequences.items():
        for event in events:
            lab_type = event['label']
            lab_values_by_type[lab_type].append(event['value'])
    
    # Fit normalizers for each lab type
    normalizers = {}
    for lab_type, values in lab_values_by_type.items():
        if len(values) < 10:  # Skip labs with too few measurements
            continue
            
        values_array = np.array(values).reshape(-1, 1)
        
        if method == 'zscore':
            normalizer = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            normalizer = MinMaxScaler()
        elif method == 'quantile':
            normalizer = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        normalizer.fit(values_array)
        normalizers[lab_type] = normalizer
    
    # Apply normalization
    normalized_sequences = {}
    for hadm_id, events in lab_sequences.items():
        normalized_events = []
        for event in events:
            normalized_event = event.copy()
            lab_type = event['label']
            
            if lab_type in normalizers:
                original_value = event['value']
                normalized_value = normalizers[lab_type].transform([[original_value]])[0][0]
                normalized_event['value_normalized'] = float(normalized_value)
                normalized_event['value_original'] = float(original_value)
            else:
                # Keep original value if no normalizer available
                normalized_event['value_normalized'] = float(event['value'])
                normalized_event['value_original'] = float(event['value'])
                
            normalized_events.append(normalized_event)
        normalized_sequences[hadm_id] = normalized_events
    
    logger.info(f"Normalized lab values using {method} method")
    logger.info(f"Applied normalization to {len(normalizers)} lab types")
    return normalized_sequences


def create_lab_tokens(
    lab_sequences: Dict[str, List[Dict]],
    vocab_size: int = 1000,
    min_freq: int = 10
) -> Tuple[Dict[str, List[Dict]], Dict[str, int], Dict[int, str]]:
    """
    Create tokenized lab events for HiBEHRT input.
    
    Args:
        lab_sequences: Normalized lab sequences
        vocab_size: Maximum vocabulary size
        min_freq: Minimum frequency for inclusion in vocab
        
    Returns:
        Tokenized sequences, token-to-id mapping, id-to-token mapping
    """
    # Collect all unique lab codes
    code_counts = defaultdict(int)
    
    for hadm_id, events in lab_sequences.items():
        for event in events:
            # Create composite code: "LAB|label|flag" (e.g., "LAB|Sodium|HIGH")
            base_code = f"LAB|{event['label']}"
            if event['flag'] in ['H', 'L', 'A', 'C']:  # High, Low, Abnormal, Critical
                composite_code = f"{base_code}|{event['flag']}"
            else:
                composite_code = base_code
                
            code_counts[composite_code] += 1
    
    # Filter by minimum frequency and create vocabulary
    filtered_codes = {code: count for code, count in code_counts.items() if count >= min_freq}
    
    # Sort by frequency and take top vocab_size
    sorted_codes = sorted(filtered_codes.items(), key=lambda x: x[1], reverse=True)
    vocab_codes = [code for code, _ in sorted_codes[:vocab_size-2]]  # Reserve 2 for special tokens
    
    # Add special tokens
    token_to_id = {
        '[PAD]': 0,
        '[UNK]': 1,
    }
    
    for i, code in enumerate(vocab_codes):
        token_to_id[code] = i + 2
        
    id_to_token = {v: k for k, v in token_to_id.items()}
    
    # Tokenize sequences
    tokenized_sequences = {}
    for hadm_id, events in lab_sequences.items():
        tokenized_events = []
        for event in events:
            # Create composite code
            base_code = f"LAB|{event['label']}"
            if event['flag'] in ['H', 'L', 'A', 'C']:
                composite_code = f"{base_code}|{event['flag']}"
            else:
                composite_code = base_code
                
            # Get token ID
            token_id = token_to_id.get(composite_code, token_to_id['[UNK]'])
            
            tokenized_event = {
                'token_id': token_id,
                'token_str': id_to_token[token_id],
                'value_normalized': event['value_normalized'],
                'value_original': event['value_original'],
                'unit': event['unit'],
                'time_window': event['time_window'],
                'hours_since_admission': event['hours_since_admission'],
                'code': composite_code
            }
            tokenized_events.append(tokenized_event)
            
        tokenized_sequences[hadm_id] = tokenized_events
    
    logger.info(f"Created vocabulary with {len(token_to_id)} tokens")
    logger.info(f"Most common tokens: {sorted_codes[:10]}")
    return tokenized_sequences, token_to_id, id_to_token


def save_lab_features(
    tokenized_sequences: Dict[str, List[Dict]],
    token_mappings: Tuple[Dict[str, int], Dict[int, str]],
    output_path: str,
    admissions_df: pd.DataFrame
):
    """
    Save processed lab features for model training.
    
    Args:
        tokenized_sequences: Tokenized lab sequences
        token_mappings: (token_to_id, id_to_token) mappings
        output_path: Path to save features
        admissions_df: Admissions dataframe for validation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create feature dictionary
    lab_features = {
        'sequences': tokenized_sequences,
        'token_to_id': token_mappings[0],
        'id_to_token': token_mappings[1],
        'vocab_size': len(token_mappings[0]),
        'feature_type': 'lab_sequences',
        'normalization_method': 'zscore',
        'time_window_hours': 24,
        'admission_count': len(tokenized_sequences)
    }
    
    # Add admission metadata
    admission_metadata = {}
    for hadm_id in tokenized_sequences.keys():
        if hadm_id in admissions_df['hadm_id'].values:
            admission_info = admissions_df[admissions_df['hadm_id'] == hadm_id].iloc[0]
            admission_metadata[hadm_id] = {
                'subject_id': admission_info['subject_id'],
                'admittime': admission_info['admittime'],
                'dischtime': admission_info['dischtime'],
                'los_days': (pd.to_datetime(admission_info['dischtime']) - 
                           pd.to_datetime(admission_info['admittime'])).days,
                'lab_event_count': len(tokenized_sequences[hadm_id])
            }
    
    lab_features['admission_metadata'] = admission_metadata
    
    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(lab_features, f)
    
    logger.info(f"Saved lab features to {output_path}")
    logger.info(f"Features include {len(tokenized_sequences)} admissions with {len(token_mappings[0])} unique tokens")


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIMIC-IV lab data for readmission prediction')
    parser.add_argument('--labevents_path', type=str, required=True,
                       help='Path to labevents.csv.gz file')
    parser.add_argument('--labitems_path', type=str, required=True,
                       help='Path to d_labitems.csv.gz file')
    parser.add_argument('--admissions_path', type=str, required=True,
                       help='Path to admissions.csv.gz file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save processed lab features')
    parser.add_argument('--vocab_size', type=int, default=1000,
                       help='Maximum vocabulary size for tokenization')
    parser.add_argument('--min_freq', type=int, default=10,
                       help='Minimum frequency for token inclusion')
    parser.add_argument('--time_window_hours', type=int, default=24,
                       help='Time window for lab sequencing (hours)')
    parser.add_argument('--normalization', type=str, default='zscore',
                       choices=['zscore', 'minmax', 'quantile'],
                       help='Value normalization method')
    
    args = parser.parse_args()
    
    # Load data
    labs = load_mimic_labs(args.labevents_path, args.labitems_path)
    admissions = pd.read_csv(args.admissions_path, compression='gzip')
    
    # Process labs
    priority_labs = filter_priority_labs(labs)
    processed_labs = process_lab_values(priority_labs)
    
    # Create sequences
    lab_sequences = create_lab_sequences(
        processed_labs, 
        admissions, 
        time_window_hours=args.time_window_hours
    )
    
    # Normalize values
    normalized_sequences = normalize_lab_values(
        lab_sequences, 
        method=args.normalization
    )
    
    # Tokenize for HiBEHRT input
    tokenized_sequences, token_to_id, id_to_token = create_lab_tokens(
        normalized_sequences,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq
    )
    
    # Save features
    save_lab_features(
        tokenized_sequences,
        (token_to_id, id_to_token),
        args.output_path,
        admissions
    )
    
    logger.info("Lab preprocessing completed successfully!")


if __name__ == '__main__':
    main()