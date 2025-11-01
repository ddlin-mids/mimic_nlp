#!/usr/bin/env python3
"""
Lab data encoder for HiBEHRT hierarchical transformer.
Processes laboratory events into HiBEHRT-compatible token sequences.
Separate from text processing which handles discharge/radiology notes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Priority lab items for readmission prediction (based on clinical relevance)
PRIORITY_LAB_ITEMS = [
    # Basic Metabolic Panel - Core electrolytes
    'Sodium', 'Potassium', 'Chloride', 'Carbon Dioxide', 'Glucose', 
    'BUN', 'Creatinine', 'Calcium',
    
    # Complete Blood Count - Hematology
    'Hemoglobin', 'Hematocrit', 'Platelet Count', 'WBC Count',
    'Neutrophils', 'Lymphocytes', 'Monocytes', 
    
    # Liver Function Tests
    'ALT', 'AST', 'Alkaline Phosphatase', 'Total Bilirubin', 'Albumin',
    
    # Cardiac/Injury Markers
    'Troponin I', 'Troponin T', 'CK-MB',
    
    # Coagulation Studies
    'PT', 'PTT', 'INR',
    
    # Blood Gas Analysis
    'pH', 'pCO2', 'pO2', 'Bicarbonate', 'Lactate',
    
    # Renal Function
    'Creatinine', 'BUN', 'GFR', 'Cystatin C',
    
    # Inflammatory Markers
    'CRP', 'ESR', 'Procalcitonin'
]

# Normal reference ranges for outlier detection (adult values)
REFERENCE_RANGES = {
    'Sodium': (135, 145, 'mmol/L'),
    'Potassium': (3.5, 5.0, 'mmol/L'),
    'Chloride': (98, 107, 'mmol/L'),
    'Carbon Dioxide': (22, 29, 'mmol/L'),
    'Glucose': (70, 140, 'mg/dL'),
    'BUN': (7, 20, 'mg/dL'),
    'Creatinine': (0.6, 1.3, 'mg/dL'),
    'Calcium': (8.5, 10.5, 'mg/dL'),
    'Hemoglobin': (12, 18, 'g/dL'),
    'Hematocrit': (36, 52, '%'),
    'Platelet Count': (150, 400, 'K/uL'),
    'WBC Count': (4, 11, 'K/uL'),
    'pH': (7.35, 7.45, ''),
    'pCO2': (35, 45, 'mmHg'),
    'pO2': (80, 100, 'mmHg'),
    'Lactate': (0.5, 2.0, 'mmol/L'),
    'ALT': (7, 56, 'U/L'),
    'AST': (10, 40, 'U/L'),
    'Albumin': (3.5, 5.0, 'g/dL'),
    'Total Bilirubin': (0.1, 1.2, 'mg/dL'),
    'CRP': (0, 10, 'mg/L'),
    'INR': (0.8, 1.2, '')
}


class LabEncoderHiBEHRT:
    """
    Lab data encoder for HiBEHRT hierarchical transformer.
    Converts lab events into time-sequenced tokens with value embeddings.
    """
    
    def __init__(
        self,
        max_events_per_admission: int = 2048,
        time_window_hours: int = 6,
        vocab_size: int = 500,
        min_frequency: int = 5,
        include_abnormal_flags: bool = True,
        normalize_values: bool = True
    ):
        """
        Initialize lab encoder.
        
        Args:
            max_events_per_admission: Maximum lab events per admission
            time_window_hours: Time window for grouping events (hours)
            vocab_size: Maximum vocabulary size for lab codes
            min_frequency: Minimum frequency for inclusion in vocabulary
            include_abnormal_flags: Include abnormal flags in tokenization
            normalize_values: Normalize numeric lab values
        """
        self.max_events_per_admission = max_events_per_admission
        self.time_window_hours = time_window_hours
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.include_abnormal_flags = include_abnormal_flags
        self.normalize_values = normalize_values
        
        # Will be populated during fit
        self.code_to_id = {}
        self.id_to_code = {}
        self.value_normalizers = {}
        self.is_fitted = False
        
    def fit(self, labevents_df: pd.DataFrame, admissions_df: pd.DataFrame) -> 'LabEncoderHiBEHRT':
        """
        Fit the encoder on lab data to build vocabulary and normalizers.
        
        Args:
            labevents_df: Lab events dataframe
            admissions_df: Admissions dataframe
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting lab encoder...")
        
        # Filter to priority labs and valid time windows
        processed_labs = self._filter_and_process_labs(labevents_df, admissions_df)
        
        # Build vocabulary from lab codes
        self._build_vocabulary(processed_labs)
        
        # Fit value normalizers
        if self.normalize_values:
            self._fit_value_normalizers(processed_labs)
        
        self.is_fitted = True
        logger.info(f"Lab encoder fitted with {len(self.code_to_id)} codes in vocabulary")
        
        return self
    
    def transform(self, labevents_df: pd.DataFrame, admissions_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Transform lab data into HiBEHRT-compatible sequences.
        
        Args:
            labevents_df: Lab events dataframe
            admissions_df: Admissions dataframe
            
        Returns:
            Dictionary mapping hadm_id to sequences of lab events
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transforming")
        
        logger.info("Transforming lab data...")
        
        # Filter and process labs
        processed_labs = self._filter_and_process_labs(labevents_df, admissions_df)
        
        # Create sequences by admission
        lab_sequences = self._create_sequences(processed_labs, admissions_df)
        
        logger.info(f"Created lab sequences for {len(lab_sequences)} admissions")
        
        return lab_sequences
    
    def _filter_and_process_labs(self, labevents_df: pd.DataFrame, admissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to priority labs and process values.
        
        Args:
            labevents_df: Raw lab events
            admissions_df: Admissions info
            
        Returns:
            Processed lab dataframe
        """
        # Convert datetime columns
        labs = labevents_df.copy()
        labs['charttime'] = pd.to_datetime(labs['charttime'])
        admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
        admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])
        
        # Filter to priority lab items
        priority_mask = labs['label'].str.contains('|'.join(PRIORITY_LAB_ITEMS), case=False, na=False)
        labs = labs[priority_mask].copy()
        
        # Convert values to numeric
        labs['valuenum'] = pd.to_numeric(labs['valuenum'], errors='coerce')
        
        # Remove invalid values (flagged as errors)
        if 'flag' in labs.columns:
            labs = labs[labs['flag'] != 'E']
        
        # Basic outlier removal
        labs = self._remove_outliers(labs)
        
        logger.info(f"Filtered to {len(labs)} priority lab measurements")
        return labs
    
    def _remove_outliers(self, labs: pd.DataFrame) -> pd.DataFrame:
        """
        Remove extreme outliers based on reference ranges.
        
        Args:
            labs: Lab dataframe
            
        Returns:
            Labs with outliers removed
        """
        valid_mask = pd.Series(True, index=labs.index)
        
        for lab_name, (low_normal, high_normal, unit) in REFERENCE_RANGES.items():
            lab_mask = labs['label'].str.contains(lab_name, case=False, na=False)
            if lab_mask.sum() == 0:
                continue
                
            # Allow values within 5x normal range (accommodate pathology)
            reasonable_low = low_normal * 0.2 if low_normal > 0 else low_normal * 5
            reasonable_high = high_normal * 5 if high_normal > 0 else high_normal * 0.2
            
            value_mask = (labs['valuenum'] >= reasonable_low) & (labs['valuenum'] <= reasonable_high)
            valid_mask = valid_mask & (~lab_mask | value_mask)
        
        return labs[valid_mask].copy()
    
    def _build_vocabulary(self, labs: pd.DataFrame):
        """
        Build vocabulary from lab codes.
        
        Args:
            labs: Processed lab dataframe
        """
        code_counts = defaultdict(int)
        
        for _, lab in labs.iterrows():
            # Create composite code: LAB|label|flag|abnormal
            base_code = f"LAB|{lab['label']}"
            
            if self.include_abnormal_flags and lab.get('flag') in ['H', 'L', 'A', 'C']:
                composite_code = f"{base_code}|{lab['flag']}"
            else:
                composite_code = base_code
            
            code_counts[composite_code] += 1
        
        # Filter by minimum frequency
        filtered_codes = {code: count for code, count in code_counts.items() 
                         if count >= self.min_frequency}
        
        # Sort by frequency and take top vocab_size
        sorted_codes = sorted(filtered_codes.items(), key=lambda x: x[1], reverse=True)
        vocab_codes = [code for code, _ in sorted_codes[:self.vocab_size-2]]  # Reserve for special tokens
        
        # Build mappings
        self.code_to_id = {'[PAD]': 0, '[UNK]': 1}
        for i, code in enumerate(vocab_codes):
            self.code_to_id[code] = i + 2
        
        self.id_to_code = {v: k for k, v in self.code_to_id.items()}
        
        logger.info(f"Built vocabulary with {len(self.code_to_id)} codes")
    
    def _fit_value_normalizers(self, labs: pd.DataFrame):
        """
        Fit normalizers for numeric lab values.
        
        Args:
            labs: Processed lab dataframe
        """
        from sklearn.preprocessing import StandardScaler
        
        # Group values by lab type
        values_by_type = defaultdict(list)
        
        for _, lab in labs.iterrows():
            if pd.notna(lab['valuenum']):
                values_by_type[lab['label']].append(lab['valuenum'])
        
        # Fit normalizers for types with sufficient data
        for lab_type, values in values_by_type.items():
            if len(values) >= 10:  # Minimum samples for reliable normalization
                scaler = StandardScaler()
                values_array = np.array(values).reshape(-1, 1)
                scaler.fit(values_array)
                self.value_normalizers[lab_type] = scaler
        
        logger.info(f"Fitted normalizers for {len(self.value_normalizers)} lab types")
    
    def _create_sequences(self, labs: pd.DataFrame, admissions_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Create time-sequenced lab events for each admission.
        
        Args:
            labs: Processed lab dataframe
            admissions_df: Admissions dataframe
            
        Returns:
            Dictionary mapping hadm_id to sequences of lab events
        """
        sequences = defaultdict(list)
        
        # Group labs by admission
        labs_by_admission = labs.groupby('hadm_id')
        
        for hadm_id, admission_labs in labs_by_admission:
            if hadm_id not in admissions_df['hadm_id'].values:
                continue
            
            admission_info = admissions_df[admissions_df['hadm_id'] == hadm_id].iloc[0]
            admit_time = pd.to_datetime(admission_info['admittime'])
            discharge_time = pd.to_datetime(admission_info['dischtime'])
            
            # Filter to admission window (prevent data leakage)
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
                admission_labs['hours_from_admission'] // self.time_window_hours
            ).astype(int)
            
            # Create events
            for _, lab in admission_labs.iterrows():
                if pd.isna(lab['valuenum']):
                    continue
                
                # Create token
                base_code = f"LAB|{lab['label']}"
                if self.include_abnormal_flags and lab.get('flag') in ['H', 'L', 'A', 'C']:
                    composite_code = f"{base_code}|{lab['flag']}"
                else:
                    composite_code = base_code
                
                token_id = self.code_to_id.get(composite_code, self.code_to_id['[UNK]'])
                
                # Normalize value if available
                if self.normalize_values and lab['label'] in self.value_normalizers:
                    normalized_value = self.value_normalizers[lab['label']].transform([[lab['valuenum']]])[0][0]
                else:
                    normalized_value = lab['valuenum']
                
                event = {
                    'token_id': token_id,
                    'token_str': self.id_to_code[token_id],
                    'code': composite_code,
                    'value': float(normalized_value),
                    'value_original': float(lab['valuenum']),
                    'unit': lab.get('valueuom', ''),
                    'time_window': int(lab['time_window']),
                    'hours_from_admission': float(lab['hours_from_admission']),
                    'label': lab['label'],
                    'flag': lab.get('flag', ''),
                    'category': 'lab'
                }
                
                sequences[hadm_id].append(event)
        
        # Sort by time and limit sequence length
        for hadm_id in sequences:
            sequences[hadm_id].sort(key=lambda x: x['hours_from_admission'])
            if len(sequences[hadm_id]) > self.max_events_per_admission:
                sequences[hadm_id] = sequences[hadm_id][:self.max_events_per_admission]
        
        return dict(sequences)
    
    def get_feature_info(self) -> Dict:
        """Get information about the encoded features."""
        return {
            'vocab_size': len(self.code_to_id),
            'max_events_per_admission': self.max_events_per_admission,
            'time_window_hours': self.time_window_hours,
            'include_abnormal_flags': self.include_abnormal_flags,
            'normalize_values': self.normalize_values,
            'normalizer_count': len(self.value_normalizers),
            'feature_type': 'lab_sequences_hibehrt'
        }


def create_lab_hibehrt_features(
    labevents_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    encoder_config: Optional[Dict] = None
) -> Tuple[Dict[str, List[Dict]], LabEncoderHiBEHRT]:
    """
    Convenience function to create lab features for HiBEHRT.
    
    Args:
        labevents_df: Lab events dataframe
        admissions_df: Admissions dataframe
        encoder_config: Optional configuration for encoder
        
    Returns:
        Lab sequences and fitted encoder
    """
    if encoder_config is None:
        encoder_config = {}
    
    # Initialize encoder
    encoder = LabEncoderHiBEHRT(**encoder_config)
    
    # Fit and transform
    encoder.fit(labevents_df, admissions_df)
    sequences = encoder.transform(labevents_df, admissions_df)
    
    return sequences, encoder


def save_lab_hibehrt_features(
    sequences: Dict[str, List[Dict]],
    encoder: LabEncoderHiBEHRT,
    output_path: str,
    split_type: str = 'train'
):
    """
    Save lab features for HiBEHRT model.
    
    Args:
        sequences: Lab sequences by admission
        encoder: Fitted lab encoder
        output_path: Path to save features
        split_type: Dataset split (train/val/test)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    features = {
        'sequences': sequences,
        'encoder_config': {
            'max_events_per_admission': encoder.max_events_per_admission,
            'time_window_hours': encoder.time_window_hours,
            'vocab_size': encoder.vocab_size,
            'min_frequency': encoder.min_frequency,
            'include_abnormal_flags': encoder.include_abnormal_flags,
            'normalize_values': encoder.normalize_values
        },
        'vocabulary': {
            'code_to_id': encoder.code_to_id,
            'id_to_code': encoder.id_to_code,
            'vocab_size': len(encoder.code_to_id)
        },
        'split_type': split_type,
        'admission_count': len(sequences),
        'total_events': sum(len(seq) for seq in sequences.values()),
        'feature_type': 'lab_sequences_hibehrt'
    }
    
    # Add feature statistics
    event_counts = [len(seq) for seq in sequences.values()]
    features['statistics'] = {
        'mean_events_per_admission': np.mean(event_counts),
        'median_events_per_admission': np.median(event_counts),
        'max_events_per_admission': np.max(event_counts),
        'min_events_per_admission': np.min(event_counts)
    }
    
    # Save as pickle
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    logger.info(f"Saved {split_type} lab features to {output_path}")
    logger.info(f"Features: {len(sequences)} admissions, {features['statistics']['total_events']} total events")


# Example usage and testing
if __name__ == '__main__':
    # Example data paths (would be replaced with actual paths)
    labevents_path = "physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz"
    labitems_path = "physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz"
    admissions_path = "physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"
    
    # Load sample data
    print("Loading lab data...")
    labevents_df = pd.read_csv(labevents_path, compression='gzip', nrows=100000)  # Sample for testing
    labitems_df = pd.read_csv(labitems_path, compression='gzip')
    admissions_df = pd.read_csv(admissions_path, compression='gzip')
    
    # Merge lab events with item descriptions
    labevents_df = labevents_df.merge(
        labitems_df[['itemid', 'label', 'fluid', 'category', 'loinc_code']], 
        on='itemid', 
        how='left'
    )
    
    # Configure encoder for HiBEHRT
    encoder_config = {
        'max_events_per_admission': 2048,
        'time_window_hours': 6,
        'vocab_size': 500,
        'min_frequency': 10,
        'include_abnormal_flags': True,
        'normalize_values': True
    }
    
    # Create lab features
    sequences, encoder = create_lab_hibehrt_features(
        labevents_df, 
        admissions_df, 
        encoder_config
    )
    
    # Save features
    save_lab_hibehrt_features(
        sequences, 
        encoder, 
        "data/interim/lab_features_train.pkl",
        split_type='train'
    )
    
    print("Lab preprocessing completed!")
    print(f"Created sequences for {len(sequences)} admissions")
    print(f"Encoder vocab size: {len(encoder.code_to_id)}")
    print(f"Feature info: {encoder.get_feature_info()}")