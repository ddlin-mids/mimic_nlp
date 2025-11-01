"""
Integration script for lab features into HiBEHRT pipeline.
Connects lab processing with existing admission-based workflow.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import pickle

# Import the lab processing functions
from lab_features_hibehrt import (
    process_lab_events_for_hibehrt,
    create_lab_sequences_by_admission,
    save_lab_sequences_hibehrt,
    get_lab_summary_stats
)


def integrate_lab_features_with_admissions(
    labevents_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    cohort_config: Optional[Dict] = None
) -> Dict:
    """
    Integrate lab features with admission-based cohort.
    
    Args:
        labevents_df: Lab events dataframe
        admissions_df: Admissions dataframe (already filtered by cohort criteria)
        cohort_config: Configuration for lab processing
        
    Returns:
        Dictionary with lab features and metadata
    """
    
    if cohort_config is None:
        cohort_config = {
            'max_events_per_admission': 2048,
            'time_window_hours': 6,
            'min_frequency': 5,
            'include_abnormal_flags': True
        }
    
    print(f"Integrating lab features for {len(admissions_df)} admissions...")
    
    # Step 1: Process lab events
    print("Step 1: Processing lab events for HiBEHRT format")
    processed_labs, code_to_id, id_to_code = process_lab_events_for_hibehrt(
        labevents_df, 
        admissions_df,
        max_events_per_admission=cohort_config['max_events_per_admission'],
        time_window_hours=cohort_config['time_window_hours'],
        min_frequency=cohort_config['min_frequency']
    )
    
    print(f"Lab vocabulary created: {len(code_to_id)} codes")
    print(f"Top 10 lab codes: {list(code_to_id.keys())[:10]}")
    
    # Step 2: Create sequences by admission
    print("Step 2: Creating lab sequences by admission")
    lab_sequences = create_lab_sequences_by_admission(
        processed_labs, 
        admissions_df,
        max_events_per_admission=cohort_config['max_events_per_admission'],
        time_window_hours=cohort_config['time_window_hours']
    )
    
    print(f"Created sequences for {len(lab_sequences)} admissions")
    
    # Step 3: Generate statistics
    print("Step 3: Generating lab feature statistics")
    stats = get_lab_summary_stats(lab_sequences)
    
    print("Lab Feature Statistics:")
    print(f"  Total admissions with labs: {stats['total_admissions']}")
    print(f"  Total lab events: {stats['total_lab_events']}")
    print(f"  Mean events per admission: {stats['mean_events_per_admission']:.1f}")
    print(f"  Median events per admission: {stats['median_events_per_admission']:.1f}")
    print(f"  Admissions with >100 lab events: {sum(1 for count in [len(seq) for seq in lab_sequences.values()] if count > 100)}")
    
    # Step 4: Validate coverage
    print("Step 4: Validating lab coverage")
    admissions_with_labs = set(lab_sequences.keys())
    total_admissions = set(admissions_df['hadm_id'].astype(str))
    coverage = len(admissions_with_labs) / len(total_admissions) * 100
    
    print(f"Lab coverage: {coverage:.1f}% ({len(admissions_with_labs)}/{len(total_admissions)} admissions)")
    
    # Identify admissions without labs
    admissions_without_labs = total_admissions - admissions_with_labs
    if admissions_without_labs:
        print(f"Admissions without labs: {len(admissions_without_labs)}")
        # These admissions will need special handling in HiBEHRT
    
    return {
        'sequences': lab_sequences,
        'vocabulary': {
            'code_to_id': code_to_id,
            'id_to_code': id_to_code,
            'vocab_size': len(code_to_id)
        },
        'statistics': stats,
        'coverage': {
            'total_admissions': len(total_admissions),
            'admissions_with_labs': len(admissions_with_labs),
            'admissions_without_labs': len(admissions_without_labs),
            'coverage_percentage': coverage
        },
        'config': cohort_config,
        'processed_labs': processed_labs  # For inspection/debugging
    }


def prepare_lab_features_for_hibehrt(
    labevents_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    output_dir: str,
    split_type: str = 'all'
) -> str:
    """
    Complete pipeline to prepare lab features for HiBEHRT.
    
    Args:
        labevents_df: Lab events dataframe
        admissions_df: Admissions dataframe (cohort-filtered)
        output_dir: Directory to save features
        split_type: Dataset split type
        
    Returns:
        Path to saved feature file
    """
    
    print(f"\n{'='*60}")
    print(f"PREPARING LAB FEATURES FOR HIBEHRT - {split_type.upper()}")
    print(f"{'='*60}")
    
    # Integrate features
    lab_features = integrate_lab_features_with_admissions(
        labevents_df, admissions_df
    )
    
    # Save features
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_path = output_dir / f"lab_features_{split_type}.pkl"
    
    # Save in HiBEHRT format
    save_lab_sequences_hibehrt(
        lab_features['sequences'],
        (lab_features['vocabulary']['code_to_id'], lab_features['vocabulary']['id_to_code']),
        str(feature_path),
        split_type
    )
    
    # Save vocabulary for inspection
    vocab_path = output_dir / f"lab_vocabulary_{split_type}.txt"
    with open(vocab_path, 'w') as f:
        f.write(f"Lab Vocabulary - {split_type.upper()} Split\n")
        f.write(f"Total codes: {lab_features['vocabulary']['vocab_size']}\n")
        f.write("="*50 + "\n")
        for code, code_id in sorted(lab_features['vocabulary']['code_to_id'].items(), 
                                   key=lambda x: x[1]):
            count = sum(1 for seq in lab_features['sequences'].values() 
                       for event in seq if event['token_str'] == code)
            f.write(f"{code_id:4d}: {code:40s} [{count:6d} occurrences]\n")
    
    # Save statistics
    stats_path = output_dir / f"lab_statistics_{split_type}.json"
    import json
    with open(stats_path, 'w') as f:
        json.dump({
            'statistics': lab_features['statistics'],
            'coverage': lab_features['coverage'],
            'config': lab_features['config'],
            'vocab_size': lab_features['vocabulary']['vocab_size']
        }, f, indent=2)
    
    print(f"\nLab features saved:")
    print(f"  Features: {feature_path}")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Statistics: {stats_path}")
    
    return str(feature_path)


def validate_lab_features_for_hibehrt(lab_features_path: str) -> Dict:
    """
    Validate lab features for HiBEHRT compatibility.
    
    Args:
        lab_features_path: Path to saved lab features
        
    Returns:
        Validation results
    """
    from lab_features_hibehrt import load_lab_hibehrt_features
    
    features = load_lab_hibehrt_features(lab_features_path)
    
    validation_results = {
        'file_exists': True,
        'required_keys': [],
        'sequence_validity': [],
        'token_validity': [],
        'warnings': [],
        'errors': []
    }
    
    # Check required structure
    required_keys = ['sequences', 'vocabulary', 'metadata']
    for key in required_keys:
        if key not in features:
            validation_results['errors'].append(f"Missing key: {key}")
        else:
            validation_results['required_keys'].append(f"✓ {key}")
    
    # Validate sequences
    sequences = features.get('sequences', {})
    if sequences:
        sample_hadm = list(sequences.keys())[0]
        sample_seq = sequences[sample_hadm]
        
        if sample_seq:
            sample_event = sample_seq[0]
            required_event_keys = ['token_id', 'value', 'time_window', 'hours_from_admission']
            
            for key in required_event_keys:
                if key in sample_event:
                    validation_results['sequence_validity'].append(f"✓ Event has {key}")
                else:
                    validation_results['errors'].append(f"Event missing {key}")
            
            # Check token validity
            vocab_size = features['vocabulary']['vocab_size']
            if sample_event.get('token_id', -1) < vocab_size:
                validation_results['token_validity'].append("✓ Token ID within vocabulary")
            else:
                validation_results['errors'].append("Token ID exceeds vocabulary size")
        else:
            validation_results['warnings'].append("Empty sequences found")
    
    # Check vocabulary consistency
    code_to_id = features['vocabulary']['code_to_id']
    id_to_code = features['vocabulary']['id_to_code']
    
    if len(code_to_id) != len(id_to_code):
        validation_results['errors'].append("Vocabulary mappings inconsistent")
    else:
        validation_results['required_keys'].append("✓ Vocabulary mappings consistent")
    
    # Check feature type
    expected_type = 'lab_sequences_hibehrt'
    if features.get('metadata', {}).get('feature_type') == expected_type:
        validation_results['required_keys'].append(f"✓ Feature type: {expected_type}")
    else:
        validation_results['warnings'].append(f"Unexpected feature type")
    
    validation_results['is_valid'] = len(validation_results['errors']) == 0
    
    return validation_results


# Example usage for notebook integration
def example_lab_hibehrt_integration():
    """Example of how to integrate lab processing in notebook."""
    
    print("Example: Lab Processing for HiBEHRT Integration")
    print("="*50)
    
    # This would be in your notebook:
    # 1. Load your cohort data
    # admissions_df = pd.read_parquet('data/cohort.parquet')
    # 
    # 2. Load lab events
    # labevents_df = pd.read_csv('physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz', compression='gzip')
    # labitems_df = pd.read_csv('physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz', compression='gzip')
    # labevents_df = labevents_df.merge(labitems_df[['itemid', 'label']], on='itemid', how='left')
    # 
    # 3. Process labs for HiBEHRT
    # lab_features_path = prepare_lab_features_for_hibehrt(
    #     labevents_df, 
    #     admissions_df, 
    #     output_dir='data/processed',
    #     split_type='all'
    # )
    # 
    # 4. Validate features
    # validation = validate_lab_features_for_hibehrt(lab_features_path)
    # print(f"Validation results: {validation['is_valid']}")
    # 
    # 5. Load features for HiBEHRT model
    # from lab_features_hibehrt import load_lab_hibehrt_features
    # lab_features = load_lab_hibehrt_features(lab_features_path)
    # lab_sequences = lab_features['sequences']
    # lab_vocab = lab_features['vocabulary']
    
    print("Integration complete - lab features ready for HiBEHRT model!")


if __name__ == '__main__':
    example_lab_hibehrt_integration()