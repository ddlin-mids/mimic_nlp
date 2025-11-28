import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def filter_cohort_for_text(cohort_path, discharge_path, radiology_path, output_path):
    print(f"Loading cohort from {cohort_path}...")
    cohort = pd.read_csv(cohort_path)
    initial_len = len(cohort)
    print(f"Initial cohort size: {initial_len}")
    
    cohort_hadms = set(cohort['hadm_id'])
    
    # 1. Check Discharge Summaries
    print(f"Scanning Discharge Summaries from {discharge_path}...")
    discharge_hadms = set()
    sample_discharge_printed = False
    
    for chunk in pd.read_csv(discharge_path, chunksize=50000, usecols=['hadm_id', 'text', 'note_type']):
        # Filter to cohort first
        chunk = chunk[chunk['hadm_id'].isin(cohort_hadms)]
        if chunk.empty:
            continue
            
        # Ensure note_type is Discharge Summary
        chunk = chunk[chunk['note_type'] == 'DS']
        
        if not chunk.empty:
            if not sample_discharge_printed:
                print("\n--- SAMPLE DISCHARGE SUMMARY ---")
                print(chunk.iloc[0]['text'][:500] + "...")
                print("--------------------------------\n")
                sample_discharge_printed = True
                
            discharge_hadms.update(chunk['hadm_id'].unique())
            
    print(f"Admissions with Discharge Summaries: {len(discharge_hadms)} ({len(discharge_hadms)/initial_len:.1%})")
    
    # 2. Check Radiology Reports
    print(f"Scanning Radiology Reports from {radiology_path}...")
    radiology_hadms = set()
    sample_radiology_printed = False
    
    # We might want to filter for Chest X-rays specifically?
    # Common terms: CHEST, PORTABLE, PA/LAT
    # For now, let's see what coverage we get with ANY radiology linked to the admission
    
    for chunk in pd.read_csv(radiology_path, chunksize=50000, usecols=['hadm_id', 'text', 'note_type']):
        chunk = chunk[chunk['hadm_id'].isin(cohort_hadms)]
        if chunk.empty:
            continue
            
        # We assume 'note_type' == 'RR' (Radiology Report) usually, but let's accept any from this file
        
        if not chunk.empty:
            # Just for verification, let's look for 'CHEST' in text
            # This is slow to do for all, but let's just grab IDs for now
            
            if not sample_radiology_printed:
                # Find a chest one for sample
                chest_sample = chunk[chunk['text'].str.contains('CHEST', case=False, na=False)]
                if not chest_sample.empty:
                    print("\n--- SAMPLE RADIOLOGY REPORT (CHEST) ---")
                    print(chest_sample.iloc[0]['text'][:500] + "...")
                    print("--------------------------------------\n")
                    sample_radiology_printed = True
            
            radiology_hadms.update(chunk['hadm_id'].unique())

    print(f"Admissions with Radiology Reports: {len(radiology_hadms)} ({len(radiology_hadms)/initial_len:.1%})")
    
    # 3. Intersection
    valid_hadms = discharge_hadms.intersection(radiology_hadms)
    print(f"Admissions with BOTH: {len(valid_hadms)} ({len(valid_hadms)/initial_len:.1%})")
    
    # 4. Filter Cohort
    final_cohort = cohort[cohort['hadm_id'].isin(valid_hadms)].copy()
    
    print(f"Saving filtered cohort ({len(final_cohort)} rows) to {output_path}...")
    final_cohort.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=str, required=True)
    parser.add_argument("--discharge", type=str, default="physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz")
    parser.add_argument("--radiology", type=str, default="physionet.org/files/mimic-iv-note/2.2/note/radiology.csv.gz")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    filter_cohort_for_text(args.cohort, args.discharge, args.radiology, args.output)
