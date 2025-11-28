import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

def main(args):
    print(f"Loading data from {args.input}...")
    # Load the flattened daily data
    df = pd.read_csv(args.input)
    
    # Columns to exclude from the feature matrix (identifiers, demographics, labels)
    # Note: We are aggregating by admission, so we need to identify feature columns
    # Features are counts (int) or flags. 
    # Demographic columns like 'gender', 'race' are categorical and static.
    # Time-varying columns are the ones we want to aggregate.
    
    # Identify feature columns (all columns that are not metadata)
    meta_cols = [
        "subject_id", "hadm_id", "admittime", "dischtime", "date", 
        "target", "split", "splits", "node_name", 
        "age_at_admit", "gender", "race", 
        "readmitted_within_30days", "readmitted_within_window",
        "num_prior_admissions", "num_prior_30d_readmissions", "days_since_last_discharge",
        "date_range", "Day_Number"
    ]
    
    feature_cols = [c for c in df.columns if c not in meta_cols and c != "Unnamed: 0"]
    
    print(f"Found {len(feature_cols)} potential dynamic features.")
    
    # Aggregation Strategy:
    # 1. Group by hadm_id
    # 2. Sum feature counts (or count occurrences for labs)
    # 3. Take first for static features (metadata)
    
    print("Aggregating daily features to admission level...")
    
    # We only aggregate the feature columns. 
    # For labs (which might be 'abnormal', 'nan', etc.), we need to handle them.
    # The preprocessing script output (ehr_combined.csv) might have already one-hot encoded them?
    # Let's check the column types. Ideally, ehr_combined.csv is fully numeric/one-hot if we look at `preprocess_ehr.py`.
    # Checking `preprocess_ehr.py`:
    # - ICD: Counts (int)
    # - Meds: Counts (int)
    # - Labs: One-hot encoded flags? No, `lab_one_hot_mimic` returns strings 'abnormal'/'nan'.
    # We need to convert these to numeric before aggregation if they aren't already.
    
    # Optimized aggregation:
    # Iterate columns, if numeric -> sum. If object -> count 'abnormal'.
    
    agg_funcs = {}
    
    # Heuristic to distinguish types
    # We will just process the dataframe to ensure numeric
    
    # Convert 'abnormal' to 1, others to 0 for object columns
    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = (df[col] == 'abnormal').astype(int)
        else:
            df[col] = df[col].fillna(0).astype(int)
            
    # Now groupby sum
    df_agg = df.groupby("hadm_id")[feature_cols].sum().reset_index()
    
    # Get metadata (order must match df_agg)
    # We fetch subject_id for the mapping
    df_meta = df.groupby("hadm_id")[["subject_id"]].first().reset_index()
    
    # Align
    df_agg = df_agg.sort_values("hadm_id")
    df_meta = df_meta.sort_values("hadm_id")
    
    X = df_agg[feature_cols].values
    hadm_ids = df_agg["hadm_id"].values
    subject_ids = df_meta["subject_id"].values
    
    print(f"Shape of aggregated matrix: {X.shape}")
    
    # Pipeline: TF-IDF (to downweight common codes) -> SVD (to reduce dim)
    print(f"fitting SVD with n_components={args.dims}...")
    pipeline = make_pipeline(
        TfidfTransformer(),
        TruncatedSVD(n_components=args.dims, random_state=42),
        StandardScaler()
    )
    
    embeddings = pipeline.fit_transform(X)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Explained variance ratio: {pipeline.named_steps['truncatedsvd'].explained_variance_ratio_.sum():.4f}")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create node names to match the other pipeline (subject_hadm)
    node_names = [f"{s}_{h}" for s, h in zip(subject_ids, hadm_ids)]
    
    np.savez(
        args.output, 
        embeddings=embeddings, 
        hadm_ids=hadm_ids, 
        node_names=node_names
    )
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to ehr_combined.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npz")
    parser.add_argument("--dims", type=int, default=128, help="Embedding dimensions")
    args = parser.parse_args()
    main(args)
