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
    # Use low_memory=False to avoid DtypeWarning, or specify dtypes if known
    df = pd.read_csv(args.input, low_memory=False)
    
    # Columns to exclude from the feature matrix (identifiers, demographics, labels)
    meta_cols = [
        "subject_id", "hadm_id", "admittime", "dischtime", "date", 
        "target", "split", "splits", "node_name", 
        "age_at_admit", "gender", "race", 
        "readmitted_within_30days", "readmitted_within_window",
        "num_prior_admissions", "num_prior_30d_readmissions", "days_since_last_discharge",
        "date_range", "Day_Number",
        # Add other potential meta columns that might appear
        "admission_type", "discharge_location", "is_cardiorenal_long",
        "has_acute_kidney_injury", "has_heart_failure", "has_hyponatremia",
        "has_posthemorrhagic_anemia", "has_sepsis", "has_any_cardiorenal_sepsis",
        "has_aki_and_hf"
    ]
    
    # Identify feature columns (all columns that are not metadata and not Unnamed)
    feature_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("Unnamed")]
    
    print(f"Found {len(feature_cols)} potential dynamic features.")
    
    print("Aggregating daily features to admission level...")
    
    # Ensure numeric types for aggregation
    # If columns are object (e.g. 'abnormal'/'nan'), convert to binary 1/0
    # Optimization: Check dtypes once
    obj_cols = df[feature_cols].select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        print(f"Binarizing {len(obj_cols)} object columns (assuming 'abnormal'=1)...")
        for col in obj_cols:
             df[col] = (df[col] == 'abnormal').astype(int)
    
    # Fill NaNs in numeric columns with 0 before summing
    # (Though groupby sum treats NaN as 0 usually, being explicit is safer)
    # df[feature_cols] = df[feature_cols].fillna(0) # High memory cost to do all at once?
    
    # Group by hadm_id and sum
    df_agg = df.groupby("hadm_id")[feature_cols].sum().reset_index()
    
    # Get metadata for splitting
    # We prefer to load the canonical cohort file if possible, but for now we'll trust
    # the 'split' column in df if it exists, or merge it in.
    # The input csv (ehr_combined.csv) SHOULD have the split column from preprocessing.
    
    if "split" not in df.columns and "splits" not in df.columns:
        # If missing in big csv, we must load cohort file (passed as arg or inferred)
        if args.cohort_file:
            print(f"Loading splits from {args.cohort_file}...")
            df_cohort = pd.read_csv(args.cohort_file, usecols=["hadm_id", "split"])
            # Merge split into df_agg
            df_agg = df_agg.merge(df_cohort, on="hadm_id", how="left")
        else:
            raise ValueError("Input CSV missing 'split' column and no --cohort_file provided.")
    else:
        # Take split from the first row of each group
        # We need to grab it during aggregation
        df_meta = df.groupby("hadm_id")[["subject_id", "split"] if "split" in df.columns else ["subject_id", "splits"]].first().reset_index()
        # Rename splits -> split if needed
        if "splits" in df_meta.columns:
            df_meta.rename(columns={"splits": "split"}, inplace=True)
            
        df_agg = df_agg.merge(df_meta, on="hadm_id", how="left")

    X = df_agg[feature_cols].values
    hadm_ids = df_agg["hadm_id"].values
    subject_ids = df_agg["subject_id"].values
    splits = df_agg["split"].values
    
    print(f"Shape of aggregated matrix: {X.shape}")
    
    # Identify Train indices
    train_mask = (splits == "train")
    X_train = X[train_mask]
    
    print(f"Training on {X_train.shape[0]} samples (split='train')...")
    
    if X_train.shape[0] == 0:
        raise ValueError("No training samples found! Check 'split' column values.")

    # Pipeline: TF-IDF -> SVD -> Scaler
    print(f"Fitting SVD pipeline with n_components={args.dims}...")
    pipeline = make_pipeline(
        TfidfTransformer(),
        TruncatedSVD(n_components=args.dims, random_state=42),
        StandardScaler()
    )
    
    # FIT only on TRAIN
    pipeline.fit(X_train)
    
    # TRANSFORM all
    print("Transforming all data...")
    embeddings = pipeline.transform(X)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Explained variance ratio (Train): {pipeline.named_steps['truncatedsvd'].explained_variance_ratio_.sum():.4f}")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create node names
    node_names = [f"{s}_{h}" for s, h in zip(subject_ids, hadm_ids)]
    
    np.savez(
        args.output, 
        embeddings=embeddings, 
        hadm_ids=hadm_ids, 
        node_names=node_names,
        splits=splits # Save splits too for sanity checking
    )
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to ehr_combined.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npz")
    parser.add_argument("--dims", type=int, default=128, help="Embedding dimensions")
    parser.add_argument("--cohort_file", type=str, default=None, help="Optional: Path to cohort CSV for explicit splits")
    args = parser.parse_args()
    main(args)
