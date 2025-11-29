import pickle
import numpy as np
import sys

pkl_path = "data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl"

print(f"Loading {pkl_path}...")
try:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print("Keys in pickle:", data.keys())
    
    feat_dict = data["feat_dict"]
    sample_key = next(iter(feat_dict))
    sample_val = feat_dict[sample_key]
    
    print(f"\nSample Key: {sample_key}")
    print(f"Sample Value Type: {type(sample_val)}")
    if isinstance(sample_val, np.ndarray):
        print(f"Sample Value Shape: {sample_val.shape}")
        print(f"Sample Value Dtype: {sample_val.dtype}")
        print("Sample Data (first 2 rows):")
        print(sample_val[:2])
        
        # Check for non-numeric content if object
        if sample_val.dtype == object:
            print("\nObject array content analysis:")
            print(f"Element type at [0,0]: {type(sample_val[0,0])}")
            print(f"Value at [0,0]: {sample_val[0,0]}")

    if "feature_cols" in data:
        print(f"\nFeature Cols ({len(data['feature_cols'])}):")
        print(data['feature_cols'][:10])
        
    if "cat_dims" in data:
        print(f"\nCat Dims: {data['cat_dims']}")
        
except Exception as e:
    print(f"Error: {e}")
