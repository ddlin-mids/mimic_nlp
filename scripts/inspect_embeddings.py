import numpy as np
import os

files = [
    "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz",
    "data/interim/ehr_long_los/embeddings_transformer/structured_ehr_embeddings.npz"
]

for f in files:
    if os.path.exists(f):
        try:
            data = np.load(f)
            print(f"File: {f}")
            print(f"  Keys: {list(data.keys())}")
            if 'embeddings' in data:
                print(f"  Shape: {data['embeddings'].shape}")
            # Check for other keys like 'hadm_ids' or similar if they exist, though previous code implied alignment via mapping
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
