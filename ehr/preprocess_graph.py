import argparse
import os
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    # 1. Load Data
    logger.info(f"Loading features from {args.input_path}...")
    with open(args.input_path, "rb") as f:
        data_dict = pickle.load(f)
        
    df = data_dict['X']
    
    # 2. Aggregation (Bag of Words)
    logger.info("Aggregating daily features to admission level...")
    feature_cols = [c for c in df.columns if c not in [
        'subject_id', 'hadm_id', 'admittime', 'dischtime', 'split', 'splits', 'date', 'node_name', 'target', 'readmitted_within_30days'
    ]]
    
    grp = df.groupby('hadm_id')[feature_cols].sum()
    
    logger.info(f"Loading cohort from {args.cohort_path}...")
    cohort = pd.read_csv(args.cohort_path)
    cohort["node_name"] = cohort["subject_id"].astype(str) + "_" + cohort["hadm_id"].astype(str)
    
    if os.path.exists(args.mapping_path):
        logger.info(f"Aligning to {args.mapping_path}...")
        mapping = pd.read_csv(args.mapping_path)
        mapping['hadm_id'] = mapping['node_name'].apply(lambda x: int(x.split('_')[1]))
        target_ids = mapping['hadm_id'].values
    else:
        logger.warning("Mapping not found. Aligning to cohort file directly.")
        target_ids = cohort['hadm_id'].values

    X_counts = grp.reindex(target_ids).fillna(0).values
    logger.info(f"Feature Matrix Shape: {X_counts.shape}")
    
    # 3. TF-IDF
    logger.info("Applying TF-IDF...")
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_counts)
    
    # 4. KNN Graph Construction
    logger.info(f"Building KNN Graph (k={args.k})...")
    knn = NearestNeighbors(n_neighbors=args.k, metric='cosine', n_jobs=8)
    knn.fit(X_tfidf)
    distances, indices = knn.kneighbors(X_tfidf)
    
    # Convert to PyG Graph
    source_nodes = np.repeat(np.arange(X_tfidf.shape[0]), args.k)
    target_nodes = indices.flatten()
    edge_weights = 1 - distances.flatten()
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=X_tfidf.shape[0])
    
    # 5. Save
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "graph_pyg.pt")
    torch.save(data, out_path)
    logger.info(f"Saved PyG graph to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/interim/ehr_long_los/ehr_preprocessed_all_one_hot.pkl")
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--mapping_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="data/interim/ehr_long_los/graph")
    parser.add_argument("--k", type=int, default=15, help="Number of neighbors")
    
    args = parser.parse_args()
    main(args)