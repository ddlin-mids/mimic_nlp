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
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def _symmetrize_edges(source_nodes, target_nodes, edge_weights):
    """
    Make edges undirected by taking the max weight for each unordered pair and dropping self-loops.
    """
    src = source_nodes.astype(int)
    tgt = target_nodes.astype(int)
    wts = edge_weights.astype(float)

    mask = src != tgt
    src = src[mask]
    tgt = tgt[mask]
    wts = wts[mask]

    edge_dict = {}
    for s, t, w in zip(src, tgt, wts):
        a, b = (s, t) if s < t else (t, s)
        key = (a, b)
        if key not in edge_dict or w > edge_dict[key]:
            edge_dict[key] = w

    undirected_src = []
    undirected_tgt = []
    undirected_w = []
    for (a, b), w in edge_dict.items():
        undirected_src.extend([a, b])
        undirected_tgt.extend([b, a])
        undirected_w.extend([w, w])

    return (
        np.array(undirected_src, dtype=np.int64),
        np.array(undirected_tgt, dtype=np.int64),
        np.array(undirected_w, dtype=np.float32),
    )


def _load_aligned_embeddings(filepath, target_ids, agg_mode='last'):
    """Load and align embeddings from .npz file."""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None
    
    data = np.load(filepath)
    # Handle key variations
    id_key = 'hadm_ids' if 'hadm_ids' in data else 'ids'
    embed_key = 'embeddings'
    
    if id_key not in data or embed_key not in data:
        logger.warning(f"Keys {id_key}/{embed_key} not found in {filepath}")
        return None

    source_ids = data[id_key]
    source_embeds = data[embed_key]
    
    # Handle multi-row per hadm_id (e.g. radiology)
    id_map = defaultdict(list)
    for k, v in zip(source_ids, source_embeds):
        # normalize ID to int
        try:
            k_int = int(k)
        except:
            k_int = k
        id_map[k_int].append(v)
        
    dim = source_embeds.shape[1]
    aligned = []
    
    for tid in target_ids:
        if tid in id_map:
            # Average multiple embeddings if present
            aligned.append(np.mean(id_map[tid], axis=0))
        else:
            aligned.append(np.zeros(dim))
            
    return np.array(aligned)

def main(args):
    # Load cohort first as it's often needed for ID verification
    logger.info(f"Loading cohort from {args.cohort_path}...")
    cohort = pd.read_csv(args.cohort_path)
    cohort["node_name"] = cohort["subject_id"].astype(str) + "_" + cohort["hadm_id"].astype(str)

    # 1. Load Data or Embeddings
    if args.use_embeddings and os.path.exists(args.use_embeddings):
        logger.info(f"Loading embeddings from {args.use_embeddings}...")
        emb_data = np.load(args.use_embeddings)
        X_ehr = emb_data['embeddings']
        node_names_emb = emb_data['node_names']
        logger.info(f"Loaded EHR embeddings shape: {X_ehr.shape}")
        
        # Extract hadm_ids from node_names (format: subject_hadm)
        target_ids = []
        for name in node_names_emb:
            try:
                target_ids.append(int(name.split('_')[1]))
            except:
                target_ids.append(-1)
        target_ids = np.array(target_ids)
        
        # Load Text Embeddings if provided
        X_text = None
        if args.text_path:
            logger.info(f"Loading text embeddings from base {args.text_path}...")
            
            disch_path = os.path.join(args.text_path, "discharge_summary.npz")
            rad_path = os.path.join(args.text_path, "radiology_report.npz")
            
            # If not found, maybe they are in 'embeddings/notes/'
            if not os.path.exists(disch_path):
                 disch_path = os.path.join(args.text_path, "embeddings/notes/discharge_summary.npz")
                 rad_path = os.path.join(args.text_path, "embeddings/notes/radiology_report.npz")

            logger.info(f"Discharge path: {disch_path}")
            logger.info(f"Radiology path: {rad_path}")

            X_disch = _load_aligned_embeddings(disch_path, target_ids)
            X_rad = _load_aligned_embeddings(rad_path, target_ids)
            
            to_concat = []
            if X_disch is not None: to_concat.append(X_disch)
            if X_rad is not None: to_concat.append(X_rad)
            
            if to_concat:
                X_text = np.hstack(to_concat)
                logger.info(f"Loaded Text embeddings shape: {X_text.shape}")
            else:
                logger.warning("No text embeddings found.")

        # Concatenate EHR + Text
        if X_text is not None:
            # Check dimensions compatibility
            if X_text.shape[0] != X_ehr.shape[0]:
                logger.error(f"Mismatch in rows: EHR {X_ehr.shape[0]} vs Text {X_text.shape[0]}")
                return
            X_knn = np.hstack([X_ehr, X_text])
            logger.info(f"Combined Multimodal Embeddings: {X_knn.shape}")
        else:
            X_knn = X_ehr

        # Use embeddings for KNN
        # Normalize for cosine metric
        norms = np.linalg.norm(X_knn, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_tfidf = X_knn / norms
        
        # Prepare mapping_df for later
        mapping_df = pd.DataFrame({
            'node_name': node_names_emb,
            'hadm_id': target_ids
        })

    else:
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
        
        # Cohort already loaded at top
        
        mapping_df = None
        if os.path.exists(args.mapping_path):
            logger.info(f"Aligning to {args.mapping_path}...")
            mapping_df = pd.read_csv(args.mapping_path)
            mapping_df['hadm_id'] = mapping_df['node_name'].apply(lambda x: int(x.split('_')[1]))
            target_ids = mapping_df['hadm_id'].values
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
    # Convert sparse to dense if needed for NearestNeighbors (it handles sparse, but let's be safe if using embeddings)
    if not isinstance(X_tfidf, np.ndarray):
        # Sparse matrix from TfidfTransformer
        pass # NearestNeighbors handles sparse
    
    knn = NearestNeighbors(n_neighbors=args.k, metric='cosine', n_jobs=8)
    knn.fit(X_tfidf)
    distances, indices = knn.kneighbors(X_tfidf)

    # Convert to PyG Graph
    source_nodes = np.repeat(np.arange(X_tfidf.shape[0]), args.k)
    target_nodes = indices.flatten()
    edge_weights = 1 - distances.flatten()

    # Symmetrize and drop self-loops
    source_nodes, target_nodes, edge_weights = _symmetrize_edges(source_nodes, target_nodes, edge_weights)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    # Use features as node features
    if isinstance(X_tfidf, np.ndarray):
         node_features = torch.tensor(X_tfidf, dtype=torch.float)
    else:
         node_features = torch.tensor(X_tfidf.toarray(), dtype=torch.float)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, x=node_features, num_nodes=X_tfidf.shape[0])

    # 5. Save
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "graph_pyg.pt")
    torch.save(data, out_path)
    logger.info(f"Saved PyG graph to {out_path}")

    # Save node ordering for downstream label alignment
    if mapping_df is not None:
        node_names = mapping_df["node_name"].tolist()
    else:
        # Fallback if mapping_df wasn't created (should be covered by else block logic but just in case)
        subj_map = cohort.set_index("hadm_id")["subject_id"].to_dict()
        node_names = [f"{subj_map.get(h, 'unknown')}_{int(h)}" for h in target_ids]
    
    mapping_out = os.path.join(args.save_dir, "graph_node_map.csv")
    map_df = pd.DataFrame({
        "node_idx": np.arange(len(target_ids)),
        "hadm_id": target_ids,
        "node_name": node_names
    })
    map_df.to_csv(mapping_out, index=False)
    logger.info(f"Saved node mapping to {mapping_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/interim/ehr_long_los/ehr_preprocessed_all_one_hot.pkl")
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--mapping_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="data/interim/ehr_long_los/graph")
    parser.add_argument("--k", type=int, default=15, help="Number of neighbors")
    parser.add_argument("--use_embeddings", type=str, default=None, help="Path to embeddings .npz to use for graph construction")
    parser.add_argument("--text_path", type=str, default=None, help="Base path for text embeddings (optional)")
    
    args = parser.parse_args()
    main(args)
