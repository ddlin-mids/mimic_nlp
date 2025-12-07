import numpy as np
import pandas as pd
import os

embedding_path = "data/interim/ehr_long_los/embeddings/structured_ehr_embeddings.npz"
mapping_path = "data/interim/ehr_long_los/graph/graph_node_map.csv"

if not os.path.exists(embedding_path):
    print(f"Error: {embedding_path} does not exist.")
else:
    print(f"Loading {embedding_path}...")
    emb_data = np.load(embedding_path)
    emb_nodes = emb_data['node_names']
    print(f"Embedding nodes: {len(emb_nodes)}")
    print(f"First 5: {emb_nodes[:5]}")

if not os.path.exists(mapping_path):
    print(f"Error: {mapping_path} does not exist.")
else:
    print(f"Loading {mapping_path}...")
    map_df = pd.read_csv(mapping_path)
    graph_nodes = map_df['node_name'].values
    print(f"Graph nodes: {len(graph_nodes)}")
    print(f"First 5: {graph_nodes[:5]}")

if os.path.exists(embedding_path) and os.path.exists(mapping_path):
    if len(emb_nodes) != len(graph_nodes):
        print("MISMATCH: Lengths differ!")
    elif not np.array_equal(emb_nodes, graph_nodes):
        print("MISMATCH: Node names do not match identically!")
        # Check if they are just sets that match
        emb_set = set(emb_nodes)
        graph_set = set(graph_nodes)
        if emb_set == graph_set:
            print("Set match: The sets of nodes are the same, but order differs.")
        else:
            print(f"Set mismatch: {len(emb_set)} vs {len(graph_set)}")
            diff = emb_set.symmetric_difference(graph_set)
            print(f"Difference size: {len(diff)}")
    else:
        print("SUCCESS: Nodes are perfectly aligned.")
