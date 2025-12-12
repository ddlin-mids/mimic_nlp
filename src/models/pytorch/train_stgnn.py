"""
Training script for Spatiotemporal Graph Neural Network (STGNN).

This script trains the STGNN model which combines:
- Graph convolution for patient similarity learning
- GRU for temporal sequence modeling
- Optional multimodal fusion with text embeddings

Uses PyTorch Geometric for graph operations.
Supports AMP (Automatic Mixed Precision) and gradient checkpointing for memory efficiency.

Usage:
    python src/models/pytorch/train_stgnn.py \
        --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl \
        --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
        --graph_path data/interim/ehr_long_los/graph/graph_pyg.pt \
        --use_amp
"""

import argparse
import os
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import mlflow

from stgnn import STGNN, STGNNMultimodal, build_knn_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(args):
    """Load and prepare data for STGNN training."""
    
    # Load cohort
    logger.info(f"Loading cohort from {args.cohort_path}...")
    cohort = pd.read_csv(args.cohort_path)
    cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
    cohort['node_name'] = cohort['subject_id'].astype(str) + "_" + cohort['hadm_id'].astype(str)
    
    # Create mappings
    labels_map = dict(zip(cohort['node_name'], cohort['readmitted_within_window'].astype(int)))
    split_map = dict(zip(cohort['node_name'], cohort['split']))
    hadm_map = dict(zip(cohort['node_name'], cohort['hadm_id']))
    
    # Load EHR sequences
    logger.info(f"Loading EHR sequences from {args.ehr_seq_path}...")
    with open(args.ehr_seq_path, 'rb') as f:
        seq_data = pickle.load(f)
    
    ehr_feat_dict = seq_data['feat_dict']
    cat_idxs = seq_data.get('cat_idxs', [])
    cat_dims = seq_data.get('cat_dims', [])
    
    # Get feature dimension
    sample_key = next(iter(ehr_feat_dict))
    ehr_dim = ehr_feat_dict[sample_key].shape[1]
    logger.info(f"EHR feature dimension: {ehr_dim}")
    
    # Load text embeddings if provided
    text_emb_dict = None
    text_dim = None
    if args.text_dir:
        logger.info(f"Loading text embeddings from {args.text_dir}...")
        text_emb_dict = {}
        
        def load_emb(name):
            path = Path(args.text_dir) / f"{name}.npz"
            if not path.exists():
                return {}
            data = np.load(path)
            ids = data['hadm_ids'] if 'hadm_ids' in data else data['ids']
            embs = data['embeddings']
            
            grouped = defaultdict(list)
            for i, e in zip(ids, embs):
                grouped[int(i)].append(e)
            return {i: np.mean(elist, axis=0) for i, elist in grouped.items()}
        
        disch_map = load_emb("discharge_summary")
        rad_map = load_emb("radiology_report")
        
        # Combine embeddings
        d_dim = next(iter(disch_map.values())).shape[0] if disch_map else 768
        r_dim = next(iter(rad_map.values())).shape[0] if rad_map else 768
        
        for node_name in labels_map.keys():
            hid = hadm_map[node_name]
            d_vec = disch_map.get(hid, np.zeros(d_dim))
            r_vec = rad_map.get(hid, np.zeros(r_dim))
            text_emb_dict[node_name] = np.concatenate([d_vec, r_vec])
        
        text_dim = d_dim + r_dim
        logger.info(f"Text embedding dimension: {text_dim}")
    
    # Get node order (consistent with graph if loading from file)
    if args.graph_path and os.path.exists(args.graph_path):
        # Load graph node mapping
        graph_dir = os.path.dirname(args.graph_path)
        node_map_path = os.path.join(graph_dir, "graph_node_map.csv")
        if os.path.exists(node_map_path):
            node_map_df = pd.read_csv(node_map_path)
            ordered_nodes = node_map_df['node_name'].tolist()
            logger.info(f"Using node order from graph: {len(ordered_nodes)} nodes")
        else:
            ordered_nodes = [k for k in ehr_feat_dict.keys() if k in labels_map]
    else:
        ordered_nodes = [k for k in ehr_feat_dict.keys() if k in labels_map]
    
    # Filter to nodes that have all required data
    valid_nodes = []
    for node in ordered_nodes:
        if node in ehr_feat_dict and node in labels_map:
            valid_nodes.append(node)
    
    logger.info(f"Valid nodes: {len(valid_nodes)}")
    
    # Split indices
    train_idx = [i for i, n in enumerate(valid_nodes) if split_map.get(n) == 'train']
    val_idx = [i for i, n in enumerate(valid_nodes) if split_map.get(n) == 'val']
    test_idx = [i for i, n in enumerate(valid_nodes) if split_map.get(n) == 'test']
    
    logger.info(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Prepare sequences (pad to max_len)
    max_len = args.max_seq_len
    ehr_sequences = []
    lengths = []
    labels = []
    text_embeddings = []
    
    for node in valid_nodes:
        seq = ehr_feat_dict[node]
        # Convert object dtype to float32 if needed
        if seq.dtype == object:
            seq = np.array(seq.tolist(), dtype=np.float32)
        else:
            seq = seq.astype(np.float32)
        
        seq_len = len(seq)
        
        # Truncate or pad
        if seq_len > max_len:
            seq = seq[-max_len:]  # Keep last max_len days
            seq_len = max_len
        elif seq_len < max_len:
            # Pad at the beginning
            pad = np.zeros((max_len - seq_len, ehr_dim), dtype=np.float32)
            seq = np.vstack([pad, seq])
        
        ehr_sequences.append(seq)
        lengths.append(min(seq_len, max_len))
        labels.append(labels_map[node])
        
        if text_emb_dict:
            text_embeddings.append(text_emb_dict.get(node, np.zeros(text_dim)).astype(np.float32))
    
    ehr_sequences = np.stack(ehr_sequences)  # (num_nodes, max_len, ehr_dim)
    lengths = np.array(lengths)
    labels = np.array(labels)
    
    if text_emb_dict:
        text_embeddings = np.stack(text_embeddings)  # (num_nodes, text_dim)
    else:
        text_embeddings = None
    
    logger.info(f"EHR sequences shape: {ehr_sequences.shape}")
    if text_embeddings is not None:
        logger.info(f"Text embeddings shape: {text_embeddings.shape}")
    
    return {
        'ehr_sequences': ehr_sequences,
        'lengths': lengths,
        'labels': labels,
        'text_embeddings': text_embeddings,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'ehr_dim': ehr_dim,
        'text_dim': text_dim,
        'cat_idxs': cat_idxs,
        'cat_dims': cat_dims,
        'valid_nodes': valid_nodes,
    }


def load_or_build_graph(args, data_dict, device):
    """Load existing graph or build k-NN graph.
    
    If loading a pre-built graph, re-indexes edges to match valid_nodes subset.
    """
    
    if args.graph_path and os.path.exists(args.graph_path):
        logger.info(f"Loading graph from {args.graph_path}...")
        pyg_data = torch.load(args.graph_path, weights_only=False)
        
        # Load node mapping to get original indices
        graph_dir = os.path.dirname(args.graph_path)
        node_map_path = os.path.join(graph_dir, "graph_node_map.csv")
        
        if os.path.exists(node_map_path):
            node_map_df = pd.read_csv(node_map_path)
            # Create mapping from node_name to original graph index
            name_to_orig_idx = dict(zip(node_map_df['node_name'], node_map_df['node_idx']))
            
            # Get valid nodes and their original indices
            valid_nodes = data_dict['valid_nodes']
            valid_orig_indices = []
            for node in valid_nodes:
                if node in name_to_orig_idx:
                    valid_orig_indices.append(name_to_orig_idx[node])
                else:
                    valid_orig_indices.append(-1)  # Not in graph
            
            valid_orig_indices = np.array(valid_orig_indices)
            valid_mask = valid_orig_indices >= 0
            
            if not valid_mask.all():
                logger.warning(f"{(~valid_mask).sum()} valid nodes not found in graph node map")
            
            # Create mapping from original index to new index (0 to num_valid-1)
            orig_to_new = {orig: new for new, orig in enumerate(valid_orig_indices) if orig >= 0}
            
            # Filter and re-index edges (vectorized for speed)
            edge_index_np = pyg_data.edge_index.numpy()
            edge_attr_np = pyg_data.edge_attr.numpy() if pyg_data.edge_attr is not None else None
            
            # Create fast lookup array: orig_idx -> new_idx (or -1 if invalid)
            max_orig_idx = int(edge_index_np.max()) + 1
            orig_to_new_arr = np.full(max_orig_idx, -1, dtype=np.int64)
            for new_idx, orig_idx in enumerate(valid_orig_indices):
                if orig_idx >= 0:
                    orig_to_new_arr[orig_idx] = new_idx
            
            # Vectorized edge filtering
            src_orig = edge_index_np[0]
            dst_orig = edge_index_np[1]
            
            # Map to new indices (-1 for invalid)
            src_new = orig_to_new_arr[src_orig]
            dst_new = orig_to_new_arr[dst_orig]
            
            # Keep edges where both endpoints are valid
            valid_edges_mask = (src_new >= 0) & (dst_new >= 0)
            
            new_src = src_new[valid_edges_mask]
            new_dst = dst_new[valid_edges_mask]
            
            edge_index = torch.tensor(np.stack([new_src, new_dst]), dtype=torch.long, device=device)
            if edge_attr_np is not None:
                new_weights = edge_attr_np[valid_edges_mask]
                edge_weight = torch.tensor(new_weights, dtype=torch.float32, device=device)
            else:
                edge_weight = None
            
            logger.info(f"Re-indexed graph: {len(valid_nodes)} nodes, {edge_index.shape[1]} edges "
                       f"(from original {pyg_data.num_nodes} nodes, {pyg_data.edge_index.shape[1]} edges)")
        else:
            # No node map, use graph as-is (may fail if sizes don't match)
            logger.warning("No node map found, using graph as-is")
            edge_index = pyg_data.edge_index.to(device)
            edge_weight = pyg_data.edge_attr.to(device) if pyg_data.edge_attr is not None else None
            logger.info(f"Loaded graph: {pyg_data.num_nodes} nodes, {edge_index.shape[1]} edges")
    else:
        logger.info(f"Building k-NN graph (k={args.k_neighbors})...")
        # Use EHR embeddings + text for graph construction
        ehr_flat = data_dict['ehr_sequences'].mean(axis=1)  # Mean over time
        
        if data_dict['text_embeddings'] is not None:
            features = np.hstack([ehr_flat, data_dict['text_embeddings']])
        else:
            features = ehr_flat
        
        edge_index, edge_weight = build_knn_graph(features, k=args.k_neighbors)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        logger.info(f"Built graph: {len(data_dict['valid_nodes'])} nodes, {edge_index.shape[1]} edges")
    
    return edge_index, edge_weight


def train_epoch(model, data_dict, edge_index, edge_weight, train_idx, criterion, optimizer, device, args, scaler=None):
    """Train for one epoch with optional AMP."""
    model.train()
    
    # Get training data
    ehr_seq = torch.tensor(data_dict['ehr_sequences'], dtype=torch.float32, device=device)
    lengths = torch.tensor(data_dict['lengths'], dtype=torch.long, device=device)
    labels = torch.tensor(data_dict['labels'], dtype=torch.float32, device=device)
    
    if data_dict['text_embeddings'] is not None:
        text_emb = torch.tensor(data_dict['text_embeddings'], dtype=torch.float32, device=device)
    else:
        text_emb = None
    
    # Training mask
    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    
    optimizer.zero_grad()
    
    # Forward pass with optional AMP
    use_amp = scaler is not None
    with autocast(enabled=use_amp):
        if text_emb is not None:
            logits, _ = model(ehr_seq, edge_index, edge_weight, text_emb, lengths)
        else:
            logits, _ = model(ehr_seq, edge_index, edge_weight, lengths=lengths)
        
        # Compute loss only on training nodes
        loss = criterion(logits[train_mask], labels[train_mask])
    
    # Backward with optional AMP scaling
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Metrics
    with torch.no_grad():
        probs = torch.sigmoid(logits[train_mask]).cpu().numpy()
        y_true = labels[train_mask].cpu().numpy()
        
        try:
            auc = roc_auc_score(y_true, probs)
        except:
            auc = 0.5
    
    return loss.item(), auc


@torch.no_grad()
def evaluate(model, data_dict, edge_index, edge_weight, eval_idx, device, threshold=0.5, use_amp=False):
    """Evaluate model on a subset of nodes with optional AMP."""
    model.eval()
    
    # Get data
    ehr_seq = torch.tensor(data_dict['ehr_sequences'], dtype=torch.float32, device=device)
    lengths = torch.tensor(data_dict['lengths'], dtype=torch.long, device=device)
    labels = torch.tensor(data_dict['labels'], dtype=torch.float32, device=device)
    
    if data_dict['text_embeddings'] is not None:
        text_emb = torch.tensor(data_dict['text_embeddings'], dtype=torch.float32, device=device)
    else:
        text_emb = None
    
    # Forward pass with optional AMP
    with autocast(enabled=use_amp):
        if text_emb is not None:
            logits, _ = model(ehr_seq, edge_index, edge_weight, text_emb, lengths)
        else:
            logits, _ = model(ehr_seq, edge_index, edge_weight, lengths=lengths)
    
    # Get predictions for eval nodes
    probs = torch.sigmoid(logits[eval_idx]).cpu().numpy()
    y_true = labels[eval_idx].cpu().numpy()
    y_pred = (probs >= threshold).astype(int)
    
    # Metrics
    try:
        auc = roc_auc_score(y_true, probs)
    except:
        auc = 0.5
    
    try:
        auprc = average_precision_score(y_true, probs)
    except:
        auprc = 0.0
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    acc = (y_pred == y_true).mean()
    
    return {
        'auc': float(auc),
        'auprc': float(auprc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'acc': float(acc),
        'probs': probs,
        'y_true': y_true,
    }


def find_optimal_threshold(model, data_dict, edge_index, edge_weight, val_idx, device, use_amp=False):
    """Find optimal classification threshold using validation set."""
    results = evaluate(model, data_dict, edge_index, edge_weight, val_idx, device, threshold=0.5, use_amp=use_amp)
    probs = results['probs']
    y_true = results['y_true']
    
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup AMP scaler if enabled
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP)")
    
    # MLflow
    mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")
    mlflow.set_experiment("stgnn_readmission")
    
    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        mlflow.log_params({
            'hidden_dim': args.hidden_dim,
            'num_gru_layers': args.num_gru_layers,
            'dropout': args.dropout,
            'lr': args.lr,
            'epochs': args.epochs,
            'max_seq_len': args.max_seq_len,
            'k_neighbors': args.k_neighbors,
            'conv_type': args.conv_type,
            'fusion_type': args.fusion_type,
            'use_text': args.text_dir is not None,
            'use_amp': use_amp,
            'use_checkpointing': args.use_checkpointing,
        })
        
        # Load data
        data_dict = load_data(args)
        
        # Log memory estimate
        ehr_memory_gb = data_dict['ehr_sequences'].nbytes / (1024**3)
        logger.info(f"EHR sequences memory: {ehr_memory_gb:.2f} GB")
        
        # Load/build graph
        edge_index, edge_weight = load_or_build_graph(args, data_dict, device)
        
        # Build model
        if args.fusion_type == 'multimodal' and data_dict['text_dim'] is not None:
            model = STGNNMultimodal(
                ehr_input_dim=data_dict['ehr_dim'],
                text_dim=data_dict['text_dim'],
                hidden_dim=args.hidden_dim,
                num_gru_layers=args.num_gru_layers,
                num_classes=1,
                conv_type=args.conv_type,
                dropout=args.dropout,
                cat_idxs=data_dict['cat_idxs'] if args.use_cat_emb else [],
                cat_dims=data_dict['cat_dims'] if args.use_cat_emb else [],
                cat_emb_dim=args.cat_emb_dim,
            )
        else:
            model = STGNN(
                ehr_input_dim=data_dict['ehr_dim'],
                hidden_dim=args.hidden_dim,
                num_gru_layers=args.num_gru_layers,
                num_classes=1,
                conv_type=args.conv_type,
                dropout=args.dropout,
                cat_idxs=data_dict['cat_idxs'] if args.use_cat_emb else [],
                cat_dims=data_dict['cat_dims'] if args.use_cat_emb else [],
                cat_emb_dim=args.cat_emb_dim,
                text_dim=data_dict['text_dim'] if args.fusion_type != 'none' else None,
                fusion_type=args.fusion_type,
                use_checkpointing=args.use_checkpointing,
            )
        
        model = model.to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        # Handle class imbalance
        pos_weight = torch.tensor([(1 - data_dict['labels'].mean()) / data_dict['labels'].mean()], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Training loop
        best_val_auc = 0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Train with optional AMP
            train_loss, train_auc = train_epoch(
                model, data_dict, edge_index, edge_weight, data_dict['train_idx'],
                criterion, optimizer, device, args, scaler=scaler
            )
            
            # Validate with optional AMP
            val_results = evaluate(model, data_dict, edge_index, edge_weight, data_dict['val_idx'], device, use_amp=use_amp)
            
            scheduler.step()
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_auc': val_results['auc'],
                'val_auprc': val_results['auprc'],
                'val_f1': val_results['f1'],
            }, step=epoch)
            
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Train AUC: {train_auc:.4f} | "
                f"Val AUC: {val_results['auc']:.4f} | "
                f"Val AUPRC: {val_results['auprc']:.4f}"
            )
            
            # Early stopping
            if val_results['auc'] > best_val_auc:
                best_val_auc = val_results['auc']
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_results['auc'],
                }, os.path.join(args.save_dir, 'stgnn_best.pt'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model for evaluation
        checkpoint = torch.load(os.path.join(args.save_dir, 'stgnn_best.pt'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1} (Val AUC: {checkpoint['val_auc']:.4f})")
        
        # Find optimal threshold
        opt_threshold, opt_f1 = find_optimal_threshold(model, data_dict, edge_index, edge_weight, data_dict['val_idx'], device, use_amp=use_amp)
        logger.info(f"Optimal threshold: {opt_threshold:.3f} (Val F1: {opt_f1:.4f})")
        
        # Test evaluation
        test_results = evaluate(model, data_dict, edge_index, edge_weight, data_dict['test_idx'], device, threshold=float(opt_threshold), use_amp=use_amp)
        
        logger.info("=" * 60)
        logger.info("Test Results:")
        logger.info(f"  AUC:       {test_results['auc']:.4f}")
        logger.info(f"  AUPRC:     {test_results['auprc']:.4f}")
        logger.info(f"  F1:        {test_results['f1']:.4f}")
        logger.info(f"  Precision: {test_results['precision']:.4f}")
        logger.info(f"  Recall:    {test_results['recall']:.4f}")
        logger.info(f"  Accuracy:  {test_results['acc']:.4f}")
        logger.info("=" * 60)
        
        # Log final metrics
        mlflow.log_metrics({
            'best_val_auc': float(best_val_auc),
            'best_epoch': float(best_epoch),
            'optimal_threshold': float(opt_threshold),
            'test_auc': float(test_results['auc']),
            'test_auprc': float(test_results['auprc']),
            'test_f1': float(test_results['f1']),
            'test_precision': float(test_results['precision']),
            'test_recall': float(test_results['recall']),
            'test_acc': float(test_results['acc']),
        })
        
        mlflow.log_artifact(os.path.join(args.save_dir, 'stgnn_best.pt'))
        
        return test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train STGNN for readmission prediction')
    
    # Data paths
    parser.add_argument('--ehr_seq_path', type=str, required=True,
                        help='Path to EHR sequence pickle file')
    parser.add_argument('--cohort_path', type=str, required=True,
                        help='Path to cohort CSV file')
    parser.add_argument('--graph_path', type=str, default=None,
                        help='Path to pre-built PyG graph (optional)')
    parser.add_argument('--text_dir', type=str, default=None,
                        help='Directory containing text embeddings (optional)')
    parser.add_argument('--save_dir', type=str, default='data/interim/ehr_long_los/models_stgnn',
                        help='Directory to save model checkpoints')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_gru_layers', type=int, default=2,
                        help='Number of GConvGRU layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--conv_type', type=str, default='sage',
                        choices=['sage', 'gat', 'gcn'],
                        help='Graph convolution type')
    parser.add_argument('--fusion_type', type=str, default='gated',
                        choices=['none', 'concat', 'gated', 'attention', 'multimodal'],
                        help='How to fuse text embeddings')
    parser.add_argument('--use_cat_emb', action='store_true',
                        help='Use categorical embeddings')
    parser.add_argument('--cat_emb_dim', type=int, default=4,
                        help='Categorical embedding dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--max_seq_len', type=int, default=30,
                        help='Maximum sequence length')
    
    # Graph
    parser.add_argument('--k_neighbors', type=int, default=15,
                        help='Number of neighbors for k-NN graph')
    
    # Memory optimization
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP) for memory efficiency')
    parser.add_argument('--use_checkpointing', action='store_true',
                        help='Use gradient checkpointing for memory efficiency')
    
    # MLflow
    parser.add_argument('--run_name', type=str, default='stgnn_experiment',
                        help='MLflow run name')
    
    args = parser.parse_args()
    main(args)
