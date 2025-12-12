"""
Hyperparameter Optimization for STGNN using Optuna.

Searches over:
- hidden_dim: [64, 128, 256, 512]
- num_gru_layers: [1, 2, 3]
- dropout: [0.1, 0.5]
- lr: [1e-4, 1e-2]
- conv_type: ['sage', 'gat', 'gcn']
- fusion_type: ['none', 'gated', 'concat']

Usage:
    python src/models/pytorch/train_stgnn_hpo.py \
        --ehr_seq_path data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_pca256.pkl \
        --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
        --graph_path data/interim/ehr_long_los/graph/graph_pyg.pt \
        --n_trials 50
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

import optuna
from optuna.trial import Trial
import mlflow

from stgnn import STGNN, build_knn_graph

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
    """Load existing graph or build k-NN graph with re-indexing."""
    
    if args.graph_path and os.path.exists(args.graph_path):
        logger.info(f"Loading graph from {args.graph_path}...")
        pyg_data = torch.load(args.graph_path, weights_only=False)
        
        # Load node mapping to get original indices
        graph_dir = os.path.dirname(args.graph_path)
        node_map_path = os.path.join(graph_dir, "graph_node_map.csv")
        
        if os.path.exists(node_map_path):
            node_map_df = pd.read_csv(node_map_path)
            name_to_orig_idx = dict(zip(node_map_df['node_name'], node_map_df['node_idx']))
            
            valid_nodes = data_dict['valid_nodes']
            valid_orig_indices = []
            for node in valid_nodes:
                if node in name_to_orig_idx:
                    valid_orig_indices.append(name_to_orig_idx[node])
                else:
                    valid_orig_indices.append(-1)
            
            valid_orig_indices = np.array(valid_orig_indices)
            
            # Create fast lookup array
            edge_index_np = pyg_data.edge_index.numpy()
            edge_attr_np = pyg_data.edge_attr.numpy() if pyg_data.edge_attr is not None else None
            
            max_orig_idx = int(edge_index_np.max()) + 1
            orig_to_new_arr = np.full(max_orig_idx, -1, dtype=np.int64)
            for new_idx, orig_idx in enumerate(valid_orig_indices):
                if orig_idx >= 0:
                    orig_to_new_arr[orig_idx] = new_idx
            
            # Vectorized edge filtering
            src_orig = edge_index_np[0]
            dst_orig = edge_index_np[1]
            
            src_new = orig_to_new_arr[src_orig]
            dst_new = orig_to_new_arr[dst_orig]
            
            valid_edges_mask = (src_new >= 0) & (dst_new >= 0)
            
            new_src = src_new[valid_edges_mask]
            new_dst = dst_new[valid_edges_mask]
            
            edge_index = torch.tensor(np.stack([new_src, new_dst]), dtype=torch.long, device=device)
            if edge_attr_np is not None:
                new_weights = edge_attr_np[valid_edges_mask]
                edge_weight = torch.tensor(new_weights, dtype=torch.float32, device=device)
            else:
                edge_weight = None
            
            logger.info(f"Re-indexed graph: {len(valid_nodes)} nodes, {edge_index.shape[1]} edges")
        else:
            edge_index = pyg_data.edge_index.to(device)
            edge_weight = pyg_data.edge_attr.to(device) if pyg_data.edge_attr is not None else None
    else:
        logger.info(f"Building k-NN graph (k={args.k_neighbors})...")
        ehr_flat = data_dict['ehr_sequences'].mean(axis=1)
        
        if data_dict['text_embeddings'] is not None:
            features = np.hstack([ehr_flat, data_dict['text_embeddings']])
        else:
            features = ehr_flat
        
        edge_index, edge_weight = build_knn_graph(features, k=args.k_neighbors)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
    
    return edge_index, edge_weight


def train_and_evaluate(
    model, data_dict, edge_index, edge_weight, 
    train_idx, val_idx, device, args, 
    lr, epochs, patience, use_amp=True
):
    """Train model and return best validation AUC."""
    
    scaler = GradScaler() if use_amp else None
    
    # Loss with class imbalance handling
    pos_weight = torch.tensor([(1 - data_dict['labels'].mean()) / data_dict['labels'].mean()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Prepare tensors
    ehr_seq = torch.tensor(data_dict['ehr_sequences'], dtype=torch.float32, device=device)
    lengths = torch.tensor(data_dict['lengths'], dtype=torch.long, device=device)
    labels = torch.tensor(data_dict['labels'], dtype=torch.float32, device=device)
    
    if data_dict['text_embeddings'] is not None:
        text_emb = torch.tensor(data_dict['text_embeddings'], dtype=torch.float32, device=device)
    else:
        text_emb = None
    
    train_mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            if text_emb is not None:
                logits, _ = model(ehr_seq, edge_index, edge_weight, text_emb, lengths)
            else:
                logits, _ = model(ehr_seq, edge_index, edge_weight, lengths=lengths)
            
            loss = criterion(logits[train_mask], labels[train_mask])
        
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
        
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            with autocast(enabled=use_amp):
                if text_emb is not None:
                    logits, _ = model(ehr_seq, edge_index, edge_weight, text_emb, lengths)
                else:
                    logits, _ = model(ehr_seq, edge_index, edge_weight, lengths=lengths)
            
            probs = torch.sigmoid(logits[val_idx]).cpu().numpy()
            y_true = labels[val_idx].cpu().numpy()
            
            try:
                val_auc = roc_auc_score(y_true, probs)
            except:
                val_auc = 0.5
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_auc


def objective(trial: Trial, args, data_dict, edge_index, edge_weight, device):
    """Optuna objective function."""
    
    # Hyperparameters to search
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    num_gru_layers = trial.suggest_int('num_gru_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    conv_type = trial.suggest_categorical('conv_type', ['sage', 'gat', 'gcn'])
    
    # Only search fusion type if text embeddings are available
    if data_dict['text_dim'] is not None:
        fusion_type = trial.suggest_categorical('fusion_type', ['none', 'gated', 'concat'])
    else:
        fusion_type = 'none'
    
    # Build model
    model = STGNN(
        ehr_input_dim=data_dict['ehr_dim'],
        hidden_dim=hidden_dim,
        num_gru_layers=num_gru_layers,
        num_classes=1,
        conv_type=conv_type,
        dropout=dropout,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=4,
        text_dim=data_dict['text_dim'] if fusion_type != 'none' else None,
        fusion_type=fusion_type,
        use_checkpointing=True,
    )
    model = model.to(device)
    
    # Train and evaluate
    try:
        val_auc = train_and_evaluate(
            model, data_dict, edge_index, edge_weight,
            data_dict['train_idx'], data_dict['val_idx'],
            device, args, lr=lr, epochs=args.epochs, patience=args.patience
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return 0.5  # Return bad score for OOM
        raise e
    
    return val_auc


def evaluate_test(model, data_dict, edge_index, edge_weight, device, use_amp=True):
    """Evaluate model on test set."""
    model.eval()
    
    ehr_seq = torch.tensor(data_dict['ehr_sequences'], dtype=torch.float32, device=device)
    lengths = torch.tensor(data_dict['lengths'], dtype=torch.long, device=device)
    labels = torch.tensor(data_dict['labels'], dtype=torch.float32, device=device)
    
    if data_dict['text_embeddings'] is not None:
        text_emb = torch.tensor(data_dict['text_embeddings'], dtype=torch.float32, device=device)
    else:
        text_emb = None
    
    test_idx = data_dict['test_idx']
    
    with torch.no_grad():
        with autocast(enabled=use_amp):
            if text_emb is not None:
                logits, _ = model(ehr_seq, edge_index, edge_weight, text_emb, lengths)
            else:
                logits, _ = model(ehr_seq, edge_index, edge_weight, lengths=lengths)
        
        probs = torch.sigmoid(logits[test_idx]).cpu().numpy()
        y_true = labels[test_idx].cpu().numpy()
        
        # Find optimal threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.01):
            y_pred = (probs >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        y_pred = (probs >= best_thresh).astype(int)
        
        try:
            auc = roc_auc_score(y_true, probs)
        except:
            auc = 0.5
        
        try:
            auprc = average_precision_score(y_true, probs)
        except:
            auprc = 0.0
        
        return {
            'test_auc': float(auc),
            'test_auprc': float(auprc),
            'test_f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'test_precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'test_recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'test_acc': float((y_pred == y_true).mean()),
            'optimal_threshold': float(best_thresh),
        }


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    data_dict = load_data(args)
    
    # Load/build graph
    edge_index, edge_weight = load_or_build_graph(args, data_dict, device)
    
    # MLflow
    mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")
    mlflow.set_experiment("stgnn_hpo")
    
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            'n_trials': args.n_trials,
            'epochs': args.epochs,
            'patience': args.patience,
            'max_seq_len': args.max_seq_len,
            'ehr_dim': data_dict['ehr_dim'],
            'text_dim': data_dict['text_dim'],
        })
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize', study_name='stgnn_hpo')
        
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, args, data_dict, edge_index, edge_weight, device),
            n_trials=args.n_trials,
            show_progress_bar=True
        )
        
        # Log best trial
        best_trial = study.best_trial
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best val AUC: {best_trial.value:.4f}")
        logger.info(f"Best params: {best_trial.params}")
        
        mlflow.log_metric('best_val_auc', best_trial.value)
        mlflow.log_params({f'best_{k}': v for k, v in best_trial.params.items()})
        
        # Train final model with best params and evaluate on test
        logger.info("Training final model with best parameters...")
        best_params = best_trial.params
        
        fusion_type = best_params.get('fusion_type', 'none')
        
        model = STGNN(
            ehr_input_dim=data_dict['ehr_dim'],
            hidden_dim=best_params['hidden_dim'],
            num_gru_layers=best_params['num_gru_layers'],
            num_classes=1,
            conv_type=best_params['conv_type'],
            dropout=best_params['dropout'],
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=4,
            text_dim=data_dict['text_dim'] if fusion_type != 'none' else None,
            fusion_type=fusion_type,
            use_checkpointing=True,
        )
        model = model.to(device)
        
        # Train on train+val, evaluate on test
        combined_train_idx = data_dict['train_idx'] + data_dict['val_idx']
        
        train_and_evaluate(
            model, data_dict, edge_index, edge_weight,
            combined_train_idx, data_dict['test_idx'],
            device, args, lr=best_params['lr'], epochs=args.epochs, patience=args.patience
        )
        
        # Evaluate on test
        test_results = evaluate_test(model, data_dict, edge_index, edge_weight, device)
        
        logger.info("=" * 60)
        logger.info("Test Results (Best Model):")
        for k, v in test_results.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)
        
        mlflow.log_metrics(test_results)
        
        # Save best model
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_params': best_params,
            'test_results': test_results,
        }, os.path.join(args.save_dir, 'stgnn_hpo_best.pt'))
        
        mlflow.log_artifact(os.path.join(args.save_dir, 'stgnn_hpo_best.pt'))
        
        return test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STGNN HPO with Optuna')
    
    # Data paths
    parser.add_argument('--ehr_seq_path', type=str, required=True)
    parser.add_argument('--cohort_path', type=str, required=True)
    parser.add_argument('--graph_path', type=str, default=None)
    parser.add_argument('--text_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='data/interim/ehr_long_los/models_stgnn')
    
    # HPO settings
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_seq_len', type=int, default=30)
    parser.add_argument('--k_neighbors', type=int, default=15)
    
    # MLflow
    parser.add_argument('--run_name', type=str, default='stgnn_hpo')
    
    args = parser.parse_args()
    main(args)
