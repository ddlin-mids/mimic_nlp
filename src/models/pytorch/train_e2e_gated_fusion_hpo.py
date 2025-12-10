"""
End-to-End Gated Fusion HPO with Optuna
=======================================
Intensive hyperparameter optimization for the E2E Gated Fusion model.

Search space:
- hidden_dim: [64, 128, 256]
- dropout: [0.1, 0.5]
- lr: [5e-5, 5e-3]
- encoder_lr: [1e-6, 1e-4]
- num_layers: [1, 3]
- num_heads: [2, 4, 8]
- pca_components: [32, 64, 128]
- warmup_ratio: [0.05, 0.2]
- batch_size: [16, 32, 64]
- weight_decay: [1e-4, 1e-1]
"""

import os
import logging
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_recall_curve, precision_score, recall_score
from transformers import get_cosine_schedule_with_warmup
import mlflow
import mlflow.pytorch
import optuna
from optuna.trial import TrialState
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

# -----------------------------------------------------------------------------
# Model Classes
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(1), :]

class EHRTransformer(nn.Module):
    def __init__(self, input_dims, embed_dims, cat_idxs, num_idxs, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.cat_idxs = torch.tensor(cat_idxs, dtype=torch.long)
        self.num_idxs = torch.tensor(num_idxs, dtype=torch.long)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num, embedding_dim=dim, padding_idx=0)
            for num, dim in zip(input_dims, embed_dims)
        ])
        cat_embed_dim_total = sum(embed_dims)
        num_input_dim = len(num_idxs)
        if num_input_dim > 0:
            self.num_proj_dim = max(16, hidden_dim // 4)
            self.num_proj = nn.Linear(num_input_dim, self.num_proj_dim)
        else:
            self.num_proj_dim = 0
            self.num_proj = None
        total_input_dim = cat_embed_dim_total + self.num_proj_dim
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=500)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, lengths):
        batch_size, seq_len, _ = x.shape
        device = x.device
        parts = []
        if len(self.cat_idxs) > 0:
            x_cat = x[:, :, self.cat_idxs.to(device)].long()
            cat_embedded = []
            for i, emb_layer in enumerate(self.embeddings):
                feat_idx = x_cat[:, :, i].clamp(0, emb_layer.num_embeddings - 1)
                cat_embedded.append(emb_layer(feat_idx))
            parts.append(torch.cat(cat_embedded, dim=-1))
        if len(self.num_idxs) > 0 and self.num_proj is not None:
            x_num = x[:, :, self.num_idxs.to(device)]
            x_num_proj = self.num_proj(x_num)
            parts.append(x_num_proj)
        x_combined = torch.cat(parts, dim=-1)
        x_emb = self.input_proj(x_combined)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        x_emb = self.pos_encoder(x_emb)
        max_len_plus_cls = x_emb.size(1)
        mask = torch.arange(max_len_plus_cls, device=device).expand(batch_size, max_len_plus_cls)
        mask = mask >= (lengths.unsqueeze(1) + 1).to(device)
        output = self.transformer_encoder(x_emb, src_key_padding_mask=mask)
        return output[:, 0, :]

class GatedMultimodalUnit(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.linear_z = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_a, x_b):
        combined = torch.cat([x_a, x_b], dim=1)
        z = self.sigmoid(self.linear_z(combined))
        h = z * x_a + (1 - z) * x_b
        return self.dropout(h), z

class EndToEndGatedFusion(nn.Module):
    def __init__(self, ehr_config, text_dim, hidden_dim=64, dropout=0.3, num_layers=2, num_heads=4):
        super().__init__()
        # Create modifiable config
        config = ehr_config.copy()
        config['num_layers'] = num_layers
        config['num_heads'] = num_heads
        
        self.ehr_transformer = EHRTransformer(
            input_dims=config['input_dims'],
            embed_dims=config['embed_dims'],
            cat_idxs=config['cat_idxs'],
            num_idxs=config['num_idxs'],
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gated_fusion = GatedMultimodalUnit(hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, ehr_seq, ehr_lens, text_emb):
        h_ehr = self.ehr_transformer(ehr_seq, ehr_lens)
        h_text = self.text_proj(text_emb)
        fused, gate_vals = self.gated_fusion(h_ehr, h_text)
        logits = self.classifier(fused)
        return logits, gate_vals

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class E2EFusionDataset(Dataset):
    def __init__(self, ehr_feat_dict, text_embeddings_dict, labels_map, hadm_ids, max_len=100):
        self.ehr_feat_dict = ehr_feat_dict
        self.text_embeddings_dict = text_embeddings_dict
        self.labels_map = labels_map
        self.keys = []
        for k in labels_map.keys():
            try:
                curr_hid = int(k.split('_')[1])
                if curr_hid in hadm_ids and k in ehr_feat_dict:
                    self.keys.append(k)
            except: pass
        self.max_len = max_len
        self.default_text_dim = 64
        if text_embeddings_dict:
            sample = next(iter(text_embeddings_dict.values()))
            self.default_text_dim = len(sample) if hasattr(sample, '__len__') else 64
                
    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        hid = int(key.split('_')[1])
        ehr_seq = self.ehr_feat_dict[key]
        if len(ehr_seq) > self.max_len: ehr_seq = ehr_seq[-self.max_len:]
        # Handle object dtype arrays
        ehr_seq = np.asarray(ehr_seq, dtype=np.float32)
        text_emb = self.text_embeddings_dict.get(hid, np.zeros(self.default_text_dim, dtype=np.float32))
        text_emb = np.asarray(text_emb, dtype=np.float32)
        label = self.labels_map[key]
        return torch.from_numpy(ehr_seq), torch.from_numpy(text_emb), torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    ehr_seqs, text_embs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in ehr_seqs])
    ehr_padded = pad_sequence(ehr_seqs, batch_first=True, padding_value=0)
    text_stacked = torch.stack(text_embs)
    labels_stacked = torch.stack(labels)
    return ehr_padded, lengths, text_stacked, labels_stacked

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_raw_data(args):
    """Load raw data without PCA transformation."""
    logger.info("Loading cohort...")
    cohort = pd.read_csv(args.cohort_path)
    cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
    cohort['node_name'] = cohort['subject_id'].astype(str) + "_" + cohort['hadm_id'].astype(str)
    labels_map = dict(zip(cohort['node_name'], cohort['readmitted_within_window'].astype(int)))
    
    logger.info(f"Loading EHR sequences from {args.ehr_seq_path}...")
    with open(args.ehr_seq_path, "rb") as f: 
        seq_data = pickle.load(f)
    ehr_feat_dict = seq_data["feat_dict"]
    
    cat_idxs = seq_data.get("cat_idxs", [])
    cat_dims = seq_data.get("cat_dims", [])
    sample_key = next(iter(ehr_feat_dict))
    total_cols = ehr_feat_dict[sample_key].shape[1]
    num_idxs = list(sorted(set(range(total_cols)) - set(cat_idxs)))
    
    logger.info(f"Loading text embeddings from {args.text_dir}...")
    def load_emb(name):
        path = Path(args.text_dir) / f"{name}.npz"
        if not path.exists(): return {}
        data = np.load(path)
        ids = data['hadm_ids'] if 'hadm_ids' in data else data['ids']
        embs = data['embeddings']
        from collections import defaultdict
        grouped = defaultdict(list)
        for i, e in zip(ids, embs): grouped[int(i)].append(e)
        return {i: np.mean(elist, axis=0) for i, elist in grouped.items()}

    disch_map = load_emb("discharge_summary")
    rad_map = load_emb("radiology_report")
    d_dim = list(disch_map.values())[0].shape[0] if disch_map else 768
    r_dim = list(rad_map.values())[0].shape[0] if rad_map else 768
    
    # Combine embeddings
    sorted_cohort = cohort.sort_values('hadm_id')
    all_text_vectors = []
    for _, row in sorted_cohort.iterrows():
        hid = row['hadm_id']
        d_vec = disch_map.get(hid, np.zeros(d_dim))
        r_vec = rad_map.get(hid, np.zeros(r_dim))
        all_text_vectors.append(np.concatenate([d_vec, r_vec]))
    all_text_vectors = np.array(all_text_vectors)
    
    train_ids = set(cohort[cohort['split'] == 'train']['hadm_id'])
    val_ids = set(cohort[cohort['split'].isin(['val', 'validation'])]['hadm_id'])
    test_ids = set(cohort[cohort['split'] == 'test']['hadm_id'])
    
    # Mask for PCA fitting
    pca_train_mask = np.array([row['split'] == 'train' for _, row in sorted_cohort.iterrows()])
    
    ehr_config = {
        'input_dims': [d + 1 for d in cat_dims],
        'embed_dims': [min(50, (d + 1) // 2) for d in cat_dims],
        'cat_idxs': cat_idxs,
        'num_idxs': num_idxs,
        'num_layers': 2,
        'num_heads': 4
    }
    
    return {
        'ehr_feat_dict': ehr_feat_dict,
        'all_text_vectors': all_text_vectors,
        'pca_train_mask': pca_train_mask,
        'sorted_hadm_ids': sorted_cohort['hadm_id'].tolist(),
        'labels_map': labels_map,
        'ehr_config': ehr_config,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'raw_text_dim': all_text_vectors.shape[1]
    }

def apply_pca(raw_data, pca_components):
    """Apply PCA transformation to text embeddings."""
    all_text = raw_data['all_text_vectors']
    train_mask = raw_data['pca_train_mask']
    hadm_ids = raw_data['sorted_hadm_ids']
    
    if pca_components > 0 and pca_components < all_text.shape[1]:
        pca = PCA(n_components=pca_components)
        pca.fit(all_text[train_mask])
        transformed = pca.transform(all_text)
        text_dim = pca_components
    else:
        transformed = all_text
        text_dim = all_text.shape[1]
    
    text_emb_dict = {hid: transformed[i] for i, hid in enumerate(hadm_ids)}
    return text_emb_dict, text_dim

# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scheduler, device, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, (ehr_seq, lengths, text_emb, y) in enumerate(loader):
        ehr_seq = ehr_seq.to(device)
        lengths = lengths.to(device)
        text_emb = text_emb.to(device)
        y = y.to(device).unsqueeze(1)
        
        logits, _ = model(ehr_seq, lengths, text_emb)
        loss = criterion(logits, y) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_gates = [], [], []
    
    for ehr_seq, lengths, text_emb, y in loader:
        ehr_seq = ehr_seq.to(device)
        lengths = lengths.to(device)
        text_emb = text_emb.to(device)
        y = y.to(device).unsqueeze(1)
        
        logits, gates = model(ehr_seq, lengths, text_emb)
        loss = criterion(logits, y)
        
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        all_gates.extend(gates.cpu().numpy())
        
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    all_gates = np.array(all_gates).flatten()
    
    auc = roc_auc_score(all_targets, all_preds)
    auprc = average_precision_score(all_targets, all_preds)
    preds_binary = (all_preds > threshold).astype(int)
    f1 = f1_score(all_targets, preds_binary, zero_division=0)
    prec = precision_score(all_targets, preds_binary, zero_division=0)
    rec = recall_score(all_targets, preds_binary, zero_division=0)
    acc = accuracy_score(all_targets, preds_binary)
    
    return {
        'loss': total_loss / len(loader),
        'auc': auc,
        'auprc': auprc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'accuracy': acc,
        'preds': all_preds,
        'targets': all_targets,
        'gates': all_gates
    }

# -----------------------------------------------------------------------------
# Optuna Objective
# -----------------------------------------------------------------------------
def create_objective(args, raw_data, device):
    """Create an Optuna objective function."""
    pca_cache = {}
    
    def objective(trial):
        # HPO Parameters
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 5e-5, 5e-3, log=True)
        encoder_lr = trial.suggest_float('encoder_lr', 1e-6, 1e-4, log=True)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        pca_components = trial.suggest_categorical('pca_components', [32, 64, 128])
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        
        # Ensure hidden_dim divisible by num_heads
        if hidden_dim % num_heads != 0:
            return 0.0
        
        # Apply PCA (with caching)
        if pca_components not in pca_cache:
            logger.info(f"Applying PCA with n_components={pca_components}")
            pca_cache[pca_components] = apply_pca(raw_data, pca_components)
        text_emb_dict, text_dim = pca_cache[pca_components]
        
        # Create datasets
        train_ds = E2EFusionDataset(raw_data['ehr_feat_dict'], text_emb_dict, raw_data['labels_map'], raw_data['train_ids'])
        val_ds = E2EFusionDataset(raw_data['ehr_feat_dict'], text_emb_dict, raw_data['labels_map'], raw_data['val_ids'])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        
        # Model
        model = EndToEndGatedFusion(
            ehr_config=raw_data['ehr_config'],
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads
        ).to(device)
        
        # Optimizer
        optimizer = torch.optim.AdamW([
            {'params': model.ehr_transformer.parameters(), 'lr': encoder_lr},
            {'params': model.text_proj.parameters(), 'lr': lr},
            {'params': model.gated_fusion.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ], weight_decay=weight_decay)
        
        criterion = nn.BCEWithLogitsLoss()
        
        num_epochs = args.epochs_per_trial
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        best_val_auc = 0.0
        patience = 5
        no_improve = 0
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, args.accumulation_steps)
            val_results = evaluate(model, val_loader, criterion, device)
            val_auc = val_results['auc']
            
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                break
                
            logger.info(f"Trial {trial.number} Epoch {epoch+1}: Val AUC={val_auc:.4f}, Best={best_val_auc:.4f}")
        
        return best_val_auc
    
    return objective

# -----------------------------------------------------------------------------
# Final Model Training
# -----------------------------------------------------------------------------
def train_final_model(args, raw_data, best_params, device):
    """Train final model with best parameters and evaluate on test set."""
    logger.info(f"Training final model with best params: {best_params}")
    
    # Apply PCA
    text_emb_dict, text_dim = apply_pca(raw_data, best_params['pca_components'])
    
    # Create datasets
    train_ds = E2EFusionDataset(raw_data['ehr_feat_dict'], text_emb_dict, raw_data['labels_map'], raw_data['train_ids'])
    val_ds = E2EFusionDataset(raw_data['ehr_feat_dict'], text_emb_dict, raw_data['labels_map'], raw_data['val_ids'])
    test_ds = E2EFusionDataset(raw_data['ehr_feat_dict'], text_emb_dict, raw_data['labels_map'], raw_data['test_ids'])
    
    batch_size = best_params['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Model
    model = EndToEndGatedFusion(
        ehr_config=raw_data['ehr_config'],
        text_dim=text_dim,
        hidden_dim=best_params['hidden_dim'],
        dropout=best_params['dropout'],
        num_layers=best_params['num_layers'],
        num_heads=best_params['num_heads']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.ehr_transformer.parameters(), 'lr': best_params['encoder_lr']},
        {'params': model.text_proj.parameters(), 'lr': best_params['lr']},
        {'params': model.gated_fusion.parameters(), 'lr': best_params['lr']},
        {'params': model.classifier.parameters(), 'lr': best_params['lr']}
    ], weight_decay=best_params['weight_decay'])
    
    criterion = nn.BCEWithLogitsLoss()
    
    num_epochs = args.final_epochs
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * best_params['warmup_ratio'])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    best_val_auc = 0.0
    best_state = None
    best_threshold = 0.5
    patience = 10
    no_improve = 0
    
    with mlflow.start_run(run_name="e2e_gated_hpo_final"):
        mlflow.log_params(best_params)
        mlflow.set_tag("model_type", "e2e_gated_fusion_hpo_final")
        
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, args.accumulation_steps)
            val_results = evaluate(model, val_loader, criterion, device)
            
            logger.info(f"Final Epoch {epoch+1}: Loss={train_loss:.4f}, Val AUC={val_results['auc']:.4f}")
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_results['loss'],
                "val_auc": val_results['auc'],
                "val_auprc": val_results['auprc'],
                "mean_gate_value": float(np.mean(val_results['gates']))
            }, step=epoch)
            
            if val_results['auc'] > best_val_auc:
                best_val_auc = val_results['auc']
                best_state = model.state_dict()
                best_threshold, _ = find_optimal_threshold(val_results['targets'], val_results['preds'])
                no_improve = 0
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(best_state, os.path.join(args.save_dir, "e2e_gated_hpo_best.pt"))
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.info("Early stopping")
                break
        
        # Final Test
        if best_state:
            model.load_state_dict(best_state)
        
        test_results = evaluate(model, test_loader, criterion, device, threshold=best_threshold)
        
        logger.info(f"=== FINAL TEST RESULTS ===")
        logger.info(f"Test AUC: {test_results['auc']:.4f}")
        logger.info(f"Test AUPRC: {test_results['auprc']:.4f}")
        logger.info(f"Test F1: {test_results['f1']:.4f}")
        logger.info(f"Test Precision: {test_results['precision']:.4f}")
        logger.info(f"Test Recall: {test_results['recall']:.4f}")
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Optimal Threshold: {best_threshold:.4f}")
        logger.info(f"Mean Gate Value: {np.mean(test_results['gates']):.3f}")
        
        mlflow.log_metrics({
            "test_auc": test_results['auc'],
            "test_auprc": test_results['auprc'],
            "test_f1": test_results['f1'],
            "test_precision": test_results['precision'],
            "test_recall": test_results['recall'],
            "test_acc": test_results['accuracy'],
            "optimal_threshold": best_threshold,
            "test_mean_gate_value": float(np.mean(test_results['gates']))
        })
        
        return test_results

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ehr_seq_path", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--epochs_per_trial", type=int, default=15, help="Epochs per HPO trial")
    parser.add_argument("--final_epochs", type=int, default=40, help="Epochs for final model")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="data/interim/ehr_long_los/models")
    parser.add_argument("--study_name", type=str, default="e2e_gated_fusion_hpo")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    args = parser.parse_args()
    
    # Setup
    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]
    if mlflow.active_run():
        mlflow.end_run()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    mlflow.set_experiment("mimic_cardiorenal_e2e_gated_hpo")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load raw data once
    raw_data = load_raw_data(args)
    
    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Run optimization
    objective = create_objective(args, raw_data, device)
    
    with mlflow.start_run(run_name="hpo_search"):
        mlflow.log_params({"n_trials": args.n_trials, "epochs_per_trial": args.epochs_per_trial})
        
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        
        logger.info(f"=== HPO COMPLETE ===")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value (Val AUC): {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        mlflow.log_metric("best_val_auc", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        
        # Save study results
        os.makedirs(args.save_dir, exist_ok=True)
        results_path = os.path.join(args.save_dir, "hpo_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': study.best_params,
                'best_value': study.best_value,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials)
            }, f, indent=2)
        mlflow.log_artifact(results_path)
    
    # Train final model
    logger.info("=== Training Final Model with Best Params ===")
    test_results = train_final_model(args, raw_data, study.best_params, device)
    
    # Save final results
    final_path = os.path.join(args.save_dir, "final_test_results.json")
    with open(final_path, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'test_results': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                          for k, v in test_results.items() if k not in ['preds', 'targets', 'gates']}
        }, f, indent=2)
    
    logger.info(f"Results saved to {final_path}")

if __name__ == "__main__":
    main()
