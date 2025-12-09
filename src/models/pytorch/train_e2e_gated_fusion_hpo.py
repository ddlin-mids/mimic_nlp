import os
import logging
import argparse
import pickle
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
import math

# Reuse classes from main script (assuming they are importable or copy-pasted)
# To avoid import issues, I'll copy the classes here. 
# Ideally we'd move them to a shared module (src.models.pytorch.modules), but for now copy-paste ensures standalone execution.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# [COPY OF CLASSES START]
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

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
    def __init__(self, ehr_config, text_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.ehr_transformer = EHRTransformer(
            input_dims=ehr_config['input_dims'],
            embed_dims=ehr_config['embed_dims'],
            cat_idxs=ehr_config['cat_idxs'],
            num_idxs=ehr_config['num_idxs'],
            hidden_dim=hidden_dim, 
            num_layers=ehr_config['num_layers'],
            num_heads=ehr_config['num_heads'],
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
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, ehr_seq, ehr_lens, text_emb):
        h_ehr = self.ehr_transformer(ehr_seq, ehr_lens)
        h_text = self.text_proj(text_emb)
        fused, gate_vals = self.gated_fusion(h_ehr, h_text)
        logits = self.classifier(fused)
        return logits, gate_vals

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
                
    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        hid = int(key.split('_')[1])
        ehr_seq = self.ehr_feat_dict[key]
        if len(ehr_seq) > self.max_len: ehr_seq = ehr_seq[-self.max_len:]
        text_emb = self.text_embeddings_dict.get(hid, np.zeros(64)) 
        label = self.labels_map[key]
        return torch.tensor(ehr_seq, dtype=torch.float32), torch.tensor(text_emb, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    ehr_seqs, text_embs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in ehr_seqs])
    ehr_padded = pad_sequence(ehr_seqs, batch_first=True, padding_value=0)
    text_stacked = torch.stack(text_embs)
    labels_stacked = torch.stack(labels)
    return ehr_padded, lengths, text_stacked, labels_stacked

def load_data_e2e(args):
    # (Simplified loader that doesn't re-PCA every time, assumes PCA is done or passed)
    # Actually for HPO we want to do PCA once.
    # We'll use the logic from train_e2e_gated_fusion.py but assuming we load raw data once globally or passed in.
    # For standalone script execution, we reload.
    
    cohort = pd.read_csv(args.cohort_path)
    cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
    cohort['node_name'] = cohort['subject_id'].astype(str) + "_" + cohort['hadm_id'].astype(str)
    labels_map = dict(zip(cohort['node_name'], cohort['readmitted_within_window'].astype(int)))
    split_map = dict(zip(cohort['node_name'], cohort['split']))
    
    with open(args.ehr_seq_path, "rb") as f: seq_data = pickle.load(f)
    ehr_feat_dict = seq_data["feat_dict"]
    
    if "cat_idxs" in seq_data:
        cat_idxs = seq_data["cat_idxs"]
        cat_dims = seq_data["cat_dims"]
    else:
        cat_idxs = []
        cat_dims = []
        
    sample_key = next(iter(ehr_feat_dict))
    total_cols = ehr_feat_dict[sample_key].shape[1]
    all_idxs = set(range(total_cols))
    num_idxs = list(sorted(all_idxs - set(cat_idxs)))
    
    def load_emb(name):
        path = Path(args.text_dir) / f"{name}.npz"
        if not path.exists(): return {}
        data = np.load(path)
        ids = data['hadm_ids'] if 'hadm_ids' in data else data['ids']
        embs = data['embeddings']
        emb_map = {}
        from collections import defaultdict
        grouped = defaultdict(list)
        for i, e in zip(ids, embs): grouped[int(i)].append(e)
        for i, elist in grouped.items(): emb_map[i] = np.mean(elist, axis=0)
        return emb_map

    disch_map = load_emb("discharge_summary")
    rad_map = load_emb("radiology_report")
    d_dim = list(disch_map.values())[0].shape[0] if disch_map else 768
    r_dim = list(rad_map.values())[0].shape[0] if rad_map else 768
    
    all_hids = set(cohort['hadm_id'])
    text_emb_dict = {}
    all_text_vectors = []
    pca_train_mask = []
    sorted_cohort = cohort.sort_values('hadm_id')
    
    for _, row in sorted_cohort.iterrows():
        hid = row['hadm_id']
        d_vec = disch_map.get(hid, np.zeros(d_dim))
        r_vec = rad_map.get(hid, np.zeros(r_dim))
        combined = np.concatenate([d_vec, r_vec])
        text_emb_dict[hid] = combined
        all_text_vectors.append(combined)
        pca_train_mask.append(row['split'] == 'train')
        
    all_text_vectors = np.array(all_text_vectors)
    pca_train_mask = np.array(pca_train_mask)
    
    if args.pca_components > 0:
        pca = PCA(n_components=args.pca_components)
        pca.fit(all_text_vectors[pca_train_mask])
        transformed = pca.transform(all_text_vectors)
        for i, hid in enumerate(sorted_cohort['hadm_id']): text_emb_dict[hid] = transformed[i]
            
    text_dim = args.pca_components if args.pca_components > 0 else (d_dim + r_dim)
    
    ehr_config = {
        'input_dims': [d + 1 for d in cat_dims],
        'embed_dims': [min(50, (d + 1) // 2) for d in cat_dims],
        'cat_idxs': cat_idxs,
        'num_idxs': num_idxs,
        'num_layers': 2,
        'num_heads': 4
    }
    
    train_ids = set(cohort[cohort['split'] == 'train']['hadm_id'])
    val_ids = set(cohort[cohort['split'].isin(['val', 'validation'])]['hadm_id'])
    test_ids = set(cohort[cohort['split'] == 'test']['hadm_id'])
    
    return ehr_feat_dict, text_emb_dict, labels_map, split_map, ehr_config, text_dim, train_ids, val_ids, test_ids

# -----------------------------------------------------------------------------
# Optuna Objective
# -----------------------------------------------------------------------------
def objective(trial, args, data_bundle):
    ehr_feat, text_dict, labels_map, split_map, ehr_config, text_dim, train_ids, val_ids, test_ids = data_bundle
    
    # HPO Params
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    encoder_lr = trial.suggest_float("encoder_lr", 1e-6, 1e-4, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    batch_size = 32
    epochs = 20 # Faster for HPO
    accumulation_steps = 4
    
    # Setup Datasets
    train_ds = E2EFusionDataset(ehr_feat, text_dict, labels_map, train_ids)
    val_ds = E2EFusionDataset(ehr_feat, text_dict, labels_map, val_ids)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EndToEndGatedFusion(
        ehr_config=ehr_config,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    
    # Load pretrained encoder if provided (path passed in args)
    if args.pretrained_encoder_path and os.path.exists(args.pretrained_encoder_path):
        checkpoint = torch.load(args.pretrained_encoder_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        try:
            model.ehr_transformer.load_state_dict(state_dict, strict=False)
        except: pass
        
    optimizer = torch.optim.AdamW([
        {'params': model.ehr_transformer.parameters(), 'lr': encoder_lr},
        {'params': model.text_proj.parameters(), 'lr': lr},
        {'params': model.gated_fusion.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=0.01)
    
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader)*epochs*0.1), num_training_steps=len(train_loader)*epochs)
    
    best_val_auc = 0.0
    patience = 5
    no_improve = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        for i, (ehr_seq, lengths, text_emb, y) in enumerate(train_loader):
            ehr_seq, lengths, text_emb, y = ehr_seq.to(device), lengths.to(device), text_emb.to(device), y.to(device).unsqueeze(1)
            logits, _ = model(ehr_seq, lengths, text_emb)
            loss = criterion(logits, y) / accumulation_steps
            loss.backward()
            if (i+1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                
        # Val
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for ehr_seq, lengths, text_emb, y in val_loader:
                ehr_seq, lengths, text_emb, y = ehr_seq.to(device), lengths.to(device), text_emb.to(device), y.to(device).unsqueeze(1)
                logits, _ = model(ehr_seq, lengths, text_emb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(y.cpu().numpy())
        
        val_auc = roc_auc_score(all_targets, all_preds)
        trial.report(val_auc, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience: break
        
    return best_val_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ehr_seq_path", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--pretrained_encoder_path", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--pca_components", type=int, default=64)
    args = parser.parse_args()
    
    data_bundle = load_data_e2e(args)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, args, data_bundle), n_trials=args.n_trials)
    
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
    
    # Log to MLflow
    mlflow.set_experiment("mimic_cardiorenal_e2e_gated_hpo")
    with mlflow.start_run():
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_auc", study.best_value)

if __name__ == "__main__":
    main()
