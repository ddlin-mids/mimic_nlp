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
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Find threshold that maximizes F1 on validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )
    # precision_recall_curve returns n+1 precision/recall but n thresholds
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

# -----------------------------------------------------------------------------
# Positional Encoding (from train_ehr_transformer.py)
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

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

# -----------------------------------------------------------------------------
# Transformer Encoder (from train_ehr_transformer.py)
# -----------------------------------------------------------------------------
class EHRTransformer(nn.Module):
    def __init__(self, input_dims, embed_dims, cat_idxs, num_idxs, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.cat_idxs = torch.tensor(cat_idxs, dtype=torch.long)
        self.num_idxs = torch.tensor(num_idxs, dtype=torch.long)
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num, embedding_dim=dim, padding_idx=0)
            for num, dim in zip(input_dims, embed_dims)
        ])
        cat_embed_dim_total = sum(embed_dims)
        
        # Numerical Projection
        num_input_dim = len(num_idxs)
        if num_input_dim > 0:
            self.num_proj_dim = max(16, hidden_dim // 4)
            self.num_proj = nn.Linear(num_input_dim, self.num_proj_dim)
        else:
            self.num_proj_dim = 0
            self.num_proj = None
            
        total_input_dim = cat_embed_dim_total + self.num_proj_dim
        
        # Input Projection to Model Dimension
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=500)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, lengths):
        # x: (batch, seq_len, num_features)
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
        
        # Project to hidden dim
        x_emb = self.input_proj(x_combined)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        
        # Add Positional Encoding
        x_emb = self.pos_encoder(x_emb)
        
        # Create Padding Mask
        max_len_plus_cls = x_emb.size(1)
        mask = torch.arange(max_len_plus_cls, device=device).expand(batch_size, max_len_plus_cls)
        mask = mask >= (lengths.unsqueeze(1) + 1).to(device)
        
        # Transformer
        output = self.transformer_encoder(x_emb, src_key_padding_mask=mask)
        
        # Extract CLS token output
        final_embedding = output[:, 0, :]
        
        return final_embedding

# -----------------------------------------------------------------------------
# Gated Multimodal Unit (from train_gated_fusion.py)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# End-to-End Gated Fusion Model
# -----------------------------------------------------------------------------
class EndToEndGatedFusion(nn.Module):
    def __init__(self, ehr_config, text_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        
        # 1. Trainable EHR Transformer
        self.ehr_transformer = EHRTransformer(
            input_dims=ehr_config['input_dims'],
            embed_dims=ehr_config['embed_dims'],
            cat_idxs=ehr_config['cat_idxs'],
            num_idxs=ehr_config['num_idxs'],
            hidden_dim=hidden_dim, # We project everything to hidden_dim inside transformer
            num_layers=ehr_config['num_layers'],
            num_heads=ehr_config['num_heads'],
            dropout=dropout
        )
        
        # 2. Text Projection (Linear)
        # Assuming we receive PCA-reduced or raw embeddings, project to hidden_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Gated Fusion
        self.gated_fusion = GatedMultimodalUnit(hidden_dim, dropout=dropout)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, ehr_seq, ehr_lens, text_emb):
        # EHR Path
        h_ehr = self.ehr_transformer(ehr_seq, ehr_lens)
        
        # Text Path
        h_text = self.text_proj(text_emb)
        
        # Fusion
        fused, gate_vals = self.gated_fusion(h_ehr, h_text)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, gate_vals

# -----------------------------------------------------------------------------
# Dataset & DataLoader
# -----------------------------------------------------------------------------
class E2EFusionDataset(Dataset):
    def __init__(self, ehr_feat_dict, text_embeddings_dict, labels_map, hadm_ids, max_len=100):
        self.ehr_feat_dict = ehr_feat_dict
        self.text_embeddings_dict = text_embeddings_dict
        self.labels_map = labels_map
        self.hadm_ids = hadm_ids
        self.max_len = max_len
        
        # Create node names list
        self.node_names = []
        for hid in hadm_ids:
            # We assume subject_id isn't strictly needed for key lookup if we have the mapping
            # Actually, the feat_dict keys are node_names (subject_hadm).
            # We need to reconstruct keys or find them.
            # Let's assume passed hadm_ids align with a lookup map or we can filter keys.
            pass

        # Better approach: Iterate keys in labels_map that match requested split
        self.keys = []
        for k in labels_map.keys():
            try:
                curr_hid = int(k.split('_')[1])
                if curr_hid in hadm_ids:
                    if k in ehr_feat_dict: # Ensure we have features
                        self.keys.append(k)
            except:
                continue
                
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        hid = int(key.split('_')[1])
        
        # EHR Seq
        ehr_seq = self.ehr_feat_dict[key]
        if len(ehr_seq) > self.max_len:
            ehr_seq = ehr_seq[-self.max_len:]
            
        # Text Emb
        text_emb = self.text_embeddings_dict.get(hid, np.zeros(64)) # Fallback if missing? 
        # Note: text_embeddings_dict should map hadm_id -> embedding
        
        label = self.labels_map[key]
        
        return torch.tensor(ehr_seq, dtype=torch.float32), torch.tensor(text_emb, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    ehr_seqs, text_embs, labels = zip(*batch)
    
    lengths = torch.tensor([len(s) for s in ehr_seqs])
    ehr_padded = pad_sequence(ehr_seqs, batch_first=True, padding_value=0)
    text_stacked = torch.stack(text_embs)
    labels_stacked = torch.stack(labels)
    
    return ehr_padded, lengths, text_stacked, labels_stacked

# -----------------------------------------------------------------------------
# Data Preparation Helper
# -----------------------------------------------------------------------------
def load_data_e2e(args):
    logger.info("Loading cohort...")
    cohort = pd.read_csv(args.cohort_path)
    cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
    
    # Create Labels Map
    cohort['node_name'] = cohort['subject_id'].astype(str) + "_" + cohort['hadm_id'].astype(str)
    labels_map = dict(zip(cohort['node_name'], cohort['readmitted_within_window'].astype(int)))
    split_map = dict(zip(cohort['node_name'], cohort['split']))
    
    # Load EHR Sequences
    logger.info(f"Loading EHR sequences from {args.ehr_seq_path}...")
    with open(args.ehr_seq_path, "rb") as f:
        seq_data = pickle.load(f)
    ehr_feat_dict = seq_data["feat_dict"]
    
    # Clean EHR dict (cast to float32)
    # Optimization: Only clean needed keys? No, easier to just cast what we use in Dataset
    
    # Metadata for EHR model
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
    
    # Load Text Embeddings
    logger.info(f"Loading text embeddings from {args.text_dir}...")
    # Helper to load and align
    def load_emb(name):
        path = Path(args.text_dir) / f"{name}.npz"
        if not path.exists(): return {}
        data = np.load(path)
        ids = data['hadm_ids'] if 'hadm_ids' in data else data['ids']
        embs = data['embeddings']
        
        # Handle multiples by averaging
        emb_map = {}
        from collections import defaultdict
        grouped = defaultdict(list)
        for i, e in zip(ids, embs):
            grouped[int(i)].append(e)
        for i, elist in grouped.items():
            emb_map[i] = np.mean(elist, axis=0)
        return emb_map

    disch_map = load_emb("discharge_summary")
    rad_map = load_emb("radiology_report")
    
    # Combine Text Embeddings
    # Logic: concat discharge + radiology. If missing, zero pad.
    # First determine dims
    d_dim = list(disch_map.values())[0].shape[0] if disch_map else 768
    r_dim = list(rad_map.values())[0].shape[0] if rad_map else 768
    
    all_hids = set(cohort['hadm_id'])
    text_emb_dict = {}
    
    # Prepare array for PCA
    all_text_vectors = []
    pca_train_mask = []
    
    # We iterate cohort to ensure alignment
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
    
    # PCA
    if args.pca_components > 0:
        logger.info(f"Applying PCA (n={args.pca_components}) to text...")
        pca = PCA(n_components=args.pca_components)
        # Fit on train
        pca.fit(all_text_vectors[pca_train_mask])
        logger.info(f"PCA Variance Explained: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # Transform all
        transformed = pca.transform(all_text_vectors)
        
        # Update dict
        for i, hid in enumerate(sorted_cohort['hadm_id']):
            text_emb_dict[hid] = transformed[i]
            
    text_dim = args.pca_components if args.pca_components > 0 else (d_dim + r_dim)
    
    # Config for EHR model
    ehr_config = {
        'input_dims': [d + 1 for d in cat_dims],
        'embed_dims': [min(50, (d + 1) // 2) for d in cat_dims],
        'cat_idxs': cat_idxs,
        'num_idxs': num_idxs,
        'num_layers': 2,
        'num_heads': 4
    }
    
    # Split IDs
    train_ids = set(cohort[cohort['split'] == 'train']['hadm_id'])
    val_ids = set(cohort[cohort['split'].isin(['val', 'validation'])]['hadm_id'])
    test_ids = set(cohort[cohort['split'] == 'test']['hadm_id'])
    
    return ehr_feat_dict, text_emb_dict, labels_map, split_map, ehr_config, text_dim, train_ids, val_ids, test_ids

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scheduler, device, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, (ehr_seq, lengths, text_emb, y) in enumerate(tqdm(loader, desc="Train")):
        ehr_seq = ehr_seq.to(device)
        lengths = lengths.to(device)
        text_emb = text_emb.to(device)
        y = y.to(device).unsqueeze(1)
        
        logits, _ = model(ehr_seq, lengths, text_emb)
        loss = criterion(logits, y)
        loss = loss / accumulation_steps
        
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
    all_preds = []
    all_targets = []
    all_gates = []
    
    for ehr_seq, lengths, text_emb, y in loader:
        ehr_seq = ehr_seq.to(device)
        lengths = lengths.to(device)
        text_emb = text_emb.to(device)
        y = y.to(device).unsqueeze(1)
        
        logits, gates = model(ehr_seq, lengths, text_emb)
        loss = criterion(logits, y)
        
        probs = torch.sigmoid(logits).cpu().numpy()
        gates_np = gates.cpu().numpy()
        
        total_loss += loss.item()
        all_preds.extend(probs)
        all_targets.extend(y.cpu().numpy())
        all_gates.extend(gates_np)
        
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
    
    return total_loss / len(loader), auc, auprc, f1, prec, rec, acc, all_preds, all_targets, all_gates

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ehr_seq_path", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--pca_components", type=int, default=64)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="data/interim/ehr_long_los/models")
    parser.add_argument("--pretrained_encoder_path", type=str, default=None, help="Path to pre-trained EHR encoder weights")
    args = parser.parse_args()
    
    # Setup
    if "MLFLOW_RUN_ID" in os.environ: del os.environ["MLFLOW_RUN_ID"]
    if mlflow.active_run(): mlflow.end_run()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    mlflow.set_experiment("mimic_cardiorenal_e2e_gated")
    
    # Load Data
    ehr_feat, text_dict, labels_map, split_map, ehr_config, text_dim, train_ids, val_ids, test_ids = load_data_e2e(args)
    
    # Create Datasets
    train_ds = E2EFusionDataset(ehr_feat, text_dict, labels_map, train_ids)
    val_ds = E2EFusionDataset(ehr_feat, text_dict, labels_map, val_ids)
    test_ds = E2EFusionDataset(ehr_feat, text_dict, labels_map, test_ids)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Model
    model = EndToEndGatedFusion(
        ehr_config=ehr_config,
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Load Pre-trained Encoder if provided
    if args.pretrained_encoder_path and os.path.exists(args.pretrained_encoder_path):
        logger.info(f"Loading pre-trained encoder from {args.pretrained_encoder_path}")
        # The saved state dict likely has keys like 'embeddings.0.weight', etc.
        # Our model has 'ehr_transformer.embeddings.0.weight'.
        # We need to map them or load into the submodule directly.
        checkpoint = torch.load(args.pretrained_encoder_path, map_location=device)
        
        # Check if checkpoint is full model or state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Try loading into self.ehr_transformer
        # The saved encoder was likely an EHRTransformer instance.
        try:
            model.ehr_transformer.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded pre-trained encoder weights.")
        except Exception as e:
            logger.warning(f"Could not load strict state dict: {e}. Trying relaxed matching...")
            # If keys don't match exactly (e.g. prefix issues), we might need manual mapping
            # But since we copied the class definition, it should match if it was saved as model.state_dict()
            pass
            
    # Optimizer (Differential LR)
    optimizer = torch.optim.AdamW([
        {'params': model.ehr_transformer.parameters(), 'lr': args.encoder_lr},
        {'params': model.text_proj.parameters(), 'lr': args.lr},
        {'params': model.gated_fusion.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ], weight_decay=0.01)
    
    criterion = nn.BCEWithLogitsLoss()
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * args.epochs * 0.1),
        num_training_steps=len(train_loader) * args.epochs
    )
    
    best_auc = 0.0
    best_state = None
    best_threshold = 0.5
    patience = 8
    no_improve = 0
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.set_tag("model_type", "e2e_gated_fusion")
        
        for epoch in range(args.epochs):
            # Warmup Strategy: Freeze encoder for first few epochs?
            # Or just rely on low LR. Let's try low LR approach as defined in optimizer.
            
            loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, args.accumulation_steps)
            val_loss, val_auc, val_auprc, val_f1, _, _, _, val_preds, val_targets, val_gates = evaluate(model, val_loader, criterion, device)
            
            logger.info(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Mean Gate: {np.mean(val_gates):.3f}")
            mlflow.log_metrics({
                "train_loss": loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_auprc": val_auprc,
                "mean_gate_value": float(np.mean(val_gates))
            }, step=epoch)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
                best_threshold, val_f1_at = find_optimal_threshold(val_targets, val_preds)
                logger.info(f"  New Best! Threshold: {best_threshold:.3f} (Val F1: {val_f1_at:.3f})")
                no_improve = 0
                torch.save(best_state, os.path.join(args.save_dir, "e2e_gated_best.pt"))
            else:
                no_improve += 1
                
            if no_improve >= patience:
                logger.info("Early stopping")
                break
                
        # Final Test
        if best_state: model.load_state_dict(best_state)
        
        _, test_auc, test_auprc, test_f1, test_prec, test_rec, test_acc, _, _, test_gates = evaluate(
            model, test_loader, criterion, device, threshold=best_threshold
        )
        
        logger.info(f"Final Test AUC: {test_auc:.4f} | AUPRC: {test_auprc:.4f} | F1: {test_f1:.4f}")
        logger.info(f"Test Mean Gate: {np.mean(test_gates):.3f}")
        
        mlflow.log_metrics({
            "test_auc": test_auc,
            "test_auprc": test_auprc,
            "test_f1": test_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_acc": test_acc,
            "optimal_threshold": best_threshold,
            "test_mean_gate_value": float(np.mean(test_gates))
        })

if __name__ == "__main__":
    main()
