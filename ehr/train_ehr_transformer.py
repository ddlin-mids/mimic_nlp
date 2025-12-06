import argparse
import os
import pickle
import sys
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class EHRDataset(Dataset):
    def __init__(self, feat_dict, labels_map, max_len=100):
        self.node_names = list(feat_dict.keys())
        self.node_names = [n for n in self.node_names if n in labels_map]
        self.feat_dict = feat_dict
        self.labels_map = labels_map
        self.max_len = max_len

    def __len__(self):
        return len(self.node_names)

    def __getitem__(self, idx):
        node_name = self.node_names[idx]
        features = self.feat_dict[node_name]
        label = self.labels_map[node_name]

        if len(features) > self.max_len:
            features = features[-self.max_len:]
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), node_name

def collate_fn(batch):
    features, labels, node_names = zip(*batch)
    lengths = torch.tensor([len(f) for f in features])
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return features_padded, lengths, labels, node_names

# -----------------------------------------------------------------------------
# Positional Encoding
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
        # x: (Batch, SeqLen, Dim)
        # pe: (MaxLen, Dim) -> slice to (SeqLen, Dim)
        return x + self.pe[:x.size(1), :]

# -----------------------------------------------------------------------------
# Transformer Encoder
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
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

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
        # True = Ignored (Padding)
        # Sequence is [CLS, t1, t2, ..., tn, pad, pad]
        # CLS is never masked.
        # Original lengths correspond to t1...tn.
        # We need mask of size (Batch, SeqLen+1)
        max_len_plus_cls = x_emb.size(1)
        mask = torch.arange(max_len_plus_cls, device=device).expand(batch_size, max_len_plus_cls)
        # lengths needs to be adjusted? No, lengths is raw sequence length.
        # Valid indices are 0 (CLS) to lengths (inclusive? no 1-based).
        # Index 0 is CLS. Index 1 is t1.
        # Valid are indices < lengths + 1
        mask = mask >= (lengths.unsqueeze(1) + 1).to(device)
        
        # Transformer
        output = self.transformer_encoder(x_emb, src_key_padding_mask=mask)
        
        # Extract CLS token output
        final_embedding = output[:, 0, :]
        
        # Get full sequence (excluding CLS)
        output_seq = output[:, 1:, :]
        
        logits = self.classifier(final_embedding)
        
        return logits.squeeze(1), final_embedding, output_seq

# -----------------------------------------------------------------------------
# Utils (Copied from train_ehr_encoder.py)
# -----------------------------------------------------------------------------
def load_data(pkl_path, cohort_csv):
    logger.info(f"Loading sequence data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        seq_data = pickle.load(f)
    feat_dict = seq_data["feat_dict"]
    
    if "cat_idxs" in seq_data:
        cat_idxs = seq_data["cat_idxs"]
        cat_dims = seq_data["cat_dims"]
    else:
        cat_idxs = []
        cat_dims = []
        
    sample_key = next(iter(feat_dict))
    total_cols = feat_dict[sample_key].shape[1]
    all_idxs = set(range(total_cols))
    num_idxs = list(sorted(all_idxs - set(cat_idxs)))
    
    clean_feat_dict = {}
    for k, v in tqdm(feat_dict.items(), desc="Sanitizing"):
        clean_feat_dict[k] = v.astype(np.float32)
    feat_dict = clean_feat_dict
    
    df = pd.read_csv(cohort_csv)
    if "readmitted_within_30days" in df.columns: target_col = "readmitted_within_30days"
    elif "readmitted_within_window" in df.columns: target_col = "readmitted_within_window"
    
    df["node_name"] = df["subject_id"].astype(str) + "_" + df["hadm_id"].astype(str)
    labels_map = dict(zip(df["node_name"], df[target_col].astype(int)))
    split_map = dict(zip(df["node_name"], df["split"])) if "split" in df.columns else dict(zip(df["node_name"], df["subject_id"]))
    
    return feat_dict, cat_dims, cat_idxs, num_idxs, labels_map, split_map

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    feat_dict, cat_dims, cat_idxs, num_idxs, labels_map, split_map = load_data(args.input_path, args.cohort_path)
    valid_keys = [k for k in feat_dict.keys() if k in labels_map]
    
    train_keys = [k for k in valid_keys if split_map.get(k) == "train"]
    val_keys = [k for k in valid_keys if split_map.get(k) in ["val", "validation"]]
    
    train_feat = {k: feat_dict[k] for k in train_keys}
    val_feat = {k: feat_dict[k] for k in val_keys}
    
    train_loader = DataLoader(EHRDataset(train_feat, labels_map, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(EHRDataset(val_feat, labels_map, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    full_loader = DataLoader(EHRDataset(feat_dict, labels_map, args.max_len), batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    embed_dims = [min(50, (d + 1) // 2) for d in cat_dims]
    
    model = EHRTransformer(
        input_dims=[d + 1 for d in cat_dims],
        embed_dims=embed_dims,
        cat_idxs=cat_idxs,
        num_idxs=num_idxs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0
    
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for feats, lens, lbls, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            feats, lens, lbls = feats.to(device), lens.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(feats, lens)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for feats, lens, lbls, _ in val_loader:
                feats, lens, lbls = feats.to(device), lens.to(device), lbls.to(device)
                logits, _, _ = model(feats, lens)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(lbls.cpu().numpy())
        
        val_auc = roc_auc_score(val_targets, val_preds)
        val_ap = average_precision_score(val_targets, val_preds)
        logger.info(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}, Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "ehr_transformer_best.pt"))
            
    logger.info("Training complete. Generating final embeddings...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "ehr_transformer_best.pt"), weights_only=True))
    model.eval()
    
    all_embeddings, all_seqs, all_node_names = [], [], []
    with torch.no_grad():
        for feats, lens, _, nodes in tqdm(full_loader, desc="Inferencing"):
            feats, lens = feats.to(device), lens.to(device)
            _, embeddings, seqs = model(feats, lens)
            
            all_embeddings.append(embeddings.cpu().numpy())
            curr_seq = seqs.cpu().numpy()
            
            # Pad/Clip to max_len
            if curr_seq.shape[1] < args.max_len:
                pad = ((0,0), (0, args.max_len - curr_seq.shape[1]), (0,0))
                curr_seq = np.pad(curr_seq, pad, mode='constant')
            else:
                curr_seq = curr_seq[:, :args.max_len, :]
                
            all_seqs.append(curr_seq)
            all_node_names.extend(nodes)
            
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_seqs = np.concatenate(all_seqs, axis=0)
    
    np.savez(os.path.join(args.save_dir, "structured_ehr_embeddings.npz"), embeddings=final_embeddings, node_names=all_node_names)
    np.savez(os.path.join(args.save_dir, "structured_ehr_embeddings_seq.npz"), embeddings=final_seqs, node_names=all_node_names)
    
    map_df = pd.DataFrame({"node_name": all_node_names, "row_idx": range(len(all_node_names))})
    map_df.to_csv(os.path.join(args.save_dir, "structured_ehr_mapping.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="data/processed/ehr_embeddings")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train_model(args)
