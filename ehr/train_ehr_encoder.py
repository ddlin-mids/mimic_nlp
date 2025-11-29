import argparse
import os
import pickle
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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
        """
        feat_dict: {node_name: np.array(seq_len, feat_dim)}
        labels_map: {node_name: int_label}
        max_len: max sequence length to clip
        """
        self.node_names = list(feat_dict.keys())
        # Filter to only nodes that have labels
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

        # Clip length
        if len(features) > self.max_len:
            features = features[-self.max_len:]
        
        return torch.tensor(features, dtype=torch.long), torch.tensor(label, dtype=torch.float32), node_name

def collate_fn(batch):
    # batch is list of (features, label, node_name)
    features, labels, node_names = zip(*batch)
    
    # lengths
    lengths = torch.tensor([len(f) for f in features])
    
    # pad
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    
    labels = torch.stack(labels)
    
    return features_padded, lengths, labels, node_names

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class EHREncoder(nn.Module):
    def __init__(self, input_dims, embed_dims, hidden_dim=128, num_layers=1, dropout=0.1, model_type="gru"):
        """
        input_dims: list of num_classes for each feature column
        embed_dims: list of embedding dimensions for each feature column
        model_type: 'gru' or 'lstm'
        """
        super().__init__()
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num, embedding_dim=dim, padding_idx=0)
            for num, dim in zip(input_dims, embed_dims)
        ])
        
        total_embed_dim = sum(embed_dims)
        self.model_type = model_type.lower()
        
        if self.model_type == "gru":
            self.rnn = nn.GRU(
                input_size=total_embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=total_embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, lengths):
        # x: (batch, max_len, num_features)
        
        # Embed each feature and concat
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            # Clamp indices to valid range just in case
            feat_idx = x[:, :, i].clamp(0, emb_layer.num_embeddings - 1)
            embedded.append(emb_layer(feat_idx))
            
        x_emb = torch.cat(embedded, dim=-1) # (batch, max_len, total_embed_dim)
        
        # Pack
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # RNN
        if self.model_type == "lstm":
            packed_out, (hidden, cell) = self.rnn(packed)
        else:
            packed_out, hidden = self.rnn(packed)
        
        # hidden: (num_layers, batch, hidden_dim)
        # Use last layer hidden state as sequence embedding
        final_embedding = hidden[-1] # (batch, hidden_dim)
        
        # Classify
        logits = self.classifier(final_embedding)
        
        return logits.squeeze(1), final_embedding

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def load_data(pkl_path, cohort_csv):
    logger.info(f"Loading sequence data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        seq_data = pickle.load(f)
        
    feat_dict = seq_data["feat_dict"]
    cat_dims = seq_data["cat_dims"] 
    
    logger.info(f"Loaded {len(feat_dict)} admissions.")
    logger.info(f"Feature dimensions (vocab sizes): {cat_dims}")
    
    logger.info(f"Loading labels from {cohort_csv}...")
    df = pd.read_csv(cohort_csv)
    
    # Normalize target column
    if "readmitted_within_30days" in df.columns:
        target_col = "readmitted_within_30days"
    elif "readmitted_within_window" in df.columns:
        target_col = "readmitted_within_window"
    else:
        raise KeyError("Cohort file missing readmission label column")
        
    # Vectorized map creation
    df["node_name"] = df["subject_id"].astype(str) + "_" + df["hadm_id"].astype(str)
    labels_map = dict(zip(df["node_name"], df[target_col].astype(int)))
    
    # Return split mapping for strict splitting
    if "split" in df.columns:
        split_map = dict(zip(df["node_name"], df["split"]))
    elif "splits" in df.columns:
        split_map = dict(zip(df["node_name"], df["splits"]))
    else:
        # Fallback if no split column (should not happen in this pipeline)
        logger.warning("No 'split' column found. Falling back to subject ID for random split.")
        split_map = dict(zip(df["node_name"], df["subject_id"]))
        
    return feat_dict, cat_dims, labels_map, split_map

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    feat_dict, cat_dims, labels_map, split_map = load_data(args.input_path, args.cohort_path)
    
    # Filter to valid keys
    valid_keys = [k for k in feat_dict.keys() if k in labels_map]
    
    # Split based on pre-defined column
    train_keys = []
    val_keys = []
    test_keys_skipped = []
    
    # Check if split_map contains actual split names or subject IDs (fallback)
    sample_val = next(iter(split_map.values()))
    is_explicit_split = isinstance(sample_val, str) and sample_val in ["train", "val", "test", "validation"]
    
    if is_explicit_split:
        logger.info("Using pre-defined splits from cohort file.")
        for k in valid_keys:
            split = split_map.get(k)
            if split == "train":
                train_keys.append(k)
            elif split in ["val", "validation"]:
                val_keys.append(k)
            else:
                test_keys_skipped.append(k)
    else:
        logger.warning("Performing random patient-level split (NOT RECOMMENDED for final results).")
        # Fallback: Patient-level random split
        subj_map = split_map # In fallback mode, load_data returned subj_map
        subjects = list(set(subj_map[k] for k in valid_keys))
        train_subjs, val_subjs = train_test_split(subjects, test_size=0.2, random_state=42)
        train_subjs_set = set(train_subjs)
        
        train_keys = [k for k in valid_keys if subj_map[k] in train_subjs_set]
        val_keys = [k for k in valid_keys if subj_map[k] not in train_subjs_set]
    
    logger.info(f"Train admissions: {len(train_keys)}")
    logger.info(f"Val admissions: {len(val_keys)}")
    logger.info(f"Skipped (Test/Other): {len(test_keys_skipped)}")
    
    # Create dict subsets
    train_feat = {k: feat_dict[k] for k in train_keys}
    val_feat = {k: feat_dict[k] for k in val_keys}
    
    train_ds = EHRDataset(train_feat, labels_map, max_len=args.max_len)
    val_ds = EHRDataset(val_feat, labels_map, max_len=args.max_len)
    full_ds = EHRDataset(feat_dict, labels_map, max_len=args.max_len)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Model Setup
    embed_dims = [min(50, (d + 1) // 2) for d in cat_dims]
    
    model = EHREncoder(
        input_dims=[d + 1 for d in cat_dims],
        embed_dims=embed_dims,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.2,
        model_type=args.model_type
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0
    
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for feats, lens, lbls, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            feats, lens, lbls = feats.to(device), lens.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(feats, lens)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for feats, lens, lbls, _ in val_loader:
                feats, lens, lbls = feats.to(device), lens.to(device), lbls.to(device)
                logits, _ = model(feats, lens)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(lbls.cpu().numpy())
        
        val_auc = roc_auc_score(val_targets, val_preds)
        val_ap = average_precision_score(val_targets, val_preds)
        
        logger.info(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}, Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "ehr_encoder_best.pt"))
            
    logger.info("Training complete. Generating final embeddings...")
    
    # Generate embeddings for ALL data
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "ehr_encoder_best.pt"), weights_only=True))
    model.eval()
    
    all_loader = DataLoader(full_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    all_embeddings = []
    all_node_names = []
    
    with torch.no_grad():
        for feats, lens, _, nodes in tqdm(all_loader, desc="Inferencing"):
            feats, lens = feats.to(device), lens.to(device)
            _, embeddings = model(feats, lens)
            all_embeddings.append(embeddings.cpu().numpy())
            all_node_names.extend(nodes)
            
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Save
    out_npz = os.path.join(args.save_dir, "structured_ehr_embeddings.npz")
    np.savez(out_npz, embeddings=final_embeddings, node_names=all_node_names)
    logger.info(f"Saved embeddings to {out_npz}")
    
    # Save mapping
    out_map = os.path.join(args.save_dir, "structured_ehr_mapping.csv")
    map_df = pd.DataFrame({"node_name": all_node_names, "row_idx": range(len(all_node_names))})
    map_df.to_csv(out_map, index=False)
    logger.info(f"Saved mapping to {out_map}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to .pkl sequence file")
    parser.add_argument("--cohort_path", type=str, required=True, help="Path to cohort CSV")
    parser.add_argument("--save_dir", type=str, default="data/processed/ehr_embeddings")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=100, help="Max sequence length (days)")
    parser.add_argument("--model_type", type=str, default="gru", choices=["gru", "lstm"], help="RNN type")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train_model(args)
