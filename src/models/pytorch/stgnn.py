"""
Spatiotemporal Graph Neural Network (STGNN) for Readmission Prediction.

Implements GConvGRU: Graph Convolutional GRU that performs message passing
at each timestep, allowing the model to learn from similar patients' 
trajectories during temporal encoding.

Uses PyTorch Geometric (PyG) for graph operations.

Reference: Almeida et al. (2025) - Multimodal spatiotemporal graph neural 
networks for improved prediction of 30-day all-cause hospital readmission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from torch_geometric.data import Data
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GConvGRUCell(nn.Module):
    """
    Graph Convolutional GRU Cell using PyTorch Geometric.
    
    At each timestep, performs:
    1. Concatenate input x_t with hidden state h_{t-1}
    2. Apply GraphSAGE convolution (aggregate neighbor information)
    3. Compute GRU gates (reset r, update u)
    4. Output new hidden state h_t
    
    This allows temporal modeling that also considers similar patients.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        conv_type: str = 'sage',  # 'sage', 'gat', 'gcn'
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_type = conv_type
        
        combined_dim = input_dim + hidden_dim
        
        # Graph convolution for gates (reset + update = 2 * hidden_dim)
        if conv_type == 'sage':
            self.conv_gates = SAGEConv(combined_dim, hidden_dim * 2)
            self.conv_candidate = SAGEConv(combined_dim, hidden_dim)
        elif conv_type == 'gat':
            self.conv_gates = GATConv(combined_dim, hidden_dim * 2, heads=1, dropout=dropout)
            self.conv_candidate = GATConv(combined_dim, hidden_dim, heads=1, dropout=dropout)
        else:  # gcn
            self.conv_gates = GCNConv(combined_dim, hidden_dim * 2)
            self.conv_candidate = GCNConv(combined_dim, hidden_dim)
        
        # Bias terms
        self.gate_bias = nn.Parameter(torch.zeros(hidden_dim * 2))
        self.candidate_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_t, h_prev, edge_index, edge_weight=None):
        """
        Args:
            x_t: Input features at timestep t, shape (num_nodes, input_dim)
            h_prev: Hidden state from previous timestep, shape (num_nodes, hidden_dim)
            edge_index: Graph connectivity, shape (2, num_edges)
            edge_weight: Optional edge weights, shape (num_edges,)
            
        Returns:
            h_new: Updated hidden state, shape (num_nodes, hidden_dim)
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=-1)  # (num_nodes, input_dim + hidden_dim)
        
        # Compute gates via graph convolution
        if self.conv_type == 'sage':
            gates = self.conv_gates(combined, edge_index)
        else:
            gates = self.conv_gates(combined, edge_index, edge_weight)
        
        gates = gates + self.gate_bias
        gates = torch.sigmoid(gates)
        
        # Split into reset and update gates
        r, u = torch.split(gates, self.hidden_dim, dim=-1)
        
        # Compute candidate hidden state
        combined_reset = torch.cat([x_t, r * h_prev], dim=-1)
        
        if self.conv_type == 'sage':
            c = self.conv_candidate(combined_reset, edge_index)
        else:
            c = self.conv_candidate(combined_reset, edge_index, edge_weight)
        
        c = c + self.candidate_bias
        c = torch.tanh(c)
        
        # Update hidden state
        h_new = u * h_prev + (1 - u) * c
        
        return h_new


class GConvGRU(nn.Module):
    """
    Multi-layer Graph Convolutional GRU using PyTorch Geometric.
    
    Processes temporal sequences while aggregating information from
    similar patients at each timestep via graph convolution.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        conv_type: str = 'sage',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stack of GConvGRU cells
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                GConvGRUCell(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    conv_type=conv_type,
                    dropout=dropout,
                )
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_seq, edge_index, edge_weight=None, lengths=None):
        """
        Args:
            x_seq: Input sequence, shape (num_nodes, seq_len, input_dim)
            edge_index: Graph connectivity, shape (2, num_edges)
            edge_weight: Optional edge weights
            lengths: Optional sequence lengths for masking
            
        Returns:
            h_final: Final hidden state, shape (num_nodes, hidden_dim)
            h_seq: All hidden states, shape (num_nodes, seq_len, hidden_dim)
        """
        num_nodes, seq_len, _ = x_seq.shape
        device = x_seq.device
        
        # Transpose for time-first iteration
        x_seq = x_seq.transpose(0, 1)  # (seq_len, num_nodes, input_dim)
        
        # Process through layers
        curr_input = x_seq
        for layer_idx, layer in enumerate(self.layers):
            # Initialize hidden state
            h = torch.zeros(num_nodes, self.hidden_dim, device=device)
            
            outputs = []
            for t in range(seq_len):
                h = layer(curr_input[t], h, edge_index, edge_weight)
                outputs.append(h)
            
            # Stack outputs for next layer
            curr_input = torch.stack(outputs, dim=0)  # (seq_len, num_nodes, hidden_dim)
            
            # Apply dropout between layers (not on last layer)
            if layer_idx < self.num_layers - 1:
                curr_input = self.dropout(F.relu(curr_input))
        
        # Transpose back: (seq_len, num_nodes, hidden_dim) -> (num_nodes, seq_len, hidden_dim)
        h_seq = curr_input.transpose(0, 1)
        
        # Get final hidden state (last timestep or based on lengths)
        if lengths is not None:
            # Get the hidden state at the actual sequence end
            batch_indices = torch.arange(num_nodes, device=device)
            h_final = h_seq[batch_indices, lengths.long() - 1, :]
        else:
            h_final = h_seq[:, -1, :]
        
        return h_final, h_seq


class STGNN(nn.Module):
    """
    Spatiotemporal Graph Neural Network for Readmission Prediction.
    
    Architecture:
    1. Optional: Categorical embedding layer
    2. GConvGRU: Process temporal EHR sequences with graph convolution
    3. Optional: Text embedding fusion
    4. Classification head
    """
    
    def __init__(
        self,
        ehr_input_dim: int,
        hidden_dim: int = 128,
        num_gru_layers: int = 1,
        num_classes: int = 1,
        conv_type: str = 'sage',
        dropout: float = 0.2,
        # Categorical embedding params
        cat_idxs: list = None,
        cat_dims: list = None,
        cat_emb_dim: int = 4,
        # Text fusion params
        text_dim: int = None,
        fusion_type: str = 'concat',  # 'concat', 'gated', 'attention'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        self.text_dim = text_dim
        self.ehr_input_dim = ehr_input_dim
        
        # Categorical embedding (optional)
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim
        
        actual_input_dim = ehr_input_dim
        if self.cat_idxs:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(dim, cat_emb_dim) for dim in self.cat_dims
            ])
            # Adjust input dim: remove cat cols, add embeddings
            actual_input_dim = ehr_input_dim - len(self.cat_idxs) + len(self.cat_idxs) * cat_emb_dim
        else:
            self.cat_embeddings = None
        
        # GConvGRU for temporal + spatial encoding
        self.gconv_gru = GConvGRU(
            input_dim=actual_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gru_layers,
            conv_type=conv_type,
            dropout=dropout,
        )
        
        # Text fusion (optional)
        if text_dim is not None:
            if fusion_type == 'gated':
                # Gated fusion: learn to weight EHR vs text
                self.gate_fc = nn.Linear(hidden_dim + text_dim, 1)
                self.text_proj = nn.Linear(text_dim, hidden_dim)
                classifier_input_dim = hidden_dim
            elif fusion_type == 'attention':
                # Cross-attention fusion
                self.text_proj = nn.Linear(text_dim, hidden_dim)
                self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
                classifier_input_dim = hidden_dim
            else:  # concat
                self.text_proj = nn.Linear(text_dim, hidden_dim)
                classifier_input_dim = hidden_dim * 2
        else:
            classifier_input_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
    def _embed_categorical(self, x):
        """Embed categorical features."""
        if not self.cat_embeddings:
            return x
        
        # x shape: (num_nodes, seq_len, input_dim)
        num_nodes, seq_len, _ = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  # (num_nodes * seq_len, input_dim)
        
        cols = []
        cat_counter = 0
        for idx in range(self.ehr_input_dim):
            if idx in self.cat_idxs:
                # Embed categorical
                emb = self.cat_embeddings[cat_counter](x_flat[:, idx].long())
                cols.append(emb)
                cat_counter += 1
            else:
                # Keep continuous
                cols.append(x_flat[:, idx:idx+1])
        
        x_embedded = torch.cat(cols, dim=-1)
        return x_embedded.reshape(num_nodes, seq_len, -1)
    
    def forward(self, ehr_seq, edge_index, edge_weight=None, text_emb=None, lengths=None):
        """
        Args:
            ehr_seq: EHR sequence, shape (num_nodes, seq_len, ehr_input_dim)
            edge_index: Graph connectivity, shape (2, num_edges)
            edge_weight: Optional edge weights
            text_emb: Optional text embeddings, shape (num_nodes, text_dim)
            lengths: Optional sequence lengths
            
        Returns:
            logits: Prediction logits, shape (num_nodes,) or (num_nodes, num_classes)
            h_ehr: EHR hidden state for downstream use
        """
        # Embed categorical features if needed
        if self.cat_embeddings:
            ehr_seq = self._embed_categorical(ehr_seq)
        
        # Process with GConvGRU
        h_ehr, _ = self.gconv_gru(ehr_seq, edge_index, edge_weight, lengths)  # (num_nodes, hidden_dim)
        
        # Fuse with text if provided
        if text_emb is not None and self.text_dim is not None:
            h_text = self.text_proj(text_emb)  # (num_nodes, hidden_dim)
            
            if self.fusion_type == 'gated':
                # Gated fusion
                gate_input = torch.cat([h_ehr, text_emb], dim=-1)
                gate = torch.sigmoid(self.gate_fc(gate_input))
                h_fused = gate * h_ehr + (1 - gate) * h_text
            elif self.fusion_type == 'attention':
                # Cross-attention (batch_first=True)
                h_ehr_unsq = h_ehr.unsqueeze(1)  # (num_nodes, 1, hidden_dim)
                h_text_unsq = h_text.unsqueeze(1)
                h_attn, _ = self.attn(h_ehr_unsq, h_text_unsq, h_text_unsq)
                h_fused = h_attn.squeeze(1) + h_ehr  # Residual
            else:  # concat
                h_fused = torch.cat([h_ehr, h_text], dim=-1)
        else:
            h_fused = h_ehr
        
        # Classify
        logits = self.classifier(h_fused)
        
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        
        return logits, h_ehr


class STGNNMultimodal(nn.Module):
    """
    Multimodal STGNN with separate streams for EHR and text.
    
    Both streams use GConvGRU-style processing, then late fusion.
    """
    
    def __init__(
        self,
        ehr_input_dim: int,
        text_dim: int,
        hidden_dim: int = 128,
        num_gru_layers: int = 1,
        num_classes: int = 1,
        conv_type: str = 'sage',
        dropout: float = 0.2,
        cat_idxs: list = None,
        cat_dims: list = None,
        cat_emb_dim: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # EHR stream (STGNN)
        self.ehr_encoder = STGNN(
            ehr_input_dim=ehr_input_dim,
            hidden_dim=hidden_dim,
            num_gru_layers=num_gru_layers,
            num_classes=1,  # We'll use our own classifier
            conv_type=conv_type,
            dropout=dropout,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            text_dim=None,  # No fusion in encoder
        )
        
        # Text stream (simple projection)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Joint classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def forward(self, ehr_seq, edge_index, edge_weight=None, text_emb=None, lengths=None):
        """
        Args:
            ehr_seq: EHR sequences, shape (num_nodes, seq_len, ehr_dim)
            edge_index: Graph connectivity
            edge_weight: Optional edge weights
            text_emb: Text embeddings, shape (num_nodes, text_dim)
            lengths: Optional sequence lengths
            
        Returns:
            logits: Prediction logits
        """
        # Encode EHR (get hidden state only)
        _, h_ehr = self.ehr_encoder(ehr_seq, edge_index, edge_weight, lengths=lengths)
        
        # Encode text
        h_text = self.text_encoder(text_emb)
        
        # Concatenate and classify
        h_joint = torch.cat([h_ehr, h_text], dim=-1)
        logits = self.classifier(h_joint)
        
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        
        return logits


def build_knn_graph(features, k=15, metric='cosine'):
    """
    Build a k-NN graph from node features using PyTorch Geometric format.
    
    Args:
        features: Node features, shape (num_nodes, feature_dim)
        k: Number of neighbors
        metric: Distance metric ('cosine' or 'euclidean')
        
    Returns:
        edge_index: Graph connectivity, shape (2, num_edges)
        edge_weight: Edge weights, shape (num_edges,)
    """
    from sklearn.neighbors import NearestNeighbors
    
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    # Normalize for cosine
    if metric == 'cosine':
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms
    
    # Build k-NN
    knn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    knn.fit(features)
    distances, indices = knn.kneighbors(features)
    
    # Create edges
    num_nodes = features.shape[0]
    src = np.repeat(np.arange(num_nodes), k)
    dst = indices.flatten()
    
    # Edge weights (similarity, not distance)
    if metric == 'cosine':
        weights = 1 - distances.flatten()
    else:
        weights = np.exp(-distances.flatten())  # Gaussian kernel
    
    # Remove self-loops
    mask = src != dst
    src, dst, weights = src[mask], dst[mask], weights[mask]
    
    # Make undirected by adding reverse edges
    src_undirected = np.concatenate([src, dst])
    dst_undirected = np.concatenate([dst, src])
    weights_undirected = np.concatenate([weights, weights])
    
    # Remove duplicates (keep max weight)
    edge_dict = {}
    for s, d, w in zip(src_undirected, dst_undirected, weights_undirected):
        key = (s, d)
        if key not in edge_dict or w > edge_dict[key]:
            edge_dict[key] = w
    
    final_src = []
    final_dst = []
    final_weights = []
    for (s, d), w in edge_dict.items():
        final_src.append(s)
        final_dst.append(d)
        final_weights.append(w)
    
    edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)
    edge_weight = torch.tensor(final_weights, dtype=torch.float32)
    
    return edge_index, edge_weight
