# Model Architecture Deep Dive

This document provides detailed architectural explanations for the key models in our multimodal readmission prediction pipeline.

## Current Model Leaderboard

| Rank | Model | AUC | AUPRC | F1 | Notes |
|------|-------|-----|-------|-----|-------|
| 1 | **XGBoost (GRU EHR)** | **0.6406** | 0.3408 | 0.00 | Best AUC, no F1 |
| 2 | Gated Fusion (GRU) | 0.6380 | 0.3495 | 0.29 | Good balance |
| 3 | Early Fusion MLP (HPO) | 0.6378 | 0.3541 | 0.35 | |
| 4 | XGBoost Full Fusion | 0.6371 | 0.3495 | 0.09 | |
| 5 | Gated Fusion (Optuna) | 0.6316 | **0.3613** | 0.29 | Best AUPRC |
| 6 | **E2E Gated Fusion** | 0.6246 | 0.3527 | **0.41** | **Best F1!** |
| 7 | GNN (GraphSAGE) | 0.5958 | 0.3192 | 0.41 | High F1, low AUC |

---

## 1. XGBoost (GRU EHR) - AUC: 0.6406

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    XGBoost (GRU EHR)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Stage 1: FROZEN Pre-trained GRU Encoder                       │
│   ┌─────────────────────────────────────────┐                   │
│   │  EHR Sequence (labs, meds, dx per day)  │                   │
│   │         ↓                               │                   │
│   │  Bidirectional GRU (hidden=128)         │                   │
│   │         ↓                               │                   │
│   │  Final Hidden State → 128-dim embedding │                   │
│   └─────────────────────────────────────────┘                   │
│                        ↓                                        │
│   Stage 2: XGBoost Classifier (Gradient Boosted Trees)          │
│   ┌─────────────────────────────────────────┐                   │
│   │  Input: 128-dim GRU embedding           │                   │
│   │         ↓                               │                   │
│   │  1000 Decision Trees (depth=4)          │                   │
│   │         ↓                               │                   │
│   │  P(readmission)                         │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Properties

- **Two-stage pipeline**: Encoder is frozen, then XGBoost trains on embeddings
- **No text/notes used**: Pure structured EHR data
- **XGBoost strengths**: Excellent at finding nonlinear decision boundaries in fixed-dimensional embeddings
- **F1=0 issue**: XGBoost outputs uncalibrated probabilities that don't cross the 0.5 threshold well

### Why It Works Well

XGBoost excels at finding optimal decision splits in pre-computed embeddings. The GRU captures temporal patterns in the EHR sequence, and XGBoost finds the best way to use those patterns for classification.

### Limitations

- Two-stage means the GRU wasn't trained to optimize for XGBoost's needs
- No multimodal information (clinical notes ignored)
- Poor probability calibration leads to F1=0

---

## 2. Gated Fusion (GRU) - AUC: 0.6380, F1: 0.29

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gated Fusion (GRU)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────────┐      ┌──────────────────┐                │
│   │   EHR Stream     │      │   Text Stream    │                │
│   │                  │      │                  │                │
│   │  GRU Encoder     │      │  ModernBERT      │                │
│   │  (frozen)        │      │  (frozen)        │                │
│   │      ↓           │      │      ↓           │                │
│   │  h_ehr (128-dim) │      │  h_text (768-dim)│                │
│   └────────┬─────────┘      └────────┬─────────┘                │
│            │                         │                          │
│            │    ┌────────────────────┘                          │
│            │    │                                               │
│            ▼    ▼                                               │
│   ┌─────────────────────────────────────────┐                   │
│   │        GATED FUSION UNIT (GMU)          │                   │
│   │                                         │                   │
│   │  gate = σ(W_g · [h_ehr, h_text] + b)    │  ← Learned gate   │
│   │                                         │                   │
│   │  h_fused = gate * h_ehr + (1-gate) * h_text                 │
│   │                                         │                   │
│   │  (Learns WHEN to trust text vs EHR)     │                   │
│   └─────────────────────────────────────────┘                   │
│                        ↓                                        │
│   ┌─────────────────────────────────────────┐                   │
│   │  MLP Classifier: Linear → ReLU → Linear │                   │
│   │         ↓                               │                   │
│   │  P(readmission)                         │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Properties

- **Multimodal**: Uses BOTH EHR sequences AND clinical notes
- **Learned gate**: Dynamically weights modalities (typically gate ≈ 0.7, trusting EHR more)
- **End-to-end trainable fusion head**: But encoders remain frozen
- **Calibrated probabilities**: Produces reasonable F1 scores

### The Gated Multimodal Unit (GMU)

The core innovation is the **gate mechanism**:

```python
# Concatenate modality embeddings
combined = concat(h_ehr, h_text)  # (batch, 128 + 768)

# Learn a scalar gate per sample
gate = sigmoid(W_gate @ combined + b_gate)  # (batch, 1)

# Weighted combination
h_fused = gate * h_ehr + (1 - gate) * h_text
```

### Why It Works Well

The gate solves the **"Fusion Paradox"**: text embeddings are high-variance and can hurt performance if blindly concatenated. The gate learns to discount unreliable text when the EHR signal is strong.

### Limitations

- Encoders are still frozen - gate can only re-weight, not transform
- Limited expressiveness compared to attention mechanisms

---

## 3. E2E Gated Fusion (Transformer) - AUC: 0.6246, F1: 0.41

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              E2E Gated Fusion (Transformer)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────────┐      ┌──────────────────┐                │
│   │   EHR Stream     │      │   Text Stream    │                │
│   │                  │      │                  │                │
│   │  Transformer     │      │  ModernBERT      │                │
│   │  Encoder         │      │  (frozen)        │                │
│   │  (TRAINABLE!)    │      │      ↓           │                │
│   │      ↓           │      │  PCA: 768→64     │                │
│   │  h_ehr (128-dim) │      │  h_text (64-dim) │                │
│   └────────┬─────────┘      └────────┬─────────┘                │
│            │                         │                          │
│            │    ┌────────────────────┘                          │
│            ▼    ▼                                               │
│   ┌─────────────────────────────────────────┐                   │
│   │        GATED FUSION UNIT (GMU)          │                   │
│   │                                         │                   │
│   │  h_fused = gate * h_ehr + (1-gate) * h_text                 │
│   └─────────────────────────────────────────┘                   │
│                        ↓                                        │
│   ┌─────────────────────────────────────────┐                   │
│   │  MLP Classifier                         │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Difference: Joint Training

```
┌─────────────────────────────────────────┐
│  Loss backpropagates through:           │
│    Classifier → Gate → Transformer      │
│                                         │
│  Transformer learns features optimized  │
│  for fusion, not just standalone EHR    │
│                                         │
│  Differential Learning Rates:           │
│    - Transformer: 2e-5 (slow, preserve) │
│    - Fusion head: 1e-4 (fast, adapt)    │
└─────────────────────────────────────────┘
```

### Why F1 is Highest (0.41)

1. **Joint training** → Better calibrated probabilities
2. **Transformer > GRU** for long sequences (15+ day hospital stays)
3. **Self-attention** captures long-range dependencies in clinical trajectories
4. **PCA on text** (768→64) reduces noise from high-dimensional text embeddings

### Why AUC is Lower (0.6246)

- Joint training is harder to optimize (more parameters, complex loss landscape)
- Model may be slightly underfitting on ranking but better at actual classification
- Trade-off between discrimination (AUC) and calibration (F1)

### Transformer Encoder Details

```python
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2):
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, x, lengths):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Add CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Self-attention over sequence
        x = self.transformer(x)
        
        # Return CLS token as sequence representation
        return x[:, 0, :]  # (batch, hidden_dim)
```

---

## 4. STGNN (Spatiotemporal Graph Neural Network)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              STGNN (Spatiotemporal Graph NN)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Patient Similarity Graph:                                     │
│   ┌─────────────────────────────────────────┐                   │
│   │     [P1]───────[P2]                     │                   │
│   │      │╲        ╱│                       │                   │
│   │      │ ╲      ╱ │   Edges = similar     │                   │
│   │      │  ╲    ╱  │   demographics,       │                   │
│   │      │   ╲  ╱   │   diagnoses, labs     │                   │
│   │     [P3]──╳──[P4]                       │                   │
│   │           ╲╱                            │                   │
│   │          [P5]                           │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   GConvGRU: Graph Convolution + GRU at EACH timestep            │
│   ┌─────────────────────────────────────────┐                   │
│   │                                         │                   │
│   │   For t = 1, 2, ..., T (days):          │                   │
│   │                                         │                   │
│   │   1. x_t = patient's labs/meds at day t │                   │
│   │                                         │                   │
│   │   2. Aggregate neighbors' hidden states:│                   │
│   │      h_neighbors = GraphSAGE(graph,     │                   │
│   │                    [x_t, h_{t-1}])      │                   │
│   │                                         │                   │
│   │   3. GRU update with neighbor info:     │                   │
│   │      h_t = GRU(h_neighbors, h_{t-1})    │                   │
│   │                                         │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What STGNN Captures That Others Don't

```
┌─────────────────────────────────────────┐
│                                         │
│  "At day 5, patient A's creatinine is   │
│   rising. What happened to similar      │
│   patients B, C, D at their day 5?"     │
│                                         │
│  → Cross-patient temporal learning      │
│  → "Patients like you who had this      │
│     trajectory were readmitted"         │
│                                         │
└─────────────────────────────────────────┘
```

### GConvGRU Cell

The core innovation is performing **graph convolution at each timestep**:

```python
class GConvGRUCell(nn.Module):
    def forward(self, x_t, h_prev, edge_index, edge_weight):
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=-1)
        
        # Graph convolution for gates (aggregate neighbor information)
        gates = self.conv_gates(combined, edge_index)
        gates = torch.sigmoid(gates + self.gate_bias)
        r, u = torch.split(gates, self.hidden_dim, dim=-1)
        
        # Candidate hidden state with reset gate
        combined_reset = torch.cat([x_t, r * h_prev], dim=-1)
        c = self.conv_candidate(combined_reset, edge_index)
        c = torch.tanh(c + self.candidate_bias)
        
        # Update hidden state
        h_new = u * h_prev + (1 - u) * c
        return h_new
```

### Graph Construction

- **Nodes**: Each hospital admission
- **Edges**: k-NN similarity based on:
  - Demographics (age, gender, race)
  - ICD diagnosis codes
  - Medication classes
  - Lab value patterns
- **Edge weights**: Gaussian kernel on cosine distance
- **k**: 15 neighbors (top 1% most similar)

### Potential Advantages

1. **Cross-patient learning**: Leverages similar patients' outcomes
2. **Temporal graph convolution**: Message passing at each timestep
3. **Clinically intuitive**: Mimics how physicians reason ("I've seen patients like this before")

---

## Summary Comparison

| Model | Temporal | Cross-Patient | Text | Trainable Encoder | Best For |
|-------|----------|---------------|------|-------------------|----------|
| XGBoost (GRU) | GRU (frozen) | No | No | No | Pure ranking (AUC) |
| Gated Fusion | GRU (frozen) | No | Yes (Gated) | No | Balanced AUC/F1 |
| E2E Gated | Transformer (trainable) | No | Yes (Gated) | Yes | Best F1, alerting |
| **STGNN** | GConvGRU | **Yes** | Optional | Yes | Cross-patient patterns |

## HPO Recommendations

| Goal | Model to HPO | Rationale |
|------|--------------|-----------|
| **Maximize AUC** | XGBoost or Gated Fusion | Already near ceiling, tree HPO is fast |
| **Maximize F1** | E2E Gated Fusion | Best F1, joint training has more hyperparameters to tune |
| **Novel contribution** | STGNN | If it beats others, most interesting research result |

---

## Implementation References

- `src/models/pytorch/gated_fusion.py` - Gated Fusion implementation
- `src/models/pytorch/train_e2e_gated_fusion.py` - End-to-end training
- `src/models/pytorch/stgnn.py` - STGNN with GConvGRU
- `src/models/pytorch/train_stgnn.py` - STGNN training script
- `src/models/train_fusion_xgboost.py` - XGBoost ablation studies
