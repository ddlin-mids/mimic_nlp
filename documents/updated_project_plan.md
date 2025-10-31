# Updated Project Plan: MIMIC-IV 30-Day Readmission Prediction
## Integration of HiBEHRT + BioClinical Modern BERT Architecture

### Current Status (Post 10/27/2025 Meeting)
- **Architecture Decision**: HiBEHRT hierarchical Transformer for structured EHR + BioClinical Modern BERT for clinical text
- **No Imaging**: Project scope excludes chest X-rays (different from STGNN reference)
- **Repository Structure**: Modular design with configs/, src/, scripts/, tests/ as per meeting notes
- **Data Pipeline**: Cohort → Labels → Structured Events → Text → Fusion → Training

### Key Architectural Changes from STGNN Approach

While the STGNN reference implementation provided valuable insights, the project is now focusing on:

1. **HiBEHRT Encoder** instead of Graph Neural Networks
   - Hierarchical: patient → admission → event sequences
   - Transformer-based with temporal encoding
   - Handles longitudinal EHR data more effectively

2. **BioClinical Modern BERT** for text processing
   - Long-context capability (4096 tokens) or sliding window approach
   - Specifically trained on clinical text
   - Processes discharge notes, clinical notes, radiology reports

3. **Simpler Fusion Strategy**
   - Concatenate structured + text embeddings
   - MLP classifier with dropout
   - Temperature scaling for calibration

### Enhanced Implementation Plan

#### Phase 1: Repository Setup and Core Infrastructure

**Completed Tasks:**
- Repository structure defined in meeting notes
- UV package manager setup
- Basic configuration framework with Hydra

**Next Steps:**
1. Implement cohort builder (`src/data/build_cohort.py`)
2. Create label generation logic (`src/data/labels.py`)
3. Set up data validation and testing framework

#### Phase 2: Structured EHR Processing (HiBEHRT)

**Enhanced from STGNN Learnings:**
- **Event Tokenization**: Map labs, vitals, procedures to shared token space
- **Temporal Encoding**: Relative time since admission + optional bucketing
- **Numerical Processing**: Discretize values + normalized z-score embeddings
- **Sequence Handling**: max_events_per_adm = 2048 (configurable)

**Implementation:**
```python
# src/models/hibehrt.py
class HiBEHRT(nn.Module):
    # Hierarchical: patient → admission → events
    # d_model: 256, n_heads: 4, n_layers: 4
    # Handles temporal sequences with type embeddings
```

#### Phase 3: Text Processing (BioClinical Modern BERT)

**Key Features:**
- **Long Context**: 4096 tokens when available, else 512 with stride 128
- **Note Types**: Discharge summaries primary, optionally concatenate recent notes
- **Pooling**: Attention-based pooling for admission-level embeddings
- **Model**: `emilyalsentzer/Bio_ClinicalBERT` or modern variant

**Implementation:**
```python
# src/text/encoder_biocl_bert.py
class ClinicalTextEncoder:
    # Handles multiple note types
    # Configurable max sequence length
    # Sliding window aggregation if needed
```

#### Phase 4: Modality Fusion and Training

**Fusion Architecture:**
- **Concatenation**: HiBEHRT embedding + Text embedding
- **MLP Layers**: d_hidden = 256 with dropout = 0.1
- **Classification**: Binary output with BCEWithLogitsLoss
- **Calibration**: Temperature scaling on validation set

**Training Configuration:**
- **Optimizer**: AdamW with weight_decay = 0.01
- **Scheduler**: Cosine annealing or plateau
- **Loss**: BCEWithLogitsLoss with optional class weighting
- **Metrics**: AUROC, AUPRC, F1, Brier, ECE

#### Phase 5: Evaluation and Subgroup Analysis

**Enhanced from STGNN Experience:**
- **Global Metrics**: AUROC, AUPRC (primary), calibration curves
- **Subgroup Analysis**: Age bands, sex, insurance, Elixhauser, LOS, service line
- **Interpretability**: SHAP/Integrated Gradients on fusion layer
- **Vulnerable Groups**: Top-5 highest-risk strata by calibrated risk lift

### Technical Implementation Details

#### Dependencies (from meeting notes):
```toml
# Core libraries
torch, transformers, datasets, pytorch-lightning
scikit-learn, pandas/polars, pyarrow
hydra-core, omegaconf, rich, tqdm

# Optional
wandb (off by default), matplotlib, seaborn (EDA only)
```

#### Data Pipeline:
```
Raw MIMIC Data → Cohort Selection → Label Generation → 
Structured Events (HiBEHRT) → Text Processing (BERT) → 
Fusion → Training → Evaluation → Subgroup Analysis
```

#### Key Configuration Parameters:
```yaml
# HiBEHRT Configuration
hibehrt:
  d_model: 256
  n_heads: 4
  n_layers: 4
  dropout: 0.1
  max_events_per_adm: 2048

# Text Configuration  
text:
  model_name: "emilyalsentzer/Bio_ClinicalBERT"
  max_len: 512  # or 4096 for long-context
  
# Training Configuration
train:
  batch_size: 8
  max_epochs: 10
  lr: 2e-4
  weight_decay: 0.01
  precision: bf16
```

### Expected Performance Targets

**Conservative Estimates based on Literature:**
- **AUROC**: 0.70-0.75 (MIMIC readmission benchmark)
- **AUPRC**: 0.40-0.50 (more meaningful for imbalanced data)
- **Calibration**: Well-calibrated probabilities (ECE < 0.05)

**Improvement over Prototype:**
- **Data Scale**: 311K admissions vs 10K prototype
- **Feature Richness**: 100+ structured features vs 26
- **Text Quality**: Clinical BERT vs basic encoding
- **Architecture**: Hierarchical Transformer vs simple MLP

### Integration with Existing Work

**Legacy Notebooks (notebooks_dc/):**
- Keep as reference for data processing logic
- Migrate proven techniques to new src/ modules
- Maintain numbering convention (01_, 02_) for consistency

**STGNN Reference (refs/readmit-stgnn/):**
- Use for evaluation methodology
- Adapt subgroup analysis techniques
- Reference for MIMIC data handling patterns

### Next Steps and Timeline

**Week 1-2: Core Infrastructure**
- Implement cohort builder and label generation
- Set up testing framework with mock data
- Create data validation pipelines

**Week 3-4: Structured EHR Processing**
- Implement HiBEHRT encoder
- Create event tokenization and temporal encoding
- Build structured feature extraction pipeline

**Week 5-6: Text Processing**
- Implement BioClinical BERT encoder
- Create text loading and preprocessing
- Build note aggregation and pooling logic

**Week 7-8: Fusion and Training**
- Implement modality fusion architecture
- Create training loop with Lightning
- Add checkpointing and early stopping

**Week 9-10: Evaluation and Analysis**
- Implement comprehensive evaluation metrics
- Create subgroup analysis pipeline
- Add interpretability tools (SHAP/IG)

### Risk Mitigation

**Computational Challenges:**
- **Solution**: Use HPC3 cluster with GPU allocation
- **Batch Processing**: Implement efficient data loading
- **Memory Management**: Gradient checkpointing for large models

**Data Quality Issues:**
- **Validation**: Comprehensive data quality checks
- **Leakage Prevention**: Strict temporal filtering
- **Missing Data**: Robust handling strategies

**Model Complexity:**
- **Baseline Comparison**: Simple logistic regression baseline
- **Ablation Studies**: Component-wise evaluation
- **Overfitting Monitoring**: Strong validation framework

This updated plan aligns with the 10/27/2025 meeting decisions while incorporating valuable insights from the STGNN reference implementation for evaluation methodology and MIMIC data handling best practices. The focus remains on the hierarchical Transformer approach (HiBEHRT) combined with clinical text processing (BioClinical Modern BERT) as the primary architecture for achieving state-of-the-art readmission prediction performance.}