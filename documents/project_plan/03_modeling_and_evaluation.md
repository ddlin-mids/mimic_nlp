# Modeling & Evaluation Strategy

## Baseline Classifier
- **Fusion Input:** Concatenated structured embedding (HiBEHRT or successor) + text embedding (BioClinical Modern BERT).
- **Head Architecture:** 2-layer MLP (hidden 256, dropout 0.1) with sigmoid output; class weights or focal loss if imbalance impacts recall.
- **Training Regimen:** AdamW (lr 2e-4, weight_decay 0.01), cosine scheduler, early stopping on validation AUROC. Precision: bfloat16 if hardware permits.

## Alternatives Under Study
- **GraphNN Overlay:** GraphSAGE or GAT using admission similarity edges (demographics, ICD overlap, embedding cosine). Compare against MLP for calibrated performance.
- **Temporal Contrastive Pretraining:** Optional step to initialize structured encoder using sequence-based objectives.
- **Multitask Heads:** Auxiliary prediction of length-of-stay bin or discharge disposition to inject clinical priors.

## Evaluation Protocol
- **Splits:** Patient-wise train/val/test (70/15/15) with consistent seeds.
- **Primary Metrics:** AUROC, AUPRC (overall + focus-group slices); report mean ± bootstrap CI.
- **Calibration:** Reliability diagrams, Expected Calibration Error (ECE < 0.05 target), Brier score.
- **Subgroup Monitoring:** Stratify by LOS bin, focus groups, discharge disposition, insurance, age bands, race; highlight risk lift and fairness considerations.
- **Error Taxonomy:** Sample misclassified cases per group for clinical review, capture note excerpts and recent labs/meds leading to prediction.

## Deployment Readiness
- **Artifact Packaging:** Store model weights, config, and git hash; export inference graph (TorchScript/ONNX) with deterministic preprocessing pipeline.
- **Inference Path:** Batch embeddings → fusion head; ensure runtime supports long-context inference (GPU memory estimation required).
- **MLOps Hooks:** Logging of prediction distributions, threshold tuning per focus group, ability to flag high-risk psych discharges for immediate interventions.

## Research Backlog
1. Validate GraphNN lift over MLP on small-scale prototype.
2. Explore zero-shot causal contrastors (e.g., CRL-MMNAR) for comparison with literature leaders.
3. Investigate contrastive alignment between structured and text embeddings to enhance modal synergy.
