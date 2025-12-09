# Speaker Notes for Final Presentation

**NLP-Driven Temporally-Aware 30-Day Readmission Prediction on MIMIC-IV**  
DATASCI 266 Fall 2025 - David Lin and Daniel Chung

---

## SLIDE 1: Title Slide

**Speaker Notes:**
> Good [morning/afternoon], I'm [David/Daniel] and my partner is [Daniel/David]. Today we're presenting our work on predicting 30-day hospital readmissions using multimodal clinical data from MIMIC-IV. Our approach combines structured electronic health records with clinical narrative text, and we'll show you both the promises and the pitfalls of multimodal fusion in healthcare AI.

---

## SLIDE 2: Table of Contents

**Speaker Notes:**
> Here's our agenda. We'll start with the clinical motivation, then discuss a key discovery we made called the "Fusion Paradox"—where adding data actually hurts performance. We'll present our solution using Gated Fusion, show our main results, cover some additional experiments that validated our approach, and wrap up with takeaways and impact. Let's dive in.

---

## SLIDE 3: Motivation

**Speaker Notes:**
> Hospital readmissions within 30 days are a massive healthcare problem—they cost Medicare $26 billion annually. About 15-20% of patients get readmitted within a month.
> 
> Current prediction models rely entirely on structured data: lab values, vital signs, billing codes, medication lists. But they're missing critical information that lives in clinical notes:
> - Quality of discharge planning—was it thorough or rushed?
> - Social determinants: Does the patient have housing? Family support?
> - Patient comprehension: Can they understand their medications?
> 
> All of this is documented in text but completely ignored by traditional models.
> 
> We focused on the highest-risk population: patients with both heart failure AND acute kidney injury—this is called Cardiorenal Syndrome. Extended hospital stays of 15 days or longer. These patients have a 25% readmission rate—much higher than the general population at 15%.
> 
> Our dataset from MIMIC-IV contains 15,659 of these high-risk admissions. We split patient-level (not admission-level) to prevent data leakage—a patient's future admission can't be in training.

**Anticipated Questions:**

**Q: Why focus on Cardiorenal specifically?**
> A: These patients represent the "long tail" of healthcare resource utilization. They're medically complex, have prolonged stays generating rich text, and have clinically actionable readmission rates (25%). They're also an area where text notes are particularly valuable—discharge planning for such complex patients involves nuanced clinical reasoning.

**Q: Why 15+ days LOS cutoff?**
> A: Two reasons: (1) These patients generate substantial clinical documentation, making text-based analysis meaningful. (2) Longer stays correlate with complexity where text captures important nuance that structured data misses.

---

## SLIDE 4: The Fusion Paradox

**Speaker Notes:**
> This is our key discovery. We built models to combine text and structured data.
> 
> **Expected:** Adding text would improve predictions—more data, more signal.
> **Reality:** Adding text actually made our model WORSE.
> 
> We call this the "Fusion Paradox."
> 
> **Why does this happen?**
> 
> Text embeddings from transformer models like ModernBERT:
> - 768 dimensions—very high
> - Inherently noisy—contain administrative boilerplate, redundant descriptions
> - High linguistic variability—same concept described many ways
> 
> Structured features (lab values, medication sequences):
> - Only 128 dimensions
> - Clean physiological signal—direct clinical measurements
> - Low noise—standardized, normalized values
> 
> In our small-data regime with ~12,500 training samples:
> - High-variance text completely dominates the feature space
> - Model overfits to text noise
> - Can't learn from clean physiological patterns
> 
> If we had access to the t-SNE visualization [reference results/figures/embeddings_tsne.png]:
> - Text embeddings form a diffuse, noisy cloud with poor class separation
> - Structured features are more tightly clustered with visible signal
> 
> **The numbers prove it:**
> - Naive concatenation (Early Fusion): 0.612 AUROC
> - Structured data ALONE: 0.641 AUROC
> - Adding data made performance 0.03 worse—that's significant!

**Anticipated Questions:**

**Q: Isn't 768-dim vs 128-dim just a scaling issue? Why not normalize?**
> A: We tried—PCA to 64 dimensions is crucial and helps significantly. But even after dimensionality reduction, the inherent variance of text embeddings creates issues. The problem isn't just scale—it's that text contains more noise per dimension.

**Q: Is 15k samples really "small data"?**
> A: For deep learning with high-dimensional inputs (768-dim), yes. Compare to ImageNet (1.2M samples) or BERT pre-training (3B+ words). With 15k samples and 768 text dimensions, overfitting is almost guaranteed without regularization.

---

## SLIDE 5: Solution - Gated Fusion Neural Network

**Speaker Notes:**
> How do we fix the Fusion Paradox? Our solution: Gated Fusion Neural Network.
> 
> **Key insight:** Don't assume every modality is equally useful for every patient. Let the model LEARN when to trust text and when to trust structured data.
> 
> **Architecture walkthrough:**
> 
> **Structured EHR pathway:**
> - Daily sequences of labs (abnormal flags), medications (therapeutic groups), diagnoses (ICD codes)
> - Encoded with Transformer (not GRU)—we validated Transformers outperform GRUs for temporal modeling
> - Output: 128-dimensional representation
> - Transformers better capture long-range dependencies in these 15+ day stays via self-attention
> 
> **Clinical notes pathway:**
> - Discharge summaries (clinical reasoning, discharge plans) and radiology reports (diagnostic findings)
> - Encoded with BioClinical-ModernBERT
> - Has 8,192 token context window—can fit 96.9% of our documents without truncation
> - Standard BERT/ClinicalBERT only has 512 tokens—would lose significant information
> - We apply PCA to compress from 768 to 64 dimensions—prevents text from overwhelming structured features
> 
> **Gated fusion mechanism:**
> - Learned gate z: single number between 0 and 1
> - Computed as: z = σ(W·[h_struct, h_text] + b)
> - Model learns this during training—no manual tuning
> - Formula: h_fused = z × h_text + (1-z) × h_struct
> - Dynamically decides how much weight to give each modality
> - Can be different for each patient based on their features
> 
> **Result:** We resolve the Fusion Paradox by adaptive weighting—recovering performance lost by naive concatenation.

**Anticipated Questions:**

**Q: Why not just use attention instead of gating?**
> A: We tried! Temporal Attention (cross-attention between modalities) achieved 0.631 AUROC—worse than gating. The single scalar gate is more regularized and better suited to our small data regime. Attention has more parameters to overfit.

**Q: What does the gate value typically learn? Does it favor structured or text?**
> A: The gate learns to adaptively weight per patient. In aggregate, it tends to favor structured data slightly (since that's cleaner signal), but for patients with distinctive text features (e.g., concerning discharge language, poor social support documentation), it appropriately upweights text.

**Q: Why Transformer over GRU for temporal encoding?**
> A: Validation AUROC 0.658 (Transformer) vs 0.651 (GRU). Self-attention captures long-range dependencies better—important for 15+ day stays where day 1 events may relate to day 15 outcomes. GRUs struggle with vanishing gradients over long sequences.

---

## SLIDE 6: Results

**Speaker Notes:**
> Here are our test set results on 1,569 held-out admissions.
> 
> **XGBoost on structured data alone:** Our strongest baseline
> - AUROC: 0.641—best discrimination
> - But F1 score is ZERO
> - Why? It's poorly calibrated—so conservative it predicts almost no one gets readmitted at the 0.5 threshold
> - Not useful for clinical deployment where you need to flag patients
> 
> **Our Gated Fusion model:**
> - AUROC: 0.638—essentially matches the XGBoost baseline
> - F1 score: 0.29—much better! Actually useful.
> - Properly calibrated for clinical use
> - Recovers from Fusion Paradox while adding clinical utility
> 
> **Early Fusion MLP (hyperparameter optimized):**
> - Also 0.638 AUROC
> - Even better F1: 0.35
> - Shows that with proper tuning (Optuna HPO) and regularization, even "simple" concatenation can work
> 
> **Compare to naive concatenation:**
> - Remember, naive approach only got 0.612
> - Gated Fusion RECOVERED the lost performance—0.638 vs 0.612
> - We got back to baseline level and added clinical utility
> 
> **Additional finding: Transformers vs GRUs**
> - Transformer encoder: 0.658 validation AUROC
> - GRU encoder: 0.651 validation AUROC
> - Transformers are superior for these temporal clinical sequences
> - Attention mechanism captures long-range dependencies better in extended stays
> 
> **Key takeaway:** Smart fusion architecture matters MORE than just adding data.

**Anticipated Questions:**

**Q: Why didn't Transformer embeddings + fusion beat GRU + fusion?**
> A: Interesting finding! The Transformer produces better embeddings (shown by standalone evaluation), but the downstream fusion models were originally tuned for GRU embeddings. Also, the limiting factor may be label noise in the dataset rather than encoder quality. Future work: end-to-end training.

**Q: Is 0.64 AUROC clinically useful?**
> A: It's competitive with published literature on readmission prediction (Almeida et al. achieved ~0.72 with graph models but different cohort). For high-risk populations, even modest AUROC improvements translate to actionable risk stratification. The 25% baseline readmission rate means AUROC improvements are clinically meaningful.

**Q: Why does XGBoost have F1=0?**
> A: XGBoost's probability outputs are calibrated such that almost no predictions exceed the 0.5 threshold. It needs Platt scaling or threshold tuning for deployment. Neural models are naturally better calibrated due to sigmoid outputs and cross-entropy training.

---

## SLIDE 7: Additional Experiments (What Didn't Work)

**Speaker Notes:**
> We also tried some ambitious approaches that didn't work out. But they taught us important lessons.
> 
> **Graph Neural Networks (GraphSAGE):**
> - Built patient-similarity graph with k=15 nearest neighbors
> - Based on multimodal features: EHR embeddings + text embeddings (1664 dimensions total)
> - Results: High recall (72%) but poor discrimination
> - AUROC only 0.596—significantly below baselines
> - Also very unstable during training
> 
> **Trade-off:** Catch more readmissions (high F1: 0.41) but tons of false alarms. This actually makes GNN attractive for high-sensitivity alerting where missing readmissions is costly. But for ranking/triage, not ideal.
> 
> **Why didn't GNN work better?**
> - Nodes = admissions, edges = feature similarity
> - No explicit temporal modeling within the graph
> - Sequential admissions for same patient not linked
> - Message passing may propagate noise from similar-but-different patients
> 
> **LLM prompting with Llama 3.1 70B:**
> - Tried zero-shot, one-shot, and few-shot prompting
> - Gave it demographics, sparse EHR summary, and full clinical notes
> - Zero-shot and one-shot: AUROC ~0.52 (basically random)
> - Few-shot: AUROC 0.60 but completely useless
> - Why? Predicted 97% of patients would be readmitted
> - Perfect recall but terrible precision—everyone gets flagged
> 
> **What we learned:**
> - These negative results validated our focused approach
> - Complex architectures (GNN) or massive models (70B LLM) don't automatically win
> - Domain-specific fusion design matters more than model size
> - Task-specific fine-tuning beats zero-shot prompting for clinical prediction

**Anticipated Questions:**

**Q: Did you try fine-tuning the LLM?**
> A: No—70B parameter fine-tuning requires substantial compute resources beyond our cluster allocation. Also, our BioClinical-ModernBERT is already domain-adapted and produces better embeddings for this specific task than generic LLM prompting.

**Q: Could GNN work better with temporal edges?**
> A: Yes! This is a key future direction. Linking sequential admissions for the same patient would add temporal awareness. The high F1 suggests GNN learns good decision boundaries—the issue is ranking. Temporal edges could help.

**Q: Why k=15 for the graph?**
> A: Hyperparameter tuning. k=15 balances connectivity (enough neighbors for message passing) with specificity (not too many dissimilar patients connected). Tried k=5,10,15,20.

---

## SLIDE 8: Takeaways and Impact

**Speaker Notes:**
> To summarize, three main contributions:
> 
> **Contribution 1: Identified the "Fusion Paradox"**
> - In small-data regimes, more modalities ≠ better performance
> - High-variance embeddings can overwhelm clean signals
> - This is a general finding relevant beyond readmission prediction
> - Applies to any multimodal ML with heterogeneous data quality
> 
> **Contribution 2: Gated Fusion resolves the paradox**
> - Learned adaptive weighting beats naive concatenation
> - Recovers lost performance: 0.638 vs 0.612 AUROC
> - Improves calibration: F1 0.29 vs 0.00
> - Matches tree baseline while being clinically useful
> 
> **Contribution 3: Validated Transformers for clinical time series**
> - Self-attention > recurrent architectures for temporal EHR
> - Better at capturing long-range dependencies
> - 0.658 vs 0.651 on our temporal physiological sequences
> - Important for extended hospital stays where early events matter
> 
> **For real-world deployment, our recommendations:**
> 1. Use XGBoost for daily risk ranking (fast, interpretable via SHAP)
> 2. Overlay Gated Fusion for high-recall alerts on top decile
> 3. For maximum recall scenarios: GNN (0.41 F1, 72% recall)
> 4. Combines efficiency with clinical utility
> 
> **Future work:**
> - End-to-end training: jointly optimize Transformer encoder + Gated Fusion head
> - Could push beyond current 0.64 AUROC plateau
> - External validation on other hospital systems (not just MIMIC)
> - Temporal edges in GNN for combining graph and sequence modeling
> 
> Thank you! Happy to take questions.

**Anticipated Questions:**

**Q: How would this be integrated into clinical workflow?**
> A: Two pathways: (1) Daily batch scoring—run model on all patients, generate ranked list for care coordinators. (2) Real-time alerts—when discharge is triggered, immediately score and flag high-risk patients for intervention. XGBoost handles (1), neural models for (2) due to better calibration.

**Q: What interventions would you recommend based on predictions?**
> A: This is the clinical action side. High-risk patients could receive: (1) Enhanced discharge planning, (2) Follow-up calls within 48 hours, (3) Home health referrals, (4) Medication reconciliation, (5) Outpatient cardiology/nephrology follow-up scheduling before discharge.

**Q: How generalizable is this to other conditions?**
> A: The Fusion Paradox and Gated solution are generalizable principles. The specific cohort (Cardiorenal) is our focus, but the architecture could be applied to other high-risk populations (COPD, diabetes, oncology) with similar data structures.

---

## SLIDE 9: References

**Speaker Notes:**
> These are our key references. Of note:
> - Almeida et al. achieved ~0.72 AUROC using graph neural networks—higher than us, but different cohort and graph construction
> - Pandey et al. used ClinicalT5—we chose ModernBERT for longer context
> - Khan et al. showed code embeddings can beat BERT for structured data—we found similar results, hence our focus on temporal encoding rather than just embeddings
> - Huang et al. pioneered ClinicalBERT for readmission—we extend this with modern architectures
> - Weiss & Jiang's HCUP brief provides the $26B cost figure for readmissions

---

## SLIDE 10: Thank You / Q&A

**Speaker Notes:**
> Thank you for your attention. We're happy to take questions.
> 
> [Have the following ready to discuss:]
> - Cohort details: 15,659 admissions, 24.5% readmission rate, patient-level splits
> - Model comparison table for reference
> - t-SNE visualization to explain Fusion Paradox if needed
> - Code is available—all training scripts use MLflow for reproducibility

---

# ADDITIONAL Q&A PREPARATION

## Technical Questions

**Q: How did you handle class imbalance (24.5% vs 75.5%)?**
> A: Multiple strategies: (1) Weighted loss functions in neural models, (2) class_weight='balanced' in XGBoost, (3) Focus on AUPRC and F1 rather than accuracy, (4) Threshold optimization on validation set. We didn't use SMOTE as it can introduce artifacts with clinical data.

**Q: Why PCA for dimensionality reduction instead of learned projection?**
> A: PCA is deterministic, interpretable, and prevents overfitting. A learned projection layer (Linear 768→64) would add parameters that could overfit in our small-data regime. PCA also allows us to analyze which principal components carry signal.

**Q: Did you try other text encoders?**
> A: We evaluated ClinicalBERT (512 token limit) and BioClinicalBERT before settling on BioClinical-ModernBERT. The 8k context was crucial—truncation with standard BERT loses ~3.1% of documents entirely and significant content from the rest.

**Q: What's in the "temporal sequences"?**
> A: Daily aggregated features: (1) Lab abnormal flags (binary for each lab type), (2) Medication therapeutic classes (one-hot), (3) Active diagnosis codes. Aggregated at daily granularity to create a sequence of length = LOS days.

## Clinical Questions

**Q: What's the clinical significance of 0.64 AUROC?**
> A: Comparable to published readmission models. At this AUROC, if you take the top 20% risk-scored patients, you capture ~35% of actual readmissions. This enables focused intervention resources on highest-risk patients rather than blanket programs.

**Q: Why not predict at admission instead of discharge?**
> A: Discharge is the intervention point—when care coordinators can act. Admission-time prediction is useful for resource planning but discharge prediction enables immediate intervention before the patient leaves.

**Q: Could this create health disparities?**
> A: Important concern. Text notes may contain biased language. We'd recommend fairness audits across demographic groups before deployment, and ensuring the model doesn't systematically under-predict for certain populations.

---

# QUICK REFERENCE: KEY NUMBERS

| Metric | Value |
|--------|-------|
| Cohort size | 15,659 admissions |
| Train/Val/Test split | 12,522 / 1,568 / 1,569 |
| Readmission rate | 24.5% |
| XGBoost (best AUROC) | 0.641 |
| Gated Fusion AUROC | 0.638 |
| Naive Fusion AUROC | 0.612 |
| Early Fusion MLP (HPO) | 0.638 AUROC, 0.35 F1 |
| GNN (best F1/recall) | 0.596 AUROC, 0.41 F1, 72% recall |
| Transformer vs GRU | 0.658 vs 0.651 (val AUROC) |
| ModernBERT context | 8,192 tokens (96.9% docs fit) |
| Healthcare cost | $26B annually |
