# Daily Report Guide (for Coding/Data Science Agents)

## 0) Output Formats (produce both)

* **Human-readable**: a single Markdown report.
* **Machine-readable**: a compact JSON block at the end (same content, normalized keys).

---

## 1) Report Header

* **Date (use pacific time) & Agent ID(s)**
* **Project / Subtask**
* **Starting Plan of the Day** (bullet the goals as they were at start-of-day)
* **Context Sources Used** (chat threads, tickets, brief dataset notes)

---

## 2) What Was Done (Evidence-backed, DS-oriented)

Summarize actions tied to DS value—not code minutiae.

* **Data**

  * New data pulled/cleaned? Rows/cols; inclusion/exclusion rules; leakage checks.
  * Transformations/feature engineering performed (1-line rationale each).
  * Data quality stats: missingness %, outlier counts, label balance, drift notes.
* **Modeling / Experiments**

  * Models trained/evaluated (model family, size, key hyperparams only).
  * Experiment matrix (what varied): features, sampling, loss, thresholds, seeds.
  * Primary metrics (per split): e.g., AUROC/AUPRC/Recall@k/MCC; include CIs if bootstrapped.
  * Calibration & fairness slices (if applicable): ECE/Brier; subgroup deltas.
* **Analysis / Interpretation**

  * What signals mattered? (e.g., SHAP top features, error taxonomy)
  * Ablations or sanity checks (label shuffle, leak checks, feature drop tests).
* **Artifacts Produced**

  * Notebooks, data subsets, exportable models, plots. Name + 1-line purpose each.

> Keep this section short but *justifiable*: every bullet either improves data, sharpens inference, or validates a decision.

---

## 3) Results Snapshot (tables/figures)

* **Single “main table”** (concise):

| Experiment ID | Data Slice | Model | Key Change | Metric 1 | Metric 2 | Notes |
| ------------- | ---------- | ----- | ---------- | -------- | -------- | ----- |

* **1–2 plots max** (if created): learning curve or PR curve; brief caption = take-away.

---

## 4) Impact Assessment (why this matters)

* **Accuracy / Utility**: e.g., “+0.035 AUPRC on readmissions vs baseline, largest lift for ICU discharges.”
* **Reliability / Robustness**: variance across seeds/slices; overfitting signals.
* **Decision-readiness**: Can this be promoted to the next milestone (e.g., external validation, cost study)?
* **Risk & Ethics**: data gaps, bias risks, PHI handling, any constraints.

---

## 5) Deviations from Plan (and why)

* What changed vs the starting plan, and the rationale (blocked dataset, surprising result, timebox overrun, etc.).
* Trade-offs made (e.g., postponed hyperparam sweep to finish error analysis).

---

## 6) Open Questions & Unknowns

* Top 3 uncertainties that meaningfully affect conclusions (e.g., label policy ambiguity, missing covariate).
* What evidence would resolve each uncertainty (data to fetch, test to run).

---

## 7) Next Steps (ranked, time-boxed)

Give a short, ordered backlog with impact justification.

1. **Immediate (tomorrow)**: e.g., “Re-label 100 false positives from class X; rerun with threshold tuning.”
2. **Short-term (this week)**: e.g., “External validation on Site B; subgroup calibration.”
3. **Nice-to-have**: e.g., “Prototype contrastive pretraining for notes.”

Each item: **owner, expected outcome, success criterion** (e.g., “AUPRC ≥ 0.21 on discharge-to-home slice”).

---

## 8) Reproducibility Notes

* **Entry points**: notebook/script names in run order.
* **Minimal config**: data path alias, seed, key hyperparams.
* **Randomness**: seeds set? any nondeterministic ops acknowledged.
* **Data lineage**: source → filters → features (one-line flow).

*(Keep this light—enough for another agent to rerun without digging through Git history.)*

---

## 9) Appendices (only if created)

* Error buckets (top recurring failure types with 1 example each).
* Feature dictionary changes.
* Slice definitions (e.g., “high-risk discharge = {ICU stay ≥24h, age>70}”).

---


# Matching JSON Block (add at end of report, skip if no model were generated)

```json
{
  "date": "<YYYY-MM-DD>",
  "agents": ["<id1>", "<id2>"],
  "project": "<name>",
  "starting_plan": ["<goal1>", "<goal2>"],
  "data": {
    "sources": ["<...>"],
    "rows_after_filters": null,
    "transforms": ["<...>"],
    "quality": {"missing_pct": null, "label_balance": {"pos": null, "neg": null}}
  },
  "experiments": [
    {
      "id": "E123",
      "slice": "all",
      "model": "xgboost",
      "key_change": "class_weight=2.0",
      "metrics": {"auroc": 0.81, "auprc": 0.19},
      "notes": "improved ICU"
    }
  ],
  "analysis": {
    "top_features": ["<f1>", "<f2>"],
    "sanity_checks": ["label_shuffle_passed"]
  },
  "artifacts": [
    {"name": "model_v4.pkl", "purpose": "threshold tuning baseline"}
  ],
  "impact": {
    "utility": "AUPRC +0.03 vs baseline",
    "robustness": "low seed variance",
    "decision_readiness": "ready for external validation",
    "risks": ["site drift risk"]
  },
  "deviations": ["skipped HPO to finish error taxonomy"],
  "open_questions": [
    {"question": "label policy for readmit transfer?", "evidence_needed": "chart review 50 cases"}
  ],
  "next_steps": [
    {"owner": "agent-A", "action": "slice calibration", "success": "ECE <= 0.05"}
  ],
  "reproducibility": {
    "entry_points": ["01_clean.ipynb", "02_train.py"],
    "config": {"seed": 42, "data_alias": "readmit_v2"},
    "lineage": "mimic_iv → filters(v2) → features(v4)"
  }
}
```

# “How to Write It” Prompt (for the agent)

> “Using today’s chat, notes, and generated files, produce (1) the Markdown EOD report and (2) the JSON block. Focus on **data, experiments, results, and insights**—not commit details. Every claim must be supported by an artifact, metric, or example. Keep the Results to one table and ≤2 figures. Include clear next steps with owners and measurable success criteria. Save it to ”

# Quality Rubric (self-check, 0–2 each; score ≥8/10 = good)

1. **Clarity** (plain language, DS-focused)
2. **Evidence** (metrics/examples back claims)
3. **Actionability** (next steps testable & prioritized)
4. **Rigor** (sanity checks, slice/fairness awareness)
5. **Reproducibility** (someone else can rerun)

# Save Artifacts Instructions

**Save location:** `documents/reports/{date}/{meaningful_name_of_the_report}{date}/`

**Do this exactly:**

1. Create a meaningful slug for the report name from the project/task (lowercase, hyphen-separated, no spaces), e.g., `readmission-risk-ablation`.
2. Use today’s date as `YYYY-MM-DD`, e.g., `2025-10-31`.
3. Build the directory path:
   `documents/reports/{YYYY-MM-DD}/{slug}-{YYYY-MM-DD}/`
   If it already exists, append a numeric suffix `-v2`, `-v3`, … until unique.
4. Save ALL outputs there with these exact filenames:

   * Markdown report: `report.md`
   * Matching JSON: `report.json`
   * Results table (the single “main table”): `results.csv`
   * Up to two figures (if produced):

     * `figure-1.png` (e.g., PR curve)
     * `figure-2.png` (e.g., learning curve)
   * Any other artifacts produced today (models, sample errors, etc.) should be prefixed with today’s date, e.g., `2025-10-31-model.pkl`, `2025-10-31-errors.parquet`.
5. Ensure paths are relative to the repo/workspace root. Create parent dirs as needed.
6. Do not overwrite existing files silently—if a filename exists, add a suffix `-v2` before the extension.
7. At the end, print a short manifest in plain text that lists:

   * `saved_dir` (absolute or repo-relative path)
   * `files` (each filename on its own line with byte size)
   * a one-line tip to open the report (e.g., `Open documents/reports/readmission-risk-ablation-2025-10-31/report.md`)

**Template to follow (fill in tokens):**

```
[ARTIFACT SAVING]
slug = {meaningful_name_slug}
date = {YYYY-MM-DD}
dir  = documents/reports/{slug}-{date}/

Create dir if missing (append -vN if taken).
Write:
- {dir}/report.md
- {dir}/report.json
- {dir}/results.csv
- {dir}/figure-1.png (optional)
- {dir}/figure-2.png (optional)
- {dir}/{date}-*.*
Return MANIFEST with dir + file list + sizes.
[/ARTIFACT SAVING]
```

**Example directory (expected):**

```
documents/
  reports/
    readmission-risk-ablation-2025-10-31/
      report.md
      report.json
      results.csv
      figure-1.png
      figure-2.png
```
