# Daily Report: Graph Prep for GNN (Cardiorenal Readmission)

**Date:** 2025-12-06 (PT)  
**Agent:** Codex Agent  
**Project:** Cardiorenal Readmission Prediction (MIMIC-IV)  
**Starting Plan:** 1) Align graph preprocessing with PyG/STGNN expectations. 2) Produce feature-complete patient graph (x, edge_index, edge_attr, mapping). 3) Ready GPU GraphSAGE training job.  
**Context Sources:** chat thread, `ehr/preprocess_graph.py`, `scripts/slurm/preprocess_graph.sbatch`, log `logs/slurm/prep_graph_46985966.log`.

---

## 2. What Was Done

**Data / Features**
- Aggregated daily EHR one-hot features to admission level for 15,659 long-LOS admissions (6,260 TF-IDF feature dims) using `ehr/preprocess_graph.py`.
- Applied TF-IDF then KNN (k=15, cosine) to build patient similarity graph; symmetrized edges and dropped self-loops to avoid degenerate self-messages.
- Saved dense TF-IDF node features into PyG `data.x` and persisted node mapping (`graph_node_map.csv`) keyed by `node_idx, hadm_id, node_name`.

**Modeling / Experiments**
- No training today; prepared graph artifacts for GraphSAGE.

**Artifacts**
- `data/interim/ehr_long_los/graph/graph_pyg.pt` — PyG graph with TF-IDF node features, symmetric edges.
- `data/interim/ehr_long_los/graph/graph_node_map.csv` — stable node ordering for label/embedding alignment.
- Job log: `logs/slurm/prep_graph_46985966.log`.

---

## 3. Results Snapshot

| Experiment ID | Data Slice | Model | Key Change | Metric 1 | Metric 2 | Notes |
| ------------- | ---------- | ----- | ---------- | -------- | -------- | ----- |
| graph-build-gnn | all | graph prep (PyG) | TF-IDF node features + sym KNN k=15 | nodes=15659 | feats=6260 | Self-loops removed; mapping saved |

---

## 4. Impact Assessment

- Graph now includes node features and stable ordering, enabling PyG GraphSAGE training without feature/mapping hacks.
- Symmetrized, self-loop-free edges reduce noise from trivial neighbors and align with STGNN reference expectations.
- Decision-readiness: graph artifacts are ready; need environment check before launching GPU training.

---

## 5. Deviations from Plan

- Could not inspect `graph_pyg.pt` tensors via `uv run` due to pyenv shim/uv cache permission issues; relied on logged shapes instead.

---

## 6. Open Questions & Unknowns

- Can `uv run` be unblocked (pyenv shim/cache) so we can sanity-check graph tensor shapes and edge counts before training?
- Do we want to attach Transformer embeddings as `data.x` instead of TF-IDF for the first GraphSAGE run?

---

## 7. Next Steps

1. **Immediate (tomorrow, Codex):** Fix `uv run`/pyenv shim issue and verify `graph_pyg.pt` shapes (expect 15,659 nodes; symmetric edges) — success: torch load succeeds and edge count logged.
2. **Short-term (Codex):** Submit `scripts/slurm/train_gnn.sbatch` on free-gpu (A30) using new graph + Transformer embeddings — success: run completes with test AUROC/AUPRC logged.
3. **Nice-to-have (Codex):** Swap TF-IDF node features with Transformer embeddings for an ablation — success: graph regenerated and documented with matching node map.

---

## 8. Reproducibility Notes

- Entry points: `scripts/slurm/preprocess_graph.sbatch` → `ehr/preprocess_graph.py`.
- Config: k=15, cosine distance, TF-IDF features (6,260 dims), symmetric edges, self-loops dropped; inputs `ehr_preprocessed_all_one_hot.pkl`, `long_los_cohort.csv`, `structured_ehr_mapping.csv`.
- Lineage: long_los cohort → daily EHR one-hot → TF-IDF → KNN graph → PyG graph + mapping.

---

```json
{
  "date": "2025-12-06",
  "agents": ["Codex Agent"],
  "project": "Cardiorenal Readmission Prediction (MIMIC-IV)",
  "starting_plan": [
    "Align graph preprocessing with PyG/STGNN expectations",
    "Produce feature-complete patient graph (x, edge_index, edge_attr, mapping)",
    "Ready GPU GraphSAGE training job"
  ],
  "data": {
    "sources": [
      "data/interim/ehr_long_los/ehr_preprocessed_all_one_hot.pkl",
      "data/interim/readmit_analysis/long_los_cohort.csv",
      "data/interim/ehr_long_los/embeddings/structured_ehr_mapping.csv"
    ],
    "rows_after_filters": 15659,
    "transforms": [
      "TF-IDF (6260 dims)",
      "KNN k=15 cosine",
      "Symmetrize edges and drop self-loops"
    ],
    "quality": {
      "missing_pct": null,
      "label_balance": {"pos": null, "neg": null}
    }
  },
  "experiments": [
    {
      "id": "graph-build-gnn",
      "slice": "all",
      "model": "graph prep (PyG)",
      "key_change": "TF-IDF node features + sym KNN k=15",
      "metrics": {"nodes": 15659, "features": 6260},
      "notes": "Self-loops removed; mapping saved"
    }
  ],
  "analysis": {
    "top_features": [],
    "sanity_checks": [
      "Aligned node order to structured_ehr_mapping.csv",
      "Self-loops removed via symmetrization"
    ]
  },
  "artifacts": [
    {"name": "data/interim/ehr_long_los/graph/graph_pyg.pt", "purpose": "PyG graph with TF-IDF node features"},
    {"name": "data/interim/ehr_long_los/graph/graph_node_map.csv", "purpose": "node_idx to hadm_id/node_name mapping"},
    {"name": "logs/slurm/prep_graph_46985966.log", "purpose": "run log"}
  ],
  "impact": {
    "utility": "Graph is feature-complete for GNN training",
    "robustness": "Symmetrized edges and removed self-loops to reduce noise",
    "decision_readiness": "Ready to launch GraphSAGE once environment check passes",
    "risks": ["uv run/pyenv shim blocking torch inspection of graph_pyg.pt"]
  },
  "deviations": [
    "Could not load graph via uv run because of pyenv shim/cache permissions"
  ],
  "open_questions": [
    {"question": "Can uv run be unblocked to inspect graph tensors?", "evidence_needed": "Successful torch load of graph_pyg.pt via uv run"},
    {"question": "Should Transformer embeddings replace TF-IDF for initial GraphSAGE run?", "evidence_needed": "Ablation comparing TF-IDF vs Transformer node features"}
  ],
  "next_steps": [
    {"owner": "Codex", "action": "Fix uv run/pyenv shim and log graph tensor shapes", "success": "torch load of graph_pyg.pt succeeds and edge count recorded"},
    {"owner": "Codex", "action": "Submit train_gnn.sbatch on free-gpu with new graph", "success": "Run completes with test AUROC/AUPRC logged"},
    {"owner": "Codex", "action": "Regenerate graph with Transformer node features (ablation)", "success": "Graph artifacts saved with mapping and documented"}
  ],
  "reproducibility": {
    "entry_points": [
      "scripts/slurm/preprocess_graph.sbatch",
      "ehr/preprocess_graph.py"
    ],
    "config": {
      "k": 15,
      "distance": "cosine",
      "features": "TF-IDF 6260 dims",
      "edge_symmetry": "undirected, self-loops removed"
    },
    "lineage": "long_los cohort -> daily EHR one-hot -> TF-IDF -> KNN graph -> PyG graph + mapping"
  }
}
```
