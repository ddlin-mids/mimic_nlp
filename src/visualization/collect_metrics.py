import mlflow
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Define relevant experiments
EXPERIMENTS = [
    "mimic_cardiorenal_ablation",
    "mimic_cardiorenal_readmission",      # HPO XGBoost
    "mimic_cardiorenal_readmission_pytorch", # Early Fusion
    "mimic_cardiorenal_readmission_pytorch_hpo", # Early Fusion HPO
    "mimic_cardiorenal_late_fusion",       # Late Fusion
    "mimic_cardiorenal_attention_fusion",  # Attention Fusion
    "mimic_cardiorenal_gated_fusion",      # Plan 2: Gated Fusion
    "mimic_cardiorenal_gated_hpo",         # Plan 2: Gated Fusion HPO
    "mimic_cardiorenal_temporal_attention" # Plan 1: Temporal Attention
]

def get_best_run_metrics(experiment_name):
    try:
        logger.info(f"Checking experiment: {experiment_name}")
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            logger.warning(f"Experiment {experiment_name} not found.")
            return None
        
        # Search for top 5 runs by AUC
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.test_auc DESC", "attribute.start_time DESC"],
            max_results=5
        )
        
        if runs.empty:
            logger.warning(f"No runs found for {experiment_name}")
            return None
            
        # Prefer run with F1 score if available, otherwise take top
        best_run = runs.iloc[0]
        for _, run in runs.iterrows():
             # Check if test_f1 is not NaN
             if "metrics.test_f1" in run and not pd.isna(run["metrics.test_f1"]):
                 best_run = run
                 break
        
        # Extract metrics
        metrics = {
            "Experiment": experiment_name,
            "Run Name": best_run.get("tags.mlflow.runName", "Unknown"),
            "Test AUC": best_run.get("metrics.test_auc"),
            "Test AUPRC": best_run.get("metrics.test_auprc"),
            "Test F1": best_run.get("metrics.test_f1"),
            "Test Acc": best_run.get("metrics.test_accuracy")
        }
        
        # Extract relevant params
        if "ablation" in experiment_name:
            metrics["Modality"] = best_run.get("tags.mlflow.runName") # Ablation puts modality in run name
        else:
            metrics["Modality"] = "Fusion"
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error processing {experiment_name}: {e}")
        return None

def main():
    results = []
    logger.info("Scraping MLflow metrics...")
    
    # 1. Ablation Runs (Get all of them, not just best)
    exp = mlflow.get_experiment_by_name("mimic_cardiorenal_ablation")
    if exp:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        for _, run in runs.iterrows():
            if run.get("metrics.test_auc") is not None:
                results.append({
                    "Experiment": "Ablation",
                    "Model": "XGBoost",
                    "Modality": run.get("tags.mlflow.runName"),
                    "Test AUC": run.get("metrics.test_auc"),
                    "Test AUPRC": run.get("metrics.test_auprc"),
                    "Test F1": run.get("metrics.test_f1", 0.0)
                })

    # 2. HPO Best (XGBoost)
    xgb_metrics = get_best_run_metrics("mimic_cardiorenal_readmission")
    if xgb_metrics:
        xgb_metrics["Model"] = "XGBoost (Optuna)"
        results.append(xgb_metrics)

    # 3. Early Fusion (Neural)
    ef_metrics = get_best_run_metrics("mimic_cardiorenal_readmission_pytorch")
    if ef_metrics:
        ef_metrics["Model"] = "Early Fusion (MLP)"
        results.append(ef_metrics)

    # 3b. Early Fusion (Neural, HPO)
    ef_hpo_metrics = get_best_run_metrics("mimic_cardiorenal_readmission_pytorch_hpo")
    if ef_hpo_metrics:
        ef_hpo_metrics["Model"] = "Early Fusion (MLP HPO)"
        results.append(ef_hpo_metrics)

    # 4. Late Fusion (Neural)
    lf_metrics = get_best_run_metrics("mimic_cardiorenal_late_fusion")
    if lf_metrics:
        lf_metrics["Model"] = "Late Fusion (2-Tower)"
        results.append(lf_metrics)

    # 5. Attention Fusion
    attn_metrics = get_best_run_metrics("mimic_cardiorenal_attention_fusion")
    if attn_metrics:
        attn_metrics["Model"] = "Attention Fusion"
        results.append(attn_metrics)

    # 6. Gated Fusion (Plan 2)
    gate_metrics = get_best_run_metrics("mimic_cardiorenal_gated_fusion")
    if gate_metrics:
        gate_metrics["Model"] = "Gated Fusion"
        results.append(gate_metrics)

    # 6b. Gated Fusion HPO
    gate_hpo = get_best_run_metrics("mimic_cardiorenal_gated_hpo")
    if gate_hpo:
        gate_hpo["Model"] = "Gated Fusion (Optuna)"
        results.append(gate_hpo)

    # 7. Temporal Attention (Plan 1)
    temp_metrics = get_best_run_metrics("mimic_cardiorenal_temporal_attention")
    if temp_metrics:
        temp_metrics["Model"] = "Temporal Attention"
        results.append(temp_metrics)

    # Format
    df = pd.DataFrame(results)
    # Clean up columns
    cols = ["Model", "Modality", "Test AUC", "Test AUPRC", "Test F1"]
    final_df = df[cols].sort_values("Test AUC", ascending=False).round(4)
    
    print("\n--- Final Results Table ---")
    print(final_df.to_markdown(index=False))
    
    # Save
    out_path = Path("results/cohort_results/final_model_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
