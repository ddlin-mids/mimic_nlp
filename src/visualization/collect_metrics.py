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
    "mimic_cardiorenal_late_fusion"       # Late Fusion
]

def get_best_run_metrics(experiment_name):
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            logger.warning(f"Experiment {experiment_name} not found.")
            return None
        
        # Search for runs, sort by test_auc desc
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.test_auc DESC"]
        )
        
        if runs.empty:
            logger.warning(f"No runs found for {experiment_name}")
            return None
            
        best_run = runs.iloc[0]
        
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

    # 4. Late Fusion (Neural)
    lf_metrics = get_best_run_metrics("mimic_cardiorenal_late_fusion")
    if lf_metrics:
        lf_metrics["Model"] = "Late Fusion (2-Tower)"
        results.append(lf_metrics)

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
