"""
Train XGBoost on Integrated Features for Readmission Prediction.

Evaluates the new feature set including:
- Lab trajectory features
- Admission history
- Medications
- Procedures/ICU
- Risk scores (LACE, Charlson)
- Demographics

Usage:
    python src/models/train_xgboost_integrated.py \
        --features_path data/interim/ehr_long_los/integrated_features.csv \
        --output_dir data/interim/ehr_long_los/xgboost_integrated
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    features_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load integrated features and split into train/val/test.
    
    Returns:
        Tuple of (train_df, val_df, test_df, feature_columns)
    """
    logger.info(f"Loading features from {features_path}")
    df = pd.read_csv(features_path)
    
    logger.info(f"Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    # Identify feature columns (exclude ID and target columns)
    exclude_cols = [
        "hadm_id", "subject_id", "readmitted_within_window", "split",
        "gender", "admission_type", "discharge_location"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Split data
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    logger.info(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    return train_df, val_df, test_df, feature_cols


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "readmitted_within_window",
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    early_stopping_rounds: int = 50
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with early stopping.
    """
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.astype(int)
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values.astype(int)
    
    # Handle class imbalance
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    logger.info(f"Training XGBoost with scale_pos_weight={pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    logger.info(f"Best iteration: {model.best_iteration}")
    
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "readmitted_within_window",
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    """
    X = df[feature_cols].values
    y_true = df[target_col].values.astype(int)
    
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "auc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()
    
    return metrics


def get_feature_importance(
    model: xgb.XGBClassifier,
    feature_cols: List[str],
    top_n: int = 30
) -> pd.DataFrame:
    """
    Get top N most important features.
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    return importance_df.head(top_n)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on integrated features")
    parser.add_argument(
        "--features_path",
        type=str,
        default="data/interim/ehr_long_los/integrated_features.csv",
        help="Path to integrated features CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/interim/ehr_long_los/xgboost_integrated",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=500,
        help="Number of boosting rounds"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=6,
        help="Maximum tree depth"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("readmission_prediction")
    
    with mlflow.start_run(run_name="xgboost_integrated_features"):
        # Log parameters
        mlflow.log_params({
            "model_type": "xgboost",
            "feature_set": "integrated",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
        })
        
        # Load data
        train_df, val_df, test_df, feature_cols = load_and_prepare_data(args.features_path)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_val", len(val_df))
        mlflow.log_param("n_test", len(test_df))
        
        # Train model
        logger.info("Training XGBoost model...")
        model = train_xgboost(
            train_df, val_df, feature_cols,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth
        )
        mlflow.log_param("best_iteration", model.best_iteration)
        
        # Evaluate on all splits
        logger.info("\n=== Evaluation Results ===")
        results = {}
        
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            metrics = evaluate_model(model, split_df, feature_cols)
            results[split_name] = metrics
            
            logger.info(f"\n{split_name.upper()}:")
            logger.info(f"  AUC:      {metrics['auc']:.4f}")
            logger.info(f"  AUPRC:    {metrics['auprc']:.4f}")
            logger.info(f"  F1:       {metrics['f1']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:   {metrics['recall']:.4f}")
        
        # Log metrics to MLflow (standard naming convention)
        mlflow.log_metrics({
            "val_auc": float(results["val"]["auc"]),
            "test_auc": float(results["test"]["auc"]),
            "test_auprc": float(results["test"]["auprc"]),
            "test_f1": float(results["test"]["f1"]),
            "test_precision": float(results["test"]["precision"]),
            "test_recall": float(results["test"]["recall"]),
            "test_acc": float(results["test"]["accuracy"]),
        })
        
        # Feature importance
        logger.info("\n=== Top 30 Features ===")
        importance_df = get_feature_importance(model, feature_cols, top_n=30)
        for i, row in importance_df.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save results (convert numpy types to python types for JSON serialization)
        results_path = output_dir / "metrics.json"
        serializable_results = {}
        for split_name, metrics in results.items():
            serializable_results[split_name] = {
                k: float(v) if hasattr(v, 'item') else v 
                for k, v in metrics.items()
            }
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"\nSaved metrics to {results_path}")
        
        importance_path = output_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance to {importance_path}")
        
        # Note: Skipping mlflow artifact logging due to filesystem xattr issues
        # Files are saved locally in output_dir
        
        # Save model
        model_path = output_dir / "model.json"
        model.save_model(str(model_path))
        logger.info(f"Saved model to {model_path}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Test AUC:   {results['test']['auc']:.4f}")
        logger.info(f"Test AUPRC: {results['test']['auprc']:.4f}")
        logger.info(f"Test F1:    {results['test']['f1']:.4f}")
        
        # Compare to previous best
        logger.info("\nComparison to previous results:")
        logger.info("  - Lab Features Only XGBoost: 0.627 AUC")
        logger.info("  - Gated Fusion (Neural):     0.638 AUC")
        logger.info(f"  - Integrated Features XGB:   {results['test']['auc']:.3f} AUC")


if __name__ == "__main__":
    main()
