"""
Integrate All Feature Sets for Readmission Prediction.

Merges the following feature sources:
1. Lab trajectory features (135 features)
2. Admission history features (9 features)
3. Medication trajectory features (~60 features)
4. Procedure/ICU features (13 features)
5. Risk scores (LACE, Charlson CCI, comorbidities) (~25 features)
6. Cohort demographics (age, gender, LOS, etc.)

Usage:
    python src/data/integrate_features.py \
        --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
        --output_path data/interim/ehr_long_los/integrated_features.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_cohort(cohort_path: str) -> pd.DataFrame:
    """Load and prepare cohort data with demographics."""
    logger.info(f"Loading cohort from {cohort_path}")
    cohort = pd.read_csv(cohort_path)
    
    # Filter to cardiorenal long LOS if flag exists
    if "is_cardiorenal_long" in cohort.columns:
        cohort = cohort[cohort["is_cardiorenal_long"] == True].copy()
        logger.info(f"Filtered to {len(cohort):,} cardiorenal long LOS admissions")
    
    # Select demographic features
    demo_cols = [
        "hadm_id", "subject_id", "age_at_admit", "gender", "length_of_stay_days",
        "admission_type", "discharge_location", "readmitted_within_window", "split"
    ]
    
    # Only keep columns that exist
    demo_cols = [c for c in demo_cols if c in cohort.columns]
    cohort = cohort[demo_cols].copy()
    
    # Encode categorical features
    if "gender" in cohort.columns:
        cohort["is_male"] = (cohort["gender"] == "M").astype(int)
    
    if "admission_type" in cohort.columns:
        cohort["is_emergency_admission"] = cohort["admission_type"].str.contains(
            "EMER|URGENT", case=False, na=False
        ).astype(int)
    
    if "discharge_location" in cohort.columns:
        # Risk-associated discharge locations
        cohort["discharged_to_home"] = cohort["discharge_location"].str.contains(
            "HOME", case=False, na=False
        ).astype(int)
        cohort["discharged_to_snf"] = cohort["discharge_location"].str.contains(
            "SKILLED NURSING|SNF", case=False, na=False
        ).astype(int)
        cohort["discharged_to_rehab"] = cohort["discharge_location"].str.contains(
            "REHAB", case=False, na=False
        ).astype(int)
    
    logger.info(f"Loaded {len(cohort):,} admissions with demographics")
    return cohort


def load_feature_file(path: str, feature_name: str) -> Optional[pd.DataFrame]:
    """Load a feature file if it exists."""
    if not Path(path).exists():
        logger.warning(f"{feature_name} not found at {path}")
        return None
    
    logger.info(f"Loading {feature_name} from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)-1} features")
    return df


def integrate_features(
    cohort: pd.DataFrame,
    feature_dir: str,
    lab_features_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Integrate all feature sources into a single DataFrame.
    
    Args:
        cohort: Cohort DataFrame with demographics
        feature_dir: Directory containing feature files
        lab_features_path: Optional override for lab features path
    
    Returns:
        Integrated feature DataFrame
    """
    feature_dir = Path(feature_dir)
    
    # Start with cohort demographics
    integrated = cohort.copy()
    
    # Define feature files to load
    feature_files = {
        "admission_history": feature_dir / "admission_history_features.csv",
        "medication": feature_dir / "medication_features.csv",
        "procedure_icu": feature_dir / "procedure_icu_features.csv",
        "risk_scores": feature_dir / "risk_scores.csv",
    }
    
    # Add lab features path
    if lab_features_path:
        feature_files["lab"] = Path(lab_features_path)
    else:
        # Try both possible locations
        lab_path = feature_dir / "lab_features" / "lab_features_normalized.csv"
        if lab_path.exists():
            feature_files["lab"] = lab_path
        else:
            lab_path = feature_dir / "lab_features_normalized.csv"
            if lab_path.exists():
                feature_files["lab"] = lab_path
    
    # Load and merge each feature set
    for name, path in feature_files.items():
        df = load_feature_file(str(path), name)
        
        if df is not None:
            # Ensure hadm_id is the merge key
            if "hadm_id" not in df.columns:
                logger.warning(f"  {name} missing hadm_id, skipping")
                continue
            
            # Get feature columns (exclude hadm_id and any duplicate columns)
            existing_cols = set(integrated.columns)
            feature_cols = ["hadm_id"] + [
                c for c in df.columns 
                if c != "hadm_id" and c not in existing_cols
            ]
            
            # Merge
            df_to_merge = df[feature_cols]
            integrated = integrated.merge(df_to_merge, on="hadm_id", how="left")
            logger.info(f"  After merge: {integrated.shape}")
    
    # Handle missing values
    logger.info("Handling missing values...")
    
    # Get numeric columns
    numeric_cols = integrated.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fill missing values with 0 for binary indicators, median for continuous
    for col in numeric_cols:
        if col in ["hadm_id", "subject_id"]:
            continue
        
        missing_pct = integrated[col].isna().mean() * 100
        if missing_pct > 0:
            # Binary columns (0/1) - fill with 0
            if integrated[col].dropna().isin([0, 1]).all():
                integrated[col] = integrated[col].fillna(0)
            else:
                # Continuous columns - fill with median
                integrated[col] = integrated[col].fillna(integrated[col].median())
            
            if missing_pct > 5:
                logger.info(f"  {col}: {missing_pct:.1f}% missing, filled")
    
    return integrated


def create_feature_groups(integrated: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Create feature group mapping for ablation studies.
    
    Returns:
        Dictionary mapping group name to list of feature columns
    """
    groups = {}
    
    # Demographics
    groups["demographics"] = [
        c for c in integrated.columns
        if c in ["age_at_admit", "is_male", "length_of_stay_days",
                 "is_emergency_admission", "discharged_to_home",
                 "discharged_to_snf", "discharged_to_rehab"]
    ]
    
    # Admission history
    groups["admission_history"] = [
        c for c in integrated.columns
        if any(x in c for x in ["prior_admission", "days_since", "past_6mo", 
                                "past_1yr", "ed_visits", "first_admission",
                                "prior_30d"])
    ]
    
    # Medications
    groups["medications"] = [
        c for c in integrated.columns
        if any(x in c for x in ["received_", "days_on_", "last_dose_",
                                "dose_change_", "n_rx_", "n_drug_classes",
                                "any_diuretic", "raas_inhibitor"])
    ]
    
    # Procedures/ICU
    groups["procedures_icu"] = [
        c for c in integrated.columns
        if any(x in c for x in ["icu_", "had_icu", "num_icu", "mechanical_vent",
                                "vent_duration", "dialysis", "vasopressor",
                                "central_line", "high_acuity"])
    ]
    
    # Risk scores
    groups["risk_scores"] = [
        c for c in integrated.columns
        if any(x in c for x in ["lace_", "charlson", "cci_", "n_comorbidities",
                                "has_diabetes", "has_cancer", "has_liver"])
    ]
    
    # Lab features
    groups["lab_features"] = [
        c for c in integrated.columns
        if any(x in c for x in ["_first", "_last", "_min", "_max", "_mean",
                                "_std", "_trend", "_delta", "_abnormal",
                                "creatinine", "potassium", "sodium", "bun",
                                "hemoglobin", "wbc", "platelet", "glucose",
                                "albumin", "bilirubin", "inr", "lactate",
                                "troponin", "bnp", "hematocrit", "bicarbonate",
                                "magnesium", "phosphate", "calcium", "ast", "alt"])
    ]
    
    return groups


def main():
    parser = argparse.ArgumentParser(description="Integrate all features")
    parser.add_argument(
        "--cohort_path",
        type=str,
        default="data/interim/readmit_analysis/long_los_cohort.csv",
        help="Path to cohort CSV"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="data/interim/ehr_long_los",
        help="Directory containing feature files"
    )
    parser.add_argument(
        "--lab_features_path",
        type=str,
        default=None,
        help="Optional path to lab features (overrides default)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/interim/ehr_long_los/integrated_features.csv",
        help="Output path for integrated features"
    )
    args = parser.parse_args()
    
    # Load cohort with demographics
    cohort = load_cohort(args.cohort_path)
    
    # Integrate all features
    integrated = integrate_features(
        cohort, 
        args.feature_dir,
        args.lab_features_path
    )
    
    # Create feature groups
    feature_groups = create_feature_groups(integrated)
    
    # Print summary
    logger.info("\n=== Integration Summary ===")
    logger.info(f"Total admissions: {len(integrated):,}")
    logger.info(f"Total features: {len(integrated.columns) - 4}")  # Exclude ID cols
    
    logger.info("\nFeature groups:")
    for group_name, cols in feature_groups.items():
        logger.info(f"  {group_name}: {len(cols)} features")
    
    # Print class balance
    if "readmitted_within_window" in integrated.columns:
        readmit_rate = integrated["readmitted_within_window"].mean() * 100
        logger.info(f"\nReadmission rate: {readmit_rate:.1f}%")
    
    # Print split distribution
    if "split" in integrated.columns:
        logger.info("\nSplit distribution:")
        for split in ["train", "val", "test"]:
            n = (integrated["split"] == split).sum()
            logger.info(f"  {split}: {n:,}")
    
    # Save integrated features
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    integrated.to_csv(output_path, index=False)
    logger.info(f"\nSaved integrated features to {output_path}")
    
    # Save feature groups mapping
    groups_path = output_path.parent / "feature_groups.txt"
    with open(groups_path, "w") as f:
        for group_name, cols in feature_groups.items():
            f.write(f"\n=== {group_name} ({len(cols)} features) ===\n")
            for col in sorted(cols):
                f.write(f"  {col}\n")
    logger.info(f"Saved feature groups to {groups_path}")


if __name__ == "__main__":
    main()
