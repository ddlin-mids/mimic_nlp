"""
Extract Medication Trajectory Features for Cardiorenal Patients.

Features extracted per drug class:
- received_<class>: Binary flag if received during stay
- days_on_<class>: Number of days on medication
- last_dose_<class>: Last dose in mg before discharge (if numeric)
- dose_change_<class>: 1=increased, 0=stable, -1=decreased

Drug classes relevant for cardiorenal:
- Loop diuretics (Furosemide, Bumetanide, Torsemide)
- Thiazide diuretics (HCTZ, Metolazone, Chlorthalidone)
- ACE inhibitors (Lisinopril, Enalapril, Ramipril, etc.)
- ARBs (Losartan, Valsartan, Irbesartan, etc.)
- Beta-blockers (Metoprolol, Carvedilol, Bisoprolol, Atenolol)
- Anticoagulants (Heparin, Warfarin, Enoxaparin, Apixaban, Rivaroxaban)
- Insulin (all types)
- Vasopressors (Norepinephrine, Vasopressin, Epinephrine, Dopamine)
- Potassium supplements

Usage:
    python src/data/extract_medication_features.py \
        --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
        --output_path data/interim/ehr_long_los/medication_features.csv
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define drug classes with regex patterns
DRUG_CLASSES = {
    "loop_diuretic": [
        r"furosemide", r"lasix", r"bumetanide", r"bumex", r"torsemide"
    ],
    "thiazide_diuretic": [
        r"hydrochlorothiazide", r"hctz", r"metolazone", r"zaroxolyn",
        r"chlorthalidone", r"indapamide"
    ],
    "ace_inhibitor": [
        r"lisinopril", r"enalapril", r"ramipril", r"captopril",
        r"benazepril", r"fosinopril", r"quinapril", r"perindopril"
    ],
    "arb": [
        r"losartan", r"valsartan", r"irbesartan", r"olmesartan",
        r"telmisartan", r"candesartan", r"azilsartan"
    ],
    "beta_blocker": [
        r"metoprolol", r"carvedilol", r"bisoprolol", r"atenolol",
        r"propranolol", r"labetalol", r"nebivolol"
    ],
    "anticoagulant": [
        r"heparin", r"warfarin", r"coumadin", r"enoxaparin", r"lovenox",
        r"apixaban", r"eliquis", r"rivaroxaban", r"xarelto",
        r"dabigatran", r"pradaxa", r"fondaparinux"
    ],
    "insulin": [
        r"insulin", r"lantus", r"levemir", r"novolog", r"humalog",
        r"glargine", r"lispro", r"aspart", r"detemir"
    ],
    "vasopressor": [
        r"norepinephrine", r"levophed", r"vasopressin", r"epinephrine",
        r"dopamine", r"phenylephrine", r"neosynephrine", r"dobutamine"
    ],
    "potassium_supplement": [
        r"potassium chloride", r"k-dur", r"klor-con", r"micro-k"
    ],
    "statin": [
        r"atorvastatin", r"lipitor", r"simvastatin", r"zocor",
        r"rosuvastatin", r"crestor", r"pravastatin"
    ],
    "antiarrhythmic": [
        r"amiodarone", r"digoxin", r"diltiazem", r"verapamil",
        r"flecainide", r"sotalol"
    ],
    "nitrate": [
        r"nitroglycerin", r"isosorbide", r"nitropaste"
    ],
}


def load_prescriptions(mimic_path: Path) -> pd.DataFrame:
    """Load prescription data from MIMIC-IV."""
    rx_path = mimic_path / "mimiciv/3.1/hosp/prescriptions.csv.gz"
    logger.info(f"Loading prescriptions from {rx_path}")
    
    df = pd.read_csv(
        rx_path,
        usecols=[
            "subject_id", "hadm_id", "drug", "drug_type",
            "starttime", "stoptime", "dose_val_rx", "dose_unit_rx", "route"
        ],
        parse_dates=["starttime", "stoptime"],
        dtype={"dose_val_rx": str}  # Keep as string initially
    )
    
    logger.info(f"Loaded {len(df):,} prescription records")
    return df


def classify_drug(drug_name: str) -> str:
    """Classify a drug into a drug class based on name matching."""
    if pd.isna(drug_name):
        return "unknown"
    
    drug_lower = drug_name.lower()
    
    for drug_class, patterns in DRUG_CLASSES.items():
        for pattern in patterns:
            if re.search(pattern, drug_lower):
                return drug_class
    
    return "other"


def parse_dose(dose_str: str) -> float:
    """Parse dose string to numeric value."""
    if pd.isna(dose_str):
        return np.nan
    
    try:
        # Try direct conversion
        return float(dose_str)
    except (ValueError, TypeError):
        # Try extracting number
        match = re.search(r"[\d.]+", str(dose_str))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return np.nan
        return np.nan


def compute_medication_features(
    cohort: pd.DataFrame,
    prescriptions: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute medication features for each admission in cohort.
    
    Args:
        cohort: DataFrame with hadm_id, subject_id, admittime, dischtime
        prescriptions: Prescription records from MIMIC-IV
    
    Returns:
        DataFrame with medication features indexed by hadm_id
    """
    logger.info("Computing medication features...")
    
    # Ensure datetime types
    cohort = cohort.copy()
    cohort["admittime"] = pd.to_datetime(cohort["admittime"])
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"])
    
    # Get cohort hadm_ids for filtering
    cohort_hadm_ids = set(cohort["hadm_id"].unique())
    
    # Filter prescriptions to cohort admissions
    rx = prescriptions[prescriptions["hadm_id"].isin(cohort_hadm_ids)].copy()
    logger.info(f"Filtered to {len(rx):,} prescriptions for cohort")
    
    # Classify drugs
    logger.info("Classifying drugs...")
    rx["drug_class"] = rx["drug"].apply(classify_drug)
    
    # Parse doses
    rx["dose_numeric"] = rx["dose_val_rx"].apply(parse_dose)
    
    # Get unique drug classes (excluding 'other' and 'unknown')
    drug_classes = [c for c in DRUG_CLASSES.keys()]
    
    results = []
    
    for idx, row in cohort.iterrows():
        hadm_id = row["hadm_id"]
        admittime = row["admittime"]
        dischtime = row["dischtime"]
        
        # Get prescriptions for this admission
        adm_rx = rx[rx["hadm_id"] == hadm_id].copy()
        
        features = {"hadm_id": hadm_id}
        
        for drug_class in drug_classes:
            class_rx = adm_rx[adm_rx["drug_class"] == drug_class]
            
            # Binary: received during stay
            received = 1 if len(class_rx) > 0 else 0
            features[f"received_{drug_class}"] = received
            
            if received:
                # Days on medication (approximate from prescription records)
                if class_rx["starttime"].notna().any() and class_rx["stoptime"].notna().any():
                    total_days = 0
                    for _, rx_row in class_rx.iterrows():
                        if pd.notna(rx_row["starttime"]) and pd.notna(rx_row["stoptime"]):
                            duration = (rx_row["stoptime"] - rx_row["starttime"]).days
                            if duration > 0:
                                total_days += duration
                    features[f"days_on_{drug_class}"] = max(1, total_days)
                else:
                    features[f"days_on_{drug_class}"] = len(class_rx["starttime"].dt.date.unique())
                
                # Last dose (if available)
                valid_doses = class_rx[class_rx["dose_numeric"].notna()]
                if len(valid_doses) > 0:
                    # Get last prescription by time
                    last_rx = valid_doses.sort_values("starttime").iloc[-1]
                    features[f"last_dose_{drug_class}"] = last_rx["dose_numeric"]
                else:
                    features[f"last_dose_{drug_class}"] = np.nan
                
                # Dose change (increase/decrease/stable)
                if len(valid_doses) >= 2:
                    sorted_doses = valid_doses.sort_values("starttime")
                    first_dose = sorted_doses.iloc[0]["dose_numeric"]
                    last_dose = sorted_doses.iloc[-1]["dose_numeric"]
                    if last_dose > first_dose * 1.1:  # >10% increase
                        features[f"dose_change_{drug_class}"] = 1
                    elif last_dose < first_dose * 0.9:  # >10% decrease
                        features[f"dose_change_{drug_class}"] = -1
                    else:
                        features[f"dose_change_{drug_class}"] = 0
                else:
                    features[f"dose_change_{drug_class}"] = 0
                
                # Number of prescriptions
                features[f"n_rx_{drug_class}"] = len(class_rx)
                
            else:
                features[f"days_on_{drug_class}"] = 0
                features[f"last_dose_{drug_class}"] = 0
                features[f"dose_change_{drug_class}"] = 0
                features[f"n_rx_{drug_class}"] = 0
        
        # Aggregate features
        features["n_drug_classes"] = sum(
            features.get(f"received_{dc}", 0) for dc in drug_classes
        )
        features["received_any_diuretic"] = max(
            features.get("received_loop_diuretic", 0),
            features.get("received_thiazide_diuretic", 0)
        )
        features["received_raas_inhibitor"] = max(
            features.get("received_ace_inhibitor", 0),
            features.get("received_arb", 0)
        )
        
        results.append(features)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1:,}/{len(cohort):,} admissions")
    
    result_df = pd.DataFrame(results)
    logger.info(f"Computed features for {len(result_df):,} admissions")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Extract medication features")
    parser.add_argument(
        "--cohort_path",
        type=str,
        default="data/interim/readmit_analysis/long_los_cohort.csv",
        help="Path to cohort CSV"
    )
    parser.add_argument(
        "--mimic_path",
        type=str,
        default="physionet.org/files",
        help="Path to MIMIC-IV data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/interim/ehr_long_los/medication_features.csv",
        help="Output path for features"
    )
    args = parser.parse_args()
    
    # Load cohort
    logger.info(f"Loading cohort from {args.cohort_path}")
    cohort = pd.read_csv(args.cohort_path)
    
    # Filter to cardiorenal long LOS if flag exists
    if "is_cardiorenal_long" in cohort.columns:
        cohort = cohort[cohort["is_cardiorenal_long"] == True].copy()
        logger.info(f"Filtered to {len(cohort):,} cardiorenal long LOS admissions")
    
    # Load prescriptions
    mimic_path = Path(args.mimic_path)
    prescriptions = load_prescriptions(mimic_path)
    
    # Compute features
    features = compute_medication_features(cohort, prescriptions)
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")
    
    # Print summary stats
    logger.info("\n=== Feature Summary ===")
    for drug_class in DRUG_CLASSES.keys():
        col = f"received_{drug_class}"
        if col in features.columns:
            pct = features[col].mean() * 100
            logger.info(f"{drug_class}: {pct:.1f}% of patients received")


if __name__ == "__main__":
    main()
