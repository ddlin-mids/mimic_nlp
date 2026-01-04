"""
Extract Prior Admission History Features.

Features extracted:
1. num_prior_admissions: Total previous admissions for this patient
2. num_prior_30d_readmissions: Count of prior 30-day readmissions
3. days_since_last_discharge: Days since previous hospital discharge (-1 if first)
4. num_admissions_past_6mo: Admissions in 6 months before current
5. num_admissions_past_1yr: Admissions in 1 year before current
6. num_ed_visits_past_6mo: ED visits in 6 months before current admission

Usage:
    python src/data/extract_admission_history.py \
        --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
        --output_path data/interim/ehr_long_los/admission_history_features.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_admissions(mimic_path: Path) -> pd.DataFrame:
    """Load all admissions from MIMIC-IV."""
    admissions_path = mimic_path / "mimiciv/3.1/hosp/admissions.csv.gz"
    logger.info(f"Loading admissions from {admissions_path}")
    
    df = pd.read_csv(
        admissions_path,
        usecols=[
            "subject_id", "hadm_id", "admittime", "dischtime",
            "admission_type", "discharge_location"
        ],
        parse_dates=["admittime", "dischtime"]
    )
    
    # Sort by patient and admission time
    df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} admissions for {df['subject_id'].nunique():,} patients")
    
    return df


def load_ed_stays(mimic_path: Path) -> pd.DataFrame:
    """Load ED stays from MIMIC-IV-ED."""
    ed_path = mimic_path / "mimic-iv-ed/2.2/ed/edstays.csv.gz"
    
    if not ed_path.exists():
        logger.warning(f"ED stays file not found at {ed_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading ED stays from {ed_path}")
    df = pd.read_csv(
        ed_path,
        usecols=["subject_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"]
    )
    
    logger.info(f"Loaded {len(df):,} ED stays")
    return df


def compute_admission_history(
    cohort: pd.DataFrame,
    all_admissions: pd.DataFrame,
    ed_stays: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute admission history features for each admission in cohort.
    
    Args:
        cohort: DataFrame with hadm_id, subject_id, admittime, dischtime
        all_admissions: All admissions from MIMIC-IV
        ed_stays: ED stays from MIMIC-IV-ED
    
    Returns:
        DataFrame with admission history features indexed by hadm_id
    """
    logger.info("Computing admission history features...")
    
    # Ensure datetime types
    cohort = cohort.copy()
    cohort["admittime"] = pd.to_datetime(cohort["admittime"])
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"])
    
    all_admissions = all_admissions.copy()
    all_admissions["admittime"] = pd.to_datetime(all_admissions["admittime"])
    all_admissions["dischtime"] = pd.to_datetime(all_admissions["dischtime"])
    
    # Pre-compute 30-day readmission flag for all admissions
    all_admissions = all_admissions.sort_values(["subject_id", "admittime"])
    
    # Get next admission time for each admission
    all_admissions["next_admittime"] = all_admissions.groupby("subject_id")["admittime"].shift(-1)
    all_admissions["gap_to_next"] = (
        all_admissions["next_admittime"] - all_admissions["dischtime"]
    ).dt.days
    all_admissions["is_30d_readmit"] = (
        (all_admissions["gap_to_next"] >= 0) & 
        (all_admissions["gap_to_next"] <= 30)
    ).astype(int)
    
    results = []
    
    for idx, row in cohort.iterrows():
        hadm_id = row["hadm_id"]
        subject_id = row["subject_id"]
        admittime = row["admittime"]
        
        # Get all prior admissions for this patient
        patient_admissions = all_admissions[
            (all_admissions["subject_id"] == subject_id) &
            (all_admissions["admittime"] < admittime)
        ].copy()
        
        # 1. Total prior admissions
        num_prior_admissions = len(patient_admissions)
        
        # 2. Prior 30-day readmissions
        num_prior_30d_readmissions = patient_admissions["is_30d_readmit"].sum()
        
        # 3. Days since last discharge
        if len(patient_admissions) > 0:
            last_discharge = patient_admissions["dischtime"].max()
            days_since_last = (admittime - last_discharge).days
        else:
            days_since_last = -1  # First admission
        
        # 4. Admissions in past 6 months
        six_months_ago = admittime - pd.Timedelta(days=180)
        num_admissions_6mo = len(
            patient_admissions[patient_admissions["admittime"] >= six_months_ago]
        )
        
        # 5. Admissions in past 1 year
        one_year_ago = admittime - pd.Timedelta(days=365)
        num_admissions_1yr = len(
            patient_admissions[patient_admissions["admittime"] >= one_year_ago]
        )
        
        # 6. ED visits in past 6 months (if ED data available)
        num_ed_visits_6mo = 0
        if len(ed_stays) > 0:
            patient_ed = ed_stays[
                (ed_stays["subject_id"] == subject_id) &
                (ed_stays["intime"] >= six_months_ago) &
                (ed_stays["intime"] < admittime)
            ]
            num_ed_visits_6mo = len(patient_ed)
        
        results.append({
            "hadm_id": hadm_id,
            "num_prior_admissions": num_prior_admissions,
            "num_prior_30d_readmissions": num_prior_30d_readmissions,
            "days_since_last_discharge": days_since_last,
            "num_admissions_past_6mo": num_admissions_6mo,
            "num_admissions_past_1yr": num_admissions_1yr,
            "num_ed_visits_past_6mo": num_ed_visits_6mo,
            "is_first_admission": 1 if num_prior_admissions == 0 else 0,
            "has_prior_30d_readmit": 1 if num_prior_30d_readmissions > 0 else 0,
        })
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1:,}/{len(cohort):,} admissions")
    
    result_df = pd.DataFrame(results)
    logger.info(f"Computed features for {len(result_df):,} admissions")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Extract admission history features")
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
        default="data/interim/ehr_long_los/admission_history_features.csv",
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
    
    # Load all admissions
    mimic_path = Path(args.mimic_path)
    all_admissions = load_admissions(mimic_path)
    
    # Load ED stays
    ed_stays = load_ed_stays(mimic_path)
    
    # Compute features
    features = compute_admission_history(cohort, all_admissions, ed_stays)
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")
    
    # Print summary stats
    logger.info("\n=== Feature Summary ===")
    for col in features.columns:
        if col == "hadm_id":
            continue
        if features[col].dtype in [np.int64, np.float64]:
            logger.info(
                f"{col}: mean={features[col].mean():.2f}, "
                f"median={features[col].median():.1f}, "
                f"max={features[col].max():.0f}"
            )


if __name__ == "__main__":
    main()
