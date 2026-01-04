"""
Extract Procedure and ICU-related Features.

Features extracted:
- had_icu_stay: Binary flag if patient had ICU stay
- icu_los_hours: Total ICU length of stay in hours
- num_icu_stays: Number of ICU admissions during hospitalization
- days_from_icu_discharge: Days from last ICU discharge to hospital discharge
- had_dialysis: Binary flag for dialysis during stay
- had_mechanical_vent: Binary flag for mechanical ventilation
- vent_duration_hours: Duration of mechanical ventilation
- had_vasopressors: Binary flag for vasopressor use
- had_central_line: Binary flag for central line placement

Usage:
    python src/data/extract_procedure_icu_features.py \
        --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
        --output_path data/interim/ehr_long_los/procedure_icu_features.csv
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


# ItemIDs for key procedures from d_items
PROCEDURE_ITEMIDS = {
    "mechanical_vent": [
        225792,  # Invasive Ventilation
        225794,  # Non-Invasive Ventilation
    ],
    "dialysis": [
        225441,  # Hemodialysis
        225802,  # Dialysis - CRRT
        225803,  # Dialysis - CVVHD
        225805,  # Peritoneal Dialysis
        225809,  # Dialysis - SCUF
    ],
    "central_line": [
        225752,  # Arterial Line
        225753,  # PICC Line
        225754,  # Tunneled (Hickman/Broviac/Port)
        225755,  # Multi Lumen
        225756,  # Single Lumen
        225757,  # Triple Lumen
        225761,  # Midline
    ],
}


def load_icu_stays(mimic_path: Path) -> pd.DataFrame:
    """Load ICU stays from MIMIC-IV."""
    icu_path = mimic_path / "mimiciv/3.1/icu/icustays.csv.gz"
    logger.info(f"Loading ICU stays from {icu_path}")
    
    df = pd.read_csv(
        icu_path,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"]
    )
    
    logger.info(f"Loaded {len(df):,} ICU stays")
    return df


def load_procedure_events(mimic_path: Path) -> pd.DataFrame:
    """Load procedure events from MIMIC-IV ICU module."""
    proc_path = mimic_path / "mimiciv/3.1/icu/procedureevents.csv.gz"
    logger.info(f"Loading procedure events from {proc_path}")
    
    df = pd.read_csv(
        proc_path,
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime", "value"],
        parse_dates=["starttime", "endtime"]
    )
    
    logger.info(f"Loaded {len(df):,} procedure events")
    return df


def load_input_events(mimic_path: Path) -> pd.DataFrame:
    """Load input events for vasopressor detection."""
    input_path = mimic_path / "mimiciv/3.1/icu/inputevents.csv.gz"
    logger.info(f"Loading input events from {input_path}")
    
    # Vasopressor itemids
    vasopressor_itemids = [
        221906,  # Norepinephrine
        221289,  # Epinephrine
        222315,  # Vasopressin
        221662,  # Dopamine
        221749,  # Phenylephrine
        221653,  # Dobutamine
    ]
    
    df = pd.read_csv(
        input_path,
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime", "amount"],
        parse_dates=["starttime", "endtime"]
    )
    
    # Filter to vasopressors
    df = df[df["itemid"].isin(vasopressor_itemids)]
    logger.info(f"Filtered to {len(df):,} vasopressor events")
    
    return df


def compute_procedure_icu_features(
    cohort: pd.DataFrame,
    icu_stays: pd.DataFrame,
    procedure_events: pd.DataFrame,
    input_events: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute procedure and ICU features for each admission.
    
    Args:
        cohort: DataFrame with hadm_id, subject_id, admittime, dischtime
        icu_stays: ICU stay records
        procedure_events: Procedure events from ICU
        input_events: Input events (for vasopressors)
    
    Returns:
        DataFrame with features indexed by hadm_id
    """
    logger.info("Computing procedure/ICU features...")
    
    # Ensure datetime types
    cohort = cohort.copy()
    cohort["admittime"] = pd.to_datetime(cohort["admittime"])
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"])
    
    # Get cohort hadm_ids
    cohort_hadm_ids = set(cohort["hadm_id"].unique())
    
    # Filter to cohort
    icu = icu_stays[icu_stays["hadm_id"].isin(cohort_hadm_ids)].copy()
    procs = procedure_events[procedure_events["hadm_id"].isin(cohort_hadm_ids)].copy()
    inputs = input_events[input_events["hadm_id"].isin(cohort_hadm_ids)].copy()
    
    logger.info(f"Filtered to {len(icu):,} ICU stays, {len(procs):,} procedures, {len(inputs):,} vasopressor events")
    
    results = []
    
    for idx, row in cohort.iterrows():
        hadm_id = row["hadm_id"]
        dischtime = row["dischtime"]
        
        features = {"hadm_id": hadm_id}
        
        # ICU features
        adm_icu = icu[icu["hadm_id"] == hadm_id]
        
        features["had_icu_stay"] = 1 if len(adm_icu) > 0 else 0
        features["num_icu_stays"] = len(adm_icu)
        
        if len(adm_icu) > 0:
            # Total ICU LOS in hours
            features["icu_los_hours"] = adm_icu["los"].sum() * 24  # los is in days
            
            # Days from last ICU discharge to hospital discharge
            last_icu_out = adm_icu["outtime"].max()
            if pd.notna(last_icu_out) and pd.notna(dischtime):
                features["days_from_icu_discharge"] = (dischtime - last_icu_out).days
            else:
                features["days_from_icu_discharge"] = 0
        else:
            features["icu_los_hours"] = 0
            features["days_from_icu_discharge"] = -1  # No ICU stay
        
        # Procedure features
        adm_procs = procs[procs["hadm_id"] == hadm_id]
        
        # Mechanical ventilation
        vent_itemids = PROCEDURE_ITEMIDS["mechanical_vent"]
        vent_procs = adm_procs[adm_procs["itemid"].isin(vent_itemids)]
        features["had_mechanical_vent"] = 1 if len(vent_procs) > 0 else 0
        
        if len(vent_procs) > 0:
            # Compute ventilation duration
            vent_hours = 0
            for _, proc in vent_procs.iterrows():
                if pd.notna(proc["starttime"]) and pd.notna(proc["endtime"]):
                    duration = (proc["endtime"] - proc["starttime"]).total_seconds() / 3600
                    if duration > 0:
                        vent_hours += duration
            features["vent_duration_hours"] = vent_hours
        else:
            features["vent_duration_hours"] = 0
        
        # Dialysis
        dialysis_itemids = PROCEDURE_ITEMIDS["dialysis"]
        dialysis_procs = adm_procs[adm_procs["itemid"].isin(dialysis_itemids)]
        features["had_dialysis"] = 1 if len(dialysis_procs) > 0 else 0
        features["num_dialysis_sessions"] = len(dialysis_procs)
        
        # Central line
        central_itemids = PROCEDURE_ITEMIDS["central_line"]
        central_procs = adm_procs[adm_procs["itemid"].isin(central_itemids)]
        features["had_central_line"] = 1 if len(central_procs) > 0 else 0
        
        # Vasopressors (from input events)
        adm_vaso = inputs[inputs["hadm_id"] == hadm_id]
        features["had_vasopressors"] = 1 if len(adm_vaso) > 0 else 0
        
        if len(adm_vaso) > 0:
            # Compute vasopressor duration
            vaso_hours = 0
            for _, vaso in adm_vaso.iterrows():
                if pd.notna(vaso["starttime"]) and pd.notna(vaso["endtime"]):
                    duration = (vaso["endtime"] - vaso["starttime"]).total_seconds() / 3600
                    if duration > 0:
                        vaso_hours += duration
            features["vasopressor_duration_hours"] = vaso_hours
        else:
            features["vasopressor_duration_hours"] = 0
        
        # Severity proxy: number of high-acuity interventions
        features["n_high_acuity_interventions"] = (
            features["had_mechanical_vent"] +
            features["had_dialysis"] +
            features["had_vasopressors"]
        )
        
        results.append(features)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1:,}/{len(cohort):,} admissions")
    
    result_df = pd.DataFrame(results)
    logger.info(f"Computed features for {len(result_df):,} admissions")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Extract procedure/ICU features")
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
        default="data/interim/ehr_long_los/procedure_icu_features.csv",
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
    
    # Load data
    mimic_path = Path(args.mimic_path)
    icu_stays = load_icu_stays(mimic_path)
    procedure_events = load_procedure_events(mimic_path)
    input_events = load_input_events(mimic_path)
    
    # Compute features
    features = compute_procedure_icu_features(
        cohort, icu_stays, procedure_events, input_events
    )
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")
    
    # Print summary
    logger.info("\n=== Feature Summary ===")
    logger.info(f"Patients with ICU stay: {features['had_icu_stay'].mean()*100:.1f}%")
    logger.info(f"Patients with mechanical vent: {features['had_mechanical_vent'].mean()*100:.1f}%")
    logger.info(f"Patients with dialysis: {features['had_dialysis'].mean()*100:.1f}%")
    logger.info(f"Patients with vasopressors: {features['had_vasopressors'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
