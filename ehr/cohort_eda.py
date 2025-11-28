"""
Utilities for generating cohort exploratory tables aligned with daily reports.

This module reproduces the readmission-focused slices saved under
``data/interim/readmit_analysis``. Each helper loads the canonical cohort
(``data/interim/cohort.csv``) and derives supplemental statistics using the
read-only PhysioNet sources. Outputs mirror the CSV assets referenced in
daily reporting so downstream notebooks can reuse them.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DATA_ROOT = Path("data/interim")
COHORT_PATH = DATA_ROOT / "cohort.csv"
ANALYSIS_DIR = DATA_ROOT / "readmit_analysis"
DIAGNOSES_PATH = Path("physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz")
LABEVENTS_PATH = Path("physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz")
LABITEMS_PATH = Path("physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz")
PRESCRIPTIONS_PATH = Path("physionet.org/files/mimiciv/3.1/hosp/prescriptions.csv.gz")

LOS_BINS = [0, 4, 7, 10, 14, 21, float("inf")]
LOS_LABELS = ["<=4d", "5-7d", "8-10d", "11-14d", "15-21d", ">=22d"]
TODAY = dt.datetime.now().strftime("%Y-%m-%d")


def _ensure_inputs() -> None:
    missing: List[Path] = [
        path for path in [COHORT_PATH, DIAGNOSES_PATH, LABEVENTS_PATH, PRESCRIPTIONS_PATH]
        if not path.exists()
    ]
    if missing:
        joined = "\n  - ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Required data files not found:\n  - {joined}\nEnsure PhysioNet assets are mounted read-only."
        )
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def _load_cohort() -> pd.DataFrame:
    cohort = pd.read_csv(
        COHORT_PATH,
        parse_dates=["admittime", "dischtime"],
    )
    cohort["los_segment"] = pd.cut(
        cohort["length_of_stay_days"],
        bins=LOS_BINS,
        labels=LOS_LABELS,
        right=True,
        include_lowest=True,
    )
    return cohort


def _apply_condition_flags(cohort: pd.DataFrame) -> pd.DataFrame:
    """Tag admissions with cardio-renal / sepsis ICD indicators."""
    condition_prefixes: Dict[str, Iterable[str]] = {
        "acute_kidney_injury": ["N17", "N19", "N185", "584", "586"],
        "heart_failure": ["I50", "I13", "I11", "428"],
        "hyponatremia": ["E871", "2761"],
        "posthemorrhagic_anemia": ["D62", "2851"],
        "sepsis": ["A41", "R652", "R651", "9959", "038"],
    }
    hadm_set = set(cohort["hadm_id"])
    condition_hits: Dict[str, set[int]] = {name: set() for name in condition_prefixes}

    for chunk in pd.read_csv(DIAGNOSES_PATH, chunksize=500_000, usecols=["hadm_id", "icd_code"]):
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        if chunk.empty:
            continue
        icd = chunk["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
        for name, prefixes in condition_prefixes.items():
            mask = np.zeros(len(icd), dtype=bool)
            for prefix in prefixes:
                mask |= icd.str.startswith(prefix).to_numpy()
            if mask.any():
                condition_hits[name].update(chunk.loc[mask, "hadm_id"])

    for name, hadms in condition_hits.items():
        cohort[f"has_{name}"] = cohort["hadm_id"].isin(hadms)

    cohort["has_any_cardiorenal_sepsis"] = cohort[
        [
            "has_acute_kidney_injury",
            "has_heart_failure",
            "has_hyponatremia",
            "has_posthemorrhagic_anemia",
            "has_sepsis",
        ]
    ].any(axis=1)
    cohort["has_aki_and_hf"] = cohort["has_acute_kidney_injury"] & cohort["has_heart_failure"]
    cohort["cardiorenal_sepsis_long"] = (
        (cohort["length_of_stay_days"] >= 15) & cohort["has_any_cardiorenal_sepsis"]
    )
    cohort["long_stay_no_cardiorenal"] = (
        (cohort["length_of_stay_days"] >= 15) & (~cohort["has_any_cardiorenal_sepsis"])
    )
    return cohort


def save_long_los_cohort(cohort: pd.DataFrame) -> None:
    """
    Save the long-stay cohort with cardio-renal/sepsis flags and robust splits.

    Semantics for data/interim/readmit_analysis/long_los_cohort.csv:
    - Filter admissions with length_of_stay_days >= 15 days (canonical "long-stay").
    - Attach ICD-derived condition flags from _apply_condition_flags.
    - Define is_cardiorenal_long using the existing cardiorenal_sepsis_long flag.
    - Generate PATIENT-LEVEL Train/Val/Test splits (80/10/10) stratified by:
      1. Readmission Label (Ever readmitted?)
      2. Cardiorenal Status (Ever cardiorenal?)
      3. Gender
      4. Age (Median split)
    """
    from sklearn.model_selection import train_test_split

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    original_path = ANALYSIS_DIR / "long_los_cohort.csv"
    legacy_path = ANALYSIS_DIR / "long_los_cohort_los14.csv"

    # Preserve the legacy LOS>=14 cohort once for reference if it exists.
    if original_path.exists() and not legacy_path.exists():
        legacy_df = pd.read_csv(original_path)
        legacy_df.to_csv(legacy_path, index=False)

    # Canonical long-stay cohort: LOS >= 15 days.
    long_mask = cohort["length_of_stay_days"] >= 15
    long_cohort = cohort.loc[long_mask].copy()
    long_cohort["is_cardiorenal_long"] = long_cohort["cardiorenal_sepsis_long"].fillna(False)

    # --- Robust Splitting Logic ---
    if "split" in long_cohort.columns:
        long_cohort.drop(columns=["split"], inplace=True)

    print("Generating robust patient-level splits (80/10/10)...")
    
    # 1. Aggregate to Patient Level
    # We define a patient's "phenotype" for stratification based on their history
    # For age, we take the age at their *first* admission in this cohort to bin them
    patient_level = long_cohort.sort_values("admittime").groupby("subject_id").agg(
        has_readmit=("readmitted_within_window", "max"),
        is_cardiorenal=("is_cardiorenal_long", "max"),
        gender=("gender", "first"),
        age=("age_at_admit", "first")
    ).reset_index()

    # Bin Age (Median Split)
    patient_level["age_bin"] = pd.qcut(patient_level["age"], q=2, labels=["Younger", "Older"])

    # 2. Create Composite Stratification Key
    # "1_0_F_Older"
    patient_level["strat_key"] = (
        patient_level["has_readmit"].astype(int).astype(str) + "_" +
        patient_level["is_cardiorenal"].astype(int).astype(str) + "_" +
        patient_level["gender"].astype(str) + "_" +
        patient_level["age_bin"].astype(str)
    )
    
    subjects = patient_level["subject_id"].values
    strat_labels = patient_level["strat_key"].values

    print(f"Stratifying on {patient_level['strat_key'].nunique()} unique patient subgroups.")

    # 3. Split: Train (80%) vs Temp (20%)
    train_subjs, temp_subjs, _, temp_labels = train_test_split(
        subjects, strat_labels,
        test_size=0.20,
        stratify=strat_labels,
        random_state=42
    )

    # 4. Split: Temp (20%) -> Val (10%) + Test (10%)
    val_subjs, test_subjs = train_test_split(
        temp_subjs,
        test_size=0.50,
        stratify=temp_labels,
        random_state=42
    )

    # 5. Map back to admissions
    split_map = {}
    for s in train_subjs: split_map[s] = "train"
    for s in val_subjs: split_map[s] = "val"
    for s in test_subjs: split_map[s] = "test"

    long_cohort["split"] = long_cohort["subject_id"].map(split_map)
    
    # Verification
    print("\n--- Split Verification ---")
    print("Admission Counts:")
    print(long_cohort["split"].value_counts())
    print("\nAdmission Proportions:")
    print(long_cohort["split"].value_counts(normalize=True))
    print("\nReadmission Rate by Split:")
    print(long_cohort.groupby("split")["readmitted_within_window"].mean())

    cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "length_of_stay_days",
        "readmitted_within_window",
        "readmission_gap_in_days",
        "admission_type",
        "discharge_location",
        "age_at_admit",
        "gender",
        "race",
        "is_cardiorenal_long",
        "has_acute_kidney_injury",
        "has_heart_failure",
        "has_hyponatremia",
        "has_posthemorrhagic_anemia",
        "has_sepsis",
        "has_any_cardiorenal_sepsis",
        "has_aki_and_hf",
        "split",  # Included now
    ]

    missing = [c for c in cols if c not in long_cohort.columns]
    if missing:
        raise KeyError(f"Missing expected columns for long_los_cohort: {missing}")

    out = long_cohort[cols].sort_values(["subject_id", "hadm_id"])
    output_path = ANALYSIS_DIR / "long_los_cohort.csv"
    out.to_csv(output_path, index=False)
    print(f"\nSaved cohort with splits to {output_path}")


def save_los_and_cluster_tables(cohort: pd.DataFrame) -> None:
    los_summary = (
        cohort.groupby("los_segment")
        .agg(
            admissions=("hadm_id", "count"),
            readmits=("readmitted_within_window", "sum"),
            readmit_rate=("readmitted_within_window", "mean"),
            cardiorenal_share=("cardiorenal_sepsis_long", "mean"),
        )
        .reset_index()
    )
    los_summary["readmits"] = los_summary["readmits"].astype(int)
    los_summary["readmit_rate_pct"] = (los_summary["readmit_rate"] * 100).round(1)
    los_summary["cardiorenal_share_pct"] = (los_summary["cardiorenal_share"] * 100).round(1)
    los_summary = los_summary[
        ["los_segment", "admissions", "readmits", "readmit_rate_pct", "cardiorenal_share_pct"]
    ]

    cluster_masks = {
        "cardiorenal_long": cohort["cardiorenal_sepsis_long"],
        "long_no_cardiorenal": cohort["long_stay_no_cardiorenal"],
        "aki_hf_combo": cohort["has_aki_and_hf"],
        "sepsis_any": cohort["has_sepsis"],
    }
    cluster_records = []
    for name, mask in cluster_masks.items():
        subset = cohort[mask]
        cluster_records.append(
            {
                "cluster": name,
                "admissions": int(mask.sum()),
                "readmits": int(subset["readmitted_within_window"].sum()),
                "readmit_rate_pct": (
                    round(subset["readmitted_within_window"].mean() * 100, 1)
                    if len(subset) > 0
                    else np.nan
                ),
            }
        )
    cluster_summary = pd.DataFrame(cluster_records)

    los_path = ANALYSIS_DIR / f"{TODAY}_readmit_los_readmit_rates.csv"
    cluster_path = ANALYSIS_DIR / f"{TODAY}_readmit_cluster_readmit_rates.csv"
    flags_path = ANALYSIS_DIR / f"{TODAY}_readmit_condition_flags.csv"

    los_summary.to_csv(los_path, index=False)
    cluster_summary.to_csv(cluster_path, index=False)
    mask_cols = [
        col
        for col in cohort.columns
        if col.startswith("has_") or col.endswith("_sepsis_long") or col.endswith("_cardiorenal")
    ]
    mask_df = cohort[["subject_id", "hadm_id", "length_of_stay_days", "los_segment"] + sorted(set(mask_cols))]
    mask_df.to_csv(flags_path, index=False)


def save_discharge_disposition_tables(cohort: pd.DataFrame) -> None:
    cohort["disposition_clean"] = cohort["discharge_location"].fillna("UNKNOWN").str.lower().str.strip()
    disp_summary = (
        cohort.groupby("disposition_clean")
        .agg(
            admissions=("hadm_id", "count"),
            readmits=("readmitted_within_window", "sum"),
            readmit_rate=("readmitted_within_window", "mean"),
        )
        .reset_index()
    )
    disp_summary = disp_summary[disp_summary["admissions"] >= 100].copy()
    disp_summary["readmits"] = disp_summary["readmits"].astype(int)
    disp_summary["readmit_rate_pct"] = (disp_summary["readmit_rate"] * 100).round(1)
    disp_summary = disp_summary.sort_values("readmit_rate", ascending=False)

    keywords = [
        "follow-up",
        "follow up",
        "clinic appointment",
        "pcp",
        "primary care",
        "see physician",
        "appointment",
        "call your doctor",
    ]
    text = cohort["discharge_note_text"].fillna("").str.lower()
    follow_mask = np.zeros(len(text), dtype=bool)
    for kw in keywords:
        follow_mask |= text.str.contains(kw).to_numpy()
    cohort["has_followup_note"] = follow_mask

    follow_summary = (
        cohort.groupby("has_followup_note")
        .agg(
            admissions=("hadm_id", "count"),
            readmits=("readmitted_within_window", "sum"),
            readmit_rate=("readmitted_within_window", "mean"),
            median_gap=("readmission_gap_in_days", "median"),
        )
        .reset_index()
    )
    follow_summary["readmits"] = follow_summary["readmits"].astype(int)
    follow_summary["readmit_rate_pct"] = (follow_summary["readmit_rate"] * 100).round(1)

    follow_by_los = (
        cohort.groupby(["los_segment", "has_followup_note"])
        .agg(
            admissions=("hadm_id", "count"),
            readmits=("readmitted_within_window", "sum"),
            readmit_rate=("readmitted_within_window", "mean"),
        )
        .reset_index()
    )
    follow_by_los["readmit_rate_pct"] = (follow_by_los["readmit_rate"] * 100).round(1)

    disp_path = ANALYSIS_DIR / f"{TODAY}_discharge_disposition_readmit_rates.csv"
    follow_path = ANALYSIS_DIR / f"{TODAY}_discharge_followup_language_readmit_rates.csv"
    follow_los_path = ANALYSIS_DIR / f"{TODAY}_discharge_followup_language_by_los.csv"

    disp_summary.to_csv(disp_path, index=False)
    follow_summary.to_csv(follow_path, index=False)
    follow_by_los.to_csv(follow_los_path, index=False)


def save_readmission_gap_tables(cohort: pd.DataFrame) -> None:
    positives = cohort[cohort["readmitted_within_window"] == 1].copy()
    positives["readmission_gap_in_days"] = positives["readmission_gap_in_days"].clip(lower=0, upper=30)
    gap_bins = [0, 7, 14, 21, 30]
    gap_labels = ["0-7d", "8-14d", "15-21d", "22-30d"]
    positives["gap_bucket"] = pd.cut(
        positives["readmission_gap_in_days"], bins=gap_bins, labels=gap_labels, right=True, include_lowest=True
    )

    def _share(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        grouped = df.groupby(group_cols).size().reset_index(name="readmit_count")
        totals = grouped.groupby(group_cols[:-1])["readmit_count"].transform("sum")
        grouped["share_pct"] = (grouped["readmit_count"] / totals * 100).round(1)
        return grouped

    los_gap = _share(positives, ["los_segment", "gap_bucket"])
    cardio_gap = _share(positives, ["cardiorenal_sepsis_long", "gap_bucket"])
    aki_gap = _share(positives, ["has_aki_and_hf", "gap_bucket"])

    positives[["los_segment", "gap_bucket", "readmission_gap_in_days"]].to_csv(
        ANALYSIS_DIR / f"{TODAY}_readmit_positive_gaps_raw.csv", index=False
    )
    los_gap.to_csv(ANALYSIS_DIR / f"{TODAY}_readmit_gap_distribution_by_los.csv", index=False)
    cardio_gap.to_csv(ANALYSIS_DIR / f"{TODAY}_readmit_gap_distribution_cardiorenal.csv", index=False)
    aki_gap.to_csv(ANALYSIS_DIR / f"{TODAY}_readmit_gap_distribution_aki_hf.csv", index=False)


def save_lab_summaries(cohort: pd.DataFrame) -> None:
    long_hadm_ids = set(cohort[cohort["length_of_stay_days"] >= 15]["hadm_id"])
    if not long_hadm_ids:
        return

    discharge_map = cohort.set_index("hadm_id")["dischtime"].to_dict()
    item_map = {
        "creatinine": [50912, 52546],  # serum chemistry plus generic creatinine
        "sodium": [50983, 52623],
    }
    records: List[pd.DataFrame] = []

    for chunk in pd.read_csv(
        LABEVENTS_PATH,
        chunksize=1_000_000,
        usecols=["hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
    ):
        chunk = chunk[chunk["hadm_id"].isin(long_hadm_ids)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["itemid"].isin(sum(item_map.values(), []))]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk["dischtime"] = chunk["hadm_id"].map(discharge_map)
        chunk = chunk.dropna(subset=["dischtime"])
        chunk["hours_before_discharge"] = (chunk["dischtime"] - chunk["charttime"]).dt.total_seconds() / 3600.0
        chunk = chunk[(chunk["hours_before_discharge"] >= 0) & (chunk["hours_before_discharge"] <= 48)]
        if chunk.empty:
            continue
        for analyte, itemids in item_map.items():
            sub = chunk[chunk["itemid"].isin(itemids)]
            if sub.empty:
                continue
            sub = sub.sort_values(["hadm_id", "hours_before_discharge"])
            latest = sub.groupby("hadm_id").first().reset_index()
            latest["analyte"] = analyte
            records.append(latest[["hadm_id", "analyte", "valuenum", "valueuom", "hours_before_discharge"]])

    if not records:
        return

    labs = pd.concat(records, ignore_index=True)
    labs = labs.merge(
        cohort[
            [
                "hadm_id",
                "cardiorenal_sepsis_long",
                "has_aki_and_hf",
                "length_of_stay_days",
                "readmitted_within_window",
            ]
        ],
        on="hadm_id",
        how="left",
    )

    summary = (
        labs.groupby(["analyte", "cardiorenal_sepsis_long"])
        .agg(
            hadm_with_measure=("hadm_id", "nunique"),
            median_value=("valuenum", "median"),
            q1=("valuenum", lambda x: np.percentile(x.dropna(), 25) if len(x.dropna()) > 0 else np.nan),
            q3=("valuenum", lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) > 0 else np.nan),
        )
        .reset_index()
    )
    summary["cardiorenal_sepsis_long"] = summary["cardiorenal_sepsis_long"].map(
        {True: "cardiorenal_long", False: "other_long"}
    )

    aki_summary = (
        labs.groupby(["analyte", "has_aki_and_hf"])
        .agg(
            hadm_with_measure=("hadm_id", "nunique"),
            median_value=("valuenum", "median"),
        )
        .reset_index()
    )
    aki_summary["has_aki_and_hf"] = aki_summary["has_aki_and_hf"].map({True: "aki_hf", False: "others"})

    labs.to_csv(ANALYSIS_DIR / f"{TODAY}_lab_last48h_records.csv", index=False)
    summary.to_csv(ANALYSIS_DIR / f"{TODAY}_lab_last48h_summary.csv", index=False)
    aki_summary.to_csv(ANALYSIS_DIR / f"{TODAY}_lab_last48h_aki.csv", index=False)


def save_medication_exposure(cohort: pd.DataFrame) -> None:
    long_mask = cohort["length_of_stay_days"] >= 15
    long_hadm = set(cohort.loc[long_mask, "hadm_id"])
    if not long_hadm:
        return

    long_cardiorenal = set(cohort[cohort["cardiorenal_sepsis_long"]]["hadm_id"])
    long_non_cardio = set(cohort[long_mask & (~cohort["cardiorenal_sepsis_long"])]["hadm_id"])

    meds_of_interest = {
        "loop_diuretics": ["FUROSEMIDE", "BUMETANIDE", "TORSEMIDE"],
        "k_sparing_or_thiazide": ["SPIRONOLACTONE", "CHLORTHALIDONE", "HYDROCHLOROTHIAZIDE"],
        "ace_arb": ["LISINOPRIL", "LOSARTAN", "VALSARTAN", "CAPTOPRIL", "ENALAPRIL"],
    }
    exposure = {key: set() for key in meds_of_interest}
    exposure_non = {key: set() for key in meds_of_interest}
    discharge_map = cohort.set_index("hadm_id")["dischtime"].to_dict()

    for chunk in pd.read_csv(
        PRESCRIPTIONS_PATH,
        chunksize=500_000,
        usecols=["hadm_id", "drug", "starttime", "stoptime"],
    ):
        chunk = chunk[chunk["hadm_id"].isin(long_hadm)]
        if chunk.empty:
            continue
        chunk["drug"] = chunk["drug"].astype(str).str.upper()
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["stoptime"] = pd.to_datetime(chunk["stoptime"], errors="coerce")
        chunk["dischtime"] = chunk["hadm_id"].map(discharge_map)
        chunk = chunk.dropna(subset=["dischtime"])

        for label, keywords in meds_of_interest.items():
            mask = np.zeros(len(chunk), dtype=bool)
            for kw in keywords:
                mask |= chunk["drug"].str.contains(kw).to_numpy()
            sub = chunk[mask]
            if sub.empty:
                continue
            window_start = sub["dischtime"] - pd.Timedelta(hours=48)
            overlaps = sub[
                ((sub["starttime"] <= sub["dischtime"]) & (sub["starttime"] >= window_start))
                | ((sub["stoptime"] >= window_start) & (sub["stoptime"] <= sub["dischtime"]))
                | ((sub["starttime"] <= window_start) & (sub["stoptime"] >= sub["dischtime"]))
            ]
            if overlaps.empty:
                continue
            for hadm in overlaps["hadm_id"].unique():
                if hadm in long_cardiorenal:
                    exposure[label].add(hadm)
                elif hadm in long_non_cardio:
                    exposure_non[label].add(hadm)

    rows = []
    for label in meds_of_interest:
        rows.append(
            {
                "drug_group": label,
                "cardiorenal_long_exposure": len(exposure[label]),
                "long_non_cardio_exposure": len(exposure_non[label]),
                "cardiorenal_exposure_pct": (
                    round(len(exposure[label]) / len(long_cardiorenal) * 100, 1) if long_cardiorenal else np.nan
                ),
                "non_cardio_exposure_pct": (
                    round(len(exposure_non[label]) / len(long_non_cardio) * 100, 1) if long_non_cardio else np.nan
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        ANALYSIS_DIR / f"{TODAY}_medications_near_discharge_summary.csv", index=False
    )


def save_prior_utilization_tables(cohort: pd.DataFrame) -> None:
    cohort = cohort.sort_values(["subject_id", "admittime"]).copy()
    cohort["prior_admissions"] = cohort.groupby("subject_id").cumcount()
    cohort["prior_admissions_bucket"] = pd.cut(
        cohort["prior_admissions"], bins=[-1, 0, 1, 2, 10], labels=["0", "1", "2", "3+"]
    )

    util_summary = (
        cohort.groupby("prior_admissions_bucket")
        .agg(
            admissions=("hadm_id", "count"),
            readmits=("readmitted_within_window", "sum"),
            readmit_rate=("readmitted_within_window", "mean"),
            median_age=("age_at_admit", "median"),
        )
        .reset_index()
    )
    util_summary["readmits"] = util_summary["readmits"].astype(int)
    util_summary["readmit_rate_pct"] = (util_summary["readmit_rate"] * 100).round(1)

    los_util_summary = (
        cohort.groupby(["los_segment", "prior_admissions_bucket"])
        .agg(
            admissions=("hadm_id", "count"),
            readmits=("readmitted_within_window", "sum"),
            readmit_rate=("readmitted_within_window", "mean"),
        )
        .reset_index()
    )
    los_util_summary["readmit_rate_pct"] = (los_util_summary["readmit_rate"] * 100).round(1)

    age_band = pd.cut(cohort["age_at_admit"], bins=[0, 64, 74, 120], labels=["<=64", "65-74", ">=75"])
    cohort["age_band"] = age_band
    age_summary = (
        cohort.groupby(["age_band", "los_segment"])
        .agg(
            admissions=("hadm_id", "count"),
            readmits=("readmitted_within_window", "sum"),
            readmit_rate=("readmitted_within_window", "mean"),
        )
        .reset_index()
    )
    age_summary["readmit_rate_pct"] = (age_summary["readmit_rate"] * 100).round(1)

    util_summary.to_csv(ANALYSIS_DIR / f"{TODAY}_prior_utilization_readmit_rates.csv", index=False)
    los_util_summary.to_csv(ANALYSIS_DIR / f"{TODAY}_prior_utilization_by_los.csv", index=False)
    age_summary.to_csv(ANALYSIS_DIR / f"{TODAY}_ageband_los_readmit_rates.csv", index=False)


def save_social_text_tables(cohort: pd.DataFrame) -> None:
    indicators = {
        "lives_alone": ["lives alone", "living alone", "resides alone"],
        "transport_barrier": ["transportation", "no ride", "no transport", "lack of transportation"],
        "social_support": ["family support", "caregiver", "daughter will", "son will"],
        "language_barrier": ["interpreter", "translated by", "language barrier"],
        "housing_insecure": ["homeless", "no fixed address", "shelter"],
    }
    text = cohort["discharge_note_text"].fillna("").str.lower()
    for key, phrases in indicators.items():
        mask = np.zeros(len(text), dtype=bool)
        for phrase in phrases:
            mask |= text.str.contains(phrase).to_numpy()
        cohort[key] = mask

    summary_rows = []
    for key in indicators:
        subset = cohort[cohort[key]]
        summary_rows.append(
            {
                "indicator": key,
                "admissions": len(subset),
                "readmit_rate_pct": (
                    round(subset["readmitted_within_window"].mean() * 100, 1) if len(subset) else np.nan
                ),
                "short_los_share_pct": (
                    round(subset["los_segment"].isin(["<=4d", "5-7d"]).mean() * 100, 1) if len(subset) else np.nan
                ),
            }
        )

    indicator_los = (
        cohort.melt(
            id_vars=["hadm_id", "los_segment", "readmitted_within_window"],
            value_vars=list(indicators.keys()),
            var_name="indicator",
            value_name="present",
        )
        .query("present")
    )
    los_summary = (
        indicator_los.groupby(["indicator", "los_segment"])
        .agg(
            admissions=("hadm_id", "count"),
            readmits=("readmitted_within_window", "sum"),
            readmit_rate=("readmitted_within_window", "mean"),
        )
        .reset_index()
    )
    los_summary["readmit_rate_pct"] = (los_summary["readmit_rate"] * 100).round(1)

    pd.DataFrame(summary_rows).to_csv(
        ANALYSIS_DIR / f"{TODAY}_social_text_indicators_summary.csv", index=False
    )
    los_summary.to_csv(ANALYSIS_DIR / f"{TODAY}_social_text_indicators_by_los.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cohort EDA tables for readmission analysis.")
    parser.add_argument(
        "--skip-labs",
        action="store_true",
        help="Skip pulling labevents when short on time.",
    )
    parser.add_argument(
        "--skip-meds",
        action="store_true",
        help="Skip prescriptions scan for near-discharge medication exposure.",
    )
    args = parser.parse_args()

    _ensure_inputs()
    cohort = _load_cohort()
    cohort = _apply_condition_flags(cohort)

    # Long-stay cohort for downstream text/EHR pipelines.
    save_long_los_cohort(cohort)

    save_los_and_cluster_tables(cohort)
    save_discharge_disposition_tables(cohort)
    save_readmission_gap_tables(cohort)

    if not args.skip_labs:
        save_lab_summaries(cohort)
    if not args.skip_meds:
        save_medication_exposure(cohort)

    save_prior_utilization_tables(cohort)
    save_social_text_tables(cohort)


if __name__ == "__main__":
    main()
