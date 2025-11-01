"""Utilities for constructing the MIMIC-IV readmission cohort.

The functions here mirror the logic implemented in ``ehr/get_mimic_cohort.py``
so they can be imported by other modules, unit tests, or notebooks.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Admission filtering tuned to match readmit-stgnn defaults.
ADMISSION_TYPES_EXCLUDED = {
    "AMBULATORY OBSERVATION",
    "EU OBSERVATION",
    "DIRECT OBSERVATION",
}
DISCHARGE_LOCATIONS_EXCLUDED = {
    "ACUTE HOSPITAL",
    "HEALTHCARE FACILITY",
    "AGAINST ADVICE",
    "DIED",
    "HOSPICE",
    "HOSPICE HOME",
    "HOSPICE MEDICAL FACILITY",
}
READMIT_WINDOW_DAYS = 30
MIN_LOS_DAYS = 2


@dataclass
class CohortStats:
    """Summary metrics describing the built cohort."""

    total_admissions: int
    filtered_admissions: int
    unique_patients: int
    pos_ratio_overall: float
    pos_ratio_train: float
    pos_ratio_val: float
    pos_ratio_test: float
    min_readmit_gap: float
    max_readmit_gap: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class CohortArtifacts:
    """Return type for :func:`build_mimic_cohort`."""

    cohort_frame: pd.DataFrame
    split_index: Dict[str, List[int]]
    stats: CohortStats


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_admissions(mimic_root: Path) -> pd.DataFrame:
    hosp_dir = mimic_root / "hosp"
    admissions = pd.read_csv(
        hosp_dir / "admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    admissions["discharge_location"] = admissions["discharge_location"].fillna("UNKNOWN")
    return admissions


def load_patients(mimic_root: Path) -> pd.DataFrame:
    hosp_dir = mimic_root / "hosp"
    patients = pd.read_csv(hosp_dir / "patients.csv.gz")
    return patients[["subject_id", "anchor_year", "anchor_age", "gender"]]


def load_discharge_notes(note_root: Path) -> pd.DataFrame:
    note_dir = note_root / "note"
    discharge_path = note_dir / "discharge.csv.gz"
    if not discharge_path.exists():
        raise FileNotFoundError(f"Discharge summary file not found at {discharge_path}")

    notes = pd.read_csv(
        discharge_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "note_id",
            "note_type",
            "note_seq",
            "charttime",
            "storetime",
            "text",
        ],
        parse_dates=["charttime", "storetime"],
    )
    notes = notes.dropna(subset=["subject_id", "hadm_id"]).copy()
    return notes


def load_radiology_reports(note_root: Path) -> pd.DataFrame:
    """Load MIMIC-IV radiology reports (text only) for cohort linkage."""

    note_dir = note_root / "note"
    radiology_path = note_dir / "radiology.csv.gz"
    if not radiology_path.exists():
        raise FileNotFoundError(f"Radiology report file not found at {radiology_path}")

    reports = pd.read_csv(
        radiology_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "note_id",
            "note_seq",
            "charttime",
            "storetime",
            "text",
        ],
        parse_dates=["charttime", "storetime"],
    )
    reports = reports.dropna(subset=["subject_id", "hadm_id"]).copy()
    return reports


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def filter_admissions(admissions: pd.DataFrame, min_los_days: int) -> pd.DataFrame:
    """Filter admissions based on criteria including length of stay and exclusions.
    
    Note: In-hospital deaths are excluded to focus study specifically on readmission risk
    among patients who survived their initial hospitalization.
    """
    admissions = admissions.copy()
    length_of_stay = (
        admissions["dischtime"] - admissions["admittime"]
    ).dt.total_seconds() / 86400
    admissions["length_of_stay_days"] = length_of_stay

    # Identify in-hospital deaths
    died_in_hospital = (admissions["deathtime"].notna()) | (
        admissions["discharge_location"].str.upper() == "DIED"
    )

    mask = (
        (~admissions["admission_type"].isin(ADMISSION_TYPES_EXCLUDED))
        & (~admissions["discharge_location"].isin(DISCHARGE_LOCATIONS_EXCLUDED))
        & (admissions["length_of_stay_days"] >= min_los_days)
        & (~died_in_hospital)  # Exclude in-hospital deaths
    )
    return admissions.loc[mask].reset_index(drop=True)


def attach_demographics(admissions: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    enriched = admissions.merge(patients, on="subject_id", how="left", validate="many_to_one")
    admit_year = enriched["admittime"].dt.year
    enriched["age_at_admit"] = admit_year - enriched["anchor_year"] + enriched["anchor_age"]
    if "race" in enriched.columns:
        enriched["race"] = enriched["race"].fillna("UNKNOWN")
        enriched.loc[enriched["race"] == "UNABLE TO OBTAIN", "race"] = "UNKNOWN"
    return enriched


def select_latest_notes(notes: pd.DataFrame) -> pd.DataFrame:
    ordered = notes.sort_values(
        by=[
            "subject_id",
            "hadm_id",
            "charttime",
            "storetime",
            "note_seq",
            "note_id",
        ]
    )
    latest = ordered.groupby(["subject_id", "hadm_id"], as_index=False).tail(1)
    latest = latest.reset_index(drop=True)
    return latest.rename(
        columns={
            "note_id": "discharge_note_id",
            "text": "discharge_note_text",
            "charttime": "discharge_charttime",
            "storetime": "discharge_storetime",
            "note_seq": "discharge_note_seq",
        }
    )


def attach_discharge_notes(admissions: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    latest_notes = select_latest_notes(notes)
    merged = admissions.merge(
        latest_notes,
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def select_latest_radiology(reports: pd.DataFrame) -> pd.DataFrame:
    ordered = reports.sort_values(
        by=[
            "subject_id",
            "hadm_id",
            "charttime",
            "storetime",
            "note_seq",
            "note_id",
        ]
    )
    latest = ordered.groupby(["subject_id", "hadm_id"], as_index=False).tail(1)
    latest = latest.reset_index(drop=True)
    return latest.rename(
        columns={
            "note_id": "radiology_note_id",
            "text": "radiology_note_text",
            "charttime": "radiology_charttime",
            "storetime": "radiology_storetime",
            "note_seq": "radiology_note_seq",
        }
    )


def attach_radiology_reports(admissions: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    latest_reports = select_latest_radiology(reports)
    merged = admissions.merge(
        latest_reports,
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def label_readmissions(admissions: pd.DataFrame, window_days: int) -> pd.DataFrame:
    admissions = admissions.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    admissions["next_admittime"] = admissions.groupby("subject_id")["admittime"].shift(-1)
    admissions["next_hadm_id"] = admissions.groupby("subject_id")["hadm_id"].shift(-1)

    gap = (
        admissions["next_admittime"] - admissions["dischtime"]
    ).dt.total_seconds() / 86400
    admissions["readmission_gap_in_days"] = gap.where(gap >= 0)

    # Label as positive if readmitted within the specified window
    # Note: In-hospital deaths are already excluded during filtering
    admissions["readmitted_within_window"] = (
        admissions["readmission_gap_in_days"].notna()
        & (admissions["readmission_gap_in_days"] <= window_days)
    )
    admissions.loc[admissions["readmission_gap_in_days"].isna(), "next_hadm_id"] = np.nan
    return admissions


def assign_splits(
    admissions: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    subjects = admissions["subject_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_total = len(subjects)
    n_test = int(round(n_total * test_size))
    n_val = int(round(n_total * val_size))

    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test : n_test + n_val])
    train_subjects = set(subjects[n_test + n_val :])

    def lookup_split(subject_id: int) -> str:
        if subject_id in train_subjects:
            return "train"
        if subject_id in val_subjects:
            return "val"
        return "test"

    admissions = admissions.copy()
    admissions["split"] = admissions["subject_id"].map(lookup_split)
    split_index = {
        "train": [int(x) for x in sorted(train_subjects)],
        "val": [int(x) for x in sorted(val_subjects)],
        "test": [int(x) for x in sorted(test_subjects)],
    }
    return admissions, split_index


def summarise(admissions: pd.DataFrame) -> CohortStats:
    summary = admissions.groupby("split")["readmitted_within_window"].mean()
    gaps = admissions["readmission_gap_in_days"].dropna()
    return CohortStats(
        total_admissions=int(admissions["hadm_id"].nunique()),
        filtered_admissions=len(admissions),
        unique_patients=int(admissions["subject_id"].nunique()),
        pos_ratio_overall=float(admissions["readmitted_within_window"].mean()),
        pos_ratio_train=float(summary.get("train", np.nan)),
        pos_ratio_val=float(summary.get("val", np.nan)),
        pos_ratio_test=float(summary.get("test", np.nan)),
        min_readmit_gap=float(gaps.min()) if not gaps.empty else float("nan"),
        max_readmit_gap=float(gaps.max()) if not gaps.empty else float("nan"),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_mimic_cohort(
    mimic_root: Path,
    note_root: Path,
    min_los_days: int = MIN_LOS_DAYS,
    readmit_window_days: int = READMIT_WINDOW_DAYS,
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 17,
) -> CohortArtifacts:
    """Materialize the filtered and labeled cohort.

    Parameters
    ----------
    mimic_root:
        Root path containing ``hosp/`` with admissions tables.
    note_root:
        Root path containing ``note/discharge.csv.gz``.
    min_los_days:
        Minimum length of stay (days) for inclusion.
    readmit_window_days:
        Gap between discharge and next admission defining a positive label.
    val_size:
        Fraction of unique patients assigned to validation.
    test_size:
        Fraction of unique patients assigned to test.
    seed:
        RNG seed for deterministic patient-level splits.
    """

    admissions_raw = load_admissions(mimic_root)
    filtered = filter_admissions(admissions_raw, min_los_days)
    patients = load_patients(mimic_root)
    enriched = attach_demographics(filtered, patients)
    labeled = label_readmissions(enriched, readmit_window_days)
    notes = load_discharge_notes(note_root)
    labeled = attach_discharge_notes(labeled, notes)
    radiology_reports = load_radiology_reports(note_root)
    labeled = attach_radiology_reports(labeled, radiology_reports)
    labeled, split_index = assign_splits(labeled, val_size, test_size, seed)
    stats = summarise(labeled)
    return CohortArtifacts(labeled, split_index, stats)


def write_cohort_artifacts(
    artifacts: CohortArtifacts,
    output_dir: Path,
    *,
    prefer_parquet: bool = True,
) -> Dict[str, Path]:
    """Persist cohort outputs to disk.

    Returns a mapping from artifact name to the path written.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_frame = artifacts.cohort_frame.copy()
    cohort_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "length_of_stay_days",
        "admission_type",
        "discharge_location",
        "readmission_gap_in_days",
        "next_hadm_id",
        "readmitted_within_window",
        "deathtime",
        "age_at_admit",
        "gender",
        "race",
        "split",
        "discharge_note_id",
        "discharge_note_seq",
        "discharge_charttime",
        "discharge_storetime",
        "discharge_note_text",
        "radiology_note_id",
        "radiology_note_seq",
        "radiology_charttime",
        "radiology_storetime",
        "radiology_note_text",
    ]

    outputs: Dict[str, Path] = {}
    cohort_path = output_dir / "cohort.parquet"

    if prefer_parquet:
        try:
            cohort_frame[cohort_cols].to_parquet(cohort_path, index=False)
            outputs["cohort"] = cohort_path
        except ImportError:
            cohort_path = cohort_path.with_suffix(".csv")
            cohort_frame[cohort_cols].to_csv(cohort_path, index=False)
            outputs["cohort"] = cohort_path
    else:
        cohort_path = cohort_path.with_suffix(".csv")
        cohort_frame[cohort_cols].to_csv(cohort_path, index=False)
        outputs["cohort"] = cohort_path

    splits_path = output_dir / "cohort_splits.json"
    with splits_path.open("w", encoding="utf-8") as fp:
        json.dump(artifacts.split_index, fp, indent=2)
    outputs["splits"] = splits_path

    stats_path = output_dir / "cohort_stats.json"
    payload = {
        "stats": asdict(artifacts.stats),
    }
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    outputs["stats"] = stats_path

    return outputs


def describe_artifacts(artifacts: CohortArtifacts) -> str:
    """Return a human-readable description of cohort stats."""

    return artifacts.stats.to_json()


__all__ = [
    "CohortArtifacts",
    "CohortStats",
    "build_mimic_cohort",
    "describe_artifacts",
    "write_cohort_artifacts",
    "MIN_LOS_DAYS",
    "READMIT_WINDOW_DAYS",
    "load_admissions",
    "load_patients",
    "load_discharge_notes",
    "load_radiology_reports",
    "filter_admissions",
    "attach_demographics",
    "select_latest_notes",
    "select_latest_radiology",
    "attach_discharge_notes",
    "attach_radiology_reports",
    "label_readmissions",
    "assign_splits",
    "summarise",
]
