"""CLI entry-point for building the MIMIC-IV readmission cohort.

This wrapper delegates to :mod:`src.data.build_cohort` so downstream modules
and notebooks can reuse the same logic without shelling out to this script.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set
from zipfile import ZipFile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.data.build_cohort import (
    MIN_LOS_DAYS,
    READMIT_WINDOW_DAYS,
    build_mimic_cohort,
    describe_artifacts,
    write_cohort_artifacts,
)

DEFAULT_MIMIC_ROOT = Path("physionet.org/files/mimiciv/3.1")
DEFAULT_NOTE_ROOT = Path("physionet.org/files/mimic-iv-note/2.2")
DEFAULT_CXR_ROOT = Path("physionet.org/files/mimic-cxr/2.1.0")
DEFAULT_OUTPUT_DIR = Path("data/interim")


def _normalize_report_text(text: str) -> str:
    """Collapse boilerplate and anonymization artifacts for comparisons."""

    sanitized = re.sub(r"_+", "___", text)
    sanitized = re.sub(r"___[MF]", "___", sanitized)
    lines = [line.strip() for line in sanitized.splitlines()]
    lines = [
        line
        for line in lines
        if line and line.upper() not in {"FINAL REPORT", "PRELIMINARY REPORT", "ADDENDUM"}
    ]
    collapsed = " ".join(" ".join(line.split()) for line in lines)
    return collapsed


def _hash_report_text(text: str) -> str:
    normalized = _normalize_report_text(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _load_chest_radiology_notes(note_root: Path, hadm_ids: Iterable[int]) -> pd.DataFrame:
    note_dir = note_root / "note"
    radiology_path = note_dir / "radiology.csv.gz"
    detail_path = note_dir / "radiology_detail.csv.gz"
    if not radiology_path.exists():
        raise FileNotFoundError(f"Radiology notes not found at {radiology_path}")
    if not detail_path.exists():
        raise FileNotFoundError(f"Radiology detail file not found at {detail_path}")

    hadm_set: Set[int] = {int(hadm) for hadm in hadm_ids if pd.notna(hadm)}
    if not hadm_set:
        return pd.DataFrame(columns=["note_id", "subject_id", "hadm_id", "text"])

    radiology_rows = []
    note_ids: Set[str] = set()
    for chunk in pd.read_csv(
        radiology_path,
        usecols=["note_id", "subject_id", "hadm_id", "text"],
        chunksize=50000,
    ):
        filtered = chunk[chunk["hadm_id"].isin(hadm_set)]
        if filtered.empty:
            continue
        radiology_rows.append(filtered)
        note_ids.update(filtered["note_id"].astype(str))

    if not radiology_rows:
        return pd.DataFrame(columns=["note_id", "subject_id", "hadm_id", "text"])

    cxr_note_ids: Set[str] = set()
    for chunk in pd.read_csv(
        detail_path,
        usecols=["note_id", "field_name", "field_value"],
        chunksize=75000,
    ):
        subset = chunk[chunk["note_id"].isin(note_ids)]
        if subset.empty:
            continue
        mask = (subset["field_name"].str.lower() == "exam_name") & (
            subset["field_value"].str.contains("CHEST", case=False, na=False)
        )
        if mask.any():
            cxr_note_ids.update(subset.loc[mask, "note_id"].astype(str))

    if not cxr_note_ids:
        return pd.DataFrame(columns=["note_id", "subject_id", "hadm_id", "text"])

    radiology_df = pd.concat(radiology_rows, ignore_index=True)
    return radiology_df[radiology_df["note_id"].isin(cxr_note_ids)].reset_index(drop=True)


def _build_cxr_report_index(studies: pd.DataFrame, report_zip: Path) -> Dict[int, Set[str]]:
    if not report_zip.exists():
        raise FileNotFoundError(f"MIMIC-CXR report archive not found at {report_zip}")

    index: Dict[int, Set[str]] = defaultdict(set)
    with ZipFile(report_zip) as zf:
        for row in studies.itertuples(index=False):
            path = row.path
            try:
                raw = zf.read(path)
            except KeyError:
                continue
            text = raw.decode("utf-8", errors="replace")
            index[int(row.subject_id)].add(_hash_report_text(text))
    return index


def summarise_cxr_overlap(
    cohort_frame: pd.DataFrame, note_root: Path, cxr_root: Path
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "status": "skip",
        "reason": "",
    }

    try:
        chest_notes = _load_chest_radiology_notes(note_root, cohort_frame["hadm_id"])
    except FileNotFoundError as exc:
        summary["reason"] = str(exc)
        return summary

    if chest_notes.empty:
        summary["reason"] = "No chest radiology notes found for the cohort"
        return summary

    study_list_path = cxr_root / "cxr-study-list.csv.gz"
    if not study_list_path.exists():
        summary["reason"] = f"CXR study list not found at {study_list_path}"
        return summary

    cohort_subjects: Set[int] = set(cohort_frame["subject_id"].astype(int).unique())
    studies = pd.read_csv(
        study_list_path, usecols=["subject_id", "study_id", "path"]
    )
    studies = studies[studies["subject_id"].isin(cohort_subjects)]
    if studies.empty:
        summary["reason"] = "No MIMIC-CXR studies overlap with cohort subjects"
        return summary

    try:
        report_index = _build_cxr_report_index(studies, cxr_root / "mimic-cxr-reports.zip")
    except FileNotFoundError as exc:
        summary["reason"] = str(exc)
        return summary

    if not report_index:
        summary["reason"] = "No MIMIC-CXR reports indexed for cohort subjects"
        return summary

    chest_notes = chest_notes.dropna(subset=["subject_id", "hadm_id", "text"]).copy()
    if chest_notes.empty:
        summary["reason"] = "Chest radiology notes lack required identifiers"
        return summary

    chest_notes["subject_id"] = chest_notes["subject_id"].astype(int)
    chest_notes["hadm_id"] = chest_notes["hadm_id"].astype(int)
    chest_notes["report_hash"] = chest_notes["text"].map(_hash_report_text)
    chest_notes["matches_mimic_cxr"] = chest_notes.apply(
        lambda row: row["report_hash"] in report_index.get(row["subject_id"], set()),
        axis=1,
    )

    matched = chest_notes[chest_notes["matches_mimic_cxr"]]
    chest_admissions = set(chest_notes["hadm_id"].tolist())
    matched_admissions = set(matched["hadm_id"].tolist())

    summary.update(
        {
            "status": "ok",
            "cohort_size": int(cohort_frame["hadm_id"].nunique()),
            "chest_note_count": int(len(chest_notes)),
            "matched_note_count": int(len(matched)),
            "chest_admissions": len(chest_admissions),
            "matched_admissions": len(matched_admissions),
            "cxr_subjects_indexed": len(report_index),
            "cxr_reports_indexed": int(sum(len(v) for v in report_index.values())),
            "unmatched_examples": chest_notes.loc[
                ~chest_notes["matches_mimic_cxr"], ["subject_id", "hadm_id", "note_id"]
            ]
            .head(5)
            .to_dict(orient="records"),
        }
    )
    return summary


def _print_cxr_summary(summary: Dict[str, object]) -> None:
    if summary.get("status") != "ok":
        reason = summary.get("reason", "No additional context")
        print(f"Skipped MIMIC-CXR overlap: {reason}")
        return

    chest_admissions = summary["chest_admissions"] or 0
    matched_admissions = summary["matched_admissions"] or 0
    cohort_size = summary["cohort_size"] or 1
    matched_pct = matched_admissions / chest_admissions if chest_admissions else 0.0
    coverage_pct = chest_admissions / cohort_size if cohort_size else 0.0

    print("MIMIC-CXR overlap summary:")
    print(
        f"  Admissions with chest radiology notes: {chest_admissions} "
        f"({coverage_pct:.1%} of cohort)"
    )
    print(
        f"  Admissions with matching MIMIC-CXR reports: {matched_admissions} "
        f"({matched_pct:.1%} of chest-note admissions)"
    )

    unmatched_count = int(summary["chest_note_count"]) - int(summary["matched_note_count"])
    if unmatched_count > 0:
        print(f"  Chest radiology notes without CXR match: {unmatched_count}")
        examples = summary.get("unmatched_examples") or []
        if examples:
            print("  Sample unmatched notes (subject_id, hadm_id, note_id):")
            for row in examples:
                print(
                    f"    ({row['subject_id']}, {row['hadm_id']}, {row['note_id']})"
                )
    else:
        print("  All chest radiology notes matched a MIMIC-CXR report.")

    print(
        "  Indexed MIMIC-CXR reports: "
        f"{summary['cxr_reports_indexed']} across {summary['cxr_subjects_indexed']} subjects"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MIMIC-IV cohort")
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=DEFAULT_MIMIC_ROOT,
        help="Path to the root of the extracted MIMIC-IV dataset",
    )
    parser.add_argument(
        "--mimic-note-root",
        type=Path,
        default=DEFAULT_NOTE_ROOT,
        help="Path to the root of the extracted MIMIC-IV note dataset",
    )
    parser.add_argument(
        "--mimic-cxr-root",
        type=Path,
        default=DEFAULT_CXR_ROOT,
        help="Path to the root of the extracted MIMIC-CXR dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where cohort artifacts will be written",
    )
    parser.add_argument(
        "--min-los-days",
        type=int,
        default=MIN_LOS_DAYS,
        help="Minimum length of stay (in days) required to keep an admission",
    )
    parser.add_argument(
        "--readmit-window-days",
        type=int,
        default=READMIT_WINDOW_DAYS,
        help="Gap in days that qualifies as a readmission",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of patients reserved for validation",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of patients reserved for test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Seed for deterministic patient splits",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifacts = build_mimic_cohort(
        mimic_root=args.mimic_root,
        note_root=args.mimic_note_root,
        min_los_days=args.min_los_days,
        readmit_window_days=args.readmit_window_days,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    outputs = write_cohort_artifacts(artifacts, args.output_dir)

    cohort_path = outputs["cohort"]
    splits_path = outputs["splits"]
    stats_path = outputs["stats"]

    print("Cohort saved to", cohort_path)
    print("Split index saved to", splits_path)
    print("Stats saved to", stats_path)
    print(describe_artifacts(artifacts))

    cxr_summary = summarise_cxr_overlap(
        artifacts.cohort_frame, args.mimic_note_root, args.mimic_cxr_root
    )
    _print_cxr_summary(cxr_summary)


if __name__ == "__main__":
    main()
