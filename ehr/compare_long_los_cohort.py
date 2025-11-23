from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compare_long_los_cohorts(
    original_path: Path, new_path: Path, max_diff_rows: int = 20
) -> None:
    """
    Compare legacy long_los_cohort.csv with the newly generated long_los_cohort_new.csv.

    Prints:
    - Row counts and column set differences.
    - Admissions present only in one file (by subject_id + hadm_id).
    - A small sample of rows where shared admissions differ in any value.
    """

    if not original_path.exists():
        raise FileNotFoundError(f"Original file not found: {original_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"New file not found: {new_path}")

    orig = pd.read_csv(original_path)
    new = pd.read_csv(new_path)

    print(f"Original shape: {orig.shape}")
    print(f"New shape:      {new.shape}")

    orig_cols = set(orig.columns)
    new_cols = set(new.columns)
    if orig_cols != new_cols:
        print("Column differences:")
        print("  Only in original:", sorted(orig_cols - new_cols))
        print("  Only in new:     ", sorted(new_cols - orig_cols))
    else:
        print("Columns match.")

    join_keys = ["subject_id", "hadm_id"]
    for key in join_keys:
        if key not in orig.columns or key not in new.columns:
            raise KeyError(f"Expected join key {key} missing from one of the files.")

    orig_ids = set(zip(orig["subject_id"], orig["hadm_id"]))
    new_ids = set(zip(new["subject_id"], new["hadm_id"]))

    only_in_orig = orig_ids - new_ids
    only_in_new = new_ids - orig_ids

    print(f"Admissions only in original: {len(only_in_orig)}")
    if only_in_orig:
        print("  Examples:", list(only_in_orig)[: max_diff_rows // 2])

    print(f"Admissions only in new:      {len(only_in_new)}")
    if only_in_new:
        print("  Examples:", list(only_in_new)[: max_diff_rows // 2])

    # Compare values for shared admissions
    shared_ids = orig_ids & new_ids
    if not shared_ids:
        print("No shared admissions to compare.")
        return

    orig_shared = orig.set_index(join_keys).loc[list(shared_ids)].sort_index()
    new_shared = new.set_index(join_keys).loc[list(shared_ids)].sort_index()

    # Align value columns (exclude join keys, which are now in the index)
    common_cols = sorted(orig_cols & new_cols)
    value_cols = [c for c in common_cols if c not in join_keys]
    orig_shared = orig_shared[value_cols]
    new_shared = new_shared[value_cols]

    diff_mask = (orig_shared != new_shared) & ~(orig_shared.isna() & new_shared.isna())
    num_diff_rows = diff_mask.any(axis=1).sum()
    print(f"Shared admissions with any differing values: {int(num_diff_rows)}")

    if num_diff_rows:
        diff_rows = diff_mask[diff_mask.any(axis=1)].head(max_diff_rows)
        print("\nSample differing rows (subject_id, hadm_id):")
        for (sid, hadm) in diff_rows.index:
            print(f"  ({sid}, {hadm})")
            print("    original:", orig_shared.loc[(sid, hadm)].to_dict())
            print("    new:     ", new_shared.loc[(sid, hadm)].to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy long_los_cohort.csv with long_los_cohort_new.csv."
    )
    parser.add_argument(
        "--original-path",
        type=Path,
        default=Path("data/interim/readmit_analysis/long_los_cohort_los14.csv"),
        help="Path to the legacy long_los_cohort_los14.csv file (LOS>=14).",
    )
    parser.add_argument(
        "--new-path",
        type=Path,
        default=Path("data/interim/readmit_analysis/long_los_cohort.csv"),
        help="Path to the canonical long_los_cohort.csv file (LOS>=15).",
    )
    parser.add_argument(
        "--max-diff-rows",
        type=int,
        default=20,
        help="Maximum number of differing rows to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_long_los_cohorts(args.original_path, args.new_path, args.max_diff_rows)


if __name__ == "__main__":
    main()
