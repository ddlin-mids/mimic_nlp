#!/usr/bin/env python3
"""
Extract and preprocess clinical events from MIMIC-IV-Ext-22MCTS for our cohort.

This script:
1. Loads the 22.6M clinical events from MIMIC-IV-Ext-22MCTS
2. Optionally filters to events matching our cohort's hadm_ids
3. Applies temporal leakage prevention (events before discharge only)
4. Creates a vocabulary of unique events with frequency filtering
5. Builds daily event sequences per admission (matching preprocess_ehr.py format)

Output:
- unique_clinical_events.txt: Sorted list of unique events
- event_stats.csv: Event frequency statistics
- clinical_events_filtered.csv: Events filtered to cohort (if --cohort-path provided)
- clinical_events_vocab.csv: Event vocabulary with frequencies and IDs
- clinical_events_daily.pkl: Daily event sequences per admission

Usage:
    # Basic extraction (no cohort filter)
    uv run python ehr/extract_unique_clinical_events.py

    # With cohort filtering and sequence generation
    uv run python ehr/extract_unique_clinical_events.py \
        --cohort-path data/interim/readmit_analysis/long_los_cohort.csv \
        --output_dir data/interim/clinical_events \
        --min-freq 10 \
        --max-vocab-size 2000
"""
from __future__ import annotations

import argparse
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clinical events from MIMIC-IV-Ext-22MCTS"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("physionet.org/files/mimic-iv-ext-22mcts/1.0.0/clinical_event_timestamp.csv"),
        help="Path to clinical_event_timestamp.csv"
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=None,
        help="Path to cohort CSV with hadm_id, subject_id, admittime, dischtime, split. "
             "If provided, filters events and builds daily sequences.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/interim/clinical_events"),
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=10,
        help="Minimum frequency for event to be included in vocabulary",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=2000,
        help="Maximum vocabulary size (most frequent events)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Chunk size for reading large CSV",
    )
    parser.add_argument(
        "--save_mapping",
        action="store_true",
        help="Also save hadm_id to event mapping as parquet"
    )
    return parser.parse_args()


def load_cohort(path: Path) -> pd.DataFrame:
    """Load cohort and compute LOS in hours for leakage prevention."""
    logger.info(f"Loading cohort from {path}")
    df = pd.read_csv(path, parse_dates=["admittime", "dischtime"])
    
    # Compute LOS in hours (22MCTS uses hours relative to discharge)
    df["los_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 3600
    
    logger.info(f"Loaded {len(df)} admissions")
    return df


def extract_all_unique_events(input_path: Path, output_dir: Path, save_mapping: bool = False) -> pd.DataFrame:
    """
    Extract unique events from full dataset without cohort filtering.
    Returns the full dataframe for further processing.
    """
    logger.info(f"Reading {input_path}...")
    logger.info("This may take a minute for 22M rows...")
    
    cols = ["Hadm_id", "Event", "Time", "Time_bin"]
    df = pd.read_csv(input_path, usecols=cols)
    
    total_rows = len(df)
    logger.info(f"Loaded {total_rows:,} rows")
    
    # Get unique events
    unique_events = df["Event"].dropna().unique()
    n_unique = len(unique_events)
    logger.info(f"Found {n_unique:,} unique events (compression ratio: {total_rows/n_unique:.1f}x)")
    
    # Save unique events to text file (one per line)
    unique_events_path = output_dir / "unique_clinical_events.txt"
    sorted_events = sorted(unique_events)
    with open(unique_events_path, "w", encoding="utf-8") as f:
        for event in sorted_events:
            f.write(f"{event}\n")
    logger.info(f"Saved unique events to {unique_events_path}")
    
    # Save event statistics
    stats_path = output_dir / "event_stats.csv"
    event_counts = df["Event"].value_counts()
    stats_df = event_counts.to_frame("count").reset_index()
    stats_df.columns = ["event", "count"]
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Saved event frequency stats to {stats_path}")
    
    # Optionally save the full mapping for later reconstruction
    if save_mapping:
        mapping_path = output_dir / "hadm_event_mapping.parquet"
        logger.info(f"Saving full mapping to {mapping_path}...")
        df.to_parquet(mapping_path, index=False)
        logger.info(f"Saved mapping ({mapping_path.stat().st_size / 1e6:.1f} MB)")
    
    # Print summary stats
    logger.info("=" * 60)
    logger.info("Summary Statistics (Full Dataset):")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Unique events: {n_unique:,}")
    logger.info(f"  Unique hadm_ids: {df['Hadm_id'].nunique():,}")
    logger.info(f"  Top 5 events:")
    for event, count in event_counts.head(5).items():
        logger.info(f"    - '{event}': {count:,}")
    logger.info("=" * 60)
    
    return df


def filter_events_to_cohort(
    df: pd.DataFrame, 
    cohort_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter 22MCTS events to cohort hadm_ids with temporal leakage prevention.
    
    The 22MCTS 'Time' column is in hours relative to *discharge* (negative = before).
    We need to filter events where Time <= 0 (at or before discharge).
    """
    cohort_hadm_ids = set(cohort_df["hadm_id"])
    los_map = cohort_df.set_index("hadm_id")["los_hours"].to_dict()
    
    logger.info(f"Filtering events to {len(cohort_hadm_ids)} cohort admissions")
    
    total_rows = len(df)
    
    # Filter to cohort
    df = df[df["Hadm_id"].isin(cohort_hadm_ids)].copy()
    logger.info(f"After cohort filter: {len(df):,} events")
    
    # Temporal leakage prevention:
    # Time is relative to discharge (0 = discharge, negative = before)
    # We keep events where Time <= 0 (at or before discharge)
    df = df[df["Time"] <= 0].copy()
    logger.info(f"After temporal filter (Time <= 0): {len(df):,} events")
    
    # Rename for consistency with our pipeline
    df = df.rename(columns={"Hadm_id": "hadm_id", "Event": "event", "Time": "time_hours"})
    
    # Compute Day_Number relative to admission
    # time_hours is relative to discharge, so: hours_from_admit = LOS + time_hours
    df["los_hours"] = df["hadm_id"].map(los_map)
    df["hours_from_admit"] = df["los_hours"] + df["time_hours"]
    df["Day_Number"] = (df["hours_from_admit"] / 24).apply(np.floor).astype(int) + 1
    df["Day_Number"] = df["Day_Number"].clip(lower=1)
    
    kept_rows = len(df)
    logger.info(f"Final filtered events: {kept_rows:,} / {total_rows:,} ({100*kept_rows/total_rows:.1f}%)")
    
    return df[["hadm_id", "event", "time_hours", "Day_Number", "Time_bin"]]


def build_vocabulary(
    events_df: pd.DataFrame, 
    min_freq: int, 
    max_vocab_size: int
) -> tuple[pd.DataFrame, dict]:
    """
    Build event vocabulary with frequency filtering.
    
    Returns:
        vocab_df: DataFrame with event, frequency, and assigned ID
        event2id: Dictionary mapping event string to integer ID
    """
    logger.info("Building vocabulary...")
    
    # Count frequencies
    event_counts = Counter(events_df["event"])
    logger.info(f"Total unique events in cohort: {len(event_counts)}")
    
    # Filter by minimum frequency
    filtered_events = {e: c for e, c in event_counts.items() if c >= min_freq}
    logger.info(f"Events with freq >= {min_freq}: {len(filtered_events)}")
    
    # Take top N by frequency
    sorted_events = sorted(filtered_events.items(), key=lambda x: -x[1])
    top_events = sorted_events[:max_vocab_size]
    logger.info(f"Vocabulary size (top {max_vocab_size}): {len(top_events)}")
    
    # Create vocabulary DataFrame
    vocab_df = pd.DataFrame(top_events, columns=["event", "frequency"])
    vocab_df["event_id"] = range(1, len(vocab_df) + 1)  # 0 reserved for padding/unknown
    
    # Create mapping
    event2id = {row["event"]: row["event_id"] for _, row in vocab_df.iterrows()}
    
    # Log coverage stats
    total_coverage = sum(c for e, c in event_counts.items() if e in event2id)
    total_events = sum(event_counts.values())
    logger.info(f"Vocabulary covers {100*total_coverage/total_events:.1f}% of event occurrences")
    
    return vocab_df, event2id


def build_daily_sequences(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    event2id: dict,
) -> dict:
    """
    Build daily event sequences per admission, matching preprocess_ehr.py format.
    
    Returns:
        feat_dict: Dictionary mapping node_name to (num_days, vocab_size) array
                   where each cell is a count of that event on that day
    """
    logger.info("Building daily event sequences...")
    
    vocab_size = max(event2id.values()) + 1  # +1 for padding index 0
    
    # Map events to IDs, unknown events get 0
    events_df = events_df.copy()
    events_df["event_id"] = events_df["event"].map(event2id).fillna(0).astype(int)
    
    # Create node_name for each admission
    hadm_to_node = cohort_df.set_index("hadm_id").apply(
        lambda row: f"{row['subject_id']}_{row.name}", axis=1
    ).to_dict()
    
    # Get LOS in days for each admission
    if "length_of_stay_days" in cohort_df.columns:
        los_col = "length_of_stay_days"
    elif "los_days" in cohort_df.columns:
        los_col = "los_days"
    else:
        # Compute from timestamps
        cohort_df = cohort_df.copy()
        cohort_df["length_of_stay_days"] = (
            cohort_df["dischtime"] - cohort_df["admittime"]
        ).dt.total_seconds() / (24 * 3600)
        los_col = "length_of_stay_days"
    
    los_days_map = cohort_df.set_index("hadm_id")[los_col].to_dict()
    
    feat_dict = {}
    
    for hadm_id, group in tqdm(events_df.groupby("hadm_id"), desc="Building sequences"):
        node_name = hadm_to_node.get(hadm_id)
        if node_name is None:
            continue
            
        los_days = int(los_days_map.get(hadm_id, 1))
        los_days = max(1, los_days)
        
        # Initialize matrix: (num_days, vocab_size)
        seq = np.zeros((los_days, vocab_size), dtype=np.float32)
        
        for _, row in group.iterrows():
            day_idx = min(row["Day_Number"] - 1, los_days - 1)  # 0-indexed
            day_idx = max(0, day_idx)
            event_id = row["event_id"]
            if event_id > 0:  # Skip unknown events
                seq[day_idx, event_id] += 1
                
        feat_dict[node_name] = seq
    
    logger.info(f"Built sequences for {len(feat_dict)} admissions")
    return feat_dict


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract all unique events (always do this)
    df_all = extract_all_unique_events(args.input, args.output_dir, args.save_mapping)
    
    # If no cohort path provided, stop here
    if args.cohort_path is None:
        logger.info("No cohort path provided. Skipping cohort filtering and sequence building.")
        logger.info("To generate sequences, re-run with --cohort-path")
        return
    
    # Step 2: Load cohort and filter events
    cohort_df = load_cohort(args.cohort_path)
    
    filtered_events_path = args.output_dir / "clinical_events_filtered.csv"
    if filtered_events_path.exists():
        logger.info(f"Loading cached filtered events from {filtered_events_path}")
        events_df = pd.read_csv(filtered_events_path)
    else:
        events_df = filter_events_to_cohort(df_all, cohort_df)
        events_df.to_csv(filtered_events_path, index=False)
        logger.info(f"Saved filtered events to {filtered_events_path}")
    
    # Step 3: Build vocabulary
    vocab_df, event2id = build_vocabulary(events_df, args.min_freq, args.max_vocab_size)
    vocab_path = args.output_dir / "clinical_events_vocab.csv"
    vocab_df.to_csv(vocab_path, index=False)
    logger.info(f"Saved vocabulary to {vocab_path}")
    
    # Step 4: Build daily sequences
    feat_dict = build_daily_sequences(events_df, cohort_df, event2id)
    
    # Save in format compatible with preprocess_ehr.py
    output_dict = {
        "feat_dict": feat_dict,
        "event2id": event2id,
        "vocab_size": max(event2id.values()) + 1,
        "event_cols": list(event2id.keys()),
    }
    
    seq_path = args.output_dir / "clinical_events_daily.pkl"
    with open(seq_path, "wb") as f:
        pickle.dump(output_dict, f)
    logger.info(f"Saved daily sequences to {seq_path}")
    
    # Print final summary
    logger.info("=" * 60)
    logger.info("Final Summary (Cohort Filtered):")
    logger.info(f"  Total filtered events: {len(events_df):,}")
    logger.info(f"  Unique events in data: {events_df['event'].nunique():,}")
    logger.info(f"  Vocabulary size: {len(event2id)}")
    logger.info(f"  Admissions with sequences: {len(feat_dict)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
