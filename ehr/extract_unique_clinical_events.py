#!/usr/bin/env python3
"""
Extract unique clinical event strings from MIMIC-IV-Ext-22MCTS dataset.

This is a preprocessing step for the "Chunk & Compact" strategy:
1. Read the 22M row clinical_event_timestamp.csv
2. Extract unique Event strings (~thousands)
3. Save to text file for efficient embedding generation

Usage:
    uv run python ehr/extract_unique_clinical_events.py
    uv run python ehr/extract_unique_clinical_events.py --output_dir data/interim/mimic_22m
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique clinical events from MIMIC-IV-Ext-22MCTS"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("physionet.org/files/mimic-iv-ext-22mcts/1.0.0/clinical_event_timestamp.csv"),
        help="Path to clinical_event_timestamp.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/interim/mimic_22m"),
        help="Output directory for unique events file"
    )
    parser.add_argument(
        "--save_mapping",
        action="store_true",
        help="Also save hadm_id to event mapping for later reconstruction"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    input_path = args.input
    output_dir = args.output_dir
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading {input_path}...")
    logger.info("This may take a minute for 22M rows...")
    
    # Read only necessary columns to save memory
    df = pd.read_csv(input_path, usecols=["Hadm_id", "Event", "Time", "Time_bin"])
    
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
    event_counts.to_frame("count").reset_index().rename(columns={"index": "event"}).to_csv(
        stats_path, index=False
    )
    logger.info(f"Saved event frequency stats to {stats_path}")
    
    # Optionally save the full mapping for later reconstruction
    if args.save_mapping:
        mapping_path = output_dir / "hadm_event_mapping.parquet"
        logger.info(f"Saving full mapping to {mapping_path}...")
        df.to_parquet(mapping_path, index=False)
        logger.info(f"Saved mapping ({mapping_path.stat().st_size / 1e6:.1f} MB)")
    
    # Print summary stats
    logger.info("=" * 60)
    logger.info("Summary Statistics:")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Unique events: {n_unique:,}")
    logger.info(f"  Unique hadm_ids: {df['Hadm_id'].nunique():,}")
    logger.info(f"  Top 5 events:")
    for event, count in event_counts.head(5).items():
        logger.info(f"    - '{event}': {count:,}")
    logger.info("=" * 60)
    

if __name__ == "__main__":
    main()
