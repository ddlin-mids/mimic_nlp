#!/usr/bin/env python3
"""
Embed clinical events from MIMIC-IV-Ext-22MCTS and aggregate to admission level.

This script produces embeddings in the same format as discharge_summary.npz:
- Shape: (n_admissions, 768)
- Keys: embeddings, hadm_ids

Strategy:
1. Load top-K most frequent events and generate embeddings
2. Load hadm_event_mapping to get per-admission events
3. Aggregate event embeddings per admission (mean pooling)
4. Filter to cohort if specified
5. Save in standard format

Usage:
    uv run python ehr/embed_clinical_events.py --top-k 10000
    uv run python ehr/embed_clinical_events.py --top-k 10000 --cohort-path data/interim/readmit_analysis/long_los_cohort.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Disable torch compile to avoid dynamic shape issues
torch._dynamo.config.suppress_errors = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics for the generated embeddings."""
    column: str
    rows: int
    embedding_dim: int
    dtype: str
    max_length: int
    batch_size: int
    model_name: str
    quantization: str
    mean: float
    std: float
    min: float
    max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed clinical events and aggregate to admission level"
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=Path("data/interim/mimic_22m/hadm_event_mapping.parquet"),
        help="Path to hadm_event_mapping.parquet"
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=Path("data/interim/mimic_22m/event_stats.csv"),
        help="Path to event_stats.csv for frequency-based filtering"
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=None,
        help="Optional CSV with hadm_id column to filter admissions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/embeddings/notes"),
        help="Output directory (same as other note embeddings)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="clinical_events",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10000,
        help="Embed top-K most frequent events (default: 10000)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="thomas-sounack/bioclinical-modernbert-base",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max token length for events"
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model precision"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda, cpu)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 100 events and 1000 admissions for testing"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output already exists"
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_top_k_events(stats_path: Path, top_k: int) -> List[str]:
    """Load top-K most frequent events from stats file."""
    df = pd.read_csv(stats_path, nrows=top_k)
    return df["Event"].tolist()


def load_cohort_ids(path: Path | None) -> Set[int] | None:
    """Load hadm_ids from cohort file."""
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, usecols=["hadm_id"])
    ids = set(df["hadm_id"].unique())
    logger.info(f"Loaded {len(ids):,} hadm_ids from cohort")
    return ids


def encode_events(
    events: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    """Encode event strings to embeddings using CLS pooling."""
    embeddings = []
    n_events = len(events)
    
    # Pad to make divisible by batch_size to avoid torch.compile issues
    remainder = n_events % batch_size
    if remainder > 0:
        padding_needed = batch_size - remainder
        events = events + [""] * padding_needed
        logger.info(f"Padded {padding_needed} empty events to avoid batch size issues")
    
    for start in tqdm(range(0, len(events), batch_size), desc="Encoding events"):
        end = start + batch_size
        batch = events[start:end]
        
        inputs = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            
            # Handle bfloat16 -> float32 for numpy compatibility
            if torch_dtype == torch.bfloat16:
                cls_emb = cls_emb.to(torch.float32)
            
            embeddings.append(cls_emb.cpu().numpy())
        
        del inputs, outputs, cls_emb
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Remove padding
    all_embeddings = np.concatenate(embeddings, axis=0)
    return all_embeddings[:n_events]


def aggregate_embeddings_by_admission(
    mapping_df: pd.DataFrame,
    event_to_embedding: Dict[str, np.ndarray],
    cohort_ids: Set[int] | None,
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate event embeddings by admission using mean pooling.
    
    Returns:
        embeddings: (n_admissions, embedding_dim)
        hadm_ids: (n_admissions,)
    """
    # Group events by hadm_id
    grouped = defaultdict(list)
    
    for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Grouping events"):
        hadm_id = int(row["Hadm_id"])
        event = row["Event"]
        
        # Skip if not in cohort
        if cohort_ids is not None and hadm_id not in cohort_ids:
            continue
        
        # Skip if event not in our embedding vocabulary
        if event in event_to_embedding:
            grouped[hadm_id].append(event_to_embedding[event])
    
    logger.info(f"Found {len(grouped):,} admissions with embeddable events")
    
    # Aggregate by mean pooling
    hadm_ids = []
    embeddings = []
    
    for hadm_id in sorted(grouped.keys()):
        event_embeds = grouped[hadm_id]
        if len(event_embeds) > 0:
            mean_embed = np.mean(event_embeds, axis=0)
            embeddings.append(mean_embed)
            hadm_ids.append(hadm_id)
    
    return np.array(embeddings), np.array(hadm_ids)


def main() -> None:
    args = parse_args()
    
    # Validate inputs
    if not args.stats_file.exists():
        raise FileNotFoundError(f"Stats file not found: {args.stats_file}")
    if not args.mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {args.mapping_file}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.output_prefix}.npz"
    
    if args.skip_existing and output_path.exists():
        logger.info(f"Output already exists: {output_path}, skipping.")
        return
    
    # Load top-K events
    top_k = 100 if args.dry_run else args.top_k
    events = load_top_k_events(args.stats_file, top_k)
    logger.info(f"Loaded {len(events):,} events to embed")
    
    # Load cohort
    cohort_ids = load_cohort_ids(args.cohort_path)
    
    # Setup device and dtype
    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")
    
    if args.dtype == "float16":
        torch_dtype = torch.float16
        np_dtype = np.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
        np_dtype = np.float16  # Save as float16 like discharge notes
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa"
    )
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded. Hidden size: {model.config.hidden_size}")
    embedding_dim = model.config.hidden_size
    
    # Encode events
    event_embeddings = encode_events(
        events=events,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        torch_dtype=torch_dtype,
    )
    logger.info(f"Generated {len(event_embeddings):,} event embeddings")
    
    # Create event -> embedding lookup
    event_to_embedding = {event: event_embeddings[i] for i, event in enumerate(events)}
    
    # Free GPU memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Load mapping and aggregate
    logger.info(f"Loading mapping from {args.mapping_file}...")
    mapping_df = pd.read_parquet(args.mapping_file)
    
    if args.dry_run:
        # Limit to first 1000 unique admissions
        unique_hadms = mapping_df["Hadm_id"].unique()[:1000]
        mapping_df = mapping_df[mapping_df["Hadm_id"].isin(unique_hadms)]
        logger.info(f"Dry run: limited to {len(mapping_df):,} rows")
    
    logger.info(f"Loaded {len(mapping_df):,} event mappings")
    
    # Aggregate to admission level
    embeddings, hadm_ids = aggregate_embeddings_by_admission(
        mapping_df=mapping_df,
        event_to_embedding=event_to_embedding,
        cohort_ids=cohort_ids,
        embedding_dim=embedding_dim,
    )
    
    # Cast for storage
    embeddings = embeddings.astype(np_dtype)
    
    # Save in same format as discharge notes
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        hadm_ids=hadm_ids
    )
    logger.info(f"Saved embeddings to {output_path}")
    
    # Save manifest in same format as discharge notes
    stats = EmbeddingStats(
        column="clinical_events",
        rows=len(hadm_ids),
        embedding_dim=embedding_dim,
        dtype=str(embeddings.dtype),
        max_length=args.max_length,
        batch_size=args.batch_size,
        model_name=args.model_name,
        quantization="none",
        mean=float(embeddings.mean()),
        std=float(embeddings.std()),
        min=float(embeddings.min()),
        max=float(embeddings.max()),
    )
    
    manifest = {
        "input_path": str(args.mapping_file),
        "output_dir": str(args.output_dir),
        "limit": args.top_k,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "model_name": args.model_name,
        "device": str(device),
        "rows": len(hadm_ids),
        "cohort_filter": str(args.cohort_path) if args.cohort_path else None,
        "top_k_events": args.top_k,
        "total_events_embedded": len(events),
    }
    
    metadata = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "manifest": manifest,
        "stats": asdict(stats),
    }
    
    manifest_path = args.output_dir / f"{args.output_prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved manifest to {manifest_path}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Embedding Summary:")
    logger.info(f"  Admissions: {len(hadm_ids):,}")
    logger.info(f"  Dimensions: {embedding_dim}")
    logger.info(f"  Dtype: {embeddings.dtype}")
    logger.info(f"  Events embedded: {len(events):,}")
    logger.info(f"  Mean: {stats.mean:.4f}, Std: {stats.std:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
