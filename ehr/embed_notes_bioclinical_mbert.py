#!/usr/bin/env python3
"""
BioClinical ModernBERT note encoder.

Promotes the logic from notebooks/notebooks_dc/04B_encode_notes_bioclinical_modernbert.ipynb
into a reproducible CLI so we can run the discharge + radiology embedding job on HPC.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode clinical notes with BioClinical ModernBERT")
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="CSV file containing at least hadm_id plus note text columns.",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=None,
        help="Optional CSV file containing 'hadm_id' to filter the notes (highly recommended for speed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/embeddings"),
        help="Directory for embeddings and metadata.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in the input file to encode.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files. If not set, derives from input filename.",
    )
    parser.add_argument(
        "--model-name",
        default="thomas-sounack/bioclinical-modernbert-base",
        help="Hugging Face model repository.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used for encoding.")
    parser.add_argument("--max-length", type=int, default=8192, help="Tokenizer max_length / truncation window.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Precision of the model and saved embeddings. Defaults to bfloat16 for A30.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization (NF4) via bitsandbytes. Reduces VRAM but may affect embedding precision.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows (for smoke tests).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --limit 64 to verify the pipeline without touching the full dataset.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override torch device (e.g. 'cuda', 'cuda:1', 'cpu'). Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--text-fill-value",
        default="",
        help="Value inserted when a note column contains nulls.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="How often (in batches) to log memory stats.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip encoding and just inspect the dataset (useful for validation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic row sampling when --limit is used.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if the output file already exists.",
    )
    return parser.parse_args()


@dataclass
class EmbeddingStats:
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


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cohort_ids(path: Path) -> set[int] | None:
    if not path:
        return None
    print(f"[load] reading cohort from {path}")
    df = pd.read_csv(path, usecols=["hadm_id"])
    ids = set(df["hadm_id"].unique())
    print(f"[load] found {len(ids)} unique hadm_ids in cohort")
    return ids


def load_dataframe(path: Path, limit: int | None, seed: int, cohort_ids: set[int] | None) -> pd.DataFrame:
    print(f"[load] reading notes from {path}...")
    
    # We load the full CSV. For MIMIC notes (compressed), this takes memory but is usually fine (3-4GB).
    # If OOM occurs here, we would need to use chunking, but filtering immediately helps.
    
    # Only read necessary columns if we can guess them, but 'text_column' is variable.
    # We always need 'hadm_id'.
    try:
        # peek first to check columns if needed, but read_csv is robust
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[error] failed to read {path}: {e}")
        raise

    initial_rows = len(df)
    print(f"[load] loaded {initial_rows} rows from disk")

    if cohort_ids is not None:
        if "hadm_id" not in df.columns:
             raise ValueError("Input file must contain 'hadm_id' to filter by cohort.")
        df = df[df["hadm_id"].isin(cohort_ids)]
        print(f"[filter] retained {len(df)}/{initial_rows} rows matching cohort")
    
    if limit is not None:
        df = df.sample(n=min(limit, len(df)), random_state=seed).reset_index(drop=True)
        print(f"[limit] subsampled to {len(df)} rows")
        
    return df


def prepare_texts(series: pd.Series, fill_value: str) -> List[str]:
    # Keep workflow deterministic and guard against NaNs / non-str
    return series.fillna(fill_value).astype(str).tolist()


def encode_text_column(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    dtype: torch.dtype,
    progress_interval: int,
    column_label: str,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    total_batches = (len(texts) + batch_size - 1) // batch_size
    progress = tqdm(iterator, total=total_batches, desc=f"Encoding[{column_label}]")
    
    for batch_idx, start in enumerate(progress):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            # ModernBERT handles attention mask automatically
            outputs = model(**inputs)
            # Use CLS token (index 0)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Cast to target dtype and move to CPU
            # Fix: Numpy doesn't support bfloat16. If we are in bfloat16, cast to float32 first.
            if dtype == torch.bfloat16:
                cls_embeddings = cls_embeddings.to(torch.float32).cpu().numpy()
            else:
                cls_embeddings = cls_embeddings.to(dtype).cpu().numpy()
            
            embeddings.append(cls_embeddings)
            
        # Explicit cleanup to avoid VRAM fragmentation
        del inputs, outputs, cls_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        if (batch_idx + 1) % progress_interval == 0:
            if device.type == 'cuda':
                mem_gb = torch.cuda.memory_allocated() / 1e9
                max_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                progress.set_postfix({"mem": f"{mem_gb:.1f}G", "peak": f"{max_mem_gb:.1f}G"})
            
    progress.close()
    return np.concatenate(embeddings, axis=0)


def save_embeddings(
    output_dir: Path,
    filename: str,
    hadm_ids: np.ndarray,
    embeddings: np.ndarray,
    extra_data: dict = None
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{filename}.npz"
    
    data_dict = {
        "embeddings": embeddings, 
        "hadm_ids": hadm_ids
    }
    if extra_data:
        data_dict.update(extra_data)
        
    np.savez_compressed(file_path, **data_dict)
    return file_path


def save_metadata(output_dir: Path, filename: str, manifest: dict, stats: EmbeddingStats) -> None:
    metadata = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "manifest": manifest,
        "stats": asdict(stats),
    }
    (output_dir / f"{filename}_manifest.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()

    if args.dry_run and args.limit is None:
        args.limit = 64

    # Determine output filename prefix
    if args.output_prefix:
        out_prefix = args.output_prefix
    else:
        # e.g. "discharge.csv.gz" -> "discharge"
        out_prefix = args.input_path.name.split('.')[0]
        # Append column name if not standard 'text'
        if args.text_column != "text":
            out_prefix += f"_{args.text_column}"

    if args.skip_existing:
        expected_out = args.output_dir / f"{out_prefix}.npz"
        if expected_out.exists():
            print(f"[skip] Output file already exists: {expected_out}")
            return

    print(f"[config] input={args.input_path} limit={args.limit} col={args.text_column} out={out_prefix}")
    
    # Load data
    cohort_ids = load_cohort_ids(args.cohort_path)
    df = load_dataframe(args.input_path, args.limit, args.seed, cohort_ids)
    
    if "hadm_id" not in df.columns:
        raise ValueError("Input file must contain 'hadm_id'.")
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not in dataframe. Found: {df.columns.tolist()}")

    device = resolve_device(args.device)
    print(f"[env] Using device: {device}")

    if args.metadata_only:
        print(df.describe(include="all"))
        return

    # Prepare Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Determine Dtypes
    if args.dtype == "float16":
        torch_dtype = torch.float16
        np_dtype = np.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
        np_dtype = np.float16 # Numpy has no bfloat16, save as float16
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32

    # Quantization Config
    quant_config = None
    if args.load_in_4bit:
        print("[model] Configuring 4-bit quantization (NF4)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load Model
    # Note: ModernBERT might require `trust_remote_code=True` depending on version, 
    # but BioClinical-ModernBERT is usually standard HF architecture now.
    model = AutoModel.from_pretrained(
        args.model_name, 
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map="auto" if args.load_in_4bit else None,
        attn_implementation="sdpa"
    )
    
    if not args.load_in_4bit:
        model = model.to(device)
    
    model.eval()

    print(f"[model] Loaded {args.model_name}")
    print(f"[model] Hidden size: {model.config.hidden_size}")
    print(f"[model] Dtype: {model.dtype}")
    
    # Process Texts
    hadm_ids = df["hadm_id"].to_numpy()
    
    # Extract extra metadata columns if they exist
    extra_data = {}
    for col in ["charttime", "note_type", "note_id", "subject_id"]:
        if col in df.columns:
            extra_data[col] = df[col].to_numpy()

    texts = prepare_texts(df[args.text_column], args.text_fill_value)
    print(f"[encode] Processing {len(texts)} texts from column '{args.text_column}'...")
    
    embeddings = encode_text_column(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=torch_dtype,
        progress_interval=args.progress_interval,
        column_label=args.text_column,
    )
    
    # Final cast for storage
    embeddings = embeddings.astype(np_dtype, copy=False)

    stats = EmbeddingStats(
        column=args.text_column,
        rows=embeddings.shape[0],
        embedding_dim=embeddings.shape[1],
        dtype=str(embeddings.dtype),
        max_length=args.max_length,
        batch_size=args.batch_size,
        model_name=args.model_name,
        quantization="4bit" if args.load_in_4bit else "none",
        mean=float(embeddings.mean()),
        std=float(embeddings.std()),
        min=float(embeddings.min()),
        max=float(embeddings.max()),
    )
    
    save_path = save_embeddings(args.output_dir, out_prefix, hadm_ids, embeddings, extra_data)
    print(f"[encode] Saved embeddings -> {save_path}")

    manifest = {
        "input_path": str(args.input_path),
        "output_dir": str(args.output_dir),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "model_name": args.model_name,
        "device": str(device),
        "rows": len(df),
        "cohort_filter": str(args.cohort_path) if args.cohort_path else None
    }
    save_metadata(args.output_dir, out_prefix, manifest, stats)
    print(f"[done] Metadata saved to {args.output_dir}")


if __name__ == "__main__":
    main()