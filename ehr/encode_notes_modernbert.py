#!/usr/bin/env python3
"""
ModernBERT note encoder.

Promotes the logic from notebooks/notebooks_dc/04B_encode_notes_bioclinical_modernbert.ipynb
into a reproducible CLI so we can run the discharge + radiology embedding job on HPC.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
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
        "--output-dir",
        type=Path,
        default=Path("notebooks/notebooks_dc/_data/processed/04B_encode_notes_bioclinical_modernbert"),
        help="Directory for embeddings and metadata.",
    )
    parser.add_argument(
        "--text-columns",
        nargs="+",
        default=["discharge_text", "radiology_text"],
        help="Column names to encode (each becomes a separate embedding file).",
    )
    parser.add_argument(
        "--model-name",
        default="thomas-sounack/bioclinical-modernbert-base",
        help="Hugging Face model repository.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size used for encoding.")
    parser.add_argument("--max-length", type=int, default=8192, help="Tokenizer max_length / truncation window.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Precision of the saved embeddings.",
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
        default=50,
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


def load_dataframe(path: Path, limit: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit is not None:
        df = df.sample(n=min(limit, len(df)), random_state=seed).reset_index(drop=True)
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
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            cls_embeddings = cls_embeddings.to(dtype).cpu().numpy()
            embeddings.append(cls_embeddings)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (batch_idx + 1) % progress_interval == 0:
            print(
                f"[progress] batches={batch_idx + 1}, rows={end}, "
                f"device={device}, mem_alloc={torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0:.2f}GB",
            )
    progress.close()
    return np.concatenate(embeddings, axis=0)


def save_embeddings(
    output_dir: Path,
    column: str,
    hadm_ids: np.ndarray,
    embeddings: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{column}_modernbert_embeddings.npz"
    np.savez_compressed(file_path, embeddings=embeddings, hadm_ids=hadm_ids)
    return file_path


def save_metadata(output_dir: Path, manifest: dict, stats: Iterable[EmbeddingStats]) -> None:
    metadata = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "manifest": manifest,
        "embeddings": [asdict(s) for s in stats],
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()

    if args.dry_run and args.limit is None:
        args.limit = 64

    print(f"[config] input={args.input_path} limit={args.limit} columns={args.text_columns}")
    df = load_dataframe(args.input_path, args.limit, args.seed)
    if "hadm_id" not in df.columns:
        raise ValueError("Input file must contain 'hadm_id'.")

    device = resolve_device(args.device)
    print(f"[env] Using device: {device}")

    if args.metadata_only:
        print(df.describe(include="all"))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    torch_dtype = torch.float16 if (args.dtype == "float16" and device.type == "cuda") else torch.float32
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    print(f"[model] Loaded {args.model_name} with hidden size {model.config.hidden_size}")

    results: List[EmbeddingStats] = []
    hadm_ids = df["hadm_id"].to_numpy()
    for column in args.text_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in dataframe.")
        texts = prepare_texts(df[column], args.text_fill_value)
        print(f"[encode] column={column} rows={len(texts)}")
        embeddings = encode_text_column(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dtype=torch_dtype,
            progress_interval=args.progress_interval,
            column_label=column,
        )
        np_dtype = np.float16 if args.dtype == "float16" else np.float32
        embeddings = embeddings.astype(np_dtype, copy=False)

        stats = EmbeddingStats(
            column=column,
            rows=embeddings.shape[0],
            embedding_dim=embeddings.shape[1],
            dtype=str(embeddings.dtype),
            max_length=args.max_length,
            batch_size=args.batch_size,
            model_name=args.model_name,
            mean=float(embeddings.mean()),
            std=float(embeddings.std()),
            min=float(embeddings.min()),
            max=float(embeddings.max()),
        )
        save_path = save_embeddings(args.output_dir, column, hadm_ids, embeddings)
        print(f"[encode] saved {column} embeddings -> {save_path}")
        results.append(stats)

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
    }
    save_metadata(args.output_dir, manifest, results)
    print(f"[done] metadata saved to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
