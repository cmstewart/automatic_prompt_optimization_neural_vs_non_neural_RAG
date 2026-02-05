#!/usr/bin/env python3
"""
Train Sturdy Statistics index models for every unique document in:
  1. DocFinQA  (https://huggingface.co/datasets/kensho/DocFinQA)
  2. FinDoc-RAG (https://gitlab-core.supsi.ch/dti-idsia/ai-finance-papers/findoc-rag/-/tree/main/FinDoc-RAG_data/documents?ref_type=heads)

Outputs a CSV manifest mapping each trained index back to its source
dataset, row numbers, and training metadata. This is to facilitate
matching during subsequent retrieval, ensuring that we are always
querying the index (model) for the correct document

Usage:
    export STURDY_API_KEY="insert-API-key-here"
    python train_sturdy_indices.py

Requirements:
    pip install sturdy-stats-sdk pandas requests ijson
"""

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import ijson
import pandas as pd
import requests
from sturdystats.index import Index

# =============================================================================
# CONFIGURATION
# =============================================================================

STURDY_API_KEY = os.environ["STURDY_API_KEY"]

# DocFinQA dataset URL (HuggingFace, streamed as JSON)
DOCFINQA_URL = (
    "https://huggingface.co/datasets/kensho/DocFinQA/resolve/main/train.json"
)

# FinDoc-RAG GitLab repo (public)
FINDOCRAG_REPO_URL = (
    "https://gitlab-core.supsi.ch/dti-idsia/ai-finance-papers/findoc-rag.git"
)
FINDOCRAG_DOCS_SUBDIR = "FinDoc-RAG_data/documents"

# Sturdy training parameters
INDEX_PREFIX = "bulk_train"
MAX_PARAGRAPH_LENGTH = 1024
REGEX_SPLITTER = "donotsplitzzzzuuid"

# Output paths
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV = OUTPUT_DIR / "trained_indices_manifest.csv"

# Temporary directory for cloning
CLONE_DIR = Path("./tmp_findocrag_clone")


# =============================================================================
# 1.  LOAD DocFinQA — stream JSON and deduplicate by Context
# =============================================================================

def load_docfinqa() -> pd.DataFrame:
    """
    Stream DocFinQA, deduplicate by exact Context to get unique documents.

    Returns a DataFrame with one row per unique document:
        dataset_name        : "DocFinQA"
        doc_index           : 0-based index of the unique document
        original_row_indices: comma-separated original JSON row indices that
                              share this Context
        context             : the full document text
        context_hash        : SHA-256 of Context (for verification)
    """
    print("=" * 70)
    print("STEP 1a: Streaming DocFinQA from HuggingFace")
    print("=" * 70)
    print(f"URL: {DOCFINQA_URL}\n")

    # ---- stream and collect every row ----
    seen_contexts: OrderedDict = OrderedDict()   # hash -> [row indices]
    context_text: dict[str, str] = {}
    total_rows = 0

    with requests.get(DOCFINQA_URL, stream=True) as resp:
        resp.raise_for_status()
        parser = ijson.items(resp.raw, "item")

        for idx, item in enumerate(parser):
            total_rows = idx + 1
            ctx = item.get("Context", "")
            ctx_hash = hashlib.sha256(ctx.encode("utf-8")).hexdigest()

            if ctx_hash not in seen_contexts:
                seen_contexts[ctx_hash] = []
                context_text[ctx_hash] = ctx
            seen_contexts[ctx_hash].append(idx)

            if total_rows % 500 == 0:
                print(f"  Streamed {total_rows} rows  "
                      f"({len(seen_contexts)} unique docs so far)...")

    print(f"\nTotal rows in JSON:   {total_rows}")
    print(f"Unique documents:     {len(seen_contexts)}")

    # ---- build DataFrame ----
    rows = []
    for doc_index, (ctx_hash, orig_indices) in enumerate(seen_contexts.items()):
        rows.append({
            "dataset_name": "DocFinQA",
            "doc_index": doc_index,
            "original_row_indices": ",".join(str(i) for i in orig_indices),
            "context": context_text[ctx_hash],
            "context_hash": ctx_hash,
        })

    df = pd.DataFrame(rows)
    print(f"DocFinQA document DataFrame ready: {len(df)} rows\n")
    return df


# =============================================================================
# 2.  LOAD FinDoc-RAG — clone repo, read .md files
# =============================================================================

def load_findocrag() -> pd.DataFrame:
    """
    Clone the FinDoc-RAG GitLab repo and read all .md files from the
    documents directory.

    Returns a DataFrame with one row per document:
        dataset_name : "FinDoc-RAG"
        doc_index    : 0-based index (sorted by filename)
        filename     : original .md filename (e.g. "6253.md")
        context      : full Markdown text of the document
        context_hash : SHA-256 of the text
    """
    print("=" * 70)
    print("STEP 1b: Cloning FinDoc-RAG repository")
    print("=" * 70)

    # Clean up any previous clone
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)

    # Shallow clone (depth=1) to save bandwidth
    subprocess.run(
        ["git", "clone", "--depth", "1", FINDOCRAG_REPO_URL, str(CLONE_DIR)],
        check=True,
    )

    docs_dir = CLONE_DIR / FINDOCRAG_DOCS_SUBDIR
    md_files = sorted(docs_dir.glob("*.md"))
    print(f"\nFound {len(md_files)} .md files in {docs_dir}\n")

    rows = []
    for doc_index, md_path in enumerate(md_files):
        text = md_path.read_text(encoding="utf-8")
        rows.append({
            "dataset_name": "FinDoc-RAG",
            "doc_index": doc_index,
            "filename": md_path.name,
            "context": text,
            "context_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })

    df = pd.DataFrame(rows)
    print(f"FinDoc-RAG document DataFrame ready: {len(df)} rows\n")

    # Clean up clone
    shutil.rmtree(CLONE_DIR, ignore_errors=True)
    return df


# =============================================================================
# 3.  TRAIN Sturdy Statistics indices
# =============================================================================

def make_index_name(row: pd.Series) -> str:
    """Deterministic, collision-free index name."""
    ds_tag = row["dataset_name"].lower().replace("-", "")  # docfinqa / findocrag
    return f"{INDEX_PREFIX}_{ds_tag}_{row['doc_index']}"


def train_one_index(row: pd.Series) -> dict:
    """Train a single Sturdy Statistics index model for one document."""
    index_name = make_index_name(row)

    print(f"\n  [{row.name + 1}] Training: {index_name}")
    print(f"       Dataset:        {row['dataset_name']}")
    print(f"       Doc index:      {row['doc_index']}")
    print(f"       Context length: {len(row['context']):,} chars")

    # --- guard: abort if index already exists ---
    try:
        existing = Index(
            name="_check", API_key=STURDY_API_KEY
        ).listIndices(name_filter=index_name)
        if len(existing) > 0:
            print(f"       ⚠ Index already exists — skipping.")
            return {
                "index_name": index_name,
                "dataset_name": row["dataset_name"],
                "doc_index": row["doc_index"],
                "status": "skipped_exists",
            }
    except Exception:
        pass  # listIndices may fail on a fresh account; continue anyway

    try:
        start = time.time()

        idx = Index(name=index_name, API_key=STURDY_API_KEY)
        idx.upload([{"doc": row["context"]}])
        idx.train(
            fast=False,
            regex_paragraph_splitter=REGEX_SPLITTER,
            max_paragraph_length=MAX_PARAGRAPH_LENGTH,
            wait=True,
        )

        elapsed = time.time() - start
        print(f"       ✓ Done in {elapsed:.1f}s")

        return {
            "index_name": index_name,
            "dataset_name": row["dataset_name"],
            "doc_index": row["doc_index"],
            "status": "success",
            "train_time_sec": round(elapsed, 1),
        }

    except Exception as e:
        print(f"       ✗ Error: {e}")
        return {
            "index_name": index_name,
            "dataset_name": row["dataset_name"],
            "doc_index": row["doc_index"],
            "status": "error",
            "error": str(e),
        }


# =============================================================================
# 4.  MAIN
# =============================================================================

def main():
    # ---- Load both datasets ----
    df_docfinqa = load_docfinqa()
    df_findocrag = load_findocrag()

    # ---- Combine into a single training queue ----
    # Align columns: DocFinQA has original_row_indices; FinDoc-RAG has filename.
    # We keep both columns; they'll be NaN where not applicable.
    df_all = pd.concat([df_docfinqa, df_findocrag], ignore_index=True)
    df_all["index_name"] = df_all.apply(make_index_name, axis=1)

    print("=" * 70)
    print(f"STEP 2: Training {len(df_all)} Sturdy indices")
    print("=" * 70)
    print(f"  DocFinQA documents:  {len(df_docfinqa)}")
    print(f"  FinDoc-RAG documents: {len(df_findocrag)}")
    print(f"  Total:                {len(df_all)}")
    print()

    # ---- Train each index ----
    results = []
    for i, row in df_all.iterrows():
        result = train_one_index(row)
        results.append(result)

    df_results = pd.DataFrame(results)

    # ---- Merge training results back with source metadata ----
    # Keep everything except the context text due to its size
    df_manifest = df_all.drop(columns=["context"]).copy()

    # Standardize df_results columns (some rows may lack optional fields)
    for col in ("train_time_sec", "error"):
        if col not in df_results.columns:
            df_results[col] = None

    df_manifest = df_manifest.merge(
        df_results[["index_name", "status", "train_time_sec", "error"]],
        on="index_name",
        how="left",
    )

    # ---- Write manifest CSV ----
    df_manifest.to_csv(MANIFEST_CSV, index=False)

    # ---- Summary ----
    n_ok = (df_manifest["status"] == "success").sum()
    n_skip = (df_manifest["status"] == "skipped_exists").sum()
    n_err = (df_manifest["status"] == "error").sum()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Successful:  {n_ok}")
    print(f"  Skipped:     {n_skip}")
    print(f"  Errors:      {n_err}")
    print(f"\nManifest CSV written to: {MANIFEST_CSV}")
    print(f"Columns: {df_manifest.columns.tolist()}")


if __name__ == "__main__":
    main()
