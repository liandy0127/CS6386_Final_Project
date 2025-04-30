#!/usr/bin/env python3
"""
ingest.py – Canonicalise Amazon‑style datasets with ChatGPT assistance.

This utility converts arbitrary CSV / JSON / JSON‑Lines extracts (e.g.
MovieLens‑like dumps or ad‑hoc vendor exports) into the header layouts
expected by our Amazon Review data pipeline.

CLI usage
---------
```bash
python ingest.py <path‑to‑file|dir> [--type review|meta] [-o out.csv|out/]
```
If *<path>* is a directory, every supported file inside it is processed.

Environment variable `OPENAI_API_KEY` **must** be set (no key is hard‑coded).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from openai import OpenAI  # SDK ≥ 1.0.0
except ImportError:  # pragma: no cover
    sys.exit("✖ openai package missing – run `pip install openai` (>=1.0.0)")

# ─── OpenAI client ──────────────────────────────────────────────────────────
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
api_key = ""

if not api_key:
    sys.exit(f"✖ environment variable {OPENAI_API_KEY_ENV} not set")
client = OpenAI(api_key=api_key)

# ─── Canonical schemas ──────────────────────────────────────────────────────
REVIEW_HDRS = [
    "rating",
    "title",
    "text",
    "images",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
]
META_HDRS = [
    "main_category",
    "title",
    "average_rating",
    "rating_number",
    "features",
    "description",
    "price",
    "images",
    "videos",
    "store",
    "categories",
    "details",
    "parent_asin",
    "bought_together",
]
SCHEMAS: Dict[str, List[str]] = {"review": REVIEW_HDRS, "meta": META_HDRS}

# ─── ChatGPT mapping helper ─────────────────────────────────────────────────

def ask_chatgpt_for_mapping(raw_headers: List[str], target_headers: List[str]) -> Dict[str, str]:
    """Return a mapping dict from *raw header* → *canonical header* via ChatGPT."""
    system_msg = (
        "You are an expert data engineer. Given a list of raw column names, "
        "produce a JSON object that maps each raw name to its corresponding "
        "canonical header from the provided list. Only output valid JSON – no markdown fences."
    )
    user_msg = json.dumps({"raw_headers": raw_headers, "canonical_headers": target_headers}, ensure_ascii=False)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # change if you have access to gpt‑4o or other model
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    content = response.choices[0].message.content.strip()
    try:
        mapping: Dict[str, str] = json.loads(content)
    except json.JSONDecodeError as err:
        raise ValueError(f"ChatGPT returned non‑JSON mapping: {content}") from err
    return mapping

# ─── PII scrubbing & user hashing ───────────────────────────────────────────
_PII_PATTERNS = [
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),  # emails
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # US phones
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),  # SSN
    re.compile(r"\b\d{13,19}\b"),  # CC numbers
]

def scrub_pii(val: Any) -> Any:
    if not isinstance(val, str):
        return val
    for pat in _PII_PATTERNS:
        val = pat.sub("[REDACTED]", val)
    return val

# ─── DataFrame transformations ──────────────────────────────────────────────

def map_headers(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    target = SCHEMAS[schema]
    mapping = ask_chatgpt_for_mapping(df.columns.tolist(), target)
    df = df.rename(columns=mapping)
    for h in target:
        if h not in df.columns:
            df[h] = pd.NA
    return df[target]


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].apply(scrub_pii)
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype("category").cat.codes
    return df

# ─── IO helpers ─────────────────────────────────────────────────────────────
SUPPORTED_EXTS = (".csv", ".json", ".jsonl", ".ndjson")

def load(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".json":
        return pd.read_json(path, orient="records")
    if ext in {".jsonl", ".ndjson"}:
        return pd.read_json(path, orient="records", lines=True)
    raise ValueError(f"Unsupported file type: {path}")


def guess_schema(cols: List[str]) -> str:
    return "review" if len(set(cols) & set(REVIEW_HDRS)) >= len(set(cols) & set(META_HDRS)) else "meta"

# ─── Processing helpers ─────────────────────────────────────────────────────

def process_file(path: Path, schema_override: Optional[str], out_dir: Optional[Path]) -> None:
    df = load(path)
    schema = schema_override or guess_schema(df.columns.tolist())
    df = map_headers(df, schema)
    df = sanitize(df)
    out_path = (out_dir or path.parent) / f"{path.stem}_ingested.csv"
    df.to_csv(out_path, index=False)
    rel = out_path.relative_to(Path.cwd()) if out_path.is_absolute() else out_path
    print(f"✓ {path.name} → {rel}")

# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Ingest & canonicalise datasets")
    p.add_argument("input", type=Path, help="File or directory to ingest")
    p.add_argument("--type", choices=("review", "meta"), help="Force schema type")
    p.add_argument("-o", "--output", type=Path, help="Output CSV (file) or directory (dir)")
    args = p.parse_args()

    if args.input.is_dir():
        files: List[Path] = [f for ext in SUPPORTED_EXTS for f in sorted(args.input.glob(f"*{ext}"))]
        if not files:
            sys.exit("✖ no supported files found in directory")
        for f in files:
            process_file(f, args.type, args.output)
    else:
        if args.output and args.output.is_dir():
            process_file(args.input, args.type, args.output)
        elif args.output:
            df = load(args.input)
            schema = args.type or guess_schema(df.columns.tolist())
            df = map_headers(df, schema)
            df = sanitize(df)
            df.to_csv(args.output, index=False)
            print(f"✓ Saved canonicalised file → {args.output}")
        else:
            process_file(args.input, args.type, None)

if __name__ == "__main__":
    main()

