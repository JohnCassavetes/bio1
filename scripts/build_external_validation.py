#!/usr/bin/env python3
"""Build an external validation dataset with the benchmark schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.process_dataset import (
    compute_pactivity,
    deduplicate_and_aggregate,
    filter_and_clean,
    load_bindingdb_json_files,
)


RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "external" / "bindingdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a processed external validation set")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--affinity-types",
        nargs="+",
        default=["KD"],
        help="Affinity types retained for external evaluation; default keeps KD for Davis comparability",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_bindingdb_json_files(args.raw_dir)
    cleaned_df, filter_stats = filter_and_clean(df, allowed_types=[value.upper() for value in args.affinity_types])
    dedup_df, dedup_stats = deduplicate_and_aggregate(cleaned_df)
    processed_df = compute_pactivity(dedup_df)
    processed_df = processed_df[processed_df["target_sequence"].astype(str).str.len() > 0].copy()

    ordered_columns = [
        "smiles",
        "target_id",
        "target_label",
        "target_sequence",
        "affinity_type",
        "activity_label",
        "affinity_nm",
        "p_activity",
        "measurement_count",
        "source",
    ]
    processed_df = processed_df[ordered_columns].sort_values(["target_id", "smiles"]).reset_index(drop=True)
    processed_df.to_csv(args.output_dir / "bindingdb_external.csv", index=False)

    summary = {
        "rows": int(len(processed_df)),
        "targets": int(processed_df["target_id"].nunique()),
        "ligands": int(processed_df["smiles"].nunique()),
        "affinity_types": sorted(processed_df["affinity_type"].astype(str).unique().tolist()),
        "filter_stats": filter_stats,
        "dedup_stats": dedup_stats,
    }
    with open(args.output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"Built external validation set with {summary['rows']} rows, "
        f"{summary['targets']} targets, and {summary['ligands']} ligands."
    )
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
