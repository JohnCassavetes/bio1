#!/usr/bin/env python3
"""Evaluate benchmark models on an external-source test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from kinase_ligand_ranking.dataset import load_dataset
from scripts.run_benchmark_suite import model_runner, round_nested


PROCESSED_DIR = BASE_DIR / "data" / "processed"
EXTERNAL_DIR = BASE_DIR / "data" / "external" / "bindingdb"
RESULTS_DIR = BASE_DIR / "results" / "external_validation" / "bindingdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external-source validation with Davis train/val and external test")
    parser.add_argument("--train-path", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--val-path", type=Path, default=PROCESSED_DIR / "val.csv")
    parser.add_argument("--test-path", type=Path, default=EXTERNAL_DIR / "bindingdb_external.csv")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ligand_only_ridge", "ridge_ensemble", "dual_tower_uq", "deepdta_exact", "graphdta_gcn_exact"],
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(args.train_path)
    val_df = load_dataset(args.val_path)
    test_df = load_dataset(args.test_path)

    summary_rows: List[Dict[str, object]] = []
    payload: Dict[str, object] = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "test_path": str(args.test_path),
        "models": {},
    }

    for model_name in args.models:
        model_results_dir = args.results_dir / model_name
        model_results_dir.mkdir(parents=True, exist_ok=True)
        predictions, metadata = model_runner(
            model_name,
            train_df,
            val_df,
            test_df,
            alphas=args.alphas,
            seed=args.seed,
            device=args.device,
        )
        predictions.to_csv(model_results_dir / "test_predictions.csv", index=False)
        with open(model_results_dir / "metrics.json", "w") as fh:
            json.dump(round_nested(metadata), fh, indent=2)
        summary_rows.append({"evaluation": "davis_to_external", "model": model_name, **metadata["metrics"]})
        payload["models"][model_name] = metadata

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["rmse", "model"])
        .reset_index(drop=True)
    )
    summary_df.to_csv(args.results_dir / "summary.csv", index=False)
    with open(args.results_dir / "summary.json", "w") as fh:
        json.dump(round_nested(payload), fh, indent=2)

    print(summary_df.to_string(index=False))
    print(f"\nSaved external validation results to: {args.results_dir}")


if __name__ == "__main__":
    main()
