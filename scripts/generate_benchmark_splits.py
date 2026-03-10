#!/usr/bin/env python3
"""Generate multiple benchmark split families from the processed full dataset."""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinase_ligand_ranking.dataset import load_dataset
from kinase_ligand_ranking.splits import generate_split_bundle


PROCESSED_PATH = BASE_DIR / "data" / "processed" / "kinase_binding_data.csv"
BENCHMARK_DIR = BASE_DIR / "data" / "benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark split bundles for kinase-ligand data"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=PROCESSED_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIR,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["random", "cold_target", "cold_ligand", "scaffold", "both_new", "mutation_holdout"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for split_name in args.splits:
        bundle = generate_split_bundle(
            df,
            split_type=split_name,
            random_seed=args.seed,
        )
        split_dir = args.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        bundle.train_df.to_csv(split_dir / "train.csv", index=False)
        bundle.val_df.to_csv(split_dir / "val.csv", index=False)
        bundle.test_df.to_csv(split_dir / "test.csv", index=False)
        with open(split_dir / "manifest.json", "w") as fh:
            json.dump(bundle.manifest, fh, indent=2)
        manifest[split_name] = bundle.manifest

    with open(args.output_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n=== Generated Benchmark Splits ===")
    for split_name in args.splits:
        stats = manifest[split_name]
        print(
            f"{split_name}: train={stats['train_rows']} val={stats['val_rows']} "
            f"test={stats['test_rows']} discarded={stats['discarded_rows']}"
        )
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
