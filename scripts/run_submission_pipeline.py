#!/usr/bin/env python3
"""Run the end-to-end reproducible submission pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full benchmark-and-paper pipeline")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--device", default=None, help="Torch device override for deep baselines")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    return parser.parse_args()


def run_step(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def main() -> None:
    args = parse_args()
    py = args.python

    if not args.skip_download:
        run_step([py, "scripts/download_data.py", "--source", "davis"])
        run_step([py, "scripts/process_dataset.py", "--source", "davis"])
        run_step([py, "scripts/download_data.py", "--source", "bindingdb"])
        run_step([py, "scripts/build_external_validation.py"])

    run_step([py, "scripts/generate_benchmark_splits.py"])

    if not args.skip_benchmark:
        benchmark_cmd = [
            py,
            "scripts/run_benchmark_suite.py",
            "--splits",
            "random",
            "cold_target",
            "cold_ligand",
            "scaffold",
            "both_new",
            "sequence_identity",
            "mutation_holdout",
            "--models",
            "ligand_only_ridge",
            "ridge_ensemble",
            "dual_tower_uq",
            "deepdta_exact",
            "graphdta_gcn_exact",
        ]
        if args.device:
            benchmark_cmd.extend(["--device", args.device])
        run_step(benchmark_cmd)

    if not args.skip_external:
        external_cmd = [
            py,
            "scripts/run_external_validation.py",
            "--models",
            "ligand_only_ridge",
            "ridge_ensemble",
            "dual_tower_uq",
            "deepdta_exact",
            "graphdta_gcn_exact",
        ]
        if args.device:
            external_cmd.extend(["--device", args.device])
        run_step(external_cmd)

    if not args.skip_assets:
        run_step([py, "scripts/generate_paper_assets.py"])


if __name__ == "__main__":
    main()
