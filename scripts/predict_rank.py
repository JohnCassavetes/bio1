#!/usr/bin/env python3
"""Score and rank ligand-target pairs with a trained baseline model."""

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinase_ligand_ranking.features import build_feature_matrix
from kinase_ligand_ranking.modeling import RidgeEnsembleRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict pActivity and rank ligand-target pairs"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV with columns smiles,target_id and optional target_sequence,affinity_type,source",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "models" / "baseline" / "ridge_ensemble.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "results" / "baseline" / "predictions.csv",
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input).copy()
    for column, default in [
        ("target_sequence", ""),
        ("affinity_type", "KD"),
        ("source", "inference"),
        ("target_label", None),
        ("measurement_count", 1),
    ]:
        if column not in df.columns:
            df[column] = df["target_id"] if default is None else default

    model, metadata = RidgeEnsembleRegressor.load(args.model)
    feature_metadata = metadata.get("feature_metadata", {})

    X, _ = build_feature_matrix(
        df,
        fingerprint_bits=int(feature_metadata.get("fingerprint_bits", 1024)),
        fingerprint_radius=int(feature_metadata.get("fingerprint_radius", 2)),
    )
    predicted_mean, predicted_std = model.predict(X)

    scored = df.copy()
    scored["predicted_p_activity"] = predicted_mean
    scored["prediction_std"] = predicted_std
    scored = scored.sort_values(
        ["target_id", "predicted_p_activity"],
        ascending=[True, False],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)

    print("\n=== Top Ranked Predictions ===")
    print(
        scored[["target_id", "smiles", "predicted_p_activity", "prediction_std"]]
        .head(args.top_k)
        .to_string(index=False)
    )
    print(f"\nSaved predictions to: {args.output}")


if __name__ == "__main__":
    main()
