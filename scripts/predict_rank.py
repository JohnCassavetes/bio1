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

from kinase_ligand_ranking.features import AFFINITY_TYPES, SOURCES, build_feature_matrix
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


def _metadata_categories(metadata: dict, key: str, fallback: list[str]) -> list[str]:
    feature_metadata = metadata.get("feature_metadata", {})
    values = feature_metadata.get(key, fallback)
    return [str(value) for value in values]


def prepare_inference_dataframe(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    result = df.copy()
    supported_affinity_types = _metadata_categories(metadata, "affinity_types", AFFINITY_TYPES)
    supported_sources = _metadata_categories(metadata, "source_categories", SOURCES)

    if "target_label" not in result.columns:
        result["target_label"] = result["target_id"]
    if "measurement_count" not in result.columns:
        result["measurement_count"] = 1
    if "target_sequence" not in result.columns:
        result["target_sequence"] = ""

    if "affinity_type" not in result.columns:
        result["affinity_type"] = metadata.get("default_inference_affinity_type", supported_affinity_types[0])
    result["affinity_type"] = result["affinity_type"].astype(str).str.upper().str.strip()

    if "source" not in result.columns:
        result["source"] = metadata.get("default_inference_source", supported_sources[-1])
    result["source"] = result["source"].astype(str).str.lower().str.strip()

    invalid_affinity_types = sorted(set(result["affinity_type"]) - set(supported_affinity_types))
    if invalid_affinity_types:
        raise ValueError(
            "Unsupported affinity_type values for this model: "
            f"{invalid_affinity_types}. Supported values: {supported_affinity_types}"
        )

    invalid_sources = sorted(set(result["source"]) - set(supported_sources))
    if invalid_sources:
        raise ValueError(
            "Unsupported source values for this model: "
            f"{invalid_sources}. Supported values: {supported_sources}"
        )

    result["target_sequence"] = result["target_sequence"].fillna("").astype(str)
    return result


def main() -> None:
    args = parse_args()

    model, metadata = RidgeEnsembleRegressor.load(args.model)
    df = prepare_inference_dataframe(pd.read_csv(args.input), metadata)
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
