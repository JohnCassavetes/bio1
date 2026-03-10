#!/usr/bin/env python3
"""Train and evaluate a baseline kinase-ligand ranking model."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinase_ligand_ranking.dataset import dataset_summary, load_dataset
from kinase_ligand_ranking.features import build_feature_matrix
from kinase_ligand_ranking.metrics import evaluate_split
from kinase_ligand_ranking.modeling import (
    RidgeEnsembleConfig,
    RidgeEnsembleRegressor,
    select_best_alpha,
)


PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ridge-ensemble baseline on processed kinase-ligand data"
    )
    parser.add_argument("--train-path", type=Path, default=PROCESSED_DIR / "train.csv")
    parser.add_argument("--val-path", type=Path, default=PROCESSED_DIR / "val.csv")
    parser.add_argument("--test-path", type=Path, default=PROCESSED_DIR / "test.csv")
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR / "baseline")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "baseline")
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
        help="Candidate ridge regularization strengths for validation selection",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=6.0,
        help="pActivity threshold used for ROC-AUC and enrichment metrics",
    )
    return parser.parse_args()


def attach_predictions(
    df: pd.DataFrame,
    predicted_mean,
    predicted_std,
) -> pd.DataFrame:
    """Attach prediction outputs to a copy of the split DataFrame."""
    result = df.copy()
    result["predicted_p_activity"] = predicted_mean
    result["prediction_std"] = predicted_std
    result["absolute_error"] = (result["predicted_p_activity"] - result["p_activity"]).abs()
    return result


def round_nested(obj):
    """Round floats recursively for JSON output."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {key: round_nested(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_nested(value) for value in obj]
    return obj


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(args.train_path)
    val_df = load_dataset(args.val_path)
    test_df = load_dataset(args.test_path)

    X_train, feature_metadata = build_feature_matrix(
        train_df,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )
    X_val, _ = build_feature_matrix(
        val_df,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )
    X_test, _ = build_feature_matrix(
        test_df,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )

    y_train = train_df["p_activity"].to_numpy()
    y_val = val_df["p_activity"].to_numpy()
    y_test = test_df["p_activity"].to_numpy()

    best_alpha, alpha_trials = select_best_alpha(
        X_train,
        y_train,
        X_val,
        y_val,
        candidate_alphas=args.alphas,
    )

    calibration_ensemble = RidgeEnsembleRegressor(
        RidgeEnsembleConfig(
            alpha=best_alpha,
            ensemble_size=args.ensemble_size,
            bootstrap_fraction=args.bootstrap_fraction,
            random_seed=args.seed,
        )
    )
    calibration_ensemble.fit(X_train, y_train)
    uncertainty_scale = calibration_ensemble.calibrate_uncertainty(X_val, y_val)
    calibration_ensemble.uncertainty_scale = uncertainty_scale

    val_pred_mean, val_pred_std = calibration_ensemble.predict(X_val)
    val_predictions = attach_predictions(
        val_df,
        val_pred_mean,
        val_pred_std,
    )

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val, _ = build_feature_matrix(
        train_val_df,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )
    y_train_val = train_val_df["p_activity"].to_numpy()

    ensemble = RidgeEnsembleRegressor(
        RidgeEnsembleConfig(
            alpha=best_alpha,
            ensemble_size=args.ensemble_size,
            bootstrap_fraction=args.bootstrap_fraction,
            random_seed=args.seed,
        )
    )
    ensemble.fit(X_train_val, y_train_val)
    ensemble.uncertainty_scale = uncertainty_scale

    train_val_pred_mean, train_val_pred_std = ensemble.predict(X_train_val)
    test_pred_mean, test_pred_std = ensemble.predict(X_test)

    train_val_predictions = attach_predictions(
        train_val_df,
        train_val_pred_mean,
        train_val_pred_std,
    )
    test_predictions = attach_predictions(
        test_df,
        test_pred_mean,
        test_pred_std,
    )

    train_val_metrics = evaluate_split(
        train_val_predictions,
        active_threshold=args.active_threshold,
    )
    test_metrics = evaluate_split(
        test_predictions,
        active_threshold=args.active_threshold,
    )

    metadata: Dict[str, object] = {
        "feature_metadata": feature_metadata,
        "dataset_summary": {
            "train": dataset_summary(train_df),
            "val": dataset_summary(val_df),
            "test": dataset_summary(test_df),
            "train_val": dataset_summary(train_val_df),
        },
        "selected_alpha": best_alpha,
        "uncertainty_scale": uncertainty_scale,
        "alpha_trials": alpha_trials,
    }

    model_path = args.model_dir / "ridge_ensemble.joblib"
    ensemble.save(model_path, metadata=metadata)

    val_predictions.to_csv(args.results_dir / "val_predictions.csv", index=False)
    train_val_predictions.to_csv(args.results_dir / "train_val_predictions.csv", index=False)
    test_predictions.to_csv(args.results_dir / "test_predictions.csv", index=False)
    test_metrics["per_target"].to_csv(args.results_dir / "test_per_target_metrics.csv", index=False)

    top_hits = (
        test_predictions.sort_values(["target_id", "predicted_p_activity"], ascending=[True, False])
        .groupby("target_id", as_index=False)
        .head(5)
    )
    top_hits.to_csv(args.results_dir / "test_top_ranked_hits.csv", index=False)

    metrics_payload = {
        "model": {
            "type": "ridge_ensemble",
            "selected_alpha": best_alpha,
            "ensemble_size": args.ensemble_size,
            "bootstrap_fraction": args.bootstrap_fraction,
            "fingerprint_bits": args.fingerprint_bits,
            "fingerprint_radius": args.fingerprint_radius,
            "uncertainty_scale": uncertainty_scale,
        },
        "validation_model_selection": alpha_trials,
        "train_val": train_val_metrics["overall"],
        "test": test_metrics["overall"],
    }
    with open(args.results_dir / "metrics.json", "w") as fh:
        json.dump(round_nested(metrics_payload), fh, indent=2)

    print("\n=== Baseline Training Summary ===")
    print(f"Selected alpha: {best_alpha}")
    print(f"Train+Val RMSE:  {train_val_metrics['overall']['rmse']:.3f}")
    print(f"Test RMSE:       {test_metrics['overall']['rmse']:.3f}")
    print(f"Test Spearman:   {test_metrics['overall']['spearman']:.3f}")
    mean_enrichment = test_metrics["overall"]["mean_top_10pct_enrichment"]
    if mean_enrichment is not None:
        print(f"Test EF10%:      {mean_enrichment:.3f}")
    print(f"\nSaved model to:  {model_path}")
    print(f"Saved outputs to: {args.results_dir}")


if __name__ == "__main__":
    main()
