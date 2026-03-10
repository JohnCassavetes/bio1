#!/usr/bin/env python3
"""Run comparable models across benchmark split families."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kinase_ligand_ranking.dataset import load_dataset
from kinase_ligand_ranking.features import build_feature_components, build_feature_matrix
from kinase_ligand_ranking.metrics import evaluate_split
from kinase_ligand_ranking.modeling import (
    RidgeEnsembleConfig,
    RidgeEnsembleRegressor,
    select_best_alpha,
)
from kinase_ligand_ranking.neural_modeling import (
    DualTowerRankerConfig,
    DualTowerUncertaintyRanker,
)


BENCHMARK_DIR = BASE_DIR / "data" / "benchmark"
RESULTS_DIR = BASE_DIR / "results" / "benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a comparable benchmark suite across split families"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=BENCHMARK_DIR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["random", "cold_target", "cold_ligand", "scaffold", "both_new"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ligand_only_ridge", "ridge_ensemble", "dual_tower_uq"],
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def attach_predictions(
    df: pd.DataFrame,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
) -> pd.DataFrame:
    result = df.copy()
    result["predicted_p_activity"] = predicted_mean
    result["prediction_std"] = predicted_std
    result["absolute_error"] = (result["predicted_p_activity"] - result["p_activity"]).abs()
    return result


def run_ligand_only_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    alphas: List[float],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_components, _ = build_feature_components(train_df)
    val_components, _ = build_feature_components(val_df)
    test_components, _ = build_feature_components(test_df)

    X_train = train_components["ligand"]
    X_val = val_components["ligand"]
    X_test = test_components["ligand"]
    y_train = train_df["p_activity"].to_numpy()
    y_val = val_df["p_activity"].to_numpy()

    best_alpha, alpha_trials = select_best_alpha(
        X_train,
        y_train,
        X_val,
        y_val,
        candidate_alphas=alphas,
    )

    model = RidgeEnsembleRegressor(
        RidgeEnsembleConfig(alpha=best_alpha, random_seed=seed, ensemble_size=8)
    )
    model.fit(X_train, y_train)
    model.calibrate_uncertainty(X_val, y_val)
    test_mean, test_std = model.predict(X_test)
    predictions = attach_predictions(test_df, test_mean, test_std)
    metrics = evaluate_split(predictions)
    return predictions, {
        "selected_alpha": best_alpha,
        "alpha_trials": alpha_trials,
        "metrics": metrics["overall"],
    }


def run_ridge_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    alphas: List[float],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    X_train, feature_metadata = build_feature_matrix(train_df)
    X_val, _ = build_feature_matrix(val_df)
    X_test, _ = build_feature_matrix(test_df)
    y_train = train_df["p_activity"].to_numpy()
    y_val = val_df["p_activity"].to_numpy()

    best_alpha, alpha_trials = select_best_alpha(
        X_train,
        y_train,
        X_val,
        y_val,
        candidate_alphas=alphas,
    )

    model = RidgeEnsembleRegressor(
        RidgeEnsembleConfig(alpha=best_alpha, random_seed=seed, ensemble_size=8)
    )
    model.fit(X_train, y_train)
    model.calibrate_uncertainty(X_val, y_val)
    test_mean, test_std = model.predict(X_test)
    predictions = attach_predictions(test_df, test_mean, test_std)
    metrics = evaluate_split(predictions)
    return predictions, {
        "selected_alpha": best_alpha,
        "alpha_trials": alpha_trials,
        "feature_metadata": feature_metadata,
        "metrics": metrics["overall"],
    }


def run_dual_tower_uq(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_components, feature_metadata = build_feature_components(train_df)
    val_components, _ = build_feature_components(val_df)
    test_components, _ = build_feature_components(test_df)

    config = DualTowerRankerConfig(
        ligand_dim=train_components["ligand"].shape[1],
        target_dim=train_components["target"].shape[1],
        context_dim=train_components["assay"].shape[1] + train_components["source"].shape[1],
        embed_dim=32,
        n_estimators=128,
        min_samples_leaf=2,
        random_seed=seed,
    )
    model = DualTowerUncertaintyRanker(config)
    model.fit(
        train_components=train_components,
        y_train=train_df["p_activity"].to_numpy(),
        train_target_ids=train_df["target_id"].tolist(),
        val_components=val_components,
        y_val=val_df["p_activity"].to_numpy(),
    )
    test_mean, test_std = model.predict(test_components)
    predictions = attach_predictions(test_df, test_mean, test_std)
    metrics = evaluate_split(predictions)
    return predictions, {
        "feature_metadata": feature_metadata,
        "uncertainty_scale": model.uncertainty_scale,
        "training_history": model.training_history,
        "metrics": metrics["overall"],
    }


def model_runner(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    alphas: List[float],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if model_name == "ligand_only_ridge":
        return run_ligand_only_ridge(train_df, val_df, test_df, alphas=alphas, seed=seed)
    if model_name == "ridge_ensemble":
        return run_ridge_ensemble(train_df, val_df, test_df, alphas=alphas, seed=seed)
    if model_name == "dual_tower_uq":
        return run_dual_tower_uq(train_df, val_df, test_df, seed=seed)
    raise ValueError(f"Unsupported model: {model_name}")


def round_nested(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_nested(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [round_nested(inner) for inner in value]
    return value


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    results_payload: Dict[str, object] = {}

    for split_name in args.splits:
        split_dir = args.benchmark_dir / split_name
        train_df = load_dataset(split_dir / "train.csv")
        val_df = load_dataset(split_dir / "val.csv")
        test_df = load_dataset(split_dir / "test.csv")

        split_payload: Dict[str, object] = {}
        split_results_dir = args.results_dir / split_name
        split_results_dir.mkdir(parents=True, exist_ok=True)

        for model_name in args.models:
            model_results_dir = split_results_dir / model_name
            model_results_dir.mkdir(parents=True, exist_ok=True)
            predictions, metadata = model_runner(
                model_name,
                train_df,
                val_df,
                test_df,
                alphas=args.alphas,
                seed=args.seed,
            )
            predictions.to_csv(model_results_dir / "test_predictions.csv", index=False)
            with open(model_results_dir / "metrics.json", "w") as fh:
                json.dump(round_nested(metadata), fh, indent=2)

            overall_metrics = metadata["metrics"]
            summary_rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    **overall_metrics,
                }
            )
            split_payload[model_name] = metadata

        results_payload[split_name] = split_payload

    summary_df = pd.DataFrame(summary_rows).sort_values(["split", "rmse", "model"])
    summary_df.to_csv(args.results_dir / "summary.csv", index=False)
    with open(args.results_dir / "summary.json", "w") as fh:
        json.dump(round_nested(results_payload), fh, indent=2)

    print("\n=== Benchmark Summary ===")
    for split_name, group in summary_df.groupby("split"):
        best_row = group.sort_values(["rmse", "model"]).iloc[0]
        print(
            f"{split_name}: best_rmse={best_row['rmse']:.3f} "
            f"best_model={best_row['model']} spearman={best_row['spearman']:.3f}"
        )
    print(f"\nSaved outputs to: {args.results_dir}")


if __name__ == "__main__":
    main()
