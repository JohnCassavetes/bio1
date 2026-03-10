#!/usr/bin/env python3
"""Evaluate split-conformal and normalized-conformal prediction intervals."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate conformal prediction intervals on saved predictions"
    )
    parser.add_argument("--val-predictions", type=Path, default=RESULTS_DIR / "val_predictions.csv")
    parser.add_argument("--test-predictions", type=Path, default=RESULTS_DIR / "test_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.05])
    parser.add_argument("--active-threshold", type=float, default=6.0)
    return parser.parse_args()


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile."""
    n = len(residuals)
    rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), n - 1)
    return float(np.sort(residuals)[rank])


def evaluate_intervals(
    df: pd.DataFrame,
    radius: np.ndarray,
    *,
    active_threshold: float,
) -> Dict[str, float]:
    lower = df["predicted_p_activity"].to_numpy() - radius
    upper = df["predicted_p_activity"].to_numpy() + radius
    y_true = df["p_activity"].to_numpy()

    certain_active = lower >= active_threshold
    certain_inactive = upper < active_threshold
    uncertain = ~(certain_active | certain_inactive)

    payload = {
        "coverage": float(((y_true >= lower) & (y_true <= upper)).mean()),
        "mean_interval_width": float(np.mean(upper - lower)),
        "certain_active_fraction": float(certain_active.mean()),
        "certain_inactive_fraction": float(certain_inactive.mean()),
        "uncertain_fraction": float(uncertain.mean()),
    }
    if certain_active.any():
        payload["certain_active_precision"] = float((y_true[certain_active] >= active_threshold).mean())
    else:
        payload["certain_active_precision"] = None
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.val_predictions).copy()
    test_df = pd.read_csv(args.test_predictions).copy()

    val_abs_residual = np.abs(val_df["p_activity"] - val_df["predicted_p_activity"])
    val_normalized_residual = val_abs_residual / np.maximum(val_df["prediction_std"], 1e-6)

    payload: Dict[str, object] = {
        "active_threshold": args.active_threshold,
        "alphas": args.alphas,
        "methods": {},
    }
    rows: List[Dict[str, float]] = []

    for alpha in args.alphas:
        abs_q = conformal_quantile(val_abs_residual.to_numpy(), alpha)
        norm_q = conformal_quantile(val_normalized_residual.to_numpy(), alpha)

        abs_metrics = evaluate_intervals(
            test_df,
            radius=np.full(len(test_df), abs_q, dtype=float),
            active_threshold=args.active_threshold,
        )
        norm_metrics = evaluate_intervals(
            test_df,
            radius=norm_q * np.maximum(test_df["prediction_std"].to_numpy(), 1e-6),
            active_threshold=args.active_threshold,
        )

        payload["methods"][str(alpha)] = {
            "split_conformal": {
                "quantile": abs_q,
                **abs_metrics,
            },
            "normalized_conformal": {
                "quantile": norm_q,
                **norm_metrics,
            },
        }

        rows.append({"alpha": alpha, "method": "split_conformal", "quantile": abs_q, **abs_metrics})
        rows.append({"alpha": alpha, "method": "normalized_conformal", "quantile": norm_q, **norm_metrics})

    pd.DataFrame(rows).to_csv(args.output_dir / "conformal_metrics.csv", index=False)
    with open(args.output_dir / "conformal_metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n=== Conformal Summary ===")
    for alpha in args.alphas:
        metrics = payload["methods"][str(alpha)]["normalized_conformal"]
        print(
            f"alpha={alpha}: coverage={metrics['coverage']:.3f} "
            f"width={metrics['mean_interval_width']:.3f} "
            f"uncertain={metrics['uncertain_fraction']:.3f}"
        )
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
