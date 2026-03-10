#!/usr/bin/env python3
"""Evaluate budget-constrained prioritization policies on saved predictions."""

import argparse
import json
from math import erf, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate point-estimate and uncertainty-aware selection policies"
    )
    parser.add_argument(
        "--val-predictions",
        type=Path,
        default=RESULTS_DIR / "val_predictions.csv",
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=RESULTS_DIR / "test_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--risk-grid",
        type=float,
        nargs="+",
        default=[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        help="Candidate lambdas for the score mean - lambda * std",
    )
    return parser.parse_args()


def normal_survival(z_values: np.ndarray) -> np.ndarray:
    """Return 1 - Phi(z) for a NumPy array."""
    return 0.5 * (1.0 - np.vectorize(erf)(z_values / sqrt(2.0)))


def add_policy_scores(df: pd.DataFrame, active_threshold: float) -> pd.DataFrame:
    """Add probability-of-activity policy scores."""
    result = df.copy()
    safe_std = result["prediction_std"].clip(lower=1e-6)
    z_values = (active_threshold - result["predicted_p_activity"]) / safe_std
    result["prob_active"] = normal_survival(z_values.to_numpy())
    return result


def policy_score(df: pd.DataFrame, policy: str, risk_lambda: float = 0.0) -> pd.Series:
    """Return a ranking score for a named policy."""
    if policy == "mean":
        return df["predicted_p_activity"]
    if policy == "prob_active":
        return df["prob_active"]
    if policy == "risk_adjusted":
        return df["predicted_p_activity"] - (risk_lambda * df["prediction_std"])
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_policy(
    df: pd.DataFrame,
    *,
    budget: int,
    active_threshold: float,
    policy: str,
    risk_lambda: float = 0.0,
) -> Dict[str, float]:
    """Compute average per-target utility for a prioritization policy."""
    hit_rates: List[float] = []
    mean_activities: List[float] = []
    regrets: List[float] = []

    for _, group in df.groupby("target_id"):
        scored = group.assign(score=policy_score(group, policy, risk_lambda))
        selected = scored.sort_values("score", ascending=False).head(budget)
        oracle = group.sort_values("p_activity", ascending=False).head(budget)

        hit_rates.append(float((selected["p_activity"] >= active_threshold).mean()))
        mean_activities.append(float(selected["p_activity"].mean()))
        regrets.append(float(oracle["p_activity"].mean() - selected["p_activity"].mean()))

    return {
        "hit_rate_at_budget": float(np.mean(hit_rates)),
        "mean_p_activity_at_budget": float(np.mean(mean_activities)),
        "regret_at_budget": float(np.mean(regrets)),
    }


def tune_risk_lambda(
    val_df: pd.DataFrame,
    *,
    budget: int,
    active_threshold: float,
    candidate_lambdas: Iterable[float],
) -> Tuple[float, List[Dict[str, float]]]:
    """Select lambda that maximizes validation hit rate for the given budget."""
    trials: List[Dict[str, float]] = []
    best_lambda = 0.0
    best_hit_rate = -float("inf")
    best_mean_activity = -float("inf")

    for risk_lambda in candidate_lambdas:
        metrics = evaluate_policy(
            val_df,
            budget=budget,
            active_threshold=active_threshold,
            policy="risk_adjusted",
            risk_lambda=float(risk_lambda),
        )
        trials.append(
            {
                "risk_lambda": float(risk_lambda),
                **metrics,
            }
        )
        if (
            metrics["hit_rate_at_budget"] > best_hit_rate
            or (
                np.isclose(metrics["hit_rate_at_budget"], best_hit_rate)
                and metrics["mean_p_activity_at_budget"] > best_mean_activity
            )
        ):
            best_lambda = float(risk_lambda)
            best_hit_rate = metrics["hit_rate_at_budget"]
            best_mean_activity = metrics["mean_p_activity_at_budget"]

    return best_lambda, trials


def round_nested(value):
    """Round floats recursively for JSON output."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_nested(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [round_nested(inner) for inner in value]
    return value


def uncertainty_diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    """Summarize whether predicted uncertainty tracks empirical error."""
    diagnostics_df = df.copy()
    diagnostics_df["absolute_error"] = (
        diagnostics_df["predicted_p_activity"] - diagnostics_df["p_activity"]
    ).abs()

    payload: Dict[str, object] = {
        "spearman_prediction_std_vs_absolute_error": float(
            spearmanr(
                diagnostics_df["prediction_std"],
                diagnostics_df["absolute_error"],
            ).statistic
        ),
        "quantile_bands": [],
    }

    for quantile in [0.25, 0.5, 0.75]:
        low_mask = diagnostics_df["prediction_std"] <= diagnostics_df["prediction_std"].quantile(quantile)
        high_mask = diagnostics_df["prediction_std"] >= diagnostics_df["prediction_std"].quantile(1.0 - quantile)
        payload["quantile_bands"].append(
            {
                "quantile": quantile,
                "low_uncertainty_mean_absolute_error": float(
                    diagnostics_df.loc[low_mask, "absolute_error"].mean()
                ),
                "high_uncertainty_mean_absolute_error": float(
                    diagnostics_df.loc[high_mask, "absolute_error"].mean()
                ),
            }
        )

    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_df = add_policy_scores(pd.read_csv(args.val_predictions), args.active_threshold)
    test_df = add_policy_scores(pd.read_csv(args.test_predictions), args.active_threshold)

    summary_rows: List[Dict[str, float]] = []
    payload: Dict[str, object] = {
        "active_threshold": args.active_threshold,
        "budgets": args.budgets,
        "uncertainty_diagnostics": {
            "validation": uncertainty_diagnostics(val_df),
            "test": uncertainty_diagnostics(test_df),
        },
        "validation_tuning": {},
        "test": {},
    }

    for budget in args.budgets:
        best_lambda, tuning_trials = tune_risk_lambda(
            val_df,
            budget=budget,
            active_threshold=args.active_threshold,
            candidate_lambdas=args.risk_grid,
        )

        payload["validation_tuning"][str(budget)] = {
            "selected_risk_lambda": best_lambda,
            "trials": tuning_trials,
        }

        test_mean = evaluate_policy(
            test_df,
            budget=budget,
            active_threshold=args.active_threshold,
            policy="mean",
        )
        test_prob = evaluate_policy(
            test_df,
            budget=budget,
            active_threshold=args.active_threshold,
            policy="prob_active",
        )
        test_risk = evaluate_policy(
            test_df,
            budget=budget,
            active_threshold=args.active_threshold,
            policy="risk_adjusted",
            risk_lambda=best_lambda,
        )

        payload["test"][str(budget)] = {
            "mean": test_mean,
            "prob_active": test_prob,
            "risk_adjusted": {
                "risk_lambda": best_lambda,
                **test_risk,
            },
        }

        for policy_name, metrics in [
            ("mean", test_mean),
            ("prob_active", test_prob),
            ("risk_adjusted", test_risk),
        ]:
            row = {
                "budget": budget,
                "policy": policy_name,
                **metrics,
            }
            if policy_name == "risk_adjusted":
                row["risk_lambda"] = best_lambda
            else:
                row["risk_lambda"] = 0.0
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "budget_policy_metrics.csv", index=False)

    with open(args.output_dir / "budget_policy_metrics.json", "w") as fh:
        json.dump(round_nested(payload), fh, indent=2)

    print("\n=== Budgeted Policy Summary ===")
    for budget in args.budgets:
        subset = summary_df[summary_df["budget"] == budget]
        best_row = subset.sort_values(
            ["hit_rate_at_budget", "mean_p_activity_at_budget"],
            ascending=False,
        ).iloc[0]
        print(
            f"Budget {budget}: best policy={best_row['policy']} "
            f"hit_rate={best_row['hit_rate_at_budget']:.3f} "
            f"mean_p_activity={best_row['mean_p_activity_at_budget']:.3f}"
        )
    corr = payload["uncertainty_diagnostics"]["test"][
        "spearman_prediction_std_vs_absolute_error"
    ]
    print(f"Uncertainty-error Spearman on test: {corr:.3f}")
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
