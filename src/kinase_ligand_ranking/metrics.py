"""Evaluation metrics for regression and ranking on kinase-ligand data."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Spearman correlation with guard rails for degenerate targets."""
    if len(y_true) < 2:
        return None
    correlation, _ = spearmanr(y_true, y_pred)
    if np.isnan(correlation):
        return None
    return float(correlation)


def top_fraction_enrichment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    active_threshold: float = 6.0,
    top_fraction: float = 0.1,
) -> Optional[float]:
    """Enrichment factor within the top prediction fraction."""
    if len(y_true) < 2:
        return None

    active_mask = y_true >= active_threshold
    active_rate = float(active_mask.mean())
    if active_rate == 0.0:
        return None

    k = max(1, int(np.ceil(len(y_true) * top_fraction)))
    top_indices = np.argsort(y_pred)[::-1][:k]
    top_active_rate = float(active_mask[top_indices].mean())
    return top_active_rate / active_rate


def safe_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    active_threshold: float = 6.0,
) -> Optional[float]:
    """ROC-AUC against an activity threshold."""
    labels = (y_true >= active_threshold).astype(int)
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, y_pred))


def prediction_interval_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    *,
    z_score: float = 1.96,
) -> float:
    """Empirical coverage of Gaussian prediction intervals."""
    lower = y_pred - (z_score * y_std)
    upper = y_pred + (z_score * y_std)
    covered = (y_true >= lower) & (y_true <= upper)
    return float(covered.mean())


def evaluate_split(
    df: pd.DataFrame,
    *,
    active_threshold: float = 6.0,
    top_fraction: float = 0.1,
) -> Dict[str, object]:
    """Evaluate overall and per-target regression/ranking performance."""
    y_true = df["p_activity"].to_numpy()
    y_pred = df["predicted_p_activity"].to_numpy()

    overall: Dict[str, object] = {
        "rows": int(len(df)),
        "targets": int(df["target_id"].nunique()),
        "rmse": rmse(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
        "roc_auc": safe_roc_auc(y_true, y_pred, active_threshold=active_threshold),
    }

    if "prediction_std" in df.columns:
        overall["prediction_interval_95_coverage"] = prediction_interval_coverage(
            y_true,
            y_pred,
            df["prediction_std"].to_numpy(),
        )
        overall["mean_prediction_std"] = float(df["prediction_std"].mean())

    per_target_rows: List[Dict[str, object]] = []
    for target_id, group in df.groupby("target_id"):
        target_true = group["p_activity"].to_numpy()
        target_pred = group["predicted_p_activity"].to_numpy()
        per_target_rows.append(
            {
                "target_id": target_id,
                "rows": int(len(group)),
                "rmse": rmse(target_true, target_pred),
                "spearman": safe_spearman(target_true, target_pred),
                "top_10pct_enrichment": top_fraction_enrichment(
                    target_true,
                    target_pred,
                    active_threshold=active_threshold,
                    top_fraction=top_fraction,
                ),
                "roc_auc": safe_roc_auc(
                    target_true,
                    target_pred,
                    active_threshold=active_threshold,
                ),
            }
        )

    per_target_df = pd.DataFrame(per_target_rows).sort_values("target_id")
    overall["mean_per_target_spearman"] = _nanmean(per_target_df["spearman"])
    overall["mean_top_10pct_enrichment"] = _nanmean(per_target_df["top_10pct_enrichment"])
    overall["mean_per_target_roc_auc"] = _nanmean(per_target_df["roc_auc"])

    return {
        "overall": overall,
        "per_target": per_target_df,
    }


def _nanmean(series: pd.Series) -> Optional[float]:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())
