"""Simple ensemble regression models for kinase-ligand ranking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import warnings
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class RidgeEnsembleConfig:
    alpha: float = 1.0
    ensemble_size: int = 8
    bootstrap_fraction: float = 0.8
    random_seed: int = 42


class RidgeEnsembleRegressor:
    """Bootstrap ensemble of ridge regressors with shared feature scaling."""

    def __init__(self, config: RidgeEnsembleConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.models: List[Ridge] = []
        self.uncertainty_scale = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeEnsembleRegressor":
        rng = np.random.RandomState(self.config.random_seed)
        X_scaled = self.scaler.fit_transform(X)

        sample_size = max(1, int(len(X_scaled) * self.config.bootstrap_fraction))
        self.models = []
        for model_index in range(self.config.ensemble_size):
            indices = rng.choice(len(X_scaled), size=sample_size, replace=True)
            model = Ridge(alpha=self.config.alpha, random_state=self.config.random_seed + model_index)
            model.fit(X_scaled[indices], y[indices])
            self.models.append(model)
        return self

    def _predict_raw(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("Model ensemble is not fitted.")

        X_scaled = self.scaler.transform(X)
        member_predictions = np.vstack([model.predict(X_scaled) for model in self.models])
        mean_prediction = member_predictions.mean(axis=0)
        std_prediction = member_predictions.std(axis=0, ddof=0)
        return mean_prediction.astype(np.float32), std_prediction.astype(np.float32)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_prediction, std_prediction = self._predict_raw(X)
        scaled_std = std_prediction * float(self.uncertainty_scale)
        return mean_prediction, scaled_std.astype(np.float32)

    def calibrate_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        target_coverage: float = 0.95,
        z_score: float = 1.96,
    ) -> float:
        """Scale predictive std so Gaussian intervals match validation coverage."""
        mean_prediction, std_prediction = self._predict_raw(X)
        safe_std = np.maximum(std_prediction, 1e-6)
        normalized_errors = np.abs(y - mean_prediction) / safe_std
        scale = float(np.quantile(normalized_errors, target_coverage) / z_score)
        self.uncertainty_scale = max(scale, 1.0)
        return self.uncertainty_scale

    def save(self, path: Path, metadata: Optional[Dict[str, object]] = None) -> None:
        payload = {
            "config": self.config,
            "scaler": self.scaler,
            "models": self.models,
            "uncertainty_scale": self.uncertainty_scale,
            "metadata": metadata or {},
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> Tuple["RidgeEnsembleRegressor", Dict[str, object]]:
        payload = joblib.load(path)
        model = cls(payload["config"])
        model.scaler = payload["scaler"]
        model.models = payload["models"]
        model.uncertainty_scale = payload.get("uncertainty_scale", 1.0)
        return model, payload.get("metadata", {})


def select_best_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    candidate_alphas: Sequence[float],
    tolerance: float = 1e-5,
) -> Tuple[float, List[Dict[str, float]]]:
    """Pick the alpha with the lowest validation RMSE.

    If multiple alphas are effectively tied, prefer the larger one because it is
    usually more numerically stable for dense fingerprint features.
    """
    trials: List[Dict[str, float]] = []
    best_rmse = float("inf")

    for alpha in candidate_alphas:
        model = Ridge(alpha=float(alpha))
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_val_scaled)
        trial_rmse = float(np.sqrt(np.mean((predictions - y_val) ** 2)))
        trials.append({"alpha": float(alpha), "val_rmse": trial_rmse})
        if trial_rmse < best_rmse:
            best_rmse = trial_rmse

    best_candidates = [
        float(trial["alpha"])
        for trial in trials
        if trial["val_rmse"] <= best_rmse + tolerance
    ]
    best_alpha = max(best_candidates)
    return best_alpha, trials
