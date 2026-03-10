"""Fast interaction-aware ensemble models for kinase-ligand ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


@dataclass
class DualTowerRankerConfig:
    ligand_dim: int
    target_dim: int
    context_dim: int
    embed_dim: int = 64
    n_estimators: int = 192
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    random_seed: int = 42


class DualTowerUncertaintyRanker:
    """Interaction-aware ExtraTrees model with projected ligand-target features."""

    def __init__(self, config: DualTowerRankerConfig):
        self.config = config
        self.model = ExtraTreesRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_seed,
            n_jobs=-1,
        )
        self.uncertainty_scale = 1.0
        self.training_history: List[Dict[str, float]] = []
        rng = np.random.RandomState(config.random_seed)
        self.ligand_projection = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(config.ligand_dim, 1)),
            size=(config.ligand_dim, config.embed_dim),
        ).astype(np.float32)
        self.target_projection = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(config.target_dim, 1)),
            size=(config.target_dim, config.embed_dim),
        ).astype(np.float32)

    def fit(
        self,
        train_components: Dict[str, np.ndarray],
        y_train: np.ndarray,
        train_target_ids,
        val_components: Dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> "DualTowerUncertaintyRanker":
        del train_target_ids  # reserved for future pairwise variants
        X_train = self._build_design_matrix(train_components)
        X_val = self._build_design_matrix(val_components)

        self.model.fit(X_train, y_train)
        val_mean, _ = self.predict_uncalibrated(val_components)
        val_rmse = float(np.sqrt(np.mean((val_mean - y_val) ** 2)))
        self.training_history = [
            {
                "n_estimators": float(self.config.n_estimators),
                "val_rmse": val_rmse,
            }
        ]
        self.calibrate_uncertainty(val_components, y_val)
        return self

    def _build_design_matrix(self, components: Dict[str, np.ndarray]) -> np.ndarray:
        ligand = components["ligand"]
        target = components["target"]
        context = np.concatenate([components["assay"], components["source"]], axis=1)

        ligand_embed = ligand @ self.ligand_projection
        target_embed = target @ self.target_projection
        cross_embed = ligand_embed * target_embed
        summary_features = np.column_stack(
            [
                ligand.sum(axis=1),
                target.sum(axis=1),
                cross_embed.mean(axis=1),
                cross_embed.std(axis=1),
            ]
        ).astype(np.float32)

        return np.concatenate(
            [ligand, target, context, ligand_embed, target_embed, cross_embed, summary_features],
            axis=1,
        ).astype(np.float32)

    def predict_uncalibrated(
        self,
        components: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = self._build_design_matrix(components)
        mean_prediction = self.model.predict(X).astype(np.float32)
        tree_predictions = np.vstack([estimator.predict(X) for estimator in self.model.estimators_])
        std_prediction = tree_predictions.std(axis=0, ddof=0).astype(np.float32)
        return mean_prediction, std_prediction

    def predict(self, components: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        mean_prediction, std_prediction = self.predict_uncalibrated(components)
        return mean_prediction, (std_prediction * float(self.uncertainty_scale)).astype(np.float32)

    def calibrate_uncertainty(
        self,
        components: Dict[str, np.ndarray],
        y_true: np.ndarray,
        *,
        target_coverage: float = 0.95,
        z_score: float = 1.96,
    ) -> float:
        mean, std = self.predict_uncalibrated(components)
        safe_std = np.maximum(std, 1e-6)
        normalized_errors = np.abs(y_true - mean) / safe_std
        scale = float(np.quantile(normalized_errors, target_coverage) / z_score)
        self.uncertainty_scale = max(scale, 1.0)
        return self.uncertainty_scale

    def save(self, path: Path, metadata: Optional[Dict[str, object]] = None) -> None:
        joblib.dump(
            {
                "config": asdict(self.config),
                "model": self.model,
                "uncertainty_scale": self.uncertainty_scale,
                "training_history": self.training_history,
                "ligand_projection": self.ligand_projection,
                "target_projection": self.target_projection,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Tuple["DualTowerUncertaintyRanker", Dict[str, object]]:
        payload = joblib.load(path)
        config = DualTowerRankerConfig(**payload["config"])
        model = cls(config)
        model.model = payload["model"]
        model.uncertainty_scale = payload.get("uncertainty_scale", 1.0)
        model.training_history = payload.get("training_history", [])
        model.ligand_projection = payload["ligand_projection"]
        model.target_projection = payload["target_projection"]
        return model, payload.get("metadata", {})
