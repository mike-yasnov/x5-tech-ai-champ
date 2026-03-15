from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


logger = logging.getLogger(__name__)


class BlockRanker:
    """Optional XGBoost pairwise ranker for block candidates."""

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self._is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: Sequence[int],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        group_val: Optional[Sequence[int]] = None,
        params: Optional[dict] = None,
    ) -> dict:
        try:
            from xgboost import XGBRanker
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("xgboost is required to train the block ranker") from exc

        booster_params = {
            "objective": "rank:pairwise",
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            booster_params.update(params)

        self.model = XGBRanker(**booster_params)
        fit_kwargs = {
            "X": X,
            "y": y,
            "group": list(group),
            "verbose": False,
        }
        if (
            X_val is not None
            and y_val is not None
            and group_val is not None
            and len(X_val) > 0
        ):
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_group"] = [list(group_val)]
        try:
            fit_kwargs["early_stopping_rounds"] = 50
            self.model.fit(**fit_kwargs)
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            self.model.fit(**fit_kwargs)

        self._is_trained = True
        train_pred = self.predict(X)
        metrics = {
            "train_rows": int(len(X)),
            "train_groups": int(len(group)),
            "train_mse": float(np.mean((train_pred - y) ** 2)),
        }
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_pred = self.predict(X_val)
            metrics["val_mse"] = float(np.mean((val_pred - y_val) ** 2))
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained or self.model is None:
            return np.zeros(X.shape[0], dtype=np.float32)
        return np.asarray(self.model.predict(X), dtype=np.float32)

    def rank(self, X: np.ndarray) -> np.ndarray:
        return np.argsort(-self.predict(X))

    def save(self, meta: Optional[dict] = None) -> None:
        if not self._is_trained or self.model is None:
            raise RuntimeError("cannot save an untrained block ranker")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "xgb_ranker.json"
        meta_path = self.model_dir / "meta.json"
        self.model.save_model(model_path)
        payload = dict(meta or {})
        payload.setdefault("model_type", "xgb_ranker")
        with open(meta_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def load(self) -> bool:
        model_path = self.model_dir / "xgb_ranker.json"
        if not model_path.exists():
            return False

        try:
            from xgboost import XGBRanker
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("[block_ranker] xgboost unavailable: %s", exc)
            return False

        try:
            self.model = XGBRanker()
            self.model.load_model(model_path)
            self._is_trained = True
            return True
        except Exception as exc:
            logger.warning("[block_ranker] failed to load %s: %s", model_path, exc)
            self.model = None
            self._is_trained = False
            return False

    @property
    def is_trained(self) -> bool:
        return self._is_trained


def normalize_scores(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
