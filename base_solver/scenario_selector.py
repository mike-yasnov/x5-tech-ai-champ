from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


logger = logging.getLogger(__name__)

SEED_FAMILIES: tuple[str, ...] = (
    "heavy_base",
    "liquid_fill",
    "mixed_volume",
    "fragile_density",
    "block_structured",
    "coverage_tie",
)


@dataclass(frozen=True)
class ScenarioFingerprint:
    total_items: int
    sku_count: int
    max_sku_share: float
    upright_ratio: float
    fragile_ratio: float
    non_stackable_ratio: float
    volume_ratio: float
    weight_ratio: float
    largest_base_area_ratio: float
    median_height_ratio: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.total_items),
                float(self.sku_count),
                self.max_sku_share,
                self.upright_ratio,
                self.fragile_ratio,
                self.non_stackable_ratio,
                self.volume_ratio,
                self.weight_ratio,
                self.largest_base_area_ratio,
                self.median_height_ratio,
            ],
            dtype=np.float32,
        )


def compute_request_fingerprint(request: Dict[str, object]) -> ScenarioFingerprint:
    pallet = request["pallet"]
    boxes = request["boxes"]

    total_items = sum(int(box["quantity"]) for box in boxes)
    sku_count = len(boxes)
    total_volume = sum(
        int(box["length_mm"]) * int(box["width_mm"]) * int(box["height_mm"]) * int(box["quantity"])
        for box in boxes
    )
    total_weight = sum(float(box["weight_kg"]) * int(box["quantity"]) for box in boxes)
    upright_items = sum(int(box["quantity"]) for box in boxes if box.get("strict_upright", False))
    fragile_items = sum(int(box["quantity"]) for box in boxes if box.get("fragile", False))
    non_stackable_items = sum(
        int(box["quantity"]) for box in boxes if not box.get("stackable", True)
    )
    max_sku_qty = max((int(box["quantity"]) for box in boxes), default=0)
    largest_base_area = max(
        (int(box["length_mm"]) * int(box["width_mm"]) for box in boxes),
        default=0,
    )
    heights = sorted(int(box["height_mm"]) for box in boxes)
    if heights:
        mid = len(heights) // 2
        median_height = (
            float(heights[mid])
            if len(heights) % 2 == 1
            else 0.5 * float(heights[mid - 1] + heights[mid])
        )
    else:
        median_height = 0.0

    pallet_volume = (
        int(pallet["length_mm"]) * int(pallet["width_mm"]) * int(pallet["max_height_mm"])
    )
    pallet_area = int(pallet["length_mm"]) * int(pallet["width_mm"])
    max_height = max(int(pallet["max_height_mm"]), 1)
    max_weight = max(float(pallet["max_weight_kg"]), 1.0)

    return ScenarioFingerprint(
        total_items=total_items,
        sku_count=sku_count,
        max_sku_share=max_sku_qty / max(total_items, 1),
        upright_ratio=upright_items / max(total_items, 1),
        fragile_ratio=fragile_items / max(total_items, 1),
        non_stackable_ratio=non_stackable_items / max(total_items, 1),
        volume_ratio=total_volume / max(pallet_volume, 1),
        weight_ratio=total_weight / max_weight,
        largest_base_area_ratio=largest_base_area / max(pallet_area, 1),
        median_height_ratio=median_height / max_height,
    )


class ScenarioSelector:
    """Optional lightweight XGBoost model for seed-family ordering."""

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self._is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
    ) -> dict:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("xgboost is required to train the scenario selector") from exc

        booster_params = {
            "objective": "multi:softprob",
            "num_class": len(SEED_FAMILIES),
            "n_estimators": 96,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            booster_params.update(params)

        self.model = XGBClassifier(**booster_params)
        fit_kwargs = {"X": X, "y": y, "verbose": False}
        if X_val is not None and y_val is not None and len(X_val) > 0:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        try:
            fit_kwargs["early_stopping_rounds"] = 20
            self.model.fit(**fit_kwargs)
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            self.model.fit(**fit_kwargs)

        self._is_trained = True
        train_pred = self.model.predict(X)
        metrics = {
            "train_rows": int(len(X)),
            "train_top1_accuracy": float(np.mean(train_pred == y)),
        }
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_pred = self.model.predict(X_val)
            metrics["val_top1_accuracy"] = float(np.mean(val_pred == y_val))
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained or self.model is None:
            return np.zeros((X.shape[0], len(SEED_FAMILIES)), dtype=np.float32)
        return np.asarray(self.model.predict_proba(X), dtype=np.float32)

    def rank_families(self, fingerprint: ScenarioFingerprint) -> List[str]:
        scores = self.predict_proba(fingerprint.as_array()[None, :])[0]
        order = np.argsort(-scores)
        return [SEED_FAMILIES[idx] for idx in order]

    def save(self, meta: Optional[dict] = None) -> None:
        if not self._is_trained or self.model is None:
            raise RuntimeError("cannot save an untrained scenario selector")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "selector_xgb.json"
        meta_path = self.model_dir / "selector_meta.json"
        self.model.save_model(model_path)
        payload = dict(meta or {})
        payload.setdefault("model_type", "xgb_selector")
        payload.setdefault("seed_families", list(SEED_FAMILIES))
        with open(meta_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def load(self) -> bool:
        model_path = self.model_dir / "selector_xgb.json"
        if not model_path.exists():
            return False

        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("[scenario_selector] xgboost unavailable: %s", exc)
            return False

        try:
            self.model = XGBClassifier()
            self.model.load_model(model_path)
            self._is_trained = True
            return True
        except Exception as exc:
            logger.warning("[scenario_selector] failed to load %s: %s", model_path, exc)
            self.model = None
            self._is_trained = False
            return False

    @property
    def is_trained(self) -> bool:
        return self._is_trained


def seed_family_index(seed_family: str) -> int:
    return SEED_FAMILIES.index(seed_family)


def seed_family_names(indices: Sequence[int]) -> List[str]:
    return [SEED_FAMILIES[int(idx)] for idx in indices]
