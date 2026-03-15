"""Train and export the lightweight scenario-level selector model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from base_solver.scenario_selector import ScenarioSelector


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path)
    meta_path = path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as file:
            meta = json.load(file)
    return data["X"], data["y"], data["scores"], meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the seed-family selector")
    parser.add_argument("--train", required=True, help="Training dataset .npz")
    parser.add_argument("--val", required=True, help="Validation dataset .npz")
    parser.add_argument("--model-dir", default="models", help="Output model directory")
    args = parser.parse_args()

    X_train, y_train, train_scores, train_meta = _load_dataset(Path(args.train))
    X_val, y_val, val_scores, val_meta = _load_dataset(Path(args.val))

    selector = ScenarioSelector(model_dir=args.model_dir)
    metrics = selector.train(
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    meta = {
        "feature_version": 1,
        "model_type": "xgb_selector",
        "train_meta": train_meta,
        "val_meta": val_meta,
        "metrics": metrics,
        "train_score_mean": float(np.mean(np.max(train_scores, axis=1))),
        "val_score_mean": float(np.mean(np.max(val_scores, axis=1))),
    }
    selector.save(meta=meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
