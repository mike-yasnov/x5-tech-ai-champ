"""Train and export the optional XGBoost block ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from base_solver.block_ranker import BlockRanker


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path)
    meta_path = path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as file:
            meta = json.load(file)
    return data["X"], data["y"], data["group"], meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the block ranking model")
    parser.add_argument("--train", required=True, help="Training dataset .npz")
    parser.add_argument("--val", required=True, help="Validation dataset .npz")
    parser.add_argument("--model-dir", default="models", help="Output model directory")
    args = parser.parse_args()

    X_train, y_train, group_train, train_meta = _load_dataset(Path(args.train))
    X_val, y_val, group_val, val_meta = _load_dataset(Path(args.val))

    ranker = BlockRanker(model_dir=args.model_dir)
    metrics = ranker.train(
        X=X_train,
        y=y_train,
        group=group_train.tolist(),
        X_val=X_val,
        y_val=y_val,
        group_val=group_val.tolist(),
    )
    meta = {
        "feature_version": 1,
        "model_type": "xgb_ranker",
        "runtime_enabled": False,
        "train_meta": train_meta,
        "val_meta": val_meta,
        "metrics": metrics,
        "seed_ranges": {
            "train": [train_meta.get("seed_start"), train_meta.get("seed_end")],
            "val": [val_meta.get("seed_start"), val_meta.get("seed_end")],
        },
        "generator_families": train_meta.get("families", []),
    }
    ranker.save(meta=meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
