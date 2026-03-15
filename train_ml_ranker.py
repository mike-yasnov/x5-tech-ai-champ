"""Collect training data from solver runs and train MLP scorer.

Usage:
    python train_ml_ranker.py [--episodes N] [--output models/ml_ranker.npz]
"""

import argparse
import json
import logging
import random
import time
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

from generator import generate_scenario, PALLETS, FOOD_RETAIL_ARCHETYPES
from solver.models import Pallet, Box, solution_to_dict
from solver.packer import SORT_KEYS, pack_greedy, _expand_boxes
from solver.pallet_state import PalletState
from solver.orientations import get_orientations
from solver.scoring import score_placement
from solver.ml_ranker import extract_features, MLPScorer, N_FEATURES, MODEL_PATH
from validator import evaluate_solution

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Scenario types for training data generation
SCENARIO_TYPES = [
    "heavy_water", "fragile_tower", "liquid_tetris", "random_mixed",
    "fragile_mix", "support_tetris", "cavity_fill",
]


def _request_to_models(request_dict: dict):
    p = request_dict["pallet"]
    pallet = Pallet(
        type_id=p.get("type_id", "unknown"),
        length_mm=p["length_mm"], width_mm=p["width_mm"],
        max_height_mm=p["max_height_mm"], max_weight_kg=p["max_weight_kg"],
    )
    boxes = []
    for b in request_dict["boxes"]:
        boxes.append(Box(
            sku_id=b["sku_id"], description=b.get("description", ""),
            length_mm=b["length_mm"], width_mm=b["width_mm"],
            height_mm=b["height_mm"], weight_kg=b["weight_kg"],
            quantity=b["quantity"], strict_upright=b.get("strict_upright", False),
            fragile=b.get("fragile", False), stackable=b.get("stackable", True),
        ))
    return request_dict["task_id"], pallet, boxes


def collect_placement_data(
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str,
    final_score: float,
) -> List[Tuple[np.ndarray, float]]:
    """Run greedy packer step by step, recording features for each placement.

    Returns list of (features, final_score) pairs.
    """
    sort_fn = SORT_KEYS.get(sort_key_name)
    if sort_fn is None:
        return []

    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = _expand_boxes(sorted_boxes)
    total_items = len(instances)
    total_volume = sum(
        b.length_mm * b.width_mm * b.height_mm * b.quantity for b in boxes
    )

    state = PalletState(pallet)
    data = []
    placed_count = 0

    for item_idx, (box, inst_idx) in enumerate(instances):
        orientations = get_orientations(box)
        best_score = -1.0
        best_placement = None
        best_features = None
        items_remaining = total_items - item_idx

        # Compute remaining volume
        remaining_vol = 0
        for future_box, _ in instances[item_idx:]:
            remaining_vol += future_box.length_mm * future_box.width_mm * future_box.height_mm

        for ep in list(state.extreme_points):
            ex, ey, ez = ep
            for dx, dy, dz, rot_code in orientations:
                if not state.can_place(
                    dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                ):
                    continue

                sc = score_placement(
                    state, dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile,
                )

                feats = extract_features(
                    state, dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                    items_remaining, total_items, remaining_vol,
                )

                if sc > best_score:
                    best_score = sc
                    best_placement = (dx, dy, dz, ex, ey, ez, rot_code)
                    best_features = feats

                # Record ALL candidate features with the final score
                # (both good and bad placements get labeled with the same final score,
                # but the chosen one will appear in the training data of a good solution)
                data.append((feats, final_score))

        if best_placement is not None:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(
                box.sku_id, dx, dy, dz, px, py, pz,
                box.weight_kg, box.fragile, box.stackable,
            )
            placed_count += 1

    return data


def collect_training_data(n_episodes: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random scenarios and collect training data.

    For each scenario:
    - Run ALL sort strategies
    - Record features from the BEST strategy (labeled with high score)
    - Record features from the WORST strategy (labeled with low score)
    """
    all_features = []
    all_labels = []

    sort_key_names = list(SORT_KEYS.keys())[:10]  # Use main strategies only

    for episode in range(n_episodes):
        seed = 1000 + episode
        scenario_type = random.choice(SCENARIO_TYPES)

        try:
            request_dict = generate_scenario(
                f"train_{episode}", scenario_type, seed=seed
            )
        except Exception:
            continue

        task_id, pallet, boxes = _request_to_models(request_dict)

        # Find best and worst strategies
        strategy_scores = {}
        for key_name in sort_key_names:
            solution = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name)
            response_dict = solution_to_dict(solution)
            result = evaluate_solution(request_dict, response_dict)
            if result.get("valid"):
                strategy_scores[key_name] = result.get("final_score", 0.0)

        if not strategy_scores:
            continue

        best_key = max(strategy_scores, key=strategy_scores.get)
        worst_key = min(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_key]
        worst_score = strategy_scores[worst_key]

        # Collect data from best strategy
        best_data = collect_placement_data(pallet, boxes, best_key, best_score)
        for feats, score in best_data:
            all_features.append(feats)
            all_labels.append(score)

        # Collect data from worst strategy (if different)
        if worst_key != best_key and best_score - worst_score > 0.01:
            worst_data = collect_placement_data(pallet, boxes, worst_key, worst_score)
            for feats, score in worst_data:
                all_features.append(feats)
                all_labels.append(score)

        if (episode + 1) % 50 == 0:
            logger.info(
                "[collect] episode=%d/%d samples=%d best=%.4f(%s) worst=%.4f(%s)",
                episode + 1, n_episodes, len(all_features),
                best_score, best_key, worst_score, worst_key,
            )

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    logger.info("[collect] done: %d samples, %d features", X.shape[0], X.shape[1])
    return X, y


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: Tuple[int, ...] = (64, 32),
    learning_rate: float = 0.001,
    n_epochs: int = 200,
    batch_size: int = 256,
) -> MLPScorer:
    """Train a small MLP using pure NumPy (SGD with momentum)."""
    n_samples, n_features = X.shape

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Normalize targets
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y_norm = (y - y_mean) / y_std

    # Initialize weights (He initialization)
    rng = np.random.default_rng(42)
    layer_sizes = [n_features] + list(hidden_sizes) + [1]
    weights = []
    velocities = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        std = np.sqrt(2.0 / fan_in)
        w = rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)
        b = np.zeros(fan_out, dtype=np.float32)
        weights.append((w, b))
        velocities.append((np.zeros_like(w), np.zeros_like(b)))

    momentum = 0.9
    best_loss = float("inf")
    best_weights = None

    for epoch in range(n_epochs):
        # Shuffle
        perm = rng.permutation(n_samples)
        X_shuf = X_norm[perm]
        y_shuf = y_norm[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]
            bs = end - start

            # Forward pass
            activations = [X_batch]
            for i, (w, b) in enumerate(weights):
                z = activations[-1] @ w + b
                if i < len(weights) - 1:
                    # LeakyReLU
                    a = np.where(z > 0, z, 0.01 * z)
                else:
                    a = z
                activations.append(a)

            # Loss (MSE)
            pred = activations[-1].ravel()
            error = pred - y_batch
            loss = np.mean(error ** 2)
            epoch_loss += loss
            n_batches += 1

            # Backward pass
            d_out = (2.0 / bs) * error.reshape(-1, 1)

            for i in reversed(range(len(weights))):
                w, b = weights[i]
                a_prev = activations[i]

                dw = a_prev.T @ d_out
                db = d_out.sum(axis=0)

                if i > 0:
                    d_prev = d_out @ w.T
                    # LeakyReLU derivative
                    z_prev = activations[i]
                    d_prev = d_prev * np.where(z_prev > 0, 1.0, 0.01)
                    d_out = d_prev

                # SGD with momentum
                vw, vb = velocities[i]
                vw = momentum * vw - learning_rate * dw
                vb = momentum * vb - learning_rate * db
                velocities[i] = (vw, vb)

                weights[i] = (w + vw, b + vb)

        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = [(w.copy(), b.copy()) for w, b in weights]

        if (epoch + 1) % 20 == 0:
            logger.info("[train] epoch=%d/%d loss=%.6f best=%.6f", epoch + 1, n_epochs, avg_loss, best_loss)

    # Bake normalization into first layer
    # x_norm = (x - mean) / std => w_norm @ x + b_norm where
    # w_norm = w / std, b_norm = b - (w @ mean) / std
    if best_weights:
        w0, b0 = best_weights[0]
        w0_adj = w0 / X_std.reshape(-1, 1)
        b0_adj = b0 - (X_mean / X_std) @ w0
        best_weights[0] = (w0_adj.astype(np.float32), b0_adj.astype(np.float32))

        # Bake target denormalization into last layer
        wn, bn = best_weights[-1]
        wn_adj = wn * y_std
        bn_adj = bn * y_std + y_mean
        best_weights[-1] = (wn_adj.astype(np.float32), bn_adj.astype(np.float32))

    scorer = MLPScorer(weights=best_weights)
    logger.info("[train] done: best_loss=%.6f", best_loss)
    return scorer


def evaluate_scorer(scorer: MLPScorer, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate the scorer on test data."""
    preds = scorer.predict_batch(X)
    mse = np.mean((preds - y) ** 2)
    mae = np.mean(np.abs(preds - y))
    corr = np.corrcoef(preds, y)[0, 1] if len(y) > 1 else 0.0
    return {"mse": mse, "mae": mae, "correlation": corr}


def main():
    parser = argparse.ArgumentParser(description="Train ML ranker for placement scoring")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--output", default=MODEL_PATH, help="Output model path")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    args = parser.parse_args()

    logger.info("=== Collecting training data (%d episodes) ===", args.episodes)
    t0 = time.perf_counter()
    X, y = collect_training_data(n_episodes=args.episodes)
    collect_time = time.perf_counter() - t0
    logger.info("Data collection: %.1fs, %d samples", collect_time, len(X))

    if len(X) < 100:
        logger.error("Not enough training data (%d samples). Need at least 100.", len(X))
        return

    # Train/test split (80/20)
    n = len(X)
    perm = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X[perm[:split]], y[perm[:split]]
    X_test, y_test = X[perm[split:]], y[perm[split:]]

    logger.info("=== Training MLP (%d train, %d test) ===", len(X_train), len(X_test))
    t0 = time.perf_counter()
    scorer = train_mlp(X_train, y_train, n_epochs=args.epochs)
    train_time = time.perf_counter() - t0
    logger.info("Training: %.1fs", train_time)

    # Evaluate
    train_metrics = evaluate_scorer(scorer, X_train, y_train)
    test_metrics = evaluate_scorer(scorer, X_test, y_test)
    logger.info("Train: MSE=%.6f MAE=%.4f Corr=%.4f", train_metrics["mse"], train_metrics["mae"], train_metrics["correlation"])
    logger.info("Test:  MSE=%.6f MAE=%.4f Corr=%.4f", test_metrics["mse"], test_metrics["mae"], test_metrics["correlation"])

    # Save
    scorer.save(args.output)
    logger.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
