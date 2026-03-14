"""ML Layer Ranking: Random Forest regressor for predicting layer usefulness.

Based on Section 4 of Dell'Amico et al. (2026).
Features: instance box counts + layer box counts + 5 layer properties.
Label: normalized usage count (how many times layer was used in solution).
"""

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Box, Pallet
from .build_layers import Layer, build_layers
from .layer_stacker import stack_layers, LAYER_SORT_STRATEGIES

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "layer_ranker.joblib")

# Cache loaded model to avoid repeated disk reads
_cached_model = None


def _extract_features(
    instance_counts: Dict[str, int],
    layer: Layer,
    all_sku_ids: List[str],
) -> np.ndarray:
    """Extract features for an (instance, layer) pair.

    Features (2*Φ + 5):
    - Φ instance features: quantity of each SKU type in the instance
    - Φ layer features: quantity of each SKU type in the layer
    - 5 layer properties: sum_quantity, height, weight, compression_proxy, area
    """
    phi = len(all_sku_ids)
    features = np.zeros(2 * phi + 5, dtype=np.float32)

    # Instance features
    for i, sku in enumerate(all_sku_ids):
        features[i] = instance_counts.get(sku, 0)

    # Layer features
    for i, sku in enumerate(all_sku_ids):
        features[phi + i] = layer.box_counts.get(sku, 0)

    # 5 layer properties
    features[2 * phi] = layer.box_count
    features[2 * phi + 1] = layer.height_mm
    features[2 * phi + 2] = layer.total_weight_kg
    features[2 * phi + 3] = layer.net_area / max(layer.pallet_area, 1)  # density as compression proxy
    features[2 * phi + 4] = layer.net_area

    return features


def train_ranker(
    pallet: Pallet,
    n_instances: int = 200,
    seed: int = 42,
) -> None:
    """Train the Random Forest ranker on generated instances.

    Pipeline:
    1. Generate N instances using generator.py
    2. For each instance: build layers, solve with greedy stacker
    3. Record which layers were used (label = usage count, normalized)
    4. Train RF regressor on (features, labels)
    5. Save model
    """
    import sys
    sys.path.insert(0, ".")

    try:
        from sklearn.ensemble import RandomForestRegressor
        import joblib
    except ImportError:
        logger.error("[train_ranker] sklearn/joblib not installed. Run: pip install scikit-learn joblib")
        return

    from generator import generate_scenario, FOOD_RETAIL_ARCHETYPES
    import random

    rng = random.Random(seed)
    scenarios = ["heavy_water", "fragile_tower", "liquid_tetris", "random_mixed"]

    logger.info("[train_ranker] generating %d instances for training...", n_instances)
    t0 = time.perf_counter()

    # Collect all SKU IDs across all instances for consistent feature vectors
    all_instances = []
    all_sku_ids_set = set()

    for i in range(n_instances):
        sc = scenarios[i % len(scenarios)]
        instance_seed = seed + i * 7
        request = generate_scenario(f"train_{i}", sc, seed=instance_seed)

        boxes_data = request["boxes"]
        boxes = []
        for b in boxes_data:
            boxes.append(Box(
                sku_id=b["sku_id"], description=b.get("description", ""),
                length_mm=b["length_mm"], width_mm=b["width_mm"],
                height_mm=b["height_mm"], weight_kg=b["weight_kg"],
                quantity=b["quantity"],
                strict_upright=b.get("strict_upright", False),
                fragile=b.get("fragile", False),
                stackable=b.get("stackable", True),
            ))
            all_sku_ids_set.add(b["sku_id"])

        all_instances.append((request["task_id"], boxes))

    all_sku_ids = sorted(all_sku_ids_set)
    logger.info("[train_ranker] total unique SKUs across instances: %d", len(all_sku_ids))

    # Build training data
    X_list = []
    y_list = []

    for task_id, boxes in all_instances:
        instance_counts = {}
        for b in boxes:
            instance_counts[b.sku_id] = b.quantity

        layers = build_layers(pallet, boxes, seed=seed)
        if not layers:
            continue

        # Solve with multiple strategies, pick best
        best_solution = None
        best_placed = 0
        for strategy in LAYER_SORT_STRATEGIES:
            sol = stack_layers(task_id, pallet, boxes, layers, sort_strategy=strategy)
            if len(sol.placements) > best_placed:
                best_placed = len(sol.placements)
                best_solution = sol

        if not best_solution:
            continue

        # Count layer usage
        placed_skus = defaultdict(int)
        for p in best_solution.placements:
            placed_skus[p.sku_id] += 1

        # Label each layer: how well does it match what was used
        layer_labels = []
        for layer in layers:
            usage = 0
            for sku, count in layer.box_counts.items():
                used = placed_skus.get(sku, 0)
                usage += min(count, used)
            layer_labels.append(usage)

        # Normalize labels
        max_label = max(layer_labels) if layer_labels else 1
        if max_label == 0:
            max_label = 1

        for li, layer in enumerate(layers):
            feat = _extract_features(instance_counts, layer, all_sku_ids)
            label = layer_labels[li] / max_label
            X_list.append(feat)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info("[train_ranker] training data: %d samples, %d features", X.shape[0], X.shape[1])

    # Balance: undersample label=0
    nonzero_mask = y > 0
    n_nonzero = nonzero_mask.sum()
    zero_indices = np.where(~nonzero_mask)[0]
    nonzero_indices = np.where(nonzero_mask)[0]

    if len(zero_indices) > 2 * n_nonzero:
        sampled_zeros = rng.sample(list(zero_indices), min(2 * n_nonzero, len(zero_indices)))
        keep_indices = np.concatenate([nonzero_indices, sampled_zeros])
        X = X[keep_indices]
        y = y[keep_indices]
        logger.info("[train_ranker] balanced: %d samples (was %d)", len(y), len(y_list))

    # Train RF
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # Save model + metadata
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_data = {
        "model": rf,
        "all_sku_ids": all_sku_ids,
        "n_features": X.shape[1],
    }
    joblib.dump(model_data, MODEL_PATH)

    elapsed = time.perf_counter() - t0
    logger.info(
        "[train_ranker] done in %.1fs. Model saved to %s. Samples=%d, Features=%d",
        elapsed, MODEL_PATH, X.shape[0], X.shape[1],
    )


def predict_layer_scores(
    boxes: List[Box],
    layers: List[Layer],
) -> Optional[Dict[int, float]]:
    """Predict usefulness scores for each layer using trained RF model.

    Returns dict of {layer_index: score} or None if model not available.
    """
    try:
        import joblib
    except ImportError:
        return None

    if not os.path.exists(MODEL_PATH):
        logger.debug("[predict] no trained model at %s", MODEL_PATH)
        return None

    global _cached_model
    if _cached_model is None:
        _cached_model = joblib.load(MODEL_PATH)
    model_data = _cached_model
    rf = model_data["model"]
    all_sku_ids = model_data["all_sku_ids"]

    # Build instance counts
    instance_counts = {}
    for b in boxes:
        instance_counts[b.sku_id] = b.quantity

    # Extend all_sku_ids with any new SKUs not seen in training
    current_skus = set(instance_counts.keys())
    for layer in layers:
        current_skus.update(layer.box_counts.keys())
    new_skus = current_skus - set(all_sku_ids)

    if new_skus:
        # Model can't handle unseen SKUs — use feature padding
        extended_sku_ids = all_sku_ids + sorted(new_skus)
        # But RF expects fixed feature count, so we can only use known SKUs
        logger.debug("[predict] %d new SKUs not in training, using known features only", len(new_skus))
        use_sku_ids = all_sku_ids
    else:
        use_sku_ids = all_sku_ids

    # Extract features and predict
    X = np.array([
        _extract_features(instance_counts, layer, use_sku_ids)
        for layer in layers
    ])

    # Handle feature size mismatch
    expected_features = model_data["n_features"]
    if X.shape[1] != expected_features:
        # Pad or truncate
        if X.shape[1] < expected_features:
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
            X = np.hstack([X, padding])
        else:
            X = X[:, :expected_features]

    scores = rf.predict(X)

    result = {i: float(scores[i]) for i in range(len(layers))}

    logger.info(
        "[predict] scored %d layers, top_score=%.4f, mean=%.4f",
        len(result),
        max(scores) if len(scores) > 0 else 0,
        float(np.mean(scores)) if len(scores) > 0 else 0,
    )

    return result
