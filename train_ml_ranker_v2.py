"""V2 training: pairwise ranking + RL-style self-improvement.

Approach:
1. Generate many random scenarios
2. For each scenario, run all sort strategies
3. For each step, record features of the CHOSEN placement
4. Label = final_score of that strategy's solution
5. Train MLP to predict final_score
6. Use MLP to run new solutions, collect new data, retrain (self-play)

Key difference from V1: focus on CHOSEN placements only (not all candidates),
and add self-play iterations.
"""

import argparse
import logging
import random
import time
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from generator import generate_scenario
from solver.models import Pallet, Box, solution_to_dict
from solver.packer import SORT_KEYS, pack_greedy, pack_greedy_ml, _expand_boxes
from solver.pallet_state import PalletState
from solver.orientations import get_orientations
from solver.scoring import score_placement
from solver.ml_ranker import extract_features, MLPScorer, N_FEATURES, MODEL_PATH
from validator import evaluate_solution

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


def collect_chosen_features(
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str,
    final_score: float,
) -> List[Tuple[np.ndarray, float]]:
    """Run greedy step by step, record features of CHOSEN placement only."""
    sort_fn = SORT_KEYS.get(sort_key_name)
    if sort_fn is None:
        return []

    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = _expand_boxes(sorted_boxes)
    total_items = len(instances)

    vols = [b.length_mm * b.width_mm * b.height_mm for b, _ in instances]
    cumvol = [0] * (total_items + 1)
    for i in range(total_items - 1, -1, -1):
        cumvol[i] = cumvol[i + 1] + vols[i]

    state = PalletState(pallet)
    data = []

    for item_idx, (box, inst_idx) in enumerate(instances):
        orientations = get_orientations(box)
        best_score = -1.0
        best_placement = None
        best_features = None

        items_remaining = total_items - item_idx
        remaining_vol = cumvol[item_idx]

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
                if sc < 0:
                    continue

                feats = extract_features(
                    state, dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                    items_remaining, total_items, remaining_vol,
                )

                if sc > best_score:
                    best_score = sc
                    best_placement = (dx, dy, dz, ex, ey, ez, rot_code)
                    best_features = feats

        if best_placement is not None:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(
                box.sku_id, dx, dy, dz, px, py, pz,
                box.weight_kg, box.fragile, box.stackable,
            )
            # Record chosen placement features with solution quality
            data.append((best_features, final_score))

    return data


def collect_ml_chosen_features(
    pallet: Pallet,
    boxes: List[Box],
    scorer: MLPScorer,
    sort_key_name: str,
    final_score: float,
) -> List[Tuple[np.ndarray, float]]:
    """Run ML-guided greedy, record features of chosen placements."""
    sort_fn = SORT_KEYS.get(sort_key_name)
    if sort_fn is None:
        return []

    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = _expand_boxes(sorted_boxes)
    total_items = len(instances)

    vols = [b.length_mm * b.width_mm * b.height_mm for b, _ in instances]
    cumvol = [0] * (total_items + 1)
    for i in range(total_items - 1, -1, -1):
        cumvol[i] = cumvol[i + 1] + vols[i]

    state = PalletState(pallet)
    data = []

    for item_idx, (box, inst_idx) in enumerate(instances):
        orientations = get_orientations(box)
        best_score = -float("inf")
        best_placement = None
        best_features = None

        items_remaining = total_items - item_idx
        remaining_vol = cumvol[item_idx]

        for ep in list(state.extreme_points):
            ex, ey, ez = ep
            for dx, dy, dz, rot_code in orientations:
                if not state.can_place(
                    dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                ):
                    continue

                h_score = score_placement(
                    state, dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile,
                )
                if h_score < 0:
                    continue

                feats = extract_features(
                    state, dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                    items_remaining, total_items, remaining_vol,
                )
                sc = scorer.predict(feats)

                if sc > best_score:
                    best_score = sc
                    best_placement = (dx, dy, dz, ex, ey, ez, rot_code)
                    best_features = feats

        if best_placement is not None:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(
                box.sku_id, dx, dy, dz, px, py, pz,
                box.weight_kg, box.fragile, box.stackable,
            )
            data.append((best_features, final_score))

    return data


def train_mlp(X, y, hidden=(64, 32), lr=0.001, epochs=200, batch=256):
    """Train MLP with SGD+momentum. Returns MLPScorer."""
    n, d = X.shape
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    X_n = (X - X_mean) / X_std
    y_mean, y_std = y.mean(), y.std() + 1e-8
    y_n = (y - y_mean) / y_std

    rng = np.random.default_rng(42)
    sizes = [d] + list(hidden) + [1]
    weights, vels = [], []
    for i in range(len(sizes)-1):
        std = np.sqrt(2.0/sizes[i])
        w = rng.normal(0, std, (sizes[i], sizes[i+1])).astype(np.float32)
        b = np.zeros(sizes[i+1], dtype=np.float32)
        weights.append((w, b))
        vels.append((np.zeros_like(w), np.zeros_like(b)))

    mom = 0.9
    best_loss, best_w = float("inf"), None

    for ep in range(epochs):
        perm = rng.permutation(n)
        Xs, ys = X_n[perm], y_n[perm]
        eloss, nb = 0, 0

        for s in range(0, n, batch):
            e = min(s+batch, n)
            xb, yb = Xs[s:e], ys[s:e]
            bs = e - s

            acts = [xb]
            for i, (w, b) in enumerate(weights):
                z = acts[-1] @ w + b
                acts.append(np.where(z > 0, z, 0.01*z) if i < len(weights)-1 else z)

            pred = acts[-1].ravel()
            err = pred - yb
            eloss += np.mean(err**2)
            nb += 1

            do = (2.0/bs) * err.reshape(-1,1)
            for i in reversed(range(len(weights))):
                w, b = weights[i]
                dw = acts[i].T @ do
                db = do.sum(0)
                if i > 0:
                    dp = do @ w.T
                    dp = dp * np.where(acts[i] > 0, 1.0, 0.01)
                    do = dp
                vw, vb = vels[i]
                vw = mom*vw - lr*dw
                vb = mom*vb - lr*db
                vels[i] = (vw, vb)
                weights[i] = (w+vw, b+vb)

        al = eloss/nb if nb else 0
        if al < best_loss:
            best_loss = al
            best_w = [(w.copy(), b.copy()) for w, b in weights]
        if (ep+1)%50 == 0:
            logger.info("[train] ep=%d/%d loss=%.6f best=%.6f", ep+1, epochs, al, best_loss)

    # Bake normalization
    if best_w:
        w0, b0 = best_w[0]
        best_w[0] = ((w0/X_std.reshape(-1,1)).astype(np.float32),
                      (b0 - (X_mean/X_std)@w0).astype(np.float32))
        wn, bn = best_w[-1]
        best_w[-1] = ((wn*y_std).astype(np.float32),
                       (bn*y_std + y_mean).astype(np.float32))

    return MLPScorer(weights=best_w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--self-play-rounds", type=int, default=3)
    parser.add_argument("--output", default=MODEL_PATH)
    args = parser.parse_args()

    sort_key_names = list(SORT_KEYS.keys())[:10]
    all_X, all_y = [], []
    scorer = None

    for sp_round in range(args.self_play_rounds + 1):
        logger.info("=== Round %d/%d ===", sp_round, args.self_play_rounds)
        round_X, round_y = [], []

        for episode in range(args.episodes):
            seed = 1000 + sp_round * 10000 + episode
            scenario_type = random.choice(SCENARIO_TYPES)

            try:
                req = generate_scenario(f"train_{sp_round}_{episode}", scenario_type, seed=seed)
            except Exception:
                continue

            task_id, pallet, boxes = _request_to_models(req)

            # Run heuristic strategies
            for key_name in sort_key_names:
                sol = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name)
                resp = solution_to_dict(sol)
                result = evaluate_solution(req, resp)
                if not result.get("valid"):
                    continue
                score = result.get("final_score", 0.0)

                data = collect_chosen_features(pallet, boxes, key_name, score)
                for feats, sc in data:
                    round_X.append(feats)
                    round_y.append(sc)

            # Also run ML-guided if we have a scorer
            if scorer is not None and scorer.is_loaded:
                for ml_sort in ["constrained_first", "volume_desc", "base_area_desc"]:
                    sol = pack_greedy_ml(task_id, pallet, boxes, scorer, sort_key_name=ml_sort)
                    resp = solution_to_dict(sol)
                    result = evaluate_solution(req, resp)
                    if not result.get("valid"):
                        continue
                    score = result.get("final_score", 0.0)

                    data = collect_ml_chosen_features(pallet, boxes, scorer, ml_sort, score)
                    for feats, sc in data:
                        round_X.append(feats)
                        round_y.append(sc)

            if (episode + 1) % 100 == 0:
                logger.info("[round %d] episode=%d/%d samples=%d", sp_round, episode+1, args.episodes, len(round_X))

        # Add to cumulative data
        all_X.extend(round_X)
        all_y.extend(round_y)

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.float32)

        logger.info("[round %d] training on %d samples", sp_round, len(X))

        # Train
        scorer = train_mlp(X, y, epochs=300)

        # Evaluate on benchmark scenarios
        from benchmark import ORGANIZER_SCENARIOS
        total_score = 0
        for sc_type, sc_seed in ORGANIZER_SCENARIOS:
            req = generate_scenario(f"eval_{sc_type}", sc_type, seed=sc_seed)
            task_id, pallet, boxes = _request_to_models(req)

            # Run best heuristic
            best_h_score = 0
            for key_name in sort_key_names[:5]:
                sol = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name)
                resp = solution_to_dict(sol)
                result = evaluate_solution(req, resp)
                if result.get("valid"):
                    best_h_score = max(best_h_score, result.get("final_score", 0))

            # Run ML-guided
            best_ml_score = 0
            for ml_sort in ["constrained_first", "volume_desc", "base_area_desc"]:
                sol = pack_greedy_ml(task_id, pallet, boxes, scorer, sort_key_name=ml_sort)
                resp = solution_to_dict(sol)
                result = evaluate_solution(req, resp)
                if result.get("valid"):
                    best_ml_score = max(best_ml_score, result.get("final_score", 0))

            best = max(best_h_score, best_ml_score)
            total_score += best
            marker = " <ML>" if best_ml_score > best_h_score else ""
            logger.info(
                "[eval] %s: heuristic=%.4f ml=%.4f best=%.4f%s",
                sc_type, best_h_score, best_ml_score, best, marker,
            )

        avg = total_score / len(ORGANIZER_SCENARIOS)
        logger.info("[round %d] avg organizer score: %.4f", sp_round, avg)

        # Save after each round
        scorer.save(args.output)
        logger.info("[round %d] model saved", sp_round)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
