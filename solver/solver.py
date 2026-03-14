"""Solver: greedy multi-restart + paper-based layer method (Dell'Amico et al.)."""

import logging
import sys
import time
from typing import List, Optional

from .models import Box, Pallet, Solution, solution_to_dict
from .packer import SORT_KEYS, pack_greedy
from .build_layers import build_layers
from .layer_stacker import stack_layers, LAYER_SORT_STRATEGIES
from .ml_ranker import predict_layer_scores

# Add project root to path for validator import
sys.path.insert(0, ".")

logger = logging.getLogger(__name__)


def _evaluate_score(request_dict: dict, solution: Solution) -> Optional[float]:
    """Try to evaluate solution using validator. Returns final_score or None."""
    try:
        from validator import evaluate_solution
        response_dict = solution_to_dict(solution)
        result = evaluate_solution(request_dict, response_dict)
        if result.get("valid"):
            return result.get("final_score", 0.0)
        else:
            logger.warning(
                "[evaluate] invalid solution: %s", result.get("error", "unknown")
            )
            return None
    except ImportError:
        logger.debug("[evaluate] validator not available, skipping scoring")
        return None


def _score_or_fallback(request_dict: dict, solution: Solution, boxes: List[Box]) -> float:
    score = _evaluate_score(request_dict, solution)
    if score is not None:
        return score
    return len(solution.placements) / max(1, sum(b.quantity for b in boxes))


def solve(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    request_dict: dict,
    n_restarts: int = 10,
    time_budget_ms: int = 900,
) -> Solution:
    """Run multiple strategies and return the best solution.

    Pipeline:
    1. Greedy multi-restart (baseline, fast)
    2. Paper method: BuildLayers → ML ranking → layer stacking
    """
    t0 = time.perf_counter()
    best_solution: Optional[Solution] = None
    best_score: float = -1.0
    best_method: str = ""

    sort_key_names = list(SORT_KEYS.keys())

    def _elapsed() -> float:
        return (time.perf_counter() - t0) * 1000

    def _update_best(solution: Solution, method: str) -> None:
        nonlocal best_solution, best_score, best_method
        # Only accept valid solutions
        validity_score = _evaluate_score(request_dict, solution)
        if validity_score is None:
            # Validator not available, use fallback
            score = len(solution.placements) / max(1, sum(b.quantity for b in boxes))
        elif validity_score == 0.0:
            # Score of 0 from validator means constraints failed somewhere
            logger.debug("[solve] %s: score=0, skipping", method)
            return
        else:
            score = validity_score

        if score > best_score:
            best_score = score
            best_solution = solution
            best_method = method
            logger.info("[solve] new best: %s score=%.4f placed=%d", method, score, len(solution.placements))

    # ── Phase 1: Greedy multi-restart (budget: 50%) ──────────────────
    logger.info(
        "[solve] task=%s budget=%dms greedy_strategies=%d",
        task_id, time_budget_ms, len(sort_key_names),
    )

    for i, key_name in enumerate(sort_key_names):
        if _elapsed() > time_budget_ms * 0.45:
            logger.debug("[solve] greedy phase limit, ran %d/%d", i, len(sort_key_names))
            break
        solution = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name)
        _update_best(solution, f"greedy/{key_name}")

    # ── Phase 2: Paper method — BuildLayers + stacking (only if budget allows) ─
    total_items = sum(b.quantity for b in boxes)
    elapsed_after_greedy = _elapsed()
    # Conservative gate: need ≥70% budget remaining AND ≤40 items
    if elapsed_after_greedy < time_budget_ms * 0.25 and total_items <= 40:
        try:
            layers = build_layers(pallet, boxes, seed=42)
            logger.info("[solve] paper method: %d layers generated in %.0fms",
                        len(layers), _elapsed() - elapsed_after_greedy)

            # Bail if build_layers consumed too much time
            if layers and _elapsed() < time_budget_ms * 0.55:
                # Try density stacking
                solution = stack_layers(
                    task_id, pallet, boxes, layers,
                    sort_strategy="density",
                )
                _update_best(solution, "paper/density")

                # Try ML ranking if we still have time
                if _elapsed() < time_budget_ms * 0.65:
                    ml_scores = predict_layer_scores(boxes, layers)
                    if ml_scores:
                        solution = stack_layers(
                            task_id, pallet, boxes, layers,
                            layer_scores=ml_scores,
                        )
                        _update_best(solution, "paper/ml_ranked")

        except Exception as e:
            logger.warning("[solve] paper method failed: %s", e)

    # Update total time
    total_ms = int(_elapsed())
    if best_solution is not None:
        best_solution.solve_time_ms = total_ms

    logger.info(
        "[solve] done method=%s score=%.4f total_time=%dms",
        best_method, best_score, total_ms,
    )

    if best_solution is None:
        from . import __version__
        return Solution(
            task_id=task_id,
            solver_version=__version__,
            solve_time_ms=total_ms,
        )

    return best_solution
