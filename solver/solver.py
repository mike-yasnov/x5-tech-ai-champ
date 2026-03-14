"""Advanced solver: greedy multi-restart + beam search + layer packing + LNS."""

import logging
import sys
import time
from typing import List, Optional

from .models import Box, Pallet, Solution, solution_to_dict
from .packer import SORT_KEYS, pack_greedy
from .beam_search import pack_beam_search
from .layer_packer import pack_layers
from .lns import lns_optimize

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
    n_restarts: int = 15,
    time_budget_ms: int = 900,
) -> Solution:
    """Run multiple packing strategies and return the best solution.

    Pipeline:
    1. Greedy multi-restart with all sort strategies
    2. Beam search with top sort strategies
    3. Layer-based packing
    4. LNS optimization on the best solution
    """
    t0 = time.perf_counter()
    best_solution: Optional[Solution] = None
    best_score: float = -1.0
    best_method: str = ""

    sort_key_names = list(SORT_KEYS.keys())

    logger.info(
        "[solve] task=%s budget=%dms strategies=%d",
        task_id, time_budget_ms, len(sort_key_names),
    )

    def _elapsed() -> float:
        return (time.perf_counter() - t0) * 1000

    def _update_best(solution: Solution, method: str) -> None:
        nonlocal best_solution, best_score, best_method
        score = _score_or_fallback(request_dict, solution, boxes)
        logger.info("[solve] %s score=%.4f placed=%d", method, score, len(solution.placements))
        if score > best_score:
            best_score = score
            best_solution = solution
            best_method = method

    # ── Phase 1: Greedy multi-restart (budget: 40%) ─────────────────
    for i, key_name in enumerate(sort_key_names):
        if _elapsed() > time_budget_ms * 0.40:
            logger.debug("[solve] greedy phase time limit, ran %d/%d", i, len(sort_key_names))
            break
        solution = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name)
        _update_best(solution, f"greedy/{key_name}")

    # ── Phase 2: Beam search (budget: 20%) ───────────────────────────
    if _elapsed() < time_budget_ms * 0.55:
        solution = pack_beam_search(
            task_id, pallet, boxes,
            sort_key=lambda b: (1 if b.fragile else 0, -b.weight_kg, -b.volume),
            beam_width=2,
            candidates_per_step=3,
        )
        _update_best(solution, "beam")

    # ── Phase 3: Layer-based packing (budget: 10%) ───────────────────
    if _elapsed() < time_budget_ms * 0.70:
        solution = pack_layers(task_id, pallet, boxes)
        _update_best(solution, "layers")

    # ── Phase 4: LNS on best solution (remaining budget) ─────────────
    remaining_ms = max(30, int(time_budget_ms * 0.85 - _elapsed()))
    if best_solution and remaining_ms > 30:
        improved = lns_optimize(
            task_id, pallet, boxes, best_solution,
            destroy_fraction=0.25,
            max_iterations=15,
            time_budget_ms=remaining_ms,
        )
        improved_score = _score_or_fallback(request_dict, improved, boxes)
        if improved_score > best_score:
            logger.info(
                "[solve] LNS improved: %.4f -> %.4f",
                best_score, improved_score,
            )
            best_score = improved_score
            best_solution = improved
            best_method = f"lns+{best_method}"

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
