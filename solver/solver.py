"""Multi-restart solver: runs greedy packer with different sort strategies."""

import logging
import sys
import time
from typing import List, Optional

from .models import Box, Pallet, Solution, solution_to_dict
from .packer import SORT_KEYS, pack_greedy

# Add project root to path for validator import
sys.path.insert(0, ".")

logger = logging.getLogger(__name__)

# CI is ~3-4x slower than local — use conservative multiplier
CI_SLOWDOWN_FACTOR = 3.5


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


def solve(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    request_dict: dict,
    n_restarts: int = 10,
    time_budget_ms: int = 900,
) -> Solution:
    """Run greedy packer with multiple sort strategies and return the best solution.

    Uses adaptive time management: estimates CI slowdown from first strategy
    to decide how many additional strategies to try.
    """
    t0 = time.perf_counter()
    best_solution: Optional[Solution] = None
    best_score: float = -1.0
    best_key: str = ""

    sort_key_names = list(SORT_KEYS.keys())

    logger.info(
        "[solve] task=%s restarts=%d budget=%dms strategies=%s",
        task_id, n_restarts, time_budget_ms, sort_key_names,
    )

    estimated_ci_time_per_strategy = 0.0

    for i in range(min(n_restarts, len(sort_key_names))):
        elapsed = (time.perf_counter() - t0) * 1000

        if i > 0:
            # Estimate how much time CI would need for remaining strategies
            estimated_ci_total = elapsed * CI_SLOWDOWN_FACTOR
            if estimated_ci_total > time_budget_ms * 0.9:
                logger.info(
                    "[solve] adaptive stop: est. CI time %.0fms > budget %.0fms at restart %d",
                    estimated_ci_total, time_budget_ms * 0.9, i,
                )
                break

            # Also check: would adding one more strategy exceed CI budget?
            est_next = estimated_ci_total + estimated_ci_time_per_strategy
            if est_next > time_budget_ms:
                logger.info(
                    "[solve] adaptive stop: next strategy would push CI to %.0fms at restart %d",
                    est_next, i,
                )
                break

        strategy_t0 = time.perf_counter()

        key_name = sort_key_names[i]
        solution = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name)

        score = _evaluate_score(request_dict, solution)
        if score is None:
            # Fallback: use placement count as proxy
            score = len(solution.placements) / max(1, sum(b.quantity for b in boxes))

        strategy_ms = (time.perf_counter() - strategy_t0) * 1000

        # Update per-strategy time estimate (use max for safety)
        estimated_ci_time_per_strategy = max(
            estimated_ci_time_per_strategy,
            strategy_ms * CI_SLOWDOWN_FACTOR,
        )

        logger.info(
            "[solve] restart=%d sort=%s score=%.4f placed=%d time=%.0fms est_ci=%.0fms",
            i, key_name, score, len(solution.placements),
            strategy_ms, strategy_ms * CI_SLOWDOWN_FACTOR,
        )

        if score > best_score:
            best_score = score
            best_solution = solution
            best_key = key_name

    # Update solve_time_ms to total time
    total_ms = int((time.perf_counter() - t0) * 1000)
    if best_solution is not None:
        best_solution.solve_time_ms = total_ms

    logger.info(
        "[solve] done best_sort=%s best_score=%.4f total_time=%dms",
        best_key, best_score, total_ms,
    )

    if best_solution is None:
        # Should not happen, but return empty solution as fallback
        from . import __version__
        return Solution(
            task_id=task_id,
            solver_version=__version__,
            solve_time_ms=total_ms,
        )

    return best_solution
