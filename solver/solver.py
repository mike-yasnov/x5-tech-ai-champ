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
    n_restarts: int = 20,
    time_budget_ms: int = 900,
) -> Solution:
    """Run greedy packer with multiple sort strategies and return the best solution.

    Args:
        task_id: Task identifier
        pallet: Pallet parameters
        boxes: List of box types
        request_dict: Original request dict for validator
        n_restarts: Maximum number of restarts
        time_budget_ms: Total time budget in milliseconds
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

    for i in range(min(n_restarts, len(sort_key_names))):
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > time_budget_ms * 0.9:
            logger.info("[solve] time budget approaching, stopping at restart %d", i)
            break

        key_name = sort_key_names[i]
        solution = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name)

        score = _evaluate_score(request_dict, solution)
        if score is None:
            # Fallback: use placement count as proxy
            score = len(solution.placements) / max(1, sum(b.quantity for b in boxes))

        logger.info(
            "[solve] restart=%d sort=%s score=%.4f placed=%d",
            i, key_name, score, len(solution.placements),
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
