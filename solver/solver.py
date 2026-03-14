"""Multi-restart solver: runs greedy packer with many heuristic configurations."""

import logging
import sys
import time
from typing import List, Optional

from .models import Box, Pallet, Solution, solution_to_dict
from .heuristics import STRATEGY_CONFIGS, StrategyConfig, select_strategy_configs
from .packer import pack_greedy

# Add project root to path for validator import
sys.path.insert(0, ".")

logger = logging.getLogger(__name__)

DEFAULT_EFFECTIVE_TIME_BUDGET_MS = 950


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
    effective_budget_ms = min(time_budget_ms, DEFAULT_EFFECTIVE_TIME_BUDGET_MS)
    selected_strategies = select_strategy_configs(
        boxes, pallet.volume, pallet.max_weight_kg
    )
    randomized_strategies = [
        strategy
        for strategy in STRATEGY_CONFIGS
        if strategy.randomized
        and strategy.sort_key_name in {cfg.sort_key_name for cfg in selected_strategies}
    ]
    strategies: List[StrategyConfig] = selected_strategies + randomized_strategies

    logger.info(
        "[solve] task=%s restarts=%d budget=%dms effective_budget=%dms strategy_count=%d",
        task_id,
        n_restarts,
        time_budget_ms,
        effective_budget_ms,
        len(strategies),
    )

    for i in range(min(n_restarts, len(strategies))):
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > effective_budget_ms * 0.92:
            logger.info("[solve] time budget approaching, stopping at restart %d", i)
            break

        strategy = strategies[i]
        key_name = strategy.name
        solution = pack_greedy(
            task_id,
            pallet,
            boxes,
            sort_key_name=strategy.sort_key_name,
            placement_policy=strategy.placement_policy,
            randomized=strategy.randomized,
            noise_factor=strategy.noise_factor,
        )

        score = _evaluate_score(request_dict, solution)
        if score is None:
            # Fallback: use placement count as proxy
            score = len(solution.placements) / max(1, sum(b.quantity for b in boxes))

        logger.info(
            "[solve] restart=%d strategy=%s score=%.4f placed=%d",
            i,
            key_name,
            score,
            len(solution.placements),
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
        best_key,
        best_score,
        total_ms,
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
