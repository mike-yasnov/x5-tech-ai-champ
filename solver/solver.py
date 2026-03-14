"""Multi-restart solver: runs greedy packer with many heuristic configurations."""

import logging
import sys
import time
from dataclasses import replace
from typing import List, Optional, Tuple

from .models import Box, Pallet, Solution, solution_to_dict
from .heuristics import STRATEGY_CONFIGS, StrategyConfig, select_strategy_configs
from .packer import pack_greedy

# Add project root to path for validator import
sys.path.insert(0, ".")

logger = logging.getLogger(__name__)

DEFAULT_EFFECTIVE_TIME_BUDGET_MS = 960


def _adaptive_budget_ms(time_budget_ms: int, boxes: List[Box]) -> int:
    total_items = sum(box.quantity for box in boxes)
    if total_items >= 150:
        return min(time_budget_ms, 900)
    if total_items >= 100:
        return min(time_budget_ms, 930)
    if total_items >= 60:
        return min(time_budget_ms, 950)
    return min(time_budget_ms, DEFAULT_EFFECTIVE_TIME_BUDGET_MS)


def _policy_variants(policy: str) -> List[str]:
    variants = {
        "balanced": ["balanced", "max_contact", "fragile_safe"],
        "dbfl": ["dbfl", "max_contact", "min_height"],
        "max_support": ["max_support", "fragile_safe", "max_contact"],
        "max_contact": ["max_contact", "fragile_safe", "center_stable"],
        "min_height": ["min_height", "max_support", "dbfl"],
        "center_stable": ["center_stable", "fragile_safe", "max_support"],
        "fragile_safe": ["fragile_safe", "max_support", "center_stable"],
    }
    return variants.get(policy, [policy, "fragile_safe", "max_contact"])


def _noise_schedule() -> List[float]:
    return [0.04, 0.08, 0.12, 0.18, 0.26, 0.34]


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
    ranked_candidates: List[Tuple[float, StrategyConfig]] = []
    effective_budget_ms = _adaptive_budget_ms(time_budget_ms, boxes)
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

        ranked_candidates.append((score, strategy))

        if best_score >= 0.9999:
            break

    ranked_candidates.sort(key=lambda item: item[0], reverse=True)

    top_candidates = [
        candidate
        for _, candidate in ranked_candidates[: min(4, len(ranked_candidates))]
    ]
    local_iterations = 0
    schedule = _noise_schedule()

    while top_candidates:
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > effective_budget_ms * 0.985 or best_score >= 0.9999:
            break

        base = top_candidates[local_iterations % len(top_candidates)]
        policy_options = _policy_variants(base.placement_policy)
        policy = policy_options[
            (local_iterations // len(top_candidates)) % len(policy_options)
        ]
        noise = schedule[
            (local_iterations // max(1, len(top_candidates) * len(policy_options)))
            % len(schedule)
        ]

        candidate = replace(
            base,
            name=f"{base.name}__local_{local_iterations}",
            placement_policy=policy,
            randomized=True,
            noise_factor=noise,
        )

        solution = pack_greedy(
            task_id,
            pallet,
            boxes,
            sort_key_name=candidate.sort_key_name,
            placement_policy=candidate.placement_policy,
            randomized=candidate.randomized,
            noise_factor=candidate.noise_factor,
        )

        score = _evaluate_score(request_dict, solution)
        if score is None:
            score = len(solution.placements) / max(
                1, sum(box.quantity for box in boxes)
            )

        if score > best_score:
            best_score = score
            best_solution = solution
            best_key = candidate.name

            top_candidates = [candidate] + top_candidates[:3]

        local_iterations += 1

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
