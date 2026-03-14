"""Multi-restart solver: runs greedy packer with different sort strategies."""

import logging
import sys
import time
from typing import List, Optional

from .models import Box, Pallet, Solution, solution_to_dict
from .packer import SORT_KEYS, pack_greedy, pack_bestfit, MAX_BESTFIT_ITEMS
from .ml_ranker import MLPScorer

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

    # Build strategy list: (sort_key, weight_profile) pairs
    # Interleave default and alternative profiles for diversity within budget
    from .scoring import WEIGHT_PROFILES
    alt_profiles = [p for p in WEIGHT_PROFILES if p != "default"]
    strategies = [
        # Top 5 with default weights
        (sort_key_names[0], "default"),  # constrained_first
        (sort_key_names[1], "default"),  # base_area_desc
        (sort_key_names[2], "default"),  # fragile_last
        (sort_key_names[3], "default"),  # volume_desc
        (sort_key_names[4], "default"),  # volume_asc
        # Top sorts with alternative profiles
        (sort_key_names[1], "contact_heavy"),  # base_area_desc + contact
        (sort_key_names[3], "fill_heavy"),     # volume_desc + fill
        (sort_key_names[0], "contact_heavy"),  # constrained_first + contact
        # Remaining default sorts
        (sort_key_names[5], "default"),  # density_desc
        (sort_key_names[6], "default"),  # non_stackable_last
        (sort_key_names[7], "default"),  # height_desc
        (sort_key_names[8], "default"),  # weight_desc
        (sort_key_names[9], "default"),  # max_dim_desc
        # More alternatives
        (sort_key_names[2], "fill_heavy"),
        (sort_key_names[4], "contact_heavy"),
    ]
    # Add remaining random sorts
    for sn in sort_key_names[10:]:
        strategies.append((sn, "default"))

    logger.info(
        "[solve] task=%s restarts=%d budget=%dms strategies=%d",
        task_id, n_restarts, time_budget_ms, len(strategies),
    )

    estimated_ci_time_per_strategy = 0.0

    for i in range(min(n_restarts, len(strategies))):
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

        key_name, weight_profile = strategies[i]
        solution = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name,
                               weight_profile=weight_profile)

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
            "[solve] restart=%d sort=%s profile=%s score=%.4f placed=%d time=%.0fms est_ci=%.0fms",
            i, key_name, weight_profile, score, len(solution.placements),
            strategy_ms, strategy_ms * CI_SLOWDOWN_FACTOR,
        )

        if score > best_score:
            best_score = score
            best_solution = solution
            best_key = key_name

    # Try best-fit for small instances (if time allows)
    total_items = sum(b.quantity for b in boxes)
    elapsed_so_far = (time.perf_counter() - t0) * 1000
    estimated_ci_elapsed = elapsed_so_far * CI_SLOWDOWN_FACTOR
    if total_items <= MAX_BESTFIT_ITEMS and estimated_ci_elapsed < time_budget_ms * 0.7:
        logger.info("[solve] trying bestfit for %d items", total_items)
        bf_solution = pack_bestfit(task_id, pallet, boxes)
        bf_score = _evaluate_score(request_dict, bf_solution)
        if bf_score is None:
            bf_score = len(bf_solution.placements) / max(1, total_items)
        logger.info(
            "[solve] bestfit score=%.4f placed=%d",
            bf_score, len(bf_solution.placements),
        )
        if bf_score > best_score:
            best_score = bf_score
            best_solution = bf_solution
            best_key = "bestfit"

    # Try ML-guided greedy (if model available and time allows)
    elapsed_so_far = (time.perf_counter() - t0) * 1000
    estimated_ci_elapsed = elapsed_so_far * CI_SLOWDOWN_FACTOR
    if estimated_ci_elapsed < time_budget_ms * 0.5:
        scorer = MLPScorer.load()
        if scorer is not None and scorer.is_loaded:
            try:
                from .packer import pack_greedy_ml
                # Try ML scorer with top 3 sort strategies
                ml_sorts = ["constrained_first", "base_area_desc", "volume_desc"]
                if best_key and best_key not in ("bestfit",) and best_key not in ml_sorts:
                    ml_sorts.insert(0, best_key)

                ml_time_limit = int((time_budget_ms / CI_SLOWDOWN_FACTOR - elapsed_so_far) * 0.8)
                per_strategy_limit = ml_time_limit // len(ml_sorts) if ml_sorts else ml_time_limit

                for ml_sort in ml_sorts:
                    elapsed_now = (time.perf_counter() - t0) * 1000
                    est_ci = elapsed_now * CI_SLOWDOWN_FACTOR
                    if est_ci > time_budget_ms * 0.85:
                        break

                    logger.info("[solve] trying ML greedy: sort=%s", ml_sort)
                    ml_solution = pack_greedy_ml(
                        task_id, pallet, boxes, scorer,
                        sort_key_name=ml_sort,
                        time_limit_ms=per_strategy_limit,
                    )
                    ml_score = _evaluate_score(request_dict, ml_solution)
                    if ml_score is None:
                        ml_score = len(ml_solution.placements) / max(1, total_items)
                    logger.info(
                        "[solve] ML greedy sort=%s score=%.4f placed=%d",
                        ml_sort, ml_score, len(ml_solution.placements),
                    )
                    if ml_score > best_score:
                        best_score = ml_score
                        best_solution = ml_solution
                        best_key = f"ml_{ml_sort}"
            except Exception as e:
                logger.warning("[solve] ML greedy failed: %s", e)

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
