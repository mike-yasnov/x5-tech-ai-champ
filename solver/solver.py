"""Multi-restart solver: runs greedy packer with different sort strategies."""

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

from .models import Box, Pallet, Solution, solution_to_dict
from .packer import SORT_KEYS, pack_greedy, pack_two_phase, pack_bestfit, MAX_BESTFIT_ITEMS
from .ml_ranker import MLPScorer

# Add project root to path for validator import
sys.path.insert(0, ".")

logger = logging.getLogger(__name__)

# Auto-detect CI: on CI, elapsed time = CI time (factor=1.0)
# Locally, estimate CI time with slowdown factor
_IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes") or os.environ.get("GITHUB_ACTIONS") == "true"
CI_SLOWDOWN_FACTOR = 1.0 if _IS_CI else 3.5

# Number of parallel workers (CI has 2 cores)
N_WORKERS = int(os.environ.get("SOLVER_WORKERS", "2"))


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


def _run_strategy(args: tuple) -> Tuple[float, Solution, str, str, str, float]:
    """Worker function: run a single packing strategy. Must be top-level for pickling."""
    task_id, pallet, boxes, key_name, weight_profile, packer_type, request_dict = args
    import time as _time

    # Ensure validator is importable
    sys.path.insert(0, ".")

    t0 = _time.perf_counter()
    if packer_type == "two_phase":
        solution = pack_two_phase(task_id, pallet, boxes, sort_key_name=key_name,
                                  weight_profile=weight_profile)
    else:
        solution = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name,
                               weight_profile=weight_profile)

    score = _evaluate_score(request_dict, solution)
    if score is None:
        score = len(solution.placements) / max(1, sum(b.quantity for b in boxes))

    strategy_ms = (_time.perf_counter() - t0) * 1000
    return score, solution, key_name, weight_profile, packer_type, strategy_ms


def _run_strategies_sequential(
    task_id, pallet, boxes, request_dict, strategies, n_to_run, strategy_budget_ms, t0,
) -> Tuple[Optional[Solution], float, str]:
    """Run strategies sequentially with adaptive time stopping."""
    best_solution = None
    best_score = -1.0
    best_key = ""
    estimated_ci_time_per_strategy = 0.0

    for i in range(n_to_run):
        elapsed = (time.perf_counter() - t0) * 1000
        if i > 0:
            estimated_ci_total = elapsed * CI_SLOWDOWN_FACTOR
            if estimated_ci_total > strategy_budget_ms * 0.9:
                logger.info("[solve] adaptive stop at restart %d (est CI %.0fms)", i, estimated_ci_total)
                break
            if estimated_ci_total + estimated_ci_time_per_strategy > strategy_budget_ms:
                logger.info("[solve] adaptive stop: next would push to %.0fms at restart %d",
                            estimated_ci_total + estimated_ci_time_per_strategy, i)
                break

        key_name, weight_profile, packer_type = strategies[i]
        strategy_t0 = time.perf_counter()

        if packer_type == "two_phase":
            solution = pack_two_phase(task_id, pallet, boxes, sort_key_name=key_name,
                                      weight_profile=weight_profile)
        else:
            solution = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name,
                                   weight_profile=weight_profile)

        score = _evaluate_score(request_dict, solution)
        if score is None:
            score = len(solution.placements) / max(1, sum(b.quantity for b in boxes))

        strategy_ms = (time.perf_counter() - strategy_t0) * 1000
        estimated_ci_time_per_strategy = max(estimated_ci_time_per_strategy, strategy_ms * CI_SLOWDOWN_FACTOR)

        logger.info("[solve] restart=%d sort=%s profile=%s packer=%s score=%.4f placed=%d time=%.0fms",
                    i, key_name, weight_profile, packer_type, score, len(solution.placements), strategy_ms)

        if score > best_score:
            best_score = score
            best_solution = solution
            best_key = key_name

    return best_solution, best_score, best_key


def _run_strategies_parallel(
    task_id, pallet, boxes, request_dict, strategies, n_to_run, strategy_budget_ms, t0,
) -> Tuple[Optional[Solution], float, str]:
    """Run strategies in parallel using ProcessPoolExecutor."""
    best_solution = None
    best_score = -1.0
    best_key = ""

    # Prepare args for all strategies
    args_list = [
        (task_id, pallet, boxes, s[0], s[1], s[2], request_dict)
        for s in strategies[:n_to_run]
    ]

    # Use 'fork' context on Linux (CI) for fast process creation
    # On macOS, 'fork' can be unsafe with some libraries but works here
    import multiprocessing
    ctx = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as pool:
        futures = {pool.submit(_run_strategy, args): i for i, args in enumerate(args_list)}

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                score, solution, key_name, weight_profile, packer_type, strategy_ms = future.result()
            except Exception as e:
                logger.warning("[solve] strategy %d failed: %s", idx, e)
                continue

            completed += 1
            logger.info("[solve] restart=%d sort=%s profile=%s packer=%s score=%.4f placed=%d time=%.0fms",
                        idx, key_name, weight_profile, packer_type, score, len(solution.placements), strategy_ms)

            if score > best_score:
                best_score = score
                best_solution = solution
                best_key = key_name

            # Check wall-clock budget — cancel remaining if exceeded
            # With N_WORKERS parallel, CI wall time is reduced by worker count
            elapsed = (time.perf_counter() - t0) * 1000
            estimated_ci_elapsed = elapsed * CI_SLOWDOWN_FACTOR / N_WORKERS
            if estimated_ci_elapsed > strategy_budget_ms:
                logger.info("[solve] parallel budget reached at %d/%d completed (est CI %.0fms)",
                            completed, n_to_run, estimated_ci_elapsed)
                for f in futures:
                    f.cancel()
                break

    logger.info("[solve] parallel done: %d strategies completed, best_score=%.4f best_key=%s",
                completed, best_score, best_key)
    return best_solution, best_score, best_key


def solve(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    request_dict: dict,
    n_restarts: int = 30,
    time_budget_ms: int = 900,
) -> Solution:
    """Run greedy packer with multiple sort strategies and return the best solution.

    Uses parallel execution with ProcessPoolExecutor (2 workers on CI).
    Falls back to sequential if parallel fails.
    """
    t0 = time.perf_counter()
    best_solution: Optional[Solution] = None
    best_score: float = -1.0
    best_key: str = ""

    sort_key_names = list(SORT_KEYS.keys())

    # Build strategy list: (sort_key, weight_profile, packer_type) triples
    from .scoring import WEIGHT_PROFILES
    strategies = [
        # Core greedy strategies with default weights
        (sort_key_names[0], "default", "greedy"),        # constrained_first
        (sort_key_names[1], "default", "greedy"),        # base_area_desc
        (sort_key_names[2], "default", "greedy"),        # fragile_last
        (sort_key_names[3], "default", "greedy"),        # volume_desc
        (sort_key_names[4], "default", "greedy"),        # volume_asc
        # Key alternative profiles (proven winners)
        (sort_key_names[1], "contact_heavy", "greedy"),  # base_area_desc + contact
        (sort_key_names[3], "fill_heavy", "greedy"),     # volume_desc + fill
        (sort_key_names[0], "contact_heavy", "greedy"),  # constrained_first + contact
        # Two-phase strategies (good for fragile-heavy scenarios)
        (sort_key_names[3], "default", "two_phase"),     # volume_desc two-phase
        (sort_key_names[1], "default", "two_phase"),     # base_area_desc two-phase
        # Additional greedy
        ("heavy_base_fragile_top", "default", "greedy"),
        ("stackable_base", "default", "greedy"),
        (sort_key_names[5], "default", "greedy"),        # density_desc
        (sort_key_names[6], "default", "greedy"),        # non_stackable_last
        (sort_key_names[7], "default", "greedy"),        # height_desc
        (sort_key_names[8], "default", "greedy"),        # weight_desc
        (sort_key_names[9], "default", "greedy"),        # max_dim_desc
        # More alternatives
        (sort_key_names[2], "fill_heavy", "greedy"),
        (sort_key_names[1], "contact_heavy", "two_phase"),
        # === New profiles (positions 19+, run with 30 restarts) ===
        (sort_key_names[2], "fragile_avoid", "greedy"),       # fragile_last + fragile_avoid
        ("heavy_base_fragile_top", "fragile_avoid", "greedy"),
        (sort_key_names[0], "fragile_avoid", "greedy"),       # constrained_first + fragile_avoid
        (sort_key_names[2], "fragile_avoid", "two_phase"),
        (sort_key_names[1], "wall_hugger", "greedy"),
        (sort_key_names[3], "compact", "greedy"),
        (sort_key_names[1], "layer_heavy", "greedy"),
        (sort_key_names[3], "layer_heavy", "greedy"),
        ("stackable_base", "fragile_avoid", "greedy"),
        (sort_key_names[0], "compact", "greedy"),
        ("heavy_base_fragile_top", "wall_hugger", "greedy"),
    ]
    # Add remaining random sorts
    for sn in sort_key_names[12:]:
        strategies.append((sn, "default", "greedy"))

    n_to_run = min(n_restarts, len(strategies))

    logger.info(
        "[solve] task=%s restarts=%d budget=%dms strategies=%d workers=%d",
        task_id, n_restarts, time_budget_ms, len(strategies), N_WORKERS,
    )

    # --- Parallel strategy evaluation ---
    # With N_WORKERS=2, we run strategies in parallel to fit more into the time budget.
    # Time budget applies to wall-clock time (parallel reduces it by ~N_WORKERS factor).
    # Strategy budget: leave 35% of total budget for LNS + postprocessing
    strategy_budget_ms = time_budget_ms * 0.65

    if N_WORKERS > 1:
        try:
            best_solution, best_score, best_key = _run_strategies_parallel(
                task_id, pallet, boxes, request_dict, strategies, n_to_run, strategy_budget_ms, t0,
            )
        except Exception as e:
            logger.warning("[solve] parallel execution failed, falling back to sequential: %s", e)
            best_solution, best_score, best_key = _run_strategies_sequential(
                task_id, pallet, boxes, request_dict, strategies, n_to_run, strategy_budget_ms, t0,
            )
    else:
        best_solution, best_score, best_key = _run_strategies_sequential(
            task_id, pallet, boxes, request_dict, strategies, n_to_run, strategy_budget_ms, t0,
        )

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

    # LNS post-processing: improve best solution by destroy + repair
    elapsed_so_far = (time.perf_counter() - t0) * 1000
    estimated_ci_elapsed = elapsed_so_far * CI_SLOWDOWN_FACTOR
    remaining_ci_ms = time_budget_ms - estimated_ci_elapsed
    if best_solution is not None and remaining_ci_ms > 50 and len(best_solution.placements) > 0:
        try:
            from .lns import lns_optimize
            lns_budget = int(remaining_ci_ms / CI_SLOWDOWN_FACTOR * 0.9) if CI_SLOWDOWN_FACTOR > 1 else int(remaining_ci_ms * 0.9)
            logger.info("[solve] LNS post-processing: budget=%dms remaining_ci=%.0fms", lns_budget, remaining_ci_ms)
            lns_solution = lns_optimize(
                task_id, pallet, boxes, best_solution,
                destroy_fraction=0.3,
                max_iterations=100,
                time_budget_ms=lns_budget,
            )
            lns_score = _evaluate_score(request_dict, lns_solution)
            if lns_score is not None and lns_score > best_score:
                logger.info(
                    "[solve] LNS improved: %.4f -> %.4f placed=%d",
                    best_score, lns_score, len(lns_solution.placements),
                )
                best_score = lns_score
                best_solution = lns_solution
                best_key = f"lns_{best_key}"
            elif lns_score is not None:
                logger.info("[solve] LNS no improvement: %.4f vs %.4f", lns_score, best_score)
        except Exception as e:
            logger.warning("[solve] LNS failed: %s", e)

    # Post-processing: compact-down + try-insert
    # Use remaining wall-clock time with a minimum of 50ms
    elapsed_so_far = (time.perf_counter() - t0) * 1000
    # On CI: use remaining time directly. Locally: estimate remaining CI time
    if _IS_CI:
        pp_budget = max(50, int((time_budget_ms - elapsed_so_far) * 0.8))
    else:
        # Locally we're faster; give postprocess enough time
        pp_budget = 200
    if best_solution is not None and len(best_solution.placements) > 0:
        try:
            from .postprocess import postprocess_solution
            logger.info("[solve] postprocess: budget=%dms elapsed=%.0fms", pp_budget, elapsed_so_far)
            pp_solution = postprocess_solution(
                task_id, pallet, boxes, best_solution,
                time_budget_ms=pp_budget,
            )
            pp_score = _evaluate_score(request_dict, pp_solution)
            if pp_score is not None and pp_score > best_score:
                logger.info(
                    "[solve] postprocess improved: %.4f -> %.4f placed=%d->%d",
                    best_score, pp_score,
                    len(best_solution.placements), len(pp_solution.placements),
                )
                best_score = pp_score
                best_solution = pp_solution
                best_key = f"pp_{best_key}"
            elif pp_score is not None:
                logger.debug("[solve] postprocess no improvement: %.4f vs %.4f", pp_score, best_score)
        except Exception as e:
            logger.warning("[solve] postprocess failed: %s", e)

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
