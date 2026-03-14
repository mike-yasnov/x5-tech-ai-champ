"""Multi-restart solver: runs greedy packer with different sort strategies."""

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

from .models import Box, Pallet, Solution, solution_to_dict
from .packer import SORT_KEYS, pack_greedy, pack_greedy_with_order, pack_two_phase, pack_beam_search, pack_bestfit, MAX_BESTFIT_ITEMS
from .ml_ranker import MLPScorer

# Add project root to path for validator import
sys.path.insert(0, ".")

logger = logging.getLogger(__name__)

# Auto-detect CI: on CI, elapsed time = CI time (factor=1.0)
# Locally, estimate CI time with slowdown factor
_IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes") or os.environ.get("GITHUB_ACTIONS") == "true"
CI_SLOWDOWN_FACTOR = 1.0 if _IS_CI else 1.0

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
    elif packer_type.startswith("beam"):
        beam_width = int(packer_type.split("_")[1]) if "_" in packer_type else 4
        solution = pack_beam_search(task_id, pallet, boxes, sort_key_name=key_name,
                                     weight_profile=weight_profile, beam_width=beam_width)
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
            if elapsed > strategy_budget_ms * 0.9:
                logger.info("[solve] adaptive stop at restart %d (%.0fms)", i, elapsed)
                break
            if elapsed + estimated_ci_time_per_strategy > strategy_budget_ms:
                logger.info("[solve] adaptive stop: next would push to %.0fms at restart %d",
                            elapsed + estimated_ci_time_per_strategy, i)
                break

        key_name, weight_profile, packer_type = strategies[i]
        strategy_t0 = time.perf_counter()

        if packer_type == "two_phase":
            solution = pack_two_phase(task_id, pallet, boxes, sort_key_name=key_name,
                                      weight_profile=weight_profile)
        elif packer_type.startswith("beam"):
            beam_width = int(packer_type.split("_")[1]) if "_" in packer_type else 4
            solution = pack_beam_search(task_id, pallet, boxes, sort_key_name=key_name,
                                         weight_profile=weight_profile, beam_width=beam_width)
        else:
            solution = pack_greedy(task_id, pallet, boxes, sort_key_name=key_name,
                                   weight_profile=weight_profile)

        score = _evaluate_score(request_dict, solution)
        if score is None:
            score = len(solution.placements) / max(1, sum(b.quantity for b in boxes))

        strategy_ms = (time.perf_counter() - strategy_t0) * 1000
        estimated_ci_time_per_strategy = max(estimated_ci_time_per_strategy, strategy_ms)

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
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > strategy_budget_ms:
                logger.info("[solve] parallel budget reached at %d/%d completed (%.0fms)",
                            completed, n_to_run, elapsed)
                for f in futures:
                    f.cancel()
                break

    logger.info("[solve] parallel done: %d strategies completed, best_score=%.4f best_key=%s",
                completed, best_score, best_key)
    return best_solution, best_score, best_key


def _local_order_search(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    best_solution: Solution,
    request_dict: dict,
    time_budget_ms: int = 200,
    n_swaps: int = 20,
) -> Tuple[Optional[Solution], Optional[float]]:
    """Local search: try permutations of item order from the best solution.

    Extracts the placement order from the solution, then tries:
    1. Swapping adjacent items near the placed/unplaced boundary
    2. Moving unplaced items earlier in the order
    3. Random adjacent swaps throughout the order

    Returns (solution, score) or (None, None) if no improvement.
    """
    import random as _random
    t0 = time.perf_counter()
    rng = _random.Random(42)

    # Extract placement order from solution
    placed_skus = [p.sku_id for p in best_solution.placements]
    placed_inst = [p.instance_index for p in best_solution.placements]

    # Build the original sorted order from the best strategy
    # Use the solution's placement order as the "good" order
    boxes_meta = {b.sku_id: b for b in boxes}

    # Reconstruct (Box, inst_idx) order from placements + unplaced
    order: List[Tuple[Box, int]] = []
    for p in best_solution.placements:
        order.append((boxes_meta[p.sku_id], p.instance_index))

    # Add unplaced items
    placed_set = set((p.sku_id, p.instance_index) for p in best_solution.placements)
    for box in boxes:
        for i in range(box.quantity):
            if (box.sku_id, i) not in placed_set:
                order.append((box, i))

    n_placed = len(best_solution.placements)
    n_total = len(order)

    if n_total < 2:
        return None, None

    best_local_score = -1.0
    best_local_solution = None

    # Determine weight profile to use (try a few)
    profiles_to_try = ["default", "contact_heavy", "fill_heavy"]

    for attempt in range(n_swaps):
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > time_budget_ms * 0.9:
            break

        new_order = list(order)

        if attempt < 3:
            # Try moving first unplaced item to different positions near boundary
            if n_placed < n_total:
                insert_pos = max(0, n_placed - attempt - 1)
                item = new_order.pop(n_placed)
                new_order.insert(insert_pos, item)
        elif attempt < 8:
            # Swap items near the placed/unplaced boundary
            boundary = max(1, n_placed - 3)
            i = rng.randint(boundary, min(n_placed + 2, n_total - 1))
            j = rng.randint(max(0, boundary - 2), min(i + 3, n_total - 1))
            if i != j:
                new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Random adjacent swaps
            i = rng.randint(0, n_total - 2)
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]

        profile = profiles_to_try[attempt % len(profiles_to_try)]
        solution = pack_greedy_with_order(task_id, pallet, new_order, weight_profile=profile)
        score = _evaluate_score(request_dict, solution)
        if score is None:
            score = len(solution.placements) / max(1, sum(b.quantity for b in boxes))

        if score > best_local_score:
            best_local_score = score
            best_local_solution = solution
            # Update order from the improved solution
            if len(solution.placements) > n_placed:
                order_new = []
                placed_set_new = set()
                for p in solution.placements:
                    order_new.append((boxes_meta[p.sku_id], p.instance_index))
                    placed_set_new.add((p.sku_id, p.instance_index))
                for box in boxes:
                    for i in range(box.quantity):
                        if (box.sku_id, i) not in placed_set_new:
                            order_new.append((box, i))
                order = order_new
                n_placed = len(solution.placements)

    logger.info(
        "[local_search] tried %d permutations in %.0fms best_score=%.4f",
        min(attempt + 1, n_swaps), (time.perf_counter() - t0) * 1000, best_local_score,
    )

    return best_local_solution, best_local_score


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
        (sort_key_names[2], "contact_heavy", "greedy"),  # fragile_last + contact → 84/84 liquid_tetris
        (sort_key_names[6], "contact_heavy", "greedy"),  # non_stackable_last + contact → also 84/84
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
        # Fragile-strict strategies
        (sort_key_names[2], "fragile_strict", "greedy"),
        ("heavy_base_fragile_top", "fragile_strict", "greedy"),
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
    # Wall-clock budget: ~40% strategies, ~60% postprocessing (LNS disabled for speed)
    strategy_budget_ms = time_budget_ms * 0.40

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
    if total_items <= MAX_BESTFIT_ITEMS and elapsed_so_far < time_budget_ms * 0.5:
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

    # LNS post-processing: only if enough time remains (skipped on tight budgets like CI)
    elapsed_so_far = (time.perf_counter() - t0) * 1000
    remaining_ms = time_budget_ms - elapsed_so_far
    if best_solution is not None and remaining_ms > 400 and len(best_solution.placements) > 0:
        try:
            from .lns import lns_optimize
            lns_budget = int(remaining_ms * 0.25)
            logger.info("[solve] LNS post-processing: budget=%dms remaining=%.0fms", lns_budget, remaining_ms)
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
    elapsed_so_far = (time.perf_counter() - t0) * 1000
    pp_budget = max(50, int((time_budget_ms - elapsed_so_far) * 0.9))
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
