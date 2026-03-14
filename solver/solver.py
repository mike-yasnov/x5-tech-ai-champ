"""Multi-restart solver: runs greedy packer with different sort strategies."""

import logging
import os
import sys
import time
from collections import defaultdict
from typing import List, Optional, Tuple

from .models import Box, Pallet, Solution, solution_to_dict
from .packer import SORT_KEYS, pack_greedy, pack_greedy_with_order, pack_two_phase, pack_beam_search, pack_bestfit, MAX_BESTFIT_ITEMS
from .pallet_state import _overlap_area

# Add project root to path for validator import
sys.path.insert(0, ".")

logger = logging.getLogger(__name__)

# Auto-detect CI
_IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes") or os.environ.get("GITHUB_ACTIONS") == "true"

# Parallel only on Linux (CI); macOS fork + short tasks = overhead > benefit
N_WORKERS = int(os.environ.get("SOLVER_WORKERS", "2" if sys.platform.startswith("linux") else "1"))


def _fast_score(pallet: Pallet, boxes: List[Box], solution: Solution) -> float:
    """Exact proxy score matching validator formula, without full validation overhead.

    Assumes solution is valid (our packers always produce valid placements).
    """
    total_requested_items = sum(b.quantity for b in boxes)
    placements = solution.placements
    if not placements:
        return 0.0

    boxes_meta = {b.sku_id: b for b in boxes}

    placed_volume = 0
    by_top_z = defaultdict(list)
    for p in placements:
        vol = p.length_mm * p.width_mm * p.height_mm
        placed_volume += vol
        by_top_z[p.z_mm + p.height_mm].append(p)

    # Count fragility violations
    fragility_violations = 0
    for top in placements:
        top_box = boxes_meta[top.sku_id]
        if top_box.weight_kg <= 2.0 or top.z_mm == 0:
            continue
        tx1, ty1 = top.x_mm, top.y_mm
        tx2, ty2 = tx1 + top.length_mm, ty1 + top.width_mm
        for bottom in by_top_z.get(top.z_mm, ()):
            bottom_box = boxes_meta[bottom.sku_id]
            if not bottom_box.fragile:
                continue
            bx1, by1 = bottom.x_mm, bottom.y_mm
            bx2, by2 = bx1 + bottom.length_mm, by1 + bottom.width_mm
            if _overlap_area(tx1, ty1, tx2, ty2, bx1, by1, bx2, by2) > 0:
                fragility_violations += 1

    vol_util = placed_volume / pallet.volume if pallet.volume > 0 else 0.0
    item_coverage = len(placements) / total_requested_items if total_requested_items > 0 else 0.0
    fragility_score = max(0.0, 1.0 - 0.05 * fragility_violations)
    # time_score=1.0 during strategy selection (final time set at the end)
    time_score = 1.0

    return (
        0.50 * vol_util
        + 0.30 * item_coverage
        + 0.10 * fragility_score
        + 0.10 * time_score
    )


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
    task_id, pallet, boxes, key_name, weight_profile, packer_type = args
    import time as _time

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

    proxy_score = _fast_score(pallet, boxes, solution)
    strategy_ms = (_time.perf_counter() - t0) * 1000
    return proxy_score, solution, key_name, weight_profile, packer_type, strategy_ms


def _run_strategies_sequential(
    task_id, pallet, boxes, strategies, n_to_run, strategy_budget_ms, t0,
) -> Tuple[Optional[Solution], float, str]:
    """Run strategies sequentially with adaptive time stopping."""
    best_solution = None
    best_score = -1.0
    best_key = ""
    estimated_time_per_strategy = 0.0

    for i in range(n_to_run):
        elapsed = (time.perf_counter() - t0) * 1000
        if i > 0:
            if elapsed > strategy_budget_ms * 0.9:
                logger.info("[solve] adaptive stop at restart %d (%.0fms)", i, elapsed)
                break
            if elapsed + estimated_time_per_strategy > strategy_budget_ms:
                logger.info("[solve] adaptive stop: next would push to %.0fms at restart %d",
                            elapsed + estimated_time_per_strategy, i)
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

        score = _fast_score(pallet, boxes, solution)

        strategy_ms = (time.perf_counter() - strategy_t0) * 1000
        estimated_time_per_strategy = max(estimated_time_per_strategy, strategy_ms)

        logger.info("[solve] restart=%d sort=%s profile=%s packer=%s score=%.4f placed=%d time=%.0fms",
                    i, key_name, weight_profile, packer_type, score, len(solution.placements), strategy_ms)

        if score > best_score:
            best_score = score
            best_solution = solution
            best_key = key_name

    return best_solution, best_score, best_key


def _run_strategies_parallel(
    task_id, pallet, boxes, strategies, n_to_run, strategy_budget_ms, t0,
) -> Tuple[Optional[Solution], float, str]:
    """Run strategies in parallel using ProcessPoolExecutor."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    best_solution = None
    best_score = -1.0
    best_key = ""

    # Prepare args (no request_dict — workers use proxy score)
    args_list = [
        (task_id, pallet, boxes, s[0], s[1], s[2])
        for s in strategies[:n_to_run]
    ]

    ctx = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as pool:
        # Submit in batches of N_WORKERS to control in-flight work
        pending = {}
        next_idx = 0

        def _submit_batch():
            nonlocal next_idx
            while len(pending) < N_WORKERS and next_idx < len(args_list):
                f = pool.submit(_run_strategy, args_list[next_idx])
                pending[f] = next_idx
                next_idx += 1

        _submit_batch()
        completed = 0

        while pending:
            # Wait for next completion
            done_futures = []
            for f in list(pending):
                if f.done():
                    done_futures.append(f)

            if not done_futures:
                # Brief wait then re-check
                time.sleep(0.001)
                continue

            for future in done_futures:
                idx = pending.pop(future)
                try:
                    score, solution, key_name, weight_profile, packer_type, strategy_ms = future.result()
                except Exception as e:
                    logger.warning("[solve] strategy %d failed: %s", idx, e)
                    _submit_batch()
                    continue

                completed += 1
                logger.info("[solve] restart=%d sort=%s profile=%s packer=%s score=%.4f placed=%d time=%.0fms",
                            idx, key_name, weight_profile, packer_type, score, len(solution.placements), strategy_ms)

                if score > best_score:
                    best_score = score
                    best_solution = solution
                    best_key = key_name

            # Check wall-clock budget before submitting more
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > strategy_budget_ms:
                logger.info("[solve] parallel budget reached at %d/%d completed (%.0fms)",
                            completed, len(args_list), elapsed)
                for f in pending:
                    f.cancel()
                pending.clear()
                break

            _submit_batch()

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
    """Run greedy packer with multiple sort strategies and return the best solution."""
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

    # --- Strategy evaluation ---
    # Wall-clock budget: ~40% strategies, rest for LNS + postprocessing
    strategy_budget_ms = time_budget_ms * 0.40

    if N_WORKERS > 1:
        try:
            best_solution, best_score, best_key = _run_strategies_parallel(
                task_id, pallet, boxes, strategies, n_to_run, strategy_budget_ms, t0,
            )
        except Exception as e:
            logger.warning("[solve] parallel execution failed, falling back to sequential: %s", e)
            best_solution, best_score, best_key = _run_strategies_sequential(
                task_id, pallet, boxes, strategies, n_to_run, strategy_budget_ms, t0,
            )
    else:
        best_solution, best_score, best_key = _run_strategies_sequential(
            task_id, pallet, boxes, strategies, n_to_run, strategy_budget_ms, t0,
        )

    # Try best-fit for small instances (if time allows)
    total_items = sum(b.quantity for b in boxes)
    elapsed_so_far = (time.perf_counter() - t0) * 1000
    if total_items <= MAX_BESTFIT_ITEMS and elapsed_so_far < time_budget_ms * 0.5:
        logger.info("[solve] trying bestfit for %d items", total_items)
        bf_solution = pack_bestfit(task_id, pallet, boxes)
        bf_score = _fast_score(pallet, boxes, bf_solution)
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
            lns_score = _fast_score(pallet, boxes, lns_solution)
            if lns_score > best_score:
                logger.info(
                    "[solve] LNS improved: %.4f -> %.4f placed=%d",
                    best_score, lns_score, len(lns_solution.placements),
                )
                best_score = lns_score
                best_solution = lns_solution
                best_key = f"lns_{best_key}"
            else:
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
            pp_score = _fast_score(pallet, boxes, pp_solution)
            if pp_score > best_score:
                logger.info(
                    "[solve] postprocess improved: %.4f -> %.4f placed=%d->%d",
                    best_score, pp_score,
                    len(best_solution.placements), len(pp_solution.placements),
                )
                best_score = pp_score
                best_solution = pp_solution
                best_key = f"pp_{best_key}"
            else:
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
        from . import __version__
        return Solution(
            task_id=task_id,
            solver_version=__version__,
            solve_time_ms=total_ms,
        )

    return best_solution
