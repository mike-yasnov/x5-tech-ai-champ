"""LNS (Large Neighborhood Search): destroy + repair optimization."""

import logging
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .orientations import get_orientations
from .pallet_state import PalletState
from .scoring import score_placement
from . import __version__

logger = logging.getLogger(__name__)


def _rebuild_state_from_placements(
    pallet: Pallet,
    placements: List[Placement],
    boxes_meta: Dict[str, Box],
) -> Tuple[PalletState, List[Placement], List[Tuple["Box", int]]]:
    """Rebuild PalletState from a list of placements.

    Validates each placement with can_place() to catch floating boxes
    that lost support after destroy. Invalid placements are returned
    as orphans to be added to the repair pool.

    Returns (state, valid_placements, orphan_items).
    """
    # Sort by z so lower boxes are placed first (support order)
    sorted_ps = sorted(placements, key=lambda p: p.z_mm)
    state = PalletState(pallet)
    valid = []
    orphans: List[Tuple[Box, int]] = []

    for p in sorted_ps:
        box = boxes_meta[p.sku_id]
        if state.can_place(
            p.length_mm, p.width_mm, p.height_mm,
            p.x_mm, p.y_mm, p.z_mm,
            box.weight_kg, box.fragile, box.stackable,
        ):
            state.place(
                p.sku_id, p.length_mm, p.width_mm, p.height_mm,
                p.x_mm, p.y_mm, p.z_mm,
                box.weight_kg, box.fragile, box.stackable,
            )
            valid.append(p)
        else:
            orphans.append((box, p.instance_index))
            logger.debug("[lns] orphan: %s at z=%d (lost support after destroy)", p.sku_id, p.z_mm)

    return state, valid, orphans


def _repair_greedy(
    state: PalletState,
    destroyed_items: List[Tuple[Box, int]],
    weight_profile: str = "default",
) -> Tuple[List[Placement], int]:
    """Try to re-place destroyed items using greedy EP approach.

    Returns (placements, total_placed_volume).
    """
    new_placements = []
    total_volume = 0

    for box, inst_idx in destroyed_items:
        orientations = get_orientations(box)
        best_score = -1.0
        best_placement = None

        for ep in list(state.extreme_points):
            ex, ey, ez = ep
            for dx, dy, dz, rot_code in orientations:
                if not state.can_place(
                    dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                ):
                    continue
                sc = score_placement(
                    state, dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile,
                    weight_profile=weight_profile,
                )
                if sc > best_score:
                    best_score = sc
                    best_placement = (dx, dy, dz, ex, ey, ez, rot_code)

        if best_placement:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(
                box.sku_id, dx, dy, dz, px, py, pz,
                box.weight_kg, box.fragile, box.stackable,
            )
            new_placements.append(Placement(
                sku_id=box.sku_id,
                instance_index=inst_idx,
                x_mm=px, y_mm=py, z_mm=pz,
                length_mm=dx, width_mm=dy, height_mm=dz,
                rotation_code=rot_code,
            ))
            total_volume += dx * dy * dz

    return new_placements, total_volume


def lns_optimize(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    initial_solution: Solution,
    destroy_fraction: float = 0.3,
    max_iterations: int = 20,
    time_budget_ms: int = 500,
    seed: int = 42,
) -> Solution:
    """Improve an existing solution using Large Neighborhood Search.

    Destroy-repair cycle:
    1. Randomly remove destroy_fraction of placed items
    2. Try to re-place them (+ previously unplaced items) using greedy
    3. Keep the new solution if it's better

    Args:
        initial_solution: Starting solution to improve
        destroy_fraction: Fraction of placements to destroy each iteration
        max_iterations: Maximum number of LNS iterations
        time_budget_ms: Time budget for LNS in milliseconds
        seed: Random seed for reproducibility
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)

    boxes_meta: Dict[str, Box] = {b.sku_id: b for b in boxes}

    best_placements = list(initial_solution.placements)
    best_placed_count = len(best_placements)

    # Track unplaced items from initial solution
    initial_unplaced_items: List[Tuple[Box, int]] = []
    placed_instance_counts: Dict[str, int] = defaultdict(int)
    for p in best_placements:
        placed_instance_counts[p.sku_id] += 1

    for box in boxes:
        placed = placed_instance_counts.get(box.sku_id, 0)
        for i in range(placed, box.quantity):
            initial_unplaced_items.append((box, i))

    logger.info(
        "[lns] task=%s iterations=%d destroy=%.0f%% initial_placed=%d unplaced=%d",
        task_id, max_iterations, destroy_fraction * 100,
        best_placed_count, len(initial_unplaced_items),
    )

    # Compute initial placed volume for accept criterion
    best_volume = sum(p.length_mm * p.width_mm * p.height_mm for p in best_placements)

    iteration = 0
    for iteration in range(max_iterations):
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > time_budget_ms * 0.9:
            logger.debug("[lns] time budget reached at iteration %d", iteration)
            break

        # Destroy strategy: alternate between random and worst-position
        n_destroy = max(1, int(len(best_placements) * destroy_fraction))
        n_destroy = min(n_destroy, len(best_placements))

        if iteration % 3 == 0:
            # Random destroy
            destroy_indices = set(rng.sample(range(len(best_placements)), n_destroy))
        elif iteration % 3 == 1:
            # Worst-position destroy: remove items placed highest (least stable)
            scored = []
            for i, p in enumerate(best_placements):
                badness = p.z_mm * 1000 - p.length_mm * p.width_mm * p.height_mm
                scored.append((badness, i))
            scored.sort(reverse=True)
            destroy_indices = set(idx for _, idx in scored[:n_destroy])
        else:
            # Fragility-aware destroy: target items involved in violations
            from .postprocess import _find_fragility_violations
            violations = _find_fragility_violations(best_placements, boxes_meta)
            violation_indices = set()
            for hi, fi in violations:
                violation_indices.add(hi)
                violation_indices.add(fi)
            destroy_indices = set()
            for idx in violation_indices:
                if len(destroy_indices) < n_destroy:
                    destroy_indices.add(idx)
            remaining_indices = [i for i in range(len(best_placements)) if i not in destroy_indices]
            if len(destroy_indices) < n_destroy and remaining_indices:
                extra = rng.sample(remaining_indices, min(n_destroy - len(destroy_indices), len(remaining_indices)))
                destroy_indices.update(extra)

        kept_placements = []
        destroyed_items: List[Tuple[Box, int]] = []

        for i, p in enumerate(best_placements):
            if i in destroy_indices:
                box = boxes_meta[p.sku_id]
                destroyed_items.append((box, p.instance_index))
            else:
                kept_placements.append(p)

        # Rebuild state from kept placements (validates support)
        state, kept_placements, orphan_items = _rebuild_state_from_placements(
            pallet, kept_placements, boxes_meta)

        # Build repair pool: orphans first (they had valid positions), then destroyed + unplaced
        repair_pool = orphan_items + destroyed_items + list(initial_unplaced_items)
        # Only shuffle destroyed + unplaced portion (keep orphans first for priority)
        non_orphan_start = len(orphan_items)
        non_orphan = repair_pool[non_orphan_start:]
        rng.shuffle(non_orphan)
        repair_pool = repair_pool[:non_orphan_start] + non_orphan

        # Rotate weight profiles across iterations for diversity
        profiles = ["default", "contact_heavy", "fill_heavy"]
        profile = profiles[iteration % len(profiles)]

        # Repair: try to place all items in repair pool
        repaired_placements, repaired_volume = _repair_greedy(state, repair_pool, weight_profile=profile)

        new_placements = kept_placements + repaired_placements
        new_placed_count = len(new_placements)
        new_volume = sum(p.length_mm * p.width_mm * p.height_mm for p in kept_placements) + repaired_volume

        # Accept if more items placed, or same items but more volume
        improved = False
        if new_placed_count > best_placed_count:
            improved = True
        elif new_placed_count == best_placed_count and new_volume > best_volume:
            improved = True

        if improved:
            logger.debug(
                "[lns] iter=%d improved: %d->%d placed, vol %d->%d profile=%s",
                iteration, best_placed_count, new_placed_count,
                best_volume, new_volume, profile,
            )
            best_placements = new_placements
            best_placed_count = new_placed_count
            best_volume = new_volume

            # Update unplaced items
            new_placed_counts: Dict[str, int] = defaultdict(int)
            for p in best_placements:
                new_placed_counts[p.sku_id] += 1
            initial_unplaced_items = []
            for box in boxes:
                placed = new_placed_counts.get(box.sku_id, 0)
                for i in range(placed, box.quantity):
                    initial_unplaced_items.append((box, i))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Build final unplaced list
    final_placed_counts: Dict[str, int] = defaultdict(int)
    for p in best_placements:
        final_placed_counts[p.sku_id] += 1

    unplaced = []
    for box in boxes:
        placed = final_placed_counts.get(box.sku_id, 0)
        if placed < box.quantity:
            unplaced.append(UnplacedItem(
                sku_id=box.sku_id,
                quantity_unplaced=box.quantity - placed,
                reason="no_space",
            ))

    total_time = initial_solution.solve_time_ms + elapsed_ms

    logger.info(
        "[lns] done iterations=%d placed=%d->%d lns_time=%dms",
        min(iteration + 1, max_iterations),
        len(initial_solution.placements), len(best_placements), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=total_time,
        placements=best_placements,
        unplaced=unplaced,
    )
