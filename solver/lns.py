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
) -> PalletState:
    """Rebuild PalletState from a list of placements."""
    state = PalletState(pallet)
    for p in placements:
        box = boxes_meta[p.sku_id]
        state.place(
            p.sku_id, p.length_mm, p.width_mm, p.height_mm,
            p.x_mm, p.y_mm, p.z_mm,
            box.weight_kg, box.fragile, box.stackable,
        )
    return state


def _repair_greedy(
    state: PalletState,
    destroyed_items: List[Tuple[Box, int]],
) -> List[Placement]:
    """Try to re-place destroyed items using greedy EP approach."""
    new_placements = []

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

    return new_placements


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

    for iteration in range(max_iterations):
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > time_budget_ms * 0.9:
            logger.debug("[lns] time budget reached at iteration %d", iteration)
            break

        # Destroy: randomly remove a fraction of placements
        n_destroy = max(1, int(len(best_placements) * destroy_fraction))
        destroy_indices = set(rng.sample(range(len(best_placements)), min(n_destroy, len(best_placements))))

        kept_placements = []
        destroyed_items: List[Tuple[Box, int]] = []

        for i, p in enumerate(best_placements):
            if i in destroy_indices:
                box = boxes_meta[p.sku_id]
                destroyed_items.append((box, p.instance_index))
            else:
                kept_placements.append(p)

        # Add previously unplaced items to repair pool
        repair_pool = destroyed_items + list(initial_unplaced_items)
        # Shuffle repair pool for diversity
        rng.shuffle(repair_pool)

        # Rebuild state from kept placements
        state = _rebuild_state_from_placements(pallet, kept_placements, boxes_meta)

        # Repair: try to place all items in repair pool
        repaired = _repair_greedy(state, repair_pool)

        new_placements = kept_placements + repaired
        new_placed_count = len(new_placements)

        if new_placed_count > best_placed_count:
            logger.debug(
                "[lns] iter=%d improved: %d -> %d placed",
                iteration, best_placed_count, new_placed_count,
            )
            best_placements = new_placements
            best_placed_count = new_placed_count

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
