from __future__ import annotations

from typing import List, Tuple

from .candidate_gen import CandidateGenerator, RemainingItem
from .constants import EPSILON, FRAGILE_WEIGHT_THRESHOLD
from .feasibility import FeasibilityChecker
from .free_space import ExtremePointManager
from .geometry import AABB
from .pallet_state import PalletState, PlacedBox


def compact_downward(
    placements: List[PlacedBox],
    pallet_length: int,
    pallet_width: int,
    max_height: int,
    max_weight: float,
) -> List[PlacedBox]:
    """For each box from top to bottom, try to lower z as much as possible."""
    # Sort by z_min descending so we process top boxes first
    indexed = sorted(range(len(placements)), key=lambda i: -placements[i].aabb.z_min)
    result = list(placements)

    for idx in indexed:
        box = result[idx]
        if box.aabb.z_min == 0:
            continue

        # Find the lowest valid z for this box
        lx = box.aabb.length_x()
        ly = box.aabb.width_y()
        lz = box.aabb.height_z()
        x = box.aabb.x_min
        y = box.aabb.y_min

        # Compute max z_max of all other boxes that overlap XY with this box
        best_z = 0
        test_aabb = AABB(x, y, 0, x + lx, y + ly, 1)
        for j, other in enumerate(result):
            if j == idx:
                continue
            if other.aabb.z_max <= box.aabb.z_min + EPSILON:
                if test_aabb.overlap_area_xy(other.aabb) > 0:
                    if other.aabb.z_max > best_z:
                        best_z = other.aabb.z_max

        if best_z >= box.aabb.z_min:
            continue

        # Try to lower to best_z
        new_aabb = AABB(x, y, best_z, x + lx, y + ly, best_z + lz)

        # Check no collision with other boxes
        collision = False
        for j, other in enumerate(result):
            if j == idx:
                continue
            if new_aabb.overlaps_3d(other.aabb):
                collision = True
                break

        if collision:
            continue

        # Check support
        if best_z > 0:
            base_area = lx * ly
            support = 0
            for j, other in enumerate(result):
                if j == idx:
                    continue
                if abs(other.aabb.z_max - best_z) < EPSILON:
                    support += new_aabb.overlap_area_xy(other.aabb)
            if base_area == 0 or support / base_area < 0.6 - EPSILON:
                continue

        # Check height
        if new_aabb.z_max > max_height + EPSILON:
            continue

        # Apply the move
        new_box = PlacedBox(
            sku_id=box.sku_id,
            instance_index=box.instance_index,
            aabb=new_aabb,
            weight=box.weight,
            fragile=box.fragile,
            stackable=box.stackable,
            strict_upright=box.strict_upright,
            rotation_code=box.rotation_code,
            placed_dims=box.placed_dims,
        )
        result[idx] = new_box

    return result


def fragile_reorder(
    placements: List[PlacedBox],
    pallet_length: int,
    pallet_width: int,
    max_height: int,
) -> List[PlacedBox]:
    """Try to swap fragile boxes upward and heavy non-fragile boxes downward.

    Goal: reduce fragility violations (heavy >2kg items resting directly on fragile items).
    """
    result = list(placements)
    improved = True
    max_iterations = 50

    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1

        # Find current fragility violations
        violations = _count_fragility_violations(result)
        if violations == 0:
            break

        # Try swapping pairs
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                bi, bj = result[i], result[j]

                # Only consider swaps where one is fragile and other is heavy
                if bi.fragile == bj.fragile:
                    continue
                # Fragile should go up, heavy should go down
                fragile_box = bi if bi.fragile else bj
                heavy_box = bj if bi.fragile else bi

                if heavy_box.weight <= FRAGILE_WEIGHT_THRESHOLD:
                    continue

                # Only swap if fragile is currently below heavy
                if fragile_box.aabb.z_min >= heavy_box.aabb.z_min:
                    continue

                # Check if they have the same footprint dimensions (can swap positions)
                fi = (fragile_box.aabb.length_x(), fragile_box.aabb.width_y())
                fj = (heavy_box.aabb.length_x(), heavy_box.aabb.width_y())
                if fi != fj:
                    continue

                # Try swapping z positions
                new_result = _try_swap_z(result, i, j, max_height)
                if new_result is not None:
                    new_violations = _count_fragility_violations(new_result)
                    if new_violations < violations:
                        result = new_result
                        improved = True
                        break
            if improved:
                break

    return result


def _count_fragility_violations(placements: List[PlacedBox]) -> int:
    count = 0
    for top in placements:
        if top.weight <= FRAGILE_WEIGHT_THRESHOLD:
            continue
        for bottom in placements:
            if not bottom.fragile:
                continue
            if abs(top.aabb.z_min - bottom.aabb.z_max) < EPSILON:
                if top.aabb.overlap_area_xy(bottom.aabb) > 0:
                    count += 1
    return count


def _try_swap_z(
    placements: List[PlacedBox],
    i: int,
    j: int,
    max_height: int,
) -> List[PlacedBox] | None:
    """Try to swap z-positions of boxes i and j. Return new list or None if infeasible."""
    bi, bj = placements[i], placements[j]

    # Simple z swap: put the lower one at the other's z, adjusting heights
    lower = bi if bi.aabb.z_min < bj.aabb.z_min else bj
    upper = bj if bi.aabb.z_min < bj.aabb.z_min else bi
    lower_idx = i if bi.aabb.z_min < bj.aabb.z_min else j
    upper_idx = j if bi.aabb.z_min < bj.aabb.z_min else i

    # Swap: lower goes to upper.z_min, upper goes to lower.z_min
    new_lower_z = lower.aabb.z_min
    new_upper_z = new_lower_z + upper.aabb.height_z()

    # Rebuild the upper box at lower's position
    new_upper_at_bottom = PlacedBox(
        sku_id=upper.sku_id,
        instance_index=upper.instance_index,
        aabb=AABB(
            upper.aabb.x_min,
            upper.aabb.y_min,
            new_lower_z,
            upper.aabb.x_max,
            upper.aabb.y_max,
            new_lower_z + upper.aabb.height_z(),
        ),
        weight=upper.weight,
        fragile=upper.fragile,
        stackable=upper.stackable,
        strict_upright=upper.strict_upright,
        rotation_code=upper.rotation_code,
        placed_dims=upper.placed_dims,
    )

    new_lower_at_top = PlacedBox(
        sku_id=lower.sku_id,
        instance_index=lower.instance_index,
        aabb=AABB(
            lower.aabb.x_min,
            lower.aabb.y_min,
            new_upper_z,
            lower.aabb.x_max,
            lower.aabb.y_max,
            new_upper_z + lower.aabb.height_z(),
        ),
        weight=lower.weight,
        fragile=lower.fragile,
        stackable=lower.stackable,
        strict_upright=lower.strict_upright,
        rotation_code=lower.rotation_code,
        placed_dims=lower.placed_dims,
    )

    # Check height bounds
    if new_lower_at_top.aabb.z_max > max_height + EPSILON:
        return None

    # Build new placements
    result = list(placements)
    result[upper_idx] = new_upper_at_bottom
    result[lower_idx] = new_lower_at_top

    # Check no collisions
    for k, box in enumerate(result):
        for m in range(k + 1, len(result)):
            if box.aabb.overlaps_3d(result[m].aabb):
                return None

    # Check support for both swapped boxes
    for idx in [upper_idx, lower_idx]:
        box = result[idx]
        if box.aabb.z_min == 0:
            continue
        base_area = box.aabb.base_area()
        support = 0
        for k, other in enumerate(result):
            if k == idx:
                continue
            if abs(other.aabb.z_max - box.aabb.z_min) < EPSILON:
                support += box.aabb.overlap_area_xy(other.aabb)
        if base_area == 0 or support / base_area < 0.6 - EPSILON:
            return None

    return result


def try_insert_unplaced(
    placements: List[PlacedBox],
    remaining: List[RemainingItem],
    state: PalletState,
    checker: FeasibilityChecker,
    ep_manager: ExtremePointManager,
    candidate_gen: CandidateGenerator,
) -> Tuple[List[PlacedBox], List[RemainingItem]]:
    """After main solve, try to squeeze remaining items into gaps."""
    from .search import _heuristic_score, _make_placed_box, _update_remaining

    extra_placements: List[PlacedBox] = []

    for _ in range(100):  # max iterations
        total_remaining = sum(it.remaining_qty for it in remaining)
        if total_remaining == 0:
            break

        candidates = candidate_gen.generate(remaining, state, ep_manager)
        if not candidates:
            break

        scored = [(c, _heuristic_score(c, state)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        best = scored[0][0]

        placed = _make_placed_box(best)
        state.place(placed)
        ep_manager.update_after_placement(best.aabb, state)
        extra_placements.append(placed)
        remaining = _update_remaining(remaining, best.sku_id)

    return placements + extra_placements, remaining


def postprocess(
    placements: List[PlacedBox],
    remaining: List[RemainingItem],
    state: PalletState,
    checker: FeasibilityChecker,
    ep_manager: ExtremePointManager,
    candidate_gen: CandidateGenerator,
) -> Tuple[List[PlacedBox], List[RemainingItem]]:
    """Run full post-processing pipeline."""
    # 1. Compact downward
    placements = compact_downward(
        placements, state.length, state.width, state.max_height, state.max_weight
    )

    # 2. Fragile reorder
    placements = fragile_reorder(
        placements, state.length, state.width, state.max_height
    )

    # 3. Try inserting unplaced items
    # Rebuild state and EP from scratch after reordering
    new_state = PalletState(state.length, state.width, state.max_height, state.max_weight)
    new_ep = ExtremePointManager(state.length, state.width)
    for pb in placements:
        new_state.place(pb)
        new_ep.update_after_placement(pb.aabb, new_state)

    placements, remaining = try_insert_unplaced(
        placements, remaining, new_state, checker, new_ep, candidate_gen
    )

    return placements, remaining
