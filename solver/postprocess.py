"""Post-processing: compact-down, try-insert-unplaced."""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .orientations import get_orientations
from .pallet_state import PalletState, _overlap_area
from .scoring import score_placement
from . import __version__

logger = logging.getLogger(__name__)


def _find_lowest_z(state: PalletState, p: Placement, box_idx: int) -> int:
    """Find the lowest valid z for a placement, keeping x,y,dims fixed.

    Temporarily removes the box from state to avoid self-collision.
    """
    x, y = p.x_mm, p.y_mm
    dx, dy, dz = p.length_mm, p.width_mm, p.height_mm
    x2, y2 = x + dx, y + dy

    # Find max z_max of boxes below that overlap in XY
    best_z = 0  # floor
    for i, box in enumerate(state.boxes):
        if i == box_idx:
            continue
        # Check XY overlap
        overlap = _overlap_area(x, y, x2, y2,
                                box.x_min, box.y_min, box.x_max, box.y_max)
        if overlap > 0 and box.z_max <= p.z_mm:
            best_z = max(best_z, box.z_max)

    return best_z


def compact_downward(
    pallet: Pallet,
    placements: List[Placement],
    boxes_meta: Dict[str, Box],
) -> List[Placement]:
    """Try to lower each box as far as possible to fill vertical gaps.

    Process boxes from top to bottom. For each box, find the lowest z
    where it has valid support and no collisions.
    """
    if not placements:
        return placements

    # Build state with all boxes
    state = PalletState(pallet)
    for p in placements:
        box = boxes_meta[p.sku_id]
        state.place(
            p.sku_id, p.length_mm, p.width_mm, p.height_mm,
            p.x_mm, p.y_mm, p.z_mm,
            box.weight_kg, box.fragile, box.stackable,
        )

    # Sort by z descending (process top boxes first)
    indexed = sorted(enumerate(placements), key=lambda x: -x[1].z_mm)
    result = list(placements)
    moved = 0

    for orig_idx, p in indexed:
        box = boxes_meta[p.sku_id]
        dx, dy, dz = p.length_mm, p.width_mm, p.height_mm

        # Find lowest valid z
        lowest_z = _find_lowest_z(state, p, orig_idx)

        if lowest_z >= p.z_mm:
            continue

        # Check support at lowest_z
        x, y = p.x_mm, p.y_mm
        x2, y2 = x + dx, y + dy
        base_area = dx * dy

        if lowest_z == 0:
            support_ok = True
        else:
            support_area = 0
            for i, sb in enumerate(state.boxes):
                if i == orig_idx:
                    continue
                if sb.z_max == lowest_z:
                    if not sb.stackable:
                        overlap = _overlap_area(x, y, x2, y2,
                                                sb.x_min, sb.y_min, sb.x_max, sb.y_max)
                        if overlap > 0:
                            support_ok = False
                            break
                    support_area += _overlap_area(x, y, x2, y2,
                                                  sb.x_min, sb.y_min, sb.x_max, sb.y_max)
            else:
                support_ok = base_area > 0 and support_area / base_area >= 0.6

        if not support_ok:
            continue

        # Check no collision at new position
        new_z2 = lowest_z + dz
        collision = False
        for i, sb in enumerate(state.boxes):
            if i == orig_idx:
                continue
            if (sb.z_min < new_z2 and sb.z_max > lowest_z
                    and sb.x_min < x2 and sb.x_max > x
                    and sb.y_min < y2 and sb.y_max > y):
                collision = True
                break

        if collision:
            continue

        # Check height limit
        if new_z2 > pallet.max_height_mm:
            continue

        # Apply the move
        state.boxes[orig_idx].z_min = lowest_z
        state.boxes[orig_idx].z_max = new_z2
        result[orig_idx] = Placement(
            sku_id=p.sku_id,
            instance_index=p.instance_index,
            x_mm=p.x_mm, y_mm=p.y_mm, z_mm=lowest_z,
            length_mm=dx, width_mm=dy, height_mm=dz,
            rotation_code=p.rotation_code,
        )
        moved += 1

    if moved:
        logger.info("[compact] moved %d boxes downward", moved)
    return result


def _generate_gap_eps(state: PalletState) -> List[Tuple[int, int, int]]:
    """Generate extra extreme points by scanning all box boundaries.

    Creates a grid of potential positions at intersections of
    existing box edges and z-levels, finding gaps the standard
    EP generation might miss.
    """
    if not state.boxes:
        return []

    # Collect all unique x, y, z boundaries
    x_coords = {0}
    y_coords = {0}
    z_coords = {0}
    for box in state.boxes:
        x_coords.update([box.x_min, box.x_max])
        y_coords.update([box.y_min, box.y_max])
        z_coords.update([box.z_min, box.z_max])

    # Only keep z_max values (top surfaces where new boxes can rest)
    z_tops = sorted({box.z_max for box in state.boxes} | {0})

    extra_eps = []
    for z in z_tops:
        for x in sorted(x_coords):
            for y in sorted(y_coords):
                if x >= state.pallet.length_mm or y >= state.pallet.width_mm:
                    continue
                if z >= state.pallet.max_height_mm:
                    continue
                # Check not inside any box
                inside = False
                for box in state.boxes:
                    if (box.x_min <= x < box.x_max
                            and box.y_min <= y < box.y_max
                            and box.z_min <= z < box.z_max):
                        inside = True
                        break
                if not inside:
                    extra_eps.append((x, y, z))

    return extra_eps


def try_insert_unplaced(
    pallet: Pallet,
    placements: List[Placement],
    boxes: List[Box],
    boxes_meta: Dict[str, Box],
    weight_profile: str = "default",
) -> Tuple[List[Placement], List[UnplacedItem]]:
    """Try to insert unplaced items into remaining gaps.

    Rebuilds state from existing placements, generates extra EPs
    from box boundary grid, then tries to fit remaining items.
    """
    # Figure out what's unplaced
    placed_counts: Dict[str, int] = defaultdict(int)
    for p in placements:
        placed_counts[p.sku_id] += 1

    unplaced_items: List[Tuple[Box, int]] = []
    for box in boxes:
        placed = placed_counts.get(box.sku_id, 0)
        for i in range(placed, box.quantity):
            unplaced_items.append((box, i))

    if not unplaced_items:
        return placements, []

    # Sort unplaced by volume desc (try to fit big items first)
    unplaced_items.sort(key=lambda x: -x[0].volume)

    # Rebuild state
    state = PalletState(pallet)
    for p in placements:
        box = boxes_meta[p.sku_id]
        state.place(
            p.sku_id, p.length_mm, p.width_mm, p.height_mm,
            p.x_mm, p.y_mm, p.z_mm,
            box.weight_kg, box.fragile, box.stackable,
        )

    # Generate extra EPs from box boundary grid
    gap_eps = _generate_gap_eps(state)
    all_eps = list(set(state.extreme_points) | set(gap_eps))
    logger.info("[try_insert] %d standard EPs + %d gap EPs = %d total",
                len(state.extreme_points), len(gap_eps), len(all_eps))

    # Try to place each unplaced item
    new_placements = list(placements)
    still_unplaced: Dict[str, int] = defaultdict(int)
    inserted = 0

    for box, inst_idx in unplaced_items:
        orientations = get_orientations(box)
        best_score = -1.0
        best_placement = None

        for ep in all_eps:
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

        if best_placement is not None:
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
            # Update gap EPs after each insertion
            gap_eps = _generate_gap_eps(state)
            all_eps = list(set(state.extreme_points) | set(gap_eps))
            inserted += 1
        else:
            still_unplaced[box.sku_id] += 1

    if inserted:
        logger.info("[try_insert] inserted %d additional items", inserted)

    unplaced_list = [
        UnplacedItem(sku_id=sid, quantity_unplaced=cnt, reason="no_space")
        for sid, cnt in still_unplaced.items()
    ]

    return new_placements, unplaced_list


def postprocess_solution(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    solution: Solution,
) -> Solution:
    """Apply all post-processing steps to improve a solution."""
    boxes_meta: Dict[str, Box] = {b.sku_id: b for b in boxes}

    # Step 1: Compact downward
    placements = compact_downward(pallet, list(solution.placements), boxes_meta)

    # Step 2: Try insert unplaced items
    placements, unplaced = try_insert_unplaced(
        pallet, placements, boxes, boxes_meta,
    )

    # Step 3: Compact again after insertions
    if len(placements) > len(solution.placements):
        placements = compact_downward(pallet, placements, boxes_meta)

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=solution.solve_time_ms,
        placements=placements,
        unplaced=unplaced,
    )
