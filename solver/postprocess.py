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
    """Generate extra extreme points by scanning box boundaries.

    Uses a spatial index for fast inside-box checks.
    """
    if not state.boxes:
        return []

    # Collect all unique x, y boundaries
    x_coords = sorted({0} | {b.x_min for b in state.boxes} | {b.x_max for b in state.boxes})
    y_coords = sorted({0} | {b.y_min for b in state.boxes} | {b.y_max for b in state.boxes})
    z_tops = sorted({0} | {b.z_max for b in state.boxes})

    # Filter to valid bounds
    x_coords = [x for x in x_coords if x < state.pallet.length_mm]
    y_coords = [y for y in y_coords if y < state.pallet.width_mm]
    z_tops = [z for z in z_tops if z < state.pallet.max_height_mm]

    # Build z-interval index for faster inside checks
    # For each (x,y) we check against all boxes — use early exit
    extra_eps = []
    boxes = state.boxes

    for z in z_tops:
        # Pre-filter boxes that span this z level
        z_boxes = [b for b in boxes if b.z_min <= z < b.z_max]

        for x in x_coords:
            for y in y_coords:
                # Check not inside any box at this z
                inside = False
                for box in z_boxes:
                    if (box.x_min <= x < box.x_max
                            and box.y_min <= y < box.y_max):
                        inside = True
                        break
                if not inside:
                    extra_eps.append((x, y, z))

    # Prioritize low z positions (bottom-up filling)
    extra_eps.sort(key=lambda ep: (ep[2], ep[0], ep[1]))

    return extra_eps


def try_insert_unplaced(
    pallet: Pallet,
    placements: List[Placement],
    boxes: List[Box],
    boxes_meta: Dict[str, Box],
    weight_profile: str = "default",
    time_budget_ms: int = 500,
) -> Tuple[List[Placement], List[UnplacedItem]]:
    """Try to insert unplaced items into remaining gaps.

    Rebuilds state from existing placements, generates extra EPs
    from box boundary grid, then tries to fit remaining items.
    """
    import time as _time
    t0 = _time.perf_counter()
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

    # Phase-aware ordering: place items that build stable base first
    # non-fragile stackable → non-fragile heavy → fragile heavy → fragile light → non-stackable
    def _insert_priority(item):
        box = item[0]
        if not box.fragile and box.stackable:
            tier = 0  # Best base items
        elif not box.fragile and box.weight_kg > 2.0:
            tier = 1  # Heavy non-fragile
        elif box.fragile and box.weight_kg > 2.0:
            tier = 2  # Heavy fragile
        elif box.fragile:
            tier = 3  # Light fragile (place on top)
        else:
            tier = 4  # Non-stackable (place last, nothing goes on them)
        return (tier, -box.volume)

    unplaced_items.sort(key=_insert_priority)

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

    for item_i, (box, inst_idx) in enumerate(unplaced_items):
        # Time check every item
        elapsed = (_time.perf_counter() - t0) * 1000
        if elapsed > time_budget_ms * 0.9:
            logger.debug("[try_insert] time budget reached after %d items", item_i)
            for remaining_box, remaining_idx in unplaced_items[item_i:]:
                still_unplaced[remaining_box.sku_id] += 1
            break

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
                # Penalize creating new fragile violations during insert
                if box.weight_kg > 2.0 and ez > 0:
                    fragile_below = state.get_fragile_boxes_at_top(
                        ez, ex, ey, ex + dx, ey + dy)
                    if fragile_below:
                        sc -= 0.3
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
            # Regenerate gap EPs after insertion to find new positions
            gap_eps = _generate_gap_eps(state)
            all_eps = list(set(state.extreme_points) | set(gap_eps))
            inserted += 1
        else:
            still_unplaced[box.sku_id] += 1

    # Grid search fallback for small number of remaining unplaced items
    if still_unplaced and len(still_unplaced) <= 3:
        elapsed = (_time.perf_counter() - t0) * 1000
        remaining_budget = time_budget_ms - elapsed
        if remaining_budget > 50:
            # Re-collect the actual unplaced items
            grid_items: List[Tuple[Box, int]] = []
            placed_counts2: Dict[str, int] = defaultdict(int)
            for p in new_placements:
                placed_counts2[p.sku_id] += 1
            for box in boxes:
                placed = placed_counts2.get(box.sku_id, 0)
                for i in range(placed, box.quantity):
                    grid_items.append((box, i))
            grid_items.sort(key=lambda x: -x[0].volume)

            for box, inst_idx in grid_items[:3]:
                elapsed = (_time.perf_counter() - t0) * 1000
                if elapsed > time_budget_ms * 0.95:
                    break
                orientations = get_orientations(box)
                found = False
                # Scan at z levels where support exists
                z_levels = sorted({0} | {b.z_max for b in state.boxes})
                z_levels = [z for z in z_levels if z + min(d[2] for d in orientations) <= pallet.max_height_mm]
                for step in [50, 20]:
                    if found:
                        break
                    for dx, dy, dz, rot_code in orientations:
                        if found:
                            break
                        for z in z_levels:
                            if z + dz > pallet.max_height_mm:
                                continue
                            if found:
                                break
                            for x in range(0, pallet.length_mm - dx + 1, step):
                                if found:
                                    break
                                for y in range(0, pallet.width_mm - dy + 1, step):
                                    if state.can_place(dx, dy, dz, x, y, z,
                                                       box.weight_kg, box.fragile, box.stackable):
                                        state.place(box.sku_id, dx, dy, dz, x, y, z,
                                                    box.weight_kg, box.fragile, box.stackable)
                                        new_placements.append(Placement(
                                            sku_id=box.sku_id, instance_index=inst_idx,
                                            x_mm=x, y_mm=y, z_mm=z,
                                            length_mm=dx, width_mm=dy, height_mm=dz,
                                            rotation_code=rot_code,
                                        ))
                                        if box.sku_id in still_unplaced:
                                            still_unplaced[box.sku_id] -= 1
                                            if still_unplaced[box.sku_id] <= 0:
                                                del still_unplaced[box.sku_id]
                                        inserted += 1
                                        found = True
                                        logger.info("[try_insert] grid search placed %s step=%d", box.sku_id, step)
                                        break
                    elapsed = (_time.perf_counter() - t0) * 1000
                    if elapsed > time_budget_ms * 0.95:
                        break

    if inserted:
        logger.info("[try_insert] inserted %d additional items total", inserted)

    unplaced_list = [
        UnplacedItem(sku_id=sid, quantity_unplaced=cnt, reason="no_space")
        for sid, cnt in still_unplaced.items()
        if cnt > 0
    ]

    return new_placements, unplaced_list


def _find_fragility_violations(
    placements: List[Placement],
    boxes_meta: Dict[str, Box],
) -> List[Tuple[int, int]]:
    """Find pairs (heavy_idx, fragile_idx) where non-fragile heavy box rests on fragile box.

    Per task spec: fragile=true means "нельзя ставить под не-хрупкий груз тяжелее 2 кг".
    Fragile-on-fragile is NOT a violation.
    """
    violations = []
    for i, p_heavy in enumerate(placements):
        box_heavy = boxes_meta[p_heavy.sku_id]
        if box_heavy.weight_kg <= 2.0 or box_heavy.fragile or p_heavy.z_mm == 0:
            continue

        hx1, hy1 = p_heavy.x_mm, p_heavy.y_mm
        hx2 = hx1 + p_heavy.length_mm
        hy2 = hy1 + p_heavy.width_mm

        for j, p_frag in enumerate(placements):
            if i == j:
                continue
            box_frag = boxes_meta[p_frag.sku_id]
            if not box_frag.fragile:
                continue

            fz_max = p_frag.z_mm + p_frag.height_mm
            if fz_max != p_heavy.z_mm:
                continue

            fx1, fy1 = p_frag.x_mm, p_frag.y_mm
            fx2 = fx1 + p_frag.length_mm
            fy2 = fy1 + p_frag.width_mm

            overlap = _overlap_area(hx1, hy1, hx2, hy2, fx1, fy1, fx2, fy2)
            if overlap > 0:
                violations.append((i, j))

    return violations


def _validate_placement_feasibility(
    pallet: Pallet,
    placements: List[Placement],
    boxes_meta: Dict[str, Box],
) -> bool:
    """Quick check that all placements are feasible (bounds, collision, support)."""
    state = PalletState(pallet)
    # Sort by z so we can validate support incrementally
    sorted_placements = sorted(placements, key=lambda p: p.z_mm)
    for p in sorted_placements:
        box = boxes_meta[p.sku_id]
        if not state.can_place(
            p.length_mm, p.width_mm, p.height_mm,
            p.x_mm, p.y_mm, p.z_mm,
            box.weight_kg, box.fragile, box.stackable,
        ):
            return False
        state.place(
            p.sku_id, p.length_mm, p.width_mm, p.height_mm,
            p.x_mm, p.y_mm, p.z_mm,
            box.weight_kg, box.fragile, box.stackable,
        )
    return True


def _try_repack_violations(
    pallet: Pallet,
    placements: List[Placement],
    boxes_meta: Dict[str, Box],
    remove_heavy_only: bool = False,
    profile: str = "fill_heavy",
    fragile_penalty: float = 0.5,
) -> List[Placement]:
    """Single attempt to reduce fragility violations by repack.

    Args:
        remove_heavy_only: If True, only remove heavy boxes from violations
                          (keep fragile in place). If False, remove both.
    """
    violations = _find_fragility_violations(placements, boxes_meta)
    if not violations:
        return placements

    # Collect indices to remove
    to_remove = set()
    for heavy_idx, frag_idx in violations:
        to_remove.add(heavy_idx)
        if not remove_heavy_only:
            to_remove.add(frag_idx)

    kept = []
    removed_items = []
    for i, p in enumerate(placements):
        if i in to_remove:
            box = boxes_meta[p.sku_id]
            removed_items.append((box, p.instance_index))
        else:
            kept.append(p)

    # Sort: non-fragile heavy first, then fragile heavy, then light fragile
    removed_items.sort(key=lambda item: (
        0 if not item[0].fragile else (1 if item[0].weight_kg > 2.0 else 2),
        -item[0].weight_kg,
    ))

    state = PalletState(pallet)
    for p in kept:
        box = boxes_meta[p.sku_id]
        state.place(
            p.sku_id, p.length_mm, p.width_mm, p.height_mm,
            p.x_mm, p.y_mm, p.z_mm,
            box.weight_kg, box.fragile, box.stackable,
        )

    gap_eps = _generate_gap_eps(state)
    all_eps = list(set(state.extreme_points) | set(gap_eps))

    repacked = list(kept)
    for box, inst_idx in removed_items:
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
                    weight_profile=profile,
                )
                if box.weight_kg > 2.0 and ez > 0:
                    fragile_below = state.get_fragile_boxes_at_top(ez, ex, ey, ex + dx, ey + dy)
                    if fragile_below:
                        sc -= fragile_penalty
                if sc > best_score:
                    best_score = sc
                    best_placement = (dx, dy, dz, ex, ey, ez, rot_code)

        if best_placement:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(
                box.sku_id, dx, dy, dz, px, py, pz,
                box.weight_kg, box.fragile, box.stackable,
            )
            repacked.append(Placement(
                sku_id=box.sku_id,
                instance_index=inst_idx,
                x_mm=px, y_mm=py, z_mm=pz,
                length_mm=dx, width_mm=dy, height_mm=dz,
                rotation_code=rot_code,
            ))
            gap_eps = _generate_gap_eps(state)
            all_eps = list(set(state.extreme_points) | set(gap_eps))

    new_violation_count = len(_find_fragility_violations(repacked, boxes_meta))
    if len(repacked) >= len(placements) and new_violation_count < len(violations):
        if _validate_placement_feasibility(pallet, repacked, boxes_meta):
            return repacked
        else:
            logger.info("[repack_violations] result has broken feasibility, reverting")
    return placements


def _lift_fragile_to_top(
    pallet: Pallet,
    placements: List[Placement],
    boxes_meta: Dict[str, Box],
) -> List[Placement]:
    """Move violated fragile items to the top of the packing.

    For each fragile item that has heavy items on top:
    remove the fragile item, then re-place it at the highest available position
    (on top of everything, where nothing heavy can go above it).
    """
    violations = _find_fragility_violations(placements, boxes_meta)
    if not violations:
        return placements

    # Find unique fragile items involved in violations
    fragile_indices = set()
    for _, fi in violations:
        fragile_indices.add(fi)

    # Remove fragile items from placements
    kept = []
    removed_fragiles: List[Tuple[int, Placement]] = []
    for i, p in enumerate(placements):
        if i in fragile_indices:
            removed_fragiles.append((i, p))
        else:
            kept.append(p)

    # Rebuild state without fragile items
    state = PalletState(pallet)
    for p in kept:
        box = boxes_meta[p.sku_id]
        state.place(
            p.sku_id, p.length_mm, p.width_mm, p.height_mm,
            p.x_mm, p.y_mm, p.z_mm,
            box.weight_kg, box.fragile, box.stackable,
        )

    # Try to re-place fragile items, preferring highest z (on top)
    result = list(kept)
    for orig_idx, p in removed_fragiles:
        box = boxes_meta[p.sku_id]
        orientations = get_orientations(box)

        # Collect all valid placements, sort by z DESC (highest first)
        valid_placements = []
        all_eps = list(state.extreme_points)
        gap_eps = _generate_gap_eps(state)
        all_eps = list(set(all_eps) | set(gap_eps))

        for ep in all_eps:
            ex, ey, ez = ep
            for dx, dy, dz, rot_code in orientations:
                if not state.can_place(
                    dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                ):
                    continue
                valid_placements.append((ez, dx, dy, dz, ex, ey, ez, rot_code))

        if valid_placements:
            # Pick highest z placement (on top, less likely to get heavy items above)
            valid_placements.sort(key=lambda x: -x[0])
            _, dx, dy, dz, px, py, pz, rot_code = valid_placements[0]
            state.place(
                box.sku_id, dx, dy, dz, px, py, pz,
                box.weight_kg, box.fragile, box.stackable,
            )
            result.append(Placement(
                sku_id=p.sku_id,
                instance_index=p.instance_index,
                x_mm=px, y_mm=py, z_mm=pz,
                length_mm=dx, width_mm=dy, height_mm=dz,
                rotation_code=rot_code,
            ))
        else:
            # Can't re-place, keep original
            result.append(p)

    new_v = len(_find_fragility_violations(result, boxes_meta))
    if new_v < len(violations) and len(result) >= len(placements):
        # Verify all placements are still feasible (support, collision)
        if _validate_placement_feasibility(pallet, result, boxes_meta):
            logger.info("[lift_fragile] improved %d->%d violations", len(violations), new_v)
            return result
        else:
            logger.info("[lift_fragile] result has broken support, reverting")

    return placements


def reorder_fragile(
    pallet: Pallet,
    placements: List[Placement],
    boxes_meta: Dict[str, Box],
    time_budget_ms: int = 50,
) -> List[Placement]:
    """Reduce fragility violations by multiple strategies.

    Tries approaches in order:
    1. Lift fragile items to top (fast, most effective)
    2. Remove both heavy and fragile violators, repack
    3. Remove only heavy violators, repack
    """
    import time as _time
    t0 = _time.perf_counter()

    if not placements:
        return placements

    violations = _find_fragility_violations(placements, boxes_meta)
    if not violations:
        return placements

    best = placements
    best_v = len(violations)

    # Strategy 1: Lift fragile items to top
    result = _lift_fragile_to_top(pallet, best, boxes_meta)
    new_v = len(_find_fragility_violations(result, boxes_meta))
    if new_v < best_v and len(result) >= len(best):
        best = result
        best_v = new_v
        if best_v == 0:
            return best

    # Strategy 2-3: Repack approaches
    configs = [
        (False, "fill_heavy", 0.5),
        (True, "fill_heavy", 0.5),
    ]

    for remove_heavy_only, profile, penalty in configs:
        elapsed = (_time.perf_counter() - t0) * 1000
        if elapsed > time_budget_ms * 0.8:
            break
        result = _try_repack_violations(
            pallet, best, boxes_meta,
            remove_heavy_only=remove_heavy_only,
            profile=profile,
            fragile_penalty=penalty,
        )
        new_v = len(_find_fragility_violations(result, boxes_meta))
        if new_v < best_v and len(result) >= len(best):
            logger.info("[fragile_reorder] improved %d->%d violations (heavy_only=%s profile=%s)",
                        best_v, new_v, remove_heavy_only, profile)
            best = result
            best_v = new_v
            if best_v == 0:
                break

    return best


def remove_and_refill(
    pallet: Pallet,
    placements: List[Placement],
    boxes: List[Box],
    boxes_meta: Dict[str, Box],
    remove_fraction: float = 0.2,
    time_budget_ms: int = 100,
) -> Tuple[List[Placement], List[UnplacedItem]]:
    """Remove worst-placed items and try to refill them + unplaced items.

    Removes items with highest z (worst packed, wasting vertical space),
    then tries to re-place them along with originally unplaced items.
    This can find better arrangements than the original greedy.
    """
    import time as _time
    t0 = _time.perf_counter()

    if not placements:
        return placements, []

    n_remove = max(1, int(len(placements) * remove_fraction))

    # Score each placement by "badness": high z + small volume = bad placement
    scored = []
    for i, p in enumerate(placements):
        vol = p.length_mm * p.width_mm * p.height_mm
        # Higher z and smaller volume = worse placement
        badness = p.z_mm * 1000 - vol
        scored.append((badness, i))
    scored.sort(reverse=True)

    remove_indices = set(idx for _, idx in scored[:n_remove])

    kept = []
    removed_items: List[Tuple[Box, int]] = []
    for i, p in enumerate(placements):
        if i in remove_indices:
            box = boxes_meta[p.sku_id]
            removed_items.append((box, p.instance_index))
        else:
            kept.append(p)

    # Add originally unplaced items
    placed_counts: Dict[str, int] = defaultdict(int)
    for p in placements:
        placed_counts[p.sku_id] += 1
    for box in boxes:
        placed = placed_counts.get(box.sku_id, 0)
        for i in range(placed, box.quantity):
            removed_items.append((box, i))

    # Sort by volume desc for greedy placement
    removed_items.sort(key=lambda x: -x[0].volume)

    # Rebuild state from kept placements
    state = PalletState(pallet)
    for p in kept:
        box = boxes_meta[p.sku_id]
        state.place(
            p.sku_id, p.length_mm, p.width_mm, p.height_mm,
            p.x_mm, p.y_mm, p.z_mm,
            box.weight_kg, box.fragile, box.stackable,
        )

    # Generate EPs and try to place
    gap_eps = _generate_gap_eps(state)
    all_eps = list(set(state.extreme_points) | set(gap_eps))

    refilled = list(kept)
    still_unplaced: Dict[str, int] = defaultdict(int)

    for box, inst_idx in removed_items:
        elapsed = (_time.perf_counter() - t0) * 1000
        if elapsed > time_budget_ms * 0.9:
            still_unplaced[box.sku_id] += 1
            continue

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
                    weight_profile="fill_heavy",
                )
                # Extra penalty for placing heavy on fragile
                if box.weight_kg > 2.0 and ez > 0:
                    fragile_below = state.get_fragile_boxes_at_top(ez, ex, ey, ex + dx, ey + dy)
                    if fragile_below:
                        sc -= 0.3
                if sc > best_score:
                    best_score = sc
                    best_placement = (dx, dy, dz, ex, ey, ez, rot_code)

        if best_placement:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(
                box.sku_id, dx, dy, dz, px, py, pz,
                box.weight_kg, box.fragile, box.stackable,
            )
            refilled.append(Placement(
                sku_id=box.sku_id, instance_index=inst_idx,
                x_mm=px, y_mm=py, z_mm=pz,
                length_mm=dx, width_mm=dy, height_mm=dz,
                rotation_code=rot_code,
            ))
            # Regenerate EPs
            gap_eps = _generate_gap_eps(state)
            all_eps = list(set(state.extreme_points) | set(gap_eps))
        else:
            still_unplaced[box.sku_id] += 1

    unplaced_list = [
        UnplacedItem(sku_id=sid, quantity_unplaced=cnt, reason="no_space")
        for sid, cnt in still_unplaced.items()
        if cnt > 0
    ]

    # Only accept if we placed at least as many items
    if len(refilled) >= len(placements):
        logger.info("[remove_refill] %d->%d placed (removed %d, refilled %d)",
                    len(placements), len(refilled), n_remove, len(refilled) - len(kept))
        return refilled, unplaced_list

    logger.debug("[remove_refill] no improvement (%d->%d)", len(placements), len(refilled))
    # Return original - rebuild unplaced
    orig_unplaced: Dict[str, int] = defaultdict(int)
    placed_counts2: Dict[str, int] = defaultdict(int)
    for p in placements:
        placed_counts2[p.sku_id] += 1
    for box in boxes:
        placed = placed_counts2.get(box.sku_id, 0)
        if placed < box.quantity:
            orig_unplaced[box.sku_id] = box.quantity - placed
    return placements, [
        UnplacedItem(sku_id=sid, quantity_unplaced=cnt, reason="no_space")
        for sid, cnt in orig_unplaced.items() if cnt > 0
    ]


def postprocess_solution(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    solution: Solution,
    time_budget_ms: int = 500,
) -> Solution:
    """Apply all post-processing steps to improve a solution."""
    import time as _time
    t0 = _time.perf_counter()

    boxes_meta: Dict[str, Box] = {b.sku_id: b for b in boxes}

    if not solution.placements:
        return solution

    # Quick check: if nothing to improve, skip entirely
    has_unplaced = any(u.quantity_unplaced > 0 for u in solution.unplaced)
    has_violations = bool(_find_fragility_violations(list(solution.placements), boxes_meta))

    if not has_unplaced and not has_violations:
        logger.info("[postprocess] nothing to improve (no unplaced, no violations)")
        return solution

    # Step 1: Compact downward (only useful if we'll try inserting more items)
    placements = list(solution.placements)
    if has_unplaced:
        placements = compact_downward(pallet, placements, boxes_meta)

    # Step 2: Reorder to reduce fragility violations (only if violations exist)
    if has_violations:
        placements = reorder_fragile(pallet, placements, boxes_meta)

    # Step 3: Try insert unplaced items (gets most of the budget)
    elapsed = (_time.perf_counter() - t0) * 1000
    remaining = time_budget_ms - elapsed
    if remaining > 50:
        placements, unplaced = try_insert_unplaced(
            pallet, placements, boxes, boxes_meta,
            time_budget_ms=int(remaining * 0.9),
        )
    else:
        # Build unplaced list without trying
        placed_counts: Dict[str, int] = defaultdict(int)
        for p in placements:
            placed_counts[p.sku_id] += 1
        unplaced = []
        for box in boxes:
            placed = placed_counts.get(box.sku_id, 0)
            if placed < box.quantity:
                unplaced.append(UnplacedItem(
                    sku_id=box.sku_id,
                    quantity_unplaced=box.quantity - placed,
                    reason="no_space",
                ))

    # Step 4: Second fragile reorder after insert (insert may create new violations)
    post_insert_violations = _find_fragility_violations(placements, boxes_meta)
    if post_insert_violations:
        elapsed2 = (_time.perf_counter() - t0) * 1000
        remaining2 = time_budget_ms - elapsed2
        if remaining2 > 30:
            pre_count = len(post_insert_violations)
            placements = reorder_fragile(pallet, placements, boxes_meta,
                                         time_budget_ms=min(int(remaining2 * 0.5), 50))
            post_count = len(_find_fragility_violations(placements, boxes_meta))
            if post_count < pre_count:
                logger.info("[postprocess] second reorder_fragile: %d->%d violations",
                            pre_count, post_count)

    # Step 5: Remove-and-refill to improve packing quality
    elapsed3 = (_time.perf_counter() - t0) * 1000
    remaining3 = time_budget_ms - elapsed3
    if remaining3 > 100:
        refilled, refill_unplaced = remove_and_refill(
            pallet, placements, boxes, boxes_meta,
            remove_fraction=0.15,
            time_budget_ms=int(remaining3 * 0.5),
        )
        if len(refilled) >= len(placements):
            # Check if fragility improved or didn't get worse
            old_v = len(_find_fragility_violations(placements, boxes_meta))
            new_v = len(_find_fragility_violations(refilled, boxes_meta))
            if new_v <= old_v:
                placements = refilled
                unplaced = refill_unplaced
                logger.info("[postprocess] remove_refill: %d placed, violations %d->%d",
                            len(placements), old_v, new_v)

    # Step 6: Compact again after insertions (only if safe)
    if len(placements) > len(solution.placements):
        compacted = compact_downward(pallet, placements, boxes_meta)
        # Verify compaction didn't break support
        if _validate_placement_feasibility(pallet, compacted, boxes_meta):
            placements = compacted
        else:
            logger.info("[postprocess] compact after insert broke feasibility, reverting")

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=solution.solve_time_ms,
        placements=placements,
        unplaced=unplaced,
    )
