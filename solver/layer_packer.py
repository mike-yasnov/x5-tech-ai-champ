"""Layer-based packer: builds horizontal layers, then stacks them."""

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .orientations import get_orientations
from .pallet_state import PalletState, _overlap_area
from . import __version__

logger = logging.getLogger(__name__)


def _expand_boxes(boxes: List[Box]) -> List[Tuple[Box, int]]:
    result = []
    for box in boxes:
        for i in range(box.quantity):
            result.append((box, i))
    return result


def _pack_layer_2d(
    pallet_length: int,
    pallet_width: int,
    layer_height: int,
    items: List[Tuple[Box, int]],
    max_weight: float,
) -> Tuple[List[Tuple[Box, int, int, int, int, int, int, str]], List[Tuple[Box, int]]]:
    """Pack items into a single 2D layer using shelf algorithm.

    Returns (placed, remaining) where placed = [(box, inst, x, y, dx, dy, dz, rot_code), ...]
    """
    placed = []
    remaining = []

    # Shelf-based: fill rows left-to-right, then advance y
    shelf_y = 0
    shelf_height = 0  # max dy in current shelf
    cursor_x = 0
    current_weight = 0.0

    for box, inst_idx in items:
        if current_weight + box.weight_kg > max_weight + 1e-6:
            remaining.append((box, inst_idx))
            continue

        orientations = get_orientations(box)
        best_fit = None

        for dx, dy, dz, rot_code in orientations:
            if dz > layer_height:
                continue
            # Try current shelf position
            if cursor_x + dx <= pallet_length and shelf_y + dy <= pallet_width:
                waste = (layer_height - dz)  # prefer filling layer height
                best_fit = (dx, dy, dz, rot_code, cursor_x, shelf_y, waste)
                break
            # Try new shelf
            new_y = shelf_y + shelf_height
            if new_y + dy <= pallet_width and dx <= pallet_length:
                best_fit = (dx, dy, dz, rot_code, 0, new_y, (layer_height - dz))
                break

        if best_fit:
            dx, dy, dz, rot_code, px, py, waste = best_fit
            placed.append((box, inst_idx, px, py, dx, dy, dz, rot_code))
            current_weight += box.weight_kg

            if py == shelf_y:
                cursor_x = px + dx
                shelf_height = max(shelf_height, dy)
            else:
                shelf_y = py
                shelf_height = dy
                cursor_x = dx
        else:
            remaining.append((box, inst_idx))

    return placed, remaining


def pack_layers(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key: Optional[Callable[[Box], tuple]] = None,
) -> Solution:
    """Layer-based packing: group by height, pack layers, stack.

    Strategy:
    1. Group boxes by compatible heights
    2. Pack each group as a 2D layer
    3. Stack layers bottom-to-top, heavy first
    """
    t0 = time.perf_counter()

    if sort_key:
        sorted_boxes = sorted(boxes, key=sort_key)
    else:
        sorted_boxes = sorted(boxes, key=lambda b: (-b.weight_kg, -b.volume))

    instances = _expand_boxes(sorted_boxes)
    total_items = len(instances)

    # Group by approximate layer height (bucket by height_mm with tolerance)
    height_groups: Dict[int, List[Tuple[Box, int]]] = defaultdict(list)
    for box, inst_idx in instances:
        orients = get_orientations(box)
        # Prefer orientation where height (Z) is smallest fitting dimension
        best_h = min(dz for _, _, dz, _ in orients)
        # Bucket to nearest 50mm
        bucket = (best_h // 50) * 50 + 50
        height_groups[bucket].append((box, inst_idx))

    # Sort groups: process groups with heavier/larger boxes first
    sorted_groups = sorted(height_groups.items(), key=lambda g: -g[0])

    state = PalletState(pallet)
    placements: List[Placement] = []
    all_remaining: List[Tuple[Box, int]] = []
    current_z = 0

    logger.info(
        "[layer_packer] task=%s groups=%d items=%d",
        task_id, len(sorted_groups), total_items,
    )

    for layer_height, group_items in sorted_groups:
        if current_z + layer_height > pallet.max_height_mm:
            all_remaining.extend(group_items)
            continue

        remaining_weight = pallet.max_weight_kg - state.current_weight
        layer_placed, layer_remaining = _pack_layer_2d(
            pallet.length_mm, pallet.width_mm,
            layer_height, group_items, remaining_weight,
        )

        if not layer_placed:
            all_remaining.extend(group_items)
            continue

        # Place items from this layer into the actual pallet state
        for box, inst_idx, px, py, dx, dy, dz, rot_code in layer_placed:
            z = current_z
            if state.can_place(dx, dy, dz, px, py, z, box.weight_kg, box.fragile, box.stackable):
                state.place(box.sku_id, dx, dy, dz, px, py, z,
                           box.weight_kg, box.fragile, box.stackable)
                placements.append(Placement(
                    sku_id=box.sku_id, instance_index=inst_idx,
                    x_mm=px, y_mm=py, z_mm=z,
                    length_mm=dx, width_mm=dy, height_mm=dz,
                    rotation_code=rot_code,
                ))
            else:
                all_remaining.append((box, inst_idx))

        current_z += layer_height
        all_remaining.extend(layer_remaining)

    # Try to place remaining items using greedy EP-based approach
    for box, inst_idx in all_remaining:
        orientations = get_orientations(box)
        placed = False
        best_score = -1.0
        best_placement = None

        for ep in list(state.extreme_points):
            ex, ey, ez = ep
            for dx, dy, dz, rot_code in orientations:
                if state.can_place(dx, dy, dz, ex, ey, ez,
                                  box.weight_kg, box.fragile, box.stackable):
                    # Simple score: lower z + more contact
                    sc = 1.0 - (ez / pallet.max_height_mm)
                    if sc > best_score:
                        best_score = sc
                        best_placement = (dx, dy, dz, ex, ey, ez, rot_code)

        if best_placement:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(box.sku_id, dx, dy, dz, px, py, pz,
                       box.weight_kg, box.fragile, box.stackable)
            placements.append(Placement(
                sku_id=box.sku_id, instance_index=inst_idx,
                x_mm=px, y_mm=py, z_mm=pz,
                length_mm=dx, width_mm=dy, height_mm=dz,
                rotation_code=rot_code,
            ))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Build unplaced
    placed_counts: Dict[str, int] = defaultdict(int)
    for p in placements:
        placed_counts[p.sku_id] += 1

    unplaced = []
    for box in boxes:
        placed_count = placed_counts.get(box.sku_id, 0)
        if placed_count < box.quantity:
            unplaced.append(UnplacedItem(
                sku_id=box.sku_id,
                quantity_unplaced=box.quantity - placed_count,
                reason="no_space",
            ))

    logger.info(
        "[layer_packer] done placed=%d/%d time=%dms",
        len(placements), total_items, elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )
