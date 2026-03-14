"""Greedy packer: places boxes one by one using Extreme Points + scoring."""

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .orientations import get_orientations
from .pallet_state import PalletState
from .scoring import score_placement
from . import __version__

logger = logging.getLogger(__name__)


# ── Sort key factories ──────────────────────────────────────────────

def _sort_volume_desc(box: Box) -> tuple:
    return (-box.volume,)


def _sort_weight_desc(box: Box) -> tuple:
    return (-box.weight_kg,)


def _sort_base_area_desc(box: Box) -> tuple:
    return (-box.base_area,)


def _sort_density_desc(box: Box) -> tuple:
    vol = box.volume if box.volume > 0 else 1
    return (-box.weight_kg / vol,)


def _sort_constrained_first(box: Box) -> tuple:
    # Constrained items first, then by volume desc
    priority = 0
    if box.strict_upright:
        priority -= 2
    if not box.stackable:
        priority -= 1
    return (priority, -box.volume)


def _sort_volume_asc(box: Box) -> tuple:
    return (box.volume,)


def _sort_height_desc(box: Box) -> tuple:
    return (-box.height_mm,)


def _sort_fragile_last(box: Box) -> tuple:
    # Non-fragile stackable first (heavy base), fragile and non-stackable last
    penalty = 0
    if box.fragile:
        penalty += 2
    if not box.stackable:
        penalty += 1
    return (penalty, -box.volume)


def _sort_non_stackable_last(box: Box) -> tuple:
    # Non-stackable items last (they block stacking), then by volume desc
    return (0 if box.stackable else 1, -box.volume)


def _sort_max_dim_desc(box: Box) -> tuple:
    return (-max(box.length_mm, box.width_mm, box.height_mm),)


# Ordered by effectiveness across scenarios (best strategies first for adaptive budget)
SORT_KEYS: Dict[str, Callable[[Box], tuple]] = {
    "constrained_first": _sort_constrained_first,   # Best for heavy_water
    "base_area_desc": _sort_base_area_desc,          # Best for liquid_tetris
    "fragile_last": _sort_fragile_last,              # Best for fragile_mix
    "volume_desc": _sort_volume_desc,                # Best for random_mixed
    "volume_asc": _sort_volume_asc,                  # Best for cavity_fill
    "density_desc": _sort_density_desc,              # Good for fragile_mix
    "non_stackable_last": _sort_non_stackable_last,  # Good for random_mixed
    "height_desc": _sort_height_desc,                # Good overall
    "weight_desc": _sort_weight_desc,                # Secondary
    "max_dim_desc": _sort_max_dim_desc,              # Secondary
}


# ── Expand boxes ────────────────────────────────────────────────────

def _expand_boxes(boxes: List[Box]) -> List[Tuple[Box, int]]:
    """Expand SKUs with quantity > 1 into individual (box, instance_index) pairs."""
    result = []
    for box in boxes:
        for i in range(box.quantity):
            result.append((box, i))
    return result


# ── Greedy packer ───────────────────────────────────────────────────

def pack_greedy(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str = "volume_desc",
    time_limit_ms: int = 0,
) -> Solution:
    """Pack boxes greedily using Extreme Points + scoring function.

    Args:
        time_limit_ms: If > 0, stop packing when this time limit is reached.
    """
    t0 = time.perf_counter()

    sort_fn = SORT_KEYS.get(sort_key_name, _sort_volume_desc)

    # Sort boxes by key, then expand
    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = _expand_boxes(sorted_boxes)

    state = PalletState(pallet)
    placements: List[Placement] = []
    unplaced_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0})

    logger.info(
        "[pack_greedy] task=%s sort=%s total_instances=%d",
        task_id, sort_key_name, len(instances),
    )

    check_interval = max(10, len(instances) // 10)  # Check time every ~10% of items

    for item_idx, (box, inst_idx) in enumerate(instances):
        # Periodic time check inside packing loop
        if time_limit_ms > 0 and item_idx % check_interval == 0 and item_idx > 0:
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > time_limit_ms:
                logger.debug("[pack_greedy] time limit reached at item %d/%d", item_idx, len(instances))
                # Mark remaining as unplaced
                for remaining_box, remaining_idx in instances[item_idx:]:
                    if state.current_weight + remaining_box.weight_kg > pallet.max_weight_kg:
                        reason = "weight_limit_exceeded"
                    else:
                        reason = "no_space"
                    unplaced_counts[remaining_box.sku_id]["count"] += 1
                    unplaced_counts[remaining_box.sku_id]["reason"] = reason
                break

        orientations = get_orientations(box)
        best_score = -1.0
        best_placement: Optional[Tuple[int, int, int, int, int, int, str]] = None

        # Try each EP × orientation
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

        if best_placement is not None:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(
                box.sku_id, dx, dy, dz, px, py, pz,
                box.weight_kg, box.fragile, box.stackable,
            )
            placements.append(Placement(
                sku_id=box.sku_id,
                instance_index=inst_idx,
                x_mm=px, y_mm=py, z_mm=pz,
                length_mm=dx, width_mm=dy, height_mm=dz,
                rotation_code=rot_code,
            ))
        else:
            # Determine reason
            if state.current_weight + box.weight_kg > pallet.max_weight_kg:
                reason = "weight_limit_exceeded"
            else:
                reason = "no_space"
            unplaced_counts[box.sku_id]["count"] += 1
            unplaced_counts[box.sku_id]["reason"] = reason
            logger.debug(
                "[pack_greedy] unplaced sku=%s instance=%d reason=%s",
                box.sku_id, inst_idx, reason,
            )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    unplaced = [
        UnplacedItem(sku_id=sid, quantity_unplaced=info["count"], reason=info["reason"])
        for sid, info in unplaced_counts.items()
    ]

    logger.info(
        "[pack_greedy] done sort=%s placed=%d unplaced=%d time=%dms",
        sort_key_name, len(placements), sum(u.quantity_unplaced for u in unplaced), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )
