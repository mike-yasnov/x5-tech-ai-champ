"""Greedy packer: places boxes one by one using Extreme Points + scoring."""

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


SORT_KEYS: Dict[str, Callable[[Box], tuple]] = {
    "volume_desc": _sort_volume_desc,
    "weight_desc": _sort_weight_desc,
    "base_area_desc": _sort_base_area_desc,
    "density_desc": _sort_density_desc,
    "constrained_first": _sort_constrained_first,
}


# ── Expand boxes ────────────────────────────────────────────────────


def expand_boxes(boxes: List[Box]) -> List[Tuple[Box, int]]:
    """Expand SKUs with quantity > 1 into individual (box, instance_index) pairs."""
    result = []
    for box in boxes:
        for i in range(box.quantity):
            result.append((box, i))
    return result


def pack_in_order(
    task_id: str,
    pallet: Pallet,
    instances: Sequence[Tuple[Box, int]],
    order_name: str = "custom",
) -> Solution:
    """Pack pre-expanded box instances in the provided order."""
    t0 = time.perf_counter()

    state = PalletState(pallet)
    placements: List[Placement] = []
    unplaced_qty: Dict[str, int] = defaultdict(int)
    unplaced_reason: Dict[str, str] = {}

    logger.info(
        "[pack_in_order] task=%s order=%s total_instances=%d",
        task_id,
        order_name,
        len(instances),
    )

    for box, inst_idx in instances:
        orientations = get_orientations(box)
        best_score = -1.0
        best_placement: Optional[Tuple[int, int, int, int, int, int, str]] = None

        for ex, ey, ez in list(state.extreme_points):
            for dx, dy, dz, rot_code in orientations:
                if not state.can_place(
                    dx, dy, dz, ex, ey, ez, box.weight_kg, box.fragile, box.stackable
                ):
                    continue

                sc = score_placement(
                    state, dx, dy, dz, ex, ey, ez, box.weight_kg, box.fragile
                )
                if sc > best_score:
                    best_score = sc
                    best_placement = (dx, dy, dz, ex, ey, ez, rot_code)

        if best_placement is not None:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(
                box.sku_id,
                dx,
                dy,
                dz,
                px,
                py,
                pz,
                box.weight_kg,
                box.fragile,
                box.stackable,
            )
            placements.append(
                Placement(
                    sku_id=box.sku_id,
                    instance_index=inst_idx,
                    x_mm=px,
                    y_mm=py,
                    z_mm=pz,
                    length_mm=dx,
                    width_mm=dy,
                    height_mm=dz,
                    rotation_code=rot_code,
                )
            )
        else:
            if state.current_weight + box.weight_kg > pallet.max_weight_kg:
                reason = "weight_limit_exceeded"
            else:
                reason = "no_space"
            unplaced_qty[box.sku_id] += 1
            unplaced_reason[box.sku_id] = reason
            logger.debug(
                "[pack_in_order] unplaced sku=%s instance=%d reason=%s",
                box.sku_id,
                inst_idx,
                reason,
            )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    unplaced = [
        UnplacedItem(
            sku_id=sid,
            quantity_unplaced=qty,
            reason=unplaced_reason.get(sid, "no_space"),
        )
        for sid, qty in unplaced_qty.items()
    ]

    logger.info(
        "[pack_in_order] done order=%s placed=%d unplaced=%d time=%dms",
        order_name,
        len(placements),
        sum(u.quantity_unplaced for u in unplaced),
        elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )


# ── Greedy packer ───────────────────────────────────────────────────


def pack_greedy(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str = "volume_desc",
) -> Solution:
    """Pack boxes greedily using Extreme Points + scoring function.

    Returns a Solution with placements and unplaced items.
    """
    sort_fn = SORT_KEYS.get(sort_key_name, _sort_volume_desc)

    # Sort boxes by key, then expand
    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = expand_boxes(sorted_boxes)
    return pack_in_order(task_id, pallet, instances, order_name=sort_key_name)
