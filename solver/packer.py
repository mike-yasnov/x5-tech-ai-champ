"""Greedy packer: places boxes one by one using Extreme Points + scoring."""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .heuristics import SORT_KEYS, perturb_boxes, sort_boxes
from .orientations import get_orientations
from .pallet_state import PalletState
from .scoring import score_placement
from . import __version__

logger = logging.getLogger(__name__)


# ── Expand boxes ────────────────────────────────────────────────────


def _expand_boxes(boxes: List[Box]) -> List[Tuple[Box, int]]:
    """Expand SKUs with quantity > 1 into individual (box, instance_index) pairs."""
    result = []
    for box in boxes:
        for i in range(box.quantity):
            result.append((box, i))
    return result


def _find_best_placement(
    state: PalletState,
    box: Box,
    placement_policy: str,
) -> Optional[Tuple[int, int, int, int, int, int, str]]:
    orientations = get_orientations(box)
    best_score = -1.0
    best_placement: Optional[Tuple[int, int, int, int, int, int, str]] = None

    for ep in list(state.extreme_points):
        ex, ey, ez = ep
        for dx, dy, dz, rot_code in orientations:
            if not state.can_place(
                dx,
                dy,
                dz,
                ex,
                ey,
                ez,
                box.weight_kg,
                box.fragile,
                box.stackable,
            ):
                continue

            score = score_placement(
                state,
                dx,
                dy,
                dz,
                ex,
                ey,
                ez,
                box.weight_kg,
                box.fragile,
                policy=placement_policy,
            )
            if score > best_score:
                best_score = score
                best_placement = (dx, dy, dz, ex, ey, ez, rot_code)

    return best_placement


def _residual_fill_order(
    failed_instances: List[Tuple[Box, int]],
) -> List[Tuple[Box, int]]:
    return sorted(
        failed_instances,
        key=lambda item: (
            item[0].fragile,
            item[0].strict_upright,
            item[0].volume,
            item[0].base_area,
            item[0].weight_kg,
        ),
    )


# ── Greedy packer ───────────────────────────────────────────────────


def pack_greedy(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str = "volume_desc",
    placement_policy: str = "balanced",
    randomized: bool = False,
    noise_factor: float = 0.0,
) -> Solution:
    """Pack boxes greedily using Extreme Points + scoring function.

    Returns a Solution with placements and unplaced items.
    """
    t0 = time.perf_counter()

    # Sort boxes by heuristic, optionally perturb for multi-start search
    sorted_boxes = sort_boxes(boxes, sort_key_name)
    if randomized:
        sorted_boxes = perturb_boxes(sorted_boxes, noise_factor)
    instances = _expand_boxes(sorted_boxes)

    state = PalletState(pallet)
    placements: List[Placement] = []
    unplaced_qty: Dict[str, int] = defaultdict(int)
    unplaced_reason: Dict[str, str] = {}
    failed_instances: List[Tuple[Box, int]] = []

    logger.info(
        "[pack_greedy] task=%s sort=%s policy=%s randomized=%s total_instances=%d",
        task_id,
        sort_key_name,
        placement_policy,
        randomized,
        len(instances),
    )

    for box, inst_idx in instances:
        best_placement = _find_best_placement(state, box, placement_policy)

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
            # Determine reason
            if state.current_weight + box.weight_kg > pallet.max_weight_kg:
                reason = "weight_limit_exceeded"
            else:
                reason = "no_space"
            failed_instances.append((box, inst_idx))
            logger.debug(
                "[pack_greedy] unplaced sku=%s instance=%d reason=%s",
                box.sku_id,
                inst_idx,
                reason,
            )

    if failed_instances:
        residual_policy = (
            "fragile_safe"
            if placement_policy in {"fragile_safe", "center_stable", "max_support"}
            else "max_contact"
        )
        for box, inst_idx in _residual_fill_order(failed_instances):
            best_placement = _find_best_placement(state, box, residual_policy)
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
        "[pack_greedy] done sort=%s policy=%s placed=%d unplaced=%d time=%dms",
        sort_key_name,
        placement_policy,
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
