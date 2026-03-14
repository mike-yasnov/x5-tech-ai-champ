"""Greedy layer stacking: select and stack layers onto a pallet respecting constraints."""

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .build_layers import Layer
from .pallet_state import PalletState
from . import __version__

logger = logging.getLogger(__name__)

# Stability parameter from paper: area(above) <= alpha * area(below)
ALPHA_STABILITY = 1.1


def _sort_by_density(layer: Layer) -> float:
    return -layer.density


def _sort_by_box_count(layer: Layer) -> float:
    return -layer.box_count


def _sort_by_weight(layer: Layer) -> float:
    return -layer.total_weight_kg


def _sort_by_area(layer: Layer) -> float:
    return -layer.net_area


LAYER_SORT_STRATEGIES: Dict[str, Callable[[Layer], float]] = {
    "density": _sort_by_density,
    "box_count": _sort_by_box_count,
    "weight": _sort_by_weight,
    "area": _sort_by_area,
}


def stack_layers(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    layers: List[Layer],
    sort_strategy: str = "density",
    layer_scores: Optional[Dict[int, float]] = None,
) -> Solution:
    """Greedily stack layers onto a pallet.

    Args:
        pallet: Pallet parameters
        boxes: Original box types (for tracking quantities)
        layers: Candidate layers from BuildLayers
        sort_strategy: How to rank layers
        layer_scores: Optional ML-predicted scores (layer_index -> score)
    """
    t0 = time.perf_counter()

    # Sort layers by strategy or ML scores
    if layer_scores:
        indexed_layers = sorted(
            enumerate(layers),
            key=lambda x: -layer_scores.get(x[0], 0.0),
        )
        sorted_layers = [l for _, l in indexed_layers]
    else:
        sort_fn = LAYER_SORT_STRATEGIES.get(sort_strategy, _sort_by_density)
        sorted_layers = sorted(layers, key=sort_fn)

    # Track remaining quantities per SKU
    remaining: Dict[str, int] = {b.sku_id: b.quantity for b in boxes}

    # Stack layers greedily
    stacked: List[Tuple[Layer, int]] = []  # (layer, z_offset)
    current_z = 0
    current_weight = 0.0
    prev_area = pallet.length_mm * pallet.width_mm  # floor provides full area

    for layer in sorted_layers:
        # Check height constraint
        if current_z + layer.height_mm > pallet.max_height_mm:
            continue

        # Check weight constraint
        if current_weight + layer.total_weight_kg > pallet.max_weight_kg + 1e-6:
            continue

        # Check stability: area(this_layer) <= alpha * area(layer_below)
        if stacked and layer.net_area > ALPHA_STABILITY * prev_area:
            continue

        # Check SKU quantity constraints
        can_use = True
        for sku_id, count in layer.box_counts.items():
            if remaining.get(sku_id, 0) < count:
                can_use = False
                break
        if not can_use:
            continue

        # Place this layer
        stacked.append((layer, current_z))
        current_z += layer.height_mm
        current_weight += layer.total_weight_kg
        prev_area = layer.net_area

        # Update remaining quantities
        for sku_id, count in layer.box_counts.items():
            remaining[sku_id] -= count

        logger.debug(
            "[stack_layers] stacked layer: h=%d w=%.1f area=%d boxes=%d z=%d",
            layer.height_mm, layer.total_weight_kg, layer.net_area,
            layer.box_count, current_z - layer.height_mm,
        )

    # Convert stacked layers to Placements, validating each via PalletState
    placements: List[Placement] = []
    instance_counters: Dict[str, int] = defaultdict(int)
    validation_state = PalletState(pallet)
    boxes_meta = {b.sku_id: b for b in boxes}

    for layer, z_offset in stacked:
        for p2d in layer.placements:
            box = boxes_meta.get(p2d.sku_id)
            if box is None:
                continue

            # Validate placement through PalletState (checks support, collisions, etc.)
            if not validation_state.can_place(
                p2d.dx, p2d.dy, p2d.dz,
                p2d.x, p2d.y, z_offset,
                box.weight_kg, box.fragile, box.stackable,
            ):
                logger.debug(
                    "[stack_layers] skipped invalid placement: %s at (%d,%d,%d)",
                    p2d.sku_id, p2d.x, p2d.y, z_offset,
                )
                continue

            validation_state.place(
                p2d.sku_id, p2d.dx, p2d.dy, p2d.dz,
                p2d.x, p2d.y, z_offset,
                box.weight_kg, box.fragile, box.stackable,
            )

            inst_idx = instance_counters[p2d.sku_id]
            instance_counters[p2d.sku_id] += 1

            placements.append(Placement(
                sku_id=p2d.sku_id,
                instance_index=inst_idx,
                x_mm=p2d.x,
                y_mm=p2d.y,
                z_mm=z_offset,
                length_mm=p2d.dx,
                width_mm=p2d.dy,
                height_mm=p2d.dz,
                rotation_code="LWH",
            ))

    # Build unplaced list
    unplaced = []
    for sku_id, rem in remaining.items():
        if rem > 0:
            unplaced.append(UnplacedItem(
                sku_id=sku_id,
                quantity_unplaced=rem,
                reason="no_space",
            ))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    total_items = sum(b.quantity for b in boxes)

    logger.info(
        "[stack_layers] done layers=%d placed=%d/%d height=%d/%d weight=%.1f/%.1f time=%dms",
        len(stacked), len(placements), total_items,
        current_z, pallet.max_height_mm,
        current_weight, pallet.max_weight_kg, elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )
