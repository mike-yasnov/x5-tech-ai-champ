"""Wrapper to apply original solver's postprocessing to any solution.

Converts raw placement dicts to solver models, runs compact_downward,
reorder_fragile, try_insert_unplaced, then converts back.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.models import Box, Pallet, Placement, Solution, UnplacedItem, solution_to_dict
from solver.postprocess import compact_downward, reorder_fragile, try_insert_unplaced
from solver.pallet_state import PalletState

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))


def _build_models(scenario: Dict) -> tuple:
    """Build Pallet + Box models from scenario dict."""
    p = scenario["pallet"]
    pallet = Pallet(
        type_id=p.get("type_id", "unknown"),
        length_mm=p["length_mm"],
        width_mm=p["width_mm"],
        max_height_mm=p["max_height_mm"],
        max_weight_kg=p["max_weight_kg"],
    )
    boxes_meta = {}
    for b in scenario["boxes"]:
        box = Box(
            sku_id=b["sku_id"],
            description=b.get("description", ""),
            length_mm=b["length_mm"],
            width_mm=b["width_mm"],
            height_mm=b["height_mm"],
            weight_kg=b["weight_kg"],
            quantity=b["quantity"],
            strict_upright=b.get("strict_upright", False),
            fragile=b.get("fragile", False),
            stackable=b.get("stackable", True),
        )
        boxes_meta[b["sku_id"]] = box
    return pallet, boxes_meta


def _response_to_solution(response: Dict, scenario: Dict) -> Solution:
    """Convert response dict to Solution model."""
    placements = []
    for p in response.get("placements", []):
        dim = p["dimensions_placed"]
        pos = p["position"]
        placements.append(Placement(
            sku_id=p["sku_id"],
            instance_index=p["instance_index"],
            x_mm=pos["x_mm"],
            y_mm=pos["y_mm"],
            z_mm=pos["z_mm"],
            length_mm=dim["length_mm"],
            width_mm=dim["width_mm"],
            height_mm=dim["height_mm"],
            rotation_code=p.get("rotation_code", "LWH"),
        ))

    unplaced = []
    for u in response.get("unplaced", []):
        unplaced.append(UnplacedItem(
            sku_id=u["sku_id"],
            quantity_unplaced=u["quantity_unplaced"],
            reason=u.get("reason", "no_space"),
        ))

    return Solution(
        task_id=response.get("task_id", scenario.get("task_id", "")),
        solver_version=response.get("solver_version", "postprocess-1.0.0"),
        placements=placements,
        unplaced=unplaced,
        solve_time_ms=response.get("solve_time_ms", 0),
    )


def apply_postprocessing(
    response: Dict,
    scenario: Dict,
    do_compact: bool = True,
    do_reorder_fragile: bool = True,
    do_try_insert: bool = True,
    time_budget_ms: int = 5000,
) -> Dict:
    """Apply original solver's postprocessing pipeline to a solution.

    Args:
        response: Solution dict (with placements/unplaced)
        scenario: Original scenario dict (with pallet/boxes)
        do_compact: Run compact_downward
        do_reorder_fragile: Run reorder_fragile
        do_try_insert: Run try_insert_unplaced
        time_budget_ms: Time budget for postprocessing

    Returns:
        Improved solution dict
    """
    t0 = time.perf_counter()
    pallet, boxes_meta = _build_models(scenario)
    solution = _response_to_solution(response, scenario)

    original_placed = len(solution.placements)
    placements = list(solution.placements)
    unplaced = list(solution.unplaced)

    # 1. Compact downward
    if do_compact and placements:
        try:
            placements = compact_downward(pallet, placements, boxes_meta)
            logger.debug("compact_downward done: %d placements", len(placements))
        except Exception as e:
            logger.warning("compact_downward failed: %s", e)

    # 2. Reorder fragile
    if do_reorder_fragile and placements:
        elapsed = (time.perf_counter() - t0) * 1000
        remaining = max(100, time_budget_ms - elapsed)
        try:
            placements = reorder_fragile(
                pallet, placements, boxes_meta,
                time_budget_ms=int(remaining * 0.4),
            )
            logger.debug("reorder_fragile done: %d placements", len(placements))
        except Exception as e:
            logger.warning("reorder_fragile failed: %s", e)

    # 3. Try insert unplaced
    if do_try_insert and unplaced:
        elapsed = (time.perf_counter() - t0) * 1000
        remaining = max(100, time_budget_ms - elapsed)
        try:
            # Convert UnplacedItem list to Box list (what try_insert_unplaced expects)
            unplaced_boxes = []
            for u in unplaced:
                if u.sku_id in boxes_meta:
                    box = boxes_meta[u.sku_id]
                    # Create a Box with quantity = quantity_unplaced
                    unplaced_boxes.append(Box(
                        sku_id=box.sku_id, description=box.description,
                        length_mm=box.length_mm, width_mm=box.width_mm,
                        height_mm=box.height_mm, weight_kg=box.weight_kg,
                        quantity=u.quantity_unplaced,
                        strict_upright=box.strict_upright,
                        fragile=box.fragile, stackable=box.stackable,
                    ))
            if unplaced_boxes:
                placements, unplaced_boxes = try_insert_unplaced(
                    pallet, placements, unplaced_boxes, boxes_meta,
                    time_budget_ms=int(remaining),
                )
                # Convert remaining Box list back to UnplacedItem list
                unplaced = [
                    UnplacedItem(sku_id=b.sku_id, quantity_unplaced=b.quantity, reason="no_space")
                    for b in unplaced_boxes if b.quantity > 0
                ]
            logger.debug("try_insert_unplaced done: placed=%d, unplaced=%d",
                         len(placements), len(unplaced))
        except Exception as e:
            logger.warning("try_insert_unplaced failed: %s", e)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Build improved solution
    improved = Solution(
        task_id=solution.task_id,
        solver_version=response.get("solver_version", "postprocess-1.0.0"),
        placements=placements,
        unplaced=unplaced,
        solve_time_ms=response.get("solve_time_ms", 0) + elapsed_ms,
    )

    new_placed = len(improved.placements)
    if new_placed != original_placed:
        logger.info("Postprocessing: %d -> %d placements (+%d)",
                     original_placed, new_placed, new_placed - original_placed)

    return solution_to_dict(improved)
