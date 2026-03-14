"""Public solver entrypoint with strategy dispatch."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import __version__
from .hybrid import solve_request as solve_hybrid_request
from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .portfolio_block import (
    solve_legacy_greedy_request,
    solve_request as solve_portfolio_request,
)


logger = logging.getLogger(__name__)

STRATEGIES = ("portfolio_block", "legacy_hybrid", "legacy_greedy")


def _legacy_effort_to_beam_width(n_restarts: int) -> int:
    """Map the old multi-restart knob to conservative hybrid-search effort."""
    return max(1, min(4, (max(n_restarts, 1) + 9) // 10))


def _request_from_models(task_id: str, pallet: Pallet, boxes: List[Box]) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "pallet": {
            "type_id": pallet.type_id,
            "length_mm": pallet.length_mm,
            "width_mm": pallet.width_mm,
            "max_height_mm": pallet.max_height_mm,
            "max_weight_kg": pallet.max_weight_kg,
        },
        "boxes": [
            {
                "sku_id": box.sku_id,
                "description": box.description,
                "length_mm": box.length_mm,
                "width_mm": box.width_mm,
                "height_mm": box.height_mm,
                "weight_kg": box.weight_kg,
                "quantity": box.quantity,
                "strict_upright": box.strict_upright,
                "fragile": box.fragile,
                "stackable": box.stackable,
            }
            for box in boxes
        ],
    }


def _solution_from_response(response: Dict[str, Any]) -> Solution:
    placements = [
        Placement(
            sku_id=item["sku_id"],
            instance_index=item["instance_index"],
            x_mm=item["position"]["x_mm"],
            y_mm=item["position"]["y_mm"],
            z_mm=item["position"]["z_mm"],
            length_mm=item["dimensions_placed"]["length_mm"],
            width_mm=item["dimensions_placed"]["width_mm"],
            height_mm=item["dimensions_placed"]["height_mm"],
            rotation_code=item["rotation_code"],
        )
        for item in response.get("placements", [])
    ]

    unplaced = [
        UnplacedItem(
            sku_id=item["sku_id"],
            quantity_unplaced=item["quantity_unplaced"],
            reason=item["reason"],
        )
        for item in response.get("unplaced", [])
    ]

    return Solution(
        task_id=response["task_id"],
        solver_version=response.get("solver_version", __version__),
        solve_time_ms=response.get("solve_time_ms", 0),
        placements=placements,
        unplaced=unplaced,
    )


def solve(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    request_dict: Optional[Dict[str, Any]],
    n_restarts: int = 10,
    time_budget_ms: int = 900,
    beam_width: Optional[int] = None,
    model_dir: str = "models",
    strategy: str = "portfolio_block",
) -> Solution:
    """Solve using the configured runtime strategy.

    `n_restarts` is kept as a legacy public knob and now maps to search effort.
    """
    request = dict(request_dict) if request_dict is not None else _request_from_models(
        task_id, pallet, boxes
    )
    request["task_id"] = task_id
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    effective_beam = (
        max(1, beam_width)
        if beam_width is not None
        else _legacy_effort_to_beam_width(n_restarts)
    )
    max_expansions = max(2, min(5, effective_beam))

    logger.info(
        "[solve] task=%s strategy=%s legacy_effort=%d beam_width=%d max_expansions=%d budget=%dms model_dir=%s",
        task_id,
        strategy,
        n_restarts,
        effective_beam,
        max_expansions,
        time_budget_ms,
        model_dir,
    )

    if strategy == "portfolio_block":
        response = solve_portfolio_request(
            request=request,
            model_dir=model_dir,
            time_budget_ms=time_budget_ms,
        )
    elif strategy == "legacy_hybrid":
        response = solve_hybrid_request(
            request=request,
            model_dir=model_dir,
            beam_width=effective_beam,
            max_expansions=max_expansions,
            time_budget_ms=time_budget_ms,
        )
    else:
        response = solve_legacy_greedy_request(
            request=request,
            time_budget_ms=time_budget_ms,
        )

    solution = _solution_from_response(response)
    solution.task_id = task_id
    solution.solver_version = __version__
    return solution
