from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .candidate_gen import CandidateGenerator, RemainingItem
from .constants import UNPLACED_REASONS
from .feasibility import FeasibilityChecker
from .free_space import ExtremePointManager
from .pallet_state import PalletState, PlacedBox
from .postprocess import postprocess
from .search import beam_search_solve, greedy_solve


logger = logging.getLogger(__name__)
SOLVER_VERSION = "hyb-solver-v0.3"


def solve_request(
    request: Dict[str, Any],
    model_dir: str | Path = "models",
    beam_width: int = 8,
    max_expansions: int = 3,
    time_budget_ms: int = 4000,
) -> Dict[str, Any]:
    """Solve a request dict and return the response dict."""
    t0 = time.perf_counter()

    pallet = request["pallet"]
    boxes = request["boxes"]

    pal_l = pallet["length_mm"]
    pal_w = pallet["width_mm"]
    pal_h = pallet["max_height_mm"]
    pal_wt = pallet["max_weight_kg"]

    # Initialize solver components
    state = PalletState(pal_l, pal_w, pal_h, pal_wt)
    ep_manager = ExtremePointManager(pal_l, pal_w)
    checker = FeasibilityChecker(pal_l, pal_w, pal_h, pal_wt)
    candidate_gen = CandidateGenerator(checker, max_candidates=200)

    # Build remaining items list
    remaining = [RemainingItem(b) for b in boxes]

    total_items = sum(b["quantity"] for b in boxes)
    effective_beam = max(1, beam_width)
    if total_items > 100:
        effective_beam = min(effective_beam, 4)
    if total_items > 200:
        effective_beam = min(effective_beam, 2)

    model, extractor = _load_optional_model(model_dir)

    if effective_beam > 1:
        placements, leftover = beam_search_solve(
            remaining,
            state,
            ep_manager,
            checker,
            candidate_gen,
            model=model,
            extractor=extractor,
            beam_width=effective_beam,
            max_expansions=max(2, max_expansions),
            time_budget_s=(time_budget_ms / 1000.0) * 0.65,
        )
    else:
        placements, leftover = greedy_solve(
            remaining, state, ep_manager, checker, candidate_gen
        )

    # Post-processing: compact, fragile reorder, insert unplaced
    placements, leftover = postprocess(
        placements, leftover, state, checker, ep_manager, candidate_gen
    )

    # Rebuild state for unplaced reason classification
    final_state = PalletState(pal_l, pal_w, pal_h, pal_wt)
    for pb in placements:
        final_state.place(pb)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return _format_output(
        request["task_id"], placements, leftover, boxes, final_state, elapsed_ms
    )


def _load_optional_model(
    model_dir: str | Path,
) -> Tuple[Optional[object], Optional[object]]:
    model_path = Path(model_dir)
    if not (model_path / "rf.pkl").exists() or not (model_path / "svr_pipeline.pkl").exists():
        return None, None

    try:
        from .features import FeatureExtractor
        from .hyb_model import HYBModel
    except Exception as exc:
        logger.warning("[hybrid] optional model support unavailable: %s", exc)
        return None, None

    model = HYBModel(model_dir=model_path)
    if not model.load():
        return None, None
    return model, FeatureExtractor()


def _format_output(
    task_id: str,
    placements: List[PlacedBox],
    leftover: List[RemainingItem],
    boxes: List[Dict],
    state: PalletState,
    solve_time_ms: int,
) -> Dict[str, Any]:
    placed_list = []
    for pb in placements:
        placed_list.append(
            {
                "sku_id": pb.sku_id,
                "instance_index": pb.instance_index,
                "position": {
                    "x_mm": pb.aabb.x_min,
                    "y_mm": pb.aabb.y_min,
                    "z_mm": pb.aabb.z_min,
                },
                "dimensions_placed": {
                    "length_mm": pb.placed_dims[0],
                    "width_mm": pb.placed_dims[1],
                    "height_mm": pb.placed_dims[2],
                },
                "rotation_code": pb.rotation_code,
            }
        )

    unplaced_list = []
    for item in leftover:
        if item.remaining_qty > 0:
            reason = _classify_unplaced_reason(item, state)
            unplaced_list.append(
                {
                    "sku_id": item.sku_id,
                    "quantity_unplaced": item.remaining_qty,
                    "reason": reason,
                }
            )

    return {
        "task_id": task_id,
        "solver_version": SOLVER_VERSION,
        "solve_time_ms": solve_time_ms,
        "placements": placed_list,
        "unplaced": unplaced_list,
    }


def _classify_unplaced_reason(item: RemainingItem, state: PalletState) -> str:
    if state.total_weight + item.weight > state.max_weight:
        return UNPLACED_REASONS["weight"]

    # Check if the item is too tall for remaining height
    min_height = min(item.height, item.width, item.length)
    if item.strict_upright:
        min_height = item.height
    if state.max_z + min_height > state.max_height:
        return UNPLACED_REASONS["height"]

    return UNPLACED_REASONS["space"]


# Backward-compatible alias for direct request-dict usage.
solve = solve_request
