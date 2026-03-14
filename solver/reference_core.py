from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from validator import evaluate_packing_quality

from .hybrid.candidate_gen import RemainingItem
from .hybrid.constants import EPSILON, FRAGILE_WEIGHT_THRESHOLD, UNPLACED_REASONS
from .hybrid.feasibility import FeasibilityChecker
from .hybrid.free_space import ExtremePointManager
from .hybrid.geometry import AABB
from .hybrid.pallet_state import PalletState, PlacedBox
from .hybrid.rotations import get_orientations

try:
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover - runtime fallback
    cp_model = None


EPS_OPTIMAL_GAP = 0.005
REFERENCE_SOLVER_VERSION = "exact-reference-v0.1"


@dataclass
class ReferenceAttempt:
    stage: str
    response: Dict[str, Any]
    packing_score_lb: float
    packing_score_ub: float
    proven_optimal: bool
    eps_optimal: bool
    wall_time_ms: int
    notes: str = ""


def request_hash(request: Dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ortools_available() -> bool:
    return cp_model is not None


def build_certificate(
    request: Dict[str, Any],
    attempt: ReferenceAttempt,
) -> Dict[str, Any]:
    packing_eval = evaluate_packing_quality(request, attempt.response)
    if not packing_eval.get("valid", False):
        raise ValueError(f"Reference attempt is invalid: {packing_eval.get('error')}")
    return {
        "request_hash": request_hash(request),
        "task_id": request["task_id"],
        "strategy": "exact_reference",
        "stage": attempt.stage,
        "packing_score_lb": round(attempt.packing_score_lb, 4),
        "packing_score_ub": round(attempt.packing_score_ub, 4),
        "gap": round(max(0.0, attempt.packing_score_ub - attempt.packing_score_lb), 4),
        "proven_optimal": attempt.proven_optimal,
        "eps_optimal": attempt.eps_optimal,
        "wall_time_ms": attempt.wall_time_ms,
        "solver_version": attempt.response.get("solver_version", REFERENCE_SOLVER_VERSION),
        "notes": attempt.notes,
        "metrics": packing_eval["metrics"],
        "placements": attempt.response.get("placements", []),
        "response": attempt.response,
    }


def coarse_upper_bound(request: Dict[str, Any]) -> float:
    pallet = request["pallet"]
    pallet_volume = (
        pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    )
    total_items = sum(box["quantity"] for box in request["boxes"])

    remaining_volume = sum(
        box["length_mm"] * box["width_mm"] * box["height_mm"] * box["quantity"]
        for box in request["boxes"]
    )
    volume_term = 0.50 * min(1.0, remaining_volume / max(pallet_volume, 1))

    weights: List[float] = []
    volumes: List[int] = []
    for box in request["boxes"]:
        volume = box["length_mm"] * box["width_mm"] * box["height_mm"]
        for _ in range(box["quantity"]):
            weights.append(box["weight_kg"])
            volumes.append(volume)
    weights.sort()
    volumes.sort()

    max_weight = pallet["max_weight_kg"]
    max_count_weight = 0
    acc_weight = 0.0
    for weight in weights:
        if acc_weight + weight > max_weight + 1e-9:
            break
        acc_weight += weight
        max_count_weight += 1

    max_count_volume = 0
    acc_volume = 0
    for volume in volumes:
        if acc_volume + volume > pallet_volume:
            break
        acc_volume += volume
        max_count_volume += 1

    count_term = 0.30 * min(
        1.0,
        min(total_items, max_count_weight, max_count_volume) / max(total_items, 1),
    )
    return round(volume_term + count_term + 0.10, 4)


def _packing_objective_coeff(
    volume: int,
    item_count: int,
    fragility_violations: int,
    pallet_volume: int,
    total_items: int,
) -> int:
    return (
        500 * total_items * volume
        + 300 * pallet_volume * item_count
        - 5 * pallet_volume * total_items * fragility_violations
    )


def _reindex_placements(placements: Sequence[PlacedBox]) -> List[PlacedBox]:
    counters: Dict[str, int] = {}
    indexed: List[PlacedBox] = []
    for placement in placements:
        instance_index = counters.get(placement.sku_id, 0)
        counters[placement.sku_id] = instance_index + 1
        indexed.append(
            PlacedBox(
                sku_id=placement.sku_id,
                instance_index=instance_index,
                aabb=placement.aabb,
                weight=placement.weight,
                fragile=placement.fragile,
                stackable=placement.stackable,
                strict_upright=placement.strict_upright,
                rotation_code=placement.rotation_code,
                placed_dims=placement.placed_dims,
            )
        )
    return indexed


def _count_fragility_violations(placements: Sequence[PlacedBox]) -> int:
    violations = 0
    for top in placements:
        if top.weight <= FRAGILE_WEIGHT_THRESHOLD:
            continue
        for bottom in placements:
            if not bottom.fragile:
                continue
            if abs(top.aabb.z_min - bottom.aabb.z_max) < EPSILON:
                if top.aabb.overlap_area_xy(bottom.aabb) > 0:
                    violations += 1
    return violations


def _packing_score_from_placements(
    request: Dict[str, Any],
    placements: Sequence[PlacedBox],
) -> float:
    pallet = request["pallet"]
    total_items = sum(box["quantity"] for box in request["boxes"])
    pallet_volume = (
        pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    )
    placed_volume = sum(placement.aabb.volume() for placement in placements)
    volume_utilization = placed_volume / max(pallet_volume, 1)
    coverage = len(placements) / max(total_items, 1)
    fragility_score = max(0.0, 1.0 - 0.05 * _count_fragility_violations(placements))
    return round(
        0.50 * volume_utilization + 0.30 * coverage + 0.10 * fragility_score,
        4,
    )


def _classify_unplaced_reason(item: RemainingItem, state: PalletState) -> str:
    if state.total_weight + item.weight > state.max_weight + EPSILON:
        return UNPLACED_REASONS["weight"]
    min_height = min(item.length, item.width, item.height)
    if item.strict_upright:
        min_height = item.height
    if state.max_z + min_height > state.max_height + EPSILON:
        return UNPLACED_REASONS["height"]
    return UNPLACED_REASONS["space"]


def placements_to_response(
    request: Dict[str, Any],
    placements: Sequence[PlacedBox],
    solve_time_ms: int,
    solver_version: str = REFERENCE_SOLVER_VERSION,
) -> Dict[str, Any]:
    indexed = _reindex_placements(placements)
    counts: Dict[str, int] = {}
    for placement in indexed:
        counts[placement.sku_id] = counts.get(placement.sku_id, 0) + 1

    pallet = request["pallet"]
    state = PalletState(
        pallet["length_mm"],
        pallet["width_mm"],
        pallet["max_height_mm"],
        pallet["max_weight_kg"],
    )
    for placement in indexed:
        state.place(placement)

    response = {
        "task_id": request["task_id"],
        "solver_version": solver_version,
        "solve_time_ms": solve_time_ms,
        "placements": [],
        "unplaced": [],
    }
    for placement in indexed:
        response["placements"].append(
            {
                "sku_id": placement.sku_id,
                "instance_index": placement.instance_index,
                "position": {
                    "x_mm": placement.aabb.x_min,
                    "y_mm": placement.aabb.y_min,
                    "z_mm": placement.aabb.z_min,
                },
                "dimensions_placed": {
                    "length_mm": placement.placed_dims[0],
                    "width_mm": placement.placed_dims[1],
                    "height_mm": placement.placed_dims[2],
                },
                "rotation_code": placement.rotation_code,
            }
        )

    for box in request["boxes"]:
        remaining_qty = box["quantity"] - counts.get(box["sku_id"], 0)
        if remaining_qty <= 0:
            continue
        item = RemainingItem(box, placed_count=counts.get(box["sku_id"], 0))
        response["unplaced"].append(
            {
                "sku_id": box["sku_id"],
                "quantity_unplaced": remaining_qty,
                "reason": _classify_unplaced_reason(item, state),
            }
        )
    return response


def _extract_attempt(
    request: Dict[str, Any],
    stage: str,
    response: Dict[str, Any],
    wall_time_ms: int,
    proven_optimal: bool,
    upper_bound: Optional[float] = None,
    notes: str = "",
) -> Optional[ReferenceAttempt]:
    packing = evaluate_packing_quality(request, response)
    if not packing.get("valid", False):
        return None
    lb = packing["packing_score"]
    ub = lb if upper_bound is None else max(lb, round(upper_bound, 4))
    gap = ub - lb
    return ReferenceAttempt(
        stage=stage,
        response=response,
        packing_score_lb=lb,
        packing_score_ub=ub,
        proven_optimal=proven_optimal and gap <= 1e-9,
        eps_optimal=gap <= EPS_OPTIMAL_GAP,
        wall_time_ms=wall_time_ms,
        notes=notes,
    )


def obvious_optimal_attempt(
    request: Dict[str, Any],
    response: Dict[str, Any],
    wall_time_ms: int,
    stage: str,
) -> Optional[ReferenceAttempt]:
    packing = evaluate_packing_quality(request, response)
    if not packing.get("valid", False):
        return None
    metrics = packing["metrics"]
    if metrics["item_coverage"] >= 0.9999 and metrics["fragility_score"] >= 0.9999:
        return ReferenceAttempt(
            stage=stage,
            response=response,
            packing_score_lb=packing["packing_score"],
            packing_score_ub=packing["packing_score"],
            proven_optimal=True,
            eps_optimal=True,
            wall_time_ms=wall_time_ms,
            notes="All requested items fit with perfect fragility, which closes the packing objective.",
        )
    return None


def _solve_rectangle_candidates(
    request: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    time_limit_s: float,
    slot_cap: int,
) -> Optional[List[PlacedBox]]:
    if not candidates or not ortools_available():
        return None

    pallet = request["pallet"]
    pallet_length = pallet["length_mm"]
    pallet_width = pallet["width_mm"]
    pallet_area = pallet_length * pallet_width
    quantities = {box["sku_id"]: box["quantity"] for box in request["boxes"]}
    min_area = min(candidate["width"] * candidate["depth"] for candidate in candidates)
    max_slots = min(slot_cap, max(1, pallet_area // max(min_area, 1)))

    model = cp_model.CpModel()
    x_vars = [model.NewIntVar(0, pallet_length, f"x_{slot}") for slot in range(max_slots)]
    y_vars = [model.NewIntVar(0, pallet_width, f"y_{slot}") for slot in range(max_slots)]

    slot_present: List[Any] = []
    option_vars: Dict[Tuple[int, int], Any] = {}
    x_intervals = []
    y_intervals = []

    for slot in range(max_slots):
        present = model.NewBoolVar(f"present_{slot}")
        slot_present.append(present)
        slot_choices = []
        for candidate_idx, candidate in enumerate(candidates):
            choice = model.NewBoolVar(f"s{slot}_c{candidate_idx}")
            option_vars[(slot, candidate_idx)] = choice
            slot_choices.append(choice)

            x_end = model.NewIntVar(0, pallet_length, f"x_end_{slot}_{candidate_idx}")
            y_end = model.NewIntVar(0, pallet_width, f"y_end_{slot}_{candidate_idx}")
            model.Add(x_end == x_vars[slot] + candidate["width"]).OnlyEnforceIf(choice)
            model.Add(y_end == y_vars[slot] + candidate["depth"]).OnlyEnforceIf(choice)
            model.Add(x_vars[slot] <= pallet_length - candidate["width"]).OnlyEnforceIf(choice)
            model.Add(y_vars[slot] <= pallet_width - candidate["depth"]).OnlyEnforceIf(choice)

            x_intervals.append(
                model.NewOptionalIntervalVar(
                    x_vars[slot],
                    candidate["width"],
                    x_end,
                    choice,
                    f"x_interval_{slot}_{candidate_idx}",
                )
            )
            y_intervals.append(
                model.NewOptionalIntervalVar(
                    y_vars[slot],
                    candidate["depth"],
                    y_end,
                    choice,
                    f"y_interval_{slot}_{candidate_idx}",
                )
            )

        model.Add(sum(slot_choices) <= 1)
        model.Add(sum(slot_choices) == 1).OnlyEnforceIf(present)
        model.Add(sum(slot_choices) == 0).OnlyEnforceIf(present.Not())

    model.AddNoOverlap2D(x_intervals, y_intervals)

    for slot in range(max_slots - 1):
        model.Add(slot_present[slot] >= slot_present[slot + 1])
        model.Add(
            x_vars[slot]
            <= x_vars[slot + 1] + pallet_length * (1 - slot_present[slot + 1])
        )

    for sku_id, quantity in quantities.items():
        model.Add(
            sum(
                option_vars[(slot, candidate_idx)]
                * candidates[candidate_idx]["usage"].get(sku_id, 0)
                for slot in range(max_slots)
                for candidate_idx in range(len(candidates))
            )
            <= quantity
        )

    max_weight_scaled = int(round(pallet["max_weight_kg"] * 1000))
    model.Add(
        sum(
            option_vars[(slot, candidate_idx)]
            * int(round(candidates[candidate_idx]["weight"] * 1000))
            for slot in range(max_slots)
            for candidate_idx in range(len(candidates))
        )
        <= max_weight_scaled
    )

    model.Maximize(
        sum(
            option_vars[(slot, candidate_idx)] * candidates[candidate_idx]["objective"]
            for slot in range(max_slots)
            for candidate_idx in range(len(candidates))
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(time_limit_s, 0.1)
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    placements: List[PlacedBox] = []
    for slot in range(max_slots):
        chosen_candidate_idx = None
        for candidate_idx in range(len(candidates)):
            if solver.Value(option_vars[(slot, candidate_idx)]) > 0:
                chosen_candidate_idx = candidate_idx
                break
        if chosen_candidate_idx is None:
            continue
        candidate = candidates[chosen_candidate_idx]
        x0 = solver.Value(x_vars[slot])
        y0 = solver.Value(y_vars[slot])
        for placement_spec in candidate["placements"]:
            aabb = AABB(
                x0,
                y0,
                placement_spec["z_offset"],
                x0 + placement_spec["dx"],
                y0 + placement_spec["dy"],
                placement_spec["z_offset"] + placement_spec["dz"],
            )
            placements.append(
                PlacedBox(
                    sku_id=placement_spec["sku_id"],
                    instance_index=0,
                    aabb=aabb,
                    weight=placement_spec["weight"],
                    fragile=placement_spec["fragile"],
                    stackable=placement_spec["stackable"],
                    strict_upright=placement_spec["strict_upright"],
                    rotation_code=placement_spec["rotation_code"],
                    placed_dims=(
                        placement_spec["dx"],
                        placement_spec["dy"],
                        placement_spec["dz"],
                    ),
                )
            )
    return _reindex_placements(placements)


def is_single_layer_only(request: Dict[str, Any]) -> bool:
    heights: List[int] = []
    total_items = 0
    for box in request["boxes"]:
        orientations = get_orientations(
            box["length_mm"],
            box["width_mm"],
            box["height_mm"],
            box.get("strict_upright", False),
        )
        if not orientations:
            continue
        min_height = min(orientation[2] for orientation in orientations)
        for _ in range(box["quantity"]):
            heights.append(min_height)
            total_items += 1
    if total_items <= 1:
        return True
    heights.sort()
    return heights[0] + heights[1] > request["pallet"]["max_height_mm"]


def try_single_layer_exact(
    request: Dict[str, Any],
    time_limit_s: float = 15.0,
    solver_version: str = REFERENCE_SOLVER_VERSION,
) -> Optional[ReferenceAttempt]:
    if not ortools_available() or not is_single_layer_only(request):
        return None

    t0 = time.perf_counter()
    pallet = request["pallet"]
    pallet_volume = (
        pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    )
    total_items = sum(box["quantity"] for box in request["boxes"])

    candidates: List[Dict[str, Any]] = []
    for box in request["boxes"]:
        for dx, dy, dz, rotation_code in get_orientations(
            box["length_mm"],
            box["width_mm"],
            box["height_mm"],
            box.get("strict_upright", False),
        ):
            if dz > pallet["max_height_mm"]:
                continue
            candidates.append(
                {
                    "name": f"{box['sku_id']}:{rotation_code}",
                    "width": dx,
                    "depth": dy,
                    "usage": {box["sku_id"]: 1},
                    "weight": box["weight_kg"],
                    "objective": _packing_objective_coeff(
                        volume=dx * dy * dz,
                        item_count=1,
                        fragility_violations=0,
                        pallet_volume=pallet_volume,
                        total_items=total_items,
                    ),
                    "placements": [
                        {
                            "sku_id": box["sku_id"],
                            "dx": dx,
                            "dy": dy,
                            "dz": dz,
                            "z_offset": 0,
                            "rotation_code": rotation_code,
                            "weight": box["weight_kg"],
                            "fragile": box.get("fragile", False),
                            "stackable": box.get("stackable", True),
                            "strict_upright": box.get("strict_upright", False),
                        }
                    ],
                }
            )

    placements = _solve_rectangle_candidates(
        request,
        candidates,
        time_limit_s=time_limit_s,
        slot_cap=max(sum(box["quantity"] for box in request["boxes"]), 1),
    )
    if placements is None:
        return None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    response = placements_to_response(request, placements, elapsed_ms, solver_version)
    return _extract_attempt(
        request,
        stage="single_layer_exact",
        response=response,
        wall_time_ms=elapsed_ms,
        proven_optimal=True,
    )


def try_layer_pattern_exact(
    request: Dict[str, Any],
    time_limit_s: float = 20.0,
    solver_version: str = REFERENCE_SOLVER_VERSION,
) -> Optional[ReferenceAttempt]:
    if not ortools_available():
        return None

    pallet = request["pallet"]
    t0 = time.perf_counter()
    pallet_volume = (
        pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    )
    total_items = sum(box["quantity"] for box in request["boxes"])
    candidates: List[Dict[str, Any]] = []

    for box in request["boxes"]:
        for dx, dy, dz, rotation_code in get_orientations(
            box["length_mm"],
            box["width_mm"],
            box["height_mm"],
            box.get("strict_upright", False),
        ):
            max_levels = pallet["max_height_mm"] // dz
            if max_levels <= 0:
                continue
            stack_cap = min(max_levels, box["quantity"])
            if not box.get("stackable", True):
                stack_cap = 1
            for stack_count in range(1, stack_cap + 1):
                fragility_violations = 0
                if box.get("fragile", False) and box["weight_kg"] > FRAGILE_WEIGHT_THRESHOLD:
                    fragility_violations = stack_count - 1
                placements = []
                for level in range(stack_count):
                    placements.append(
                        {
                            "sku_id": box["sku_id"],
                            "dx": dx,
                            "dy": dy,
                            "dz": dz,
                            "z_offset": level * dz,
                            "rotation_code": rotation_code,
                            "weight": box["weight_kg"],
                            "fragile": box.get("fragile", False),
                            "stackable": box.get("stackable", True),
                            "strict_upright": box.get("strict_upright", False),
                        }
                    )
                candidates.append(
                    {
                        "name": f"{box['sku_id']}:{rotation_code}:x{stack_count}",
                        "width": dx,
                        "depth": dy,
                        "usage": {box["sku_id"]: stack_count},
                        "weight": box["weight_kg"] * stack_count,
                        "objective": _packing_objective_coeff(
                            volume=dx * dy * dz * stack_count,
                            item_count=stack_count,
                            fragility_violations=fragility_violations,
                            pallet_volume=pallet_volume,
                            total_items=total_items,
                        ),
                        "placements": placements,
                    }
                )

    if not candidates:
        return None

    placements = _solve_rectangle_candidates(
        request,
        candidates,
        time_limit_s=time_limit_s,
        slot_cap=max(
            1,
            pallet["length_mm"] * pallet["width_mm"] // min(
                candidate["width"] * candidate["depth"] for candidate in candidates
            ),
        ),
    )
    if placements is None:
        return None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    response = placements_to_response(request, placements, elapsed_ms, solver_version)
    return _extract_attempt(
        request,
        stage="layer_pattern_exact",
        response=response,
        wall_time_ms=elapsed_ms,
        proven_optimal=False,
        upper_bound=coarse_upper_bound(request),
        notes="Exact within the repeated-column pattern model; global upper bound is still coarse.",
    )


def _response_to_placements(
    request: Dict[str, Any],
    response: Dict[str, Any],
) -> List[PlacedBox]:
    by_sku = {box["sku_id"]: box for box in request["boxes"]}
    placements: List[PlacedBox] = []
    for placement in response.get("placements", []):
        meta = by_sku[placement["sku_id"]]
        dims = placement["dimensions_placed"]
        pos = placement["position"]
        placements.append(
            PlacedBox(
                sku_id=placement["sku_id"],
                instance_index=placement["instance_index"],
                aabb=AABB(
                    pos["x_mm"],
                    pos["y_mm"],
                    pos["z_mm"],
                    pos["x_mm"] + dims["length_mm"],
                    pos["y_mm"] + dims["width_mm"],
                    pos["z_mm"] + dims["height_mm"],
                ),
                weight=meta["weight_kg"],
                fragile=meta.get("fragile", False),
                stackable=meta.get("stackable", True),
                strict_upright=meta.get("strict_upright", False),
                rotation_code=placement["rotation_code"],
                placed_dims=(
                    dims["length_mm"],
                    dims["width_mm"],
                    dims["height_mm"],
                ),
            )
        )
    return _reindex_placements(placements)


def _clone_remaining(remaining: Sequence[RemainingItem]) -> List[RemainingItem]:
    clone: List[RemainingItem] = []
    for item in remaining:
        new_item = RemainingItem.__new__(RemainingItem)
        new_item.sku_id = item.sku_id
        new_item.length = item.length
        new_item.width = item.width
        new_item.height = item.height
        new_item.weight = item.weight
        new_item.strict_upright = item.strict_upright
        new_item.fragile = item.fragile
        new_item.stackable = item.stackable
        new_item.remaining_qty = item.remaining_qty
        clone.append(new_item)
    return clone


def _decrement_remaining(remaining: List[RemainingItem], sku_id: str) -> None:
    for item in remaining:
        if item.sku_id == sku_id:
            item.remaining_qty = max(0, item.remaining_qty - 1)
            return


def _zero_out_item(remaining: List[RemainingItem], sku_id: str) -> None:
    for item in remaining:
        if item.sku_id == sku_id:
            item.remaining_qty = 0
            return


def _choose_next_remaining_item(remaining: Sequence[RemainingItem]) -> Optional[RemainingItem]:
    active = [item for item in remaining if item.remaining_qty > 0]
    if not active:
        return None
    return min(
        active,
        key=lambda item: (
            0 if item.strict_upright else 1,
            0 if not item.stackable else 1,
            item.remaining_qty,
            -(item.length * item.width * item.height),
        ),
    )


def _generate_candidates_for_item(
    item: RemainingItem,
    state: PalletState,
    ep_manager: ExtremePointManager,
    checker: FeasibilityChecker,
) -> List[PlacedBox]:
    points = list(ep_manager.get_points())
    points.sort(key=lambda point: (point[2], point[0] + point[1], point[0], point[1]))
    instance_index = state.next_instance_index(item.sku_id)
    candidates: List[PlacedBox] = []
    seen = set()

    for x0, y0, z0 in points:
        for dx, dy, dz, rotation_code in get_orientations(
            item.length,
            item.width,
            item.height,
            item.strict_upright,
        ):
            z = max(z0, state.get_max_z_at(x0, y0, dx, dy))
            aabb = AABB(x0, y0, z, x0 + dx, y0 + dy, z + dz)
            ok, _ = checker.is_feasible(
                aabb,
                item.weight,
                item.height,
                item.strict_upright,
                True,
                state,
            )
            if not ok:
                continue
            key = (aabb.x_min, aabb.y_min, aabb.z_min, dx, dy, dz)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                PlacedBox(
                    sku_id=item.sku_id,
                    instance_index=instance_index,
                    aabb=aabb,
                    weight=item.weight,
                    fragile=item.fragile,
                    stackable=item.stackable,
                    strict_upright=item.strict_upright,
                    rotation_code=rotation_code,
                    placed_dims=(dx, dy, dz),
                )
            )
    candidates.sort(
        key=lambda placement: (
            placement.aabb.z_min,
            -placement.aabb.base_area(),
            placement.aabb.x_min + placement.aabb.y_min,
            placement.aabb.x_min,
            placement.aabb.y_min,
        )
    )
    return candidates


def _small_search_state_key(
    state: PalletState,
    remaining: Sequence[RemainingItem],
) -> Tuple[Any, ...]:
    return (
        tuple(item.remaining_qty for item in remaining),
        int(round(state.total_weight * 100)),
        state.placed_volume,
        state.max_z,
        tuple(
            sorted(
                (
                    placement.aabb.x_min,
                    placement.aabb.y_min,
                    placement.aabb.z_min,
                    placement.sku_id,
                )
                for placement in state.placed
            )
        ),
    )


def _small_search_upper_bound(
    request: Dict[str, Any],
    state: PalletState,
    remaining: Sequence[RemainingItem],
    placements: Sequence[PlacedBox],
) -> float:
    pallet = request["pallet"]
    pallet_volume = (
        pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    )
    total_items = sum(box["quantity"] for box in request["boxes"])
    current_score = _packing_score_from_placements(request, placements)

    free_volume = max(0, pallet_volume - state.placed_volume)
    add_volume = min(
        free_volume,
        sum(
            item.length * item.width * item.height * item.remaining_qty
            for item in remaining
        ),
    )
    volume_term = 0.50 * add_volume / max(pallet_volume, 1)

    remaining_instances = []
    for item in remaining:
        remaining_instances.extend(
            [(item.weight, item.length * item.width * item.height)] * item.remaining_qty
        )
    if not remaining_instances:
        return current_score

    min_weight = min(weight for weight, _ in remaining_instances)
    min_volume = min(volume for _, volume in remaining_instances)
    extra_count = min(
        len(remaining_instances),
        int((state.max_weight - state.total_weight + 1e-9) // max(min_weight, 1e-9)),
        int(free_volume // max(min_volume, 1)),
    )
    count_term = 0.30 * extra_count / max(total_items, 1)
    return round(min(0.90, current_score + volume_term + count_term), 4)


def try_small_exact_search(
    request: Dict[str, Any],
    incumbent_response: Optional[Dict[str, Any]] = None,
    time_limit_s: float = 20.0,
    solver_version: str = REFERENCE_SOLVER_VERSION,
) -> Optional[ReferenceAttempt]:
    total_items = sum(box["quantity"] for box in request["boxes"])
    if total_items > 24:
        return None

    pallet = request["pallet"]
    checker = FeasibilityChecker(
        pallet["length_mm"],
        pallet["width_mm"],
        pallet["max_height_mm"],
        pallet["max_weight_kg"],
    )

    placements_best: List[PlacedBox] = []
    best_score = 0.0
    if incumbent_response is not None:
        packing = evaluate_packing_quality(request, incumbent_response)
        if packing.get("valid", False):
            best_score = packing["packing_score"]
            placements_best = _response_to_placements(request, incumbent_response)

    t0 = time.perf_counter()
    deadline = t0 + max(time_limit_s, 0.1)
    initial_state = PalletState(
        pallet["length_mm"],
        pallet["width_mm"],
        pallet["max_height_mm"],
        pallet["max_weight_kg"],
    )
    initial_ep = ExtremePointManager(pallet["length_mm"], pallet["width_mm"])
    initial_remaining = [RemainingItem(box) for box in request["boxes"]]
    memo: Dict[Tuple[Any, ...], float] = {}

    def search(
        state: PalletState,
        ep_manager: ExtremePointManager,
        remaining: List[RemainingItem],
        placements: List[PlacedBox],
    ) -> None:
        nonlocal best_score, placements_best
        if time.perf_counter() >= deadline:
            return

        current_score = _packing_score_from_placements(request, placements)
        if current_score > best_score + 1e-9:
            best_score = current_score
            placements_best = list(placements)

        ub = _small_search_upper_bound(request, state, remaining, placements)
        if ub <= best_score + 1e-9:
            return

        key = _small_search_state_key(state, remaining)
        if memo.get(key, -1.0) >= current_score - 1e-9:
            return
        memo[key] = current_score

        target = _choose_next_remaining_item(remaining)
        if target is None:
            return

        candidates = _generate_candidates_for_item(target, state, ep_manager, checker)
        if not candidates:
            skipped = _clone_remaining(remaining)
            _zero_out_item(skipped, target.sku_id)
            search(state, ep_manager, skipped, placements)
            return

        for placement in candidates:
            new_state = state.copy()
            new_ep = ep_manager.copy()
            new_remaining = _clone_remaining(remaining)
            new_state.place(placement)
            new_ep.update_after_placement(placement.aabb, new_state)
            _decrement_remaining(new_remaining, placement.sku_id)
            search(new_state, new_ep, new_remaining, placements + [placement])

        skipped = _clone_remaining(remaining)
        _zero_out_item(skipped, target.sku_id)
        search(state, ep_manager, skipped, placements)

    search(initial_state, initial_ep, initial_remaining, [])
    if not placements_best:
        return None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    exhausted = time.perf_counter() < deadline - 1e-6
    response = placements_to_response(request, placements_best, elapsed_ms, solver_version)
    return _extract_attempt(
        request,
        stage="general_exact_search",
        response=response,
        wall_time_ms=elapsed_ms,
        proven_optimal=exhausted,
        upper_bound=best_score if exhausted else _small_search_upper_bound(
            request,
            initial_state,
            initial_remaining,
            placements_best,
        ),
        notes="Exhaustive support-point search for compact requests." if exhausted else "Search hit its soft deadline; returned the best bound available.",
    )
