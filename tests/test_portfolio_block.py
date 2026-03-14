"""Tests for the portfolio-block solver and strategy dispatch."""

from __future__ import annotations

from generator import generate_scenario
from solver.hybrid.candidate_gen import RemainingItem
from solver.hybrid.feasibility import FeasibilityChecker
from solver.hybrid.geometry import AABB
from solver.hybrid.pallet_state import PalletState, PlacedBox
from solver.models import Box, Pallet, solution_to_dict
from solver.portfolio_block import (
    _BlockSpec,
    _fragile_staged_instances,
    _materialize_block_candidate,
)
from solver.solver import solve
from validator import evaluate_solution


def _request_to_models(request_dict: dict):
    p = request_dict["pallet"]
    pallet = Pallet(
        type_id=p.get("type_id", "unknown"),
        length_mm=p["length_mm"],
        width_mm=p["width_mm"],
        max_height_mm=p["max_height_mm"],
        max_weight_kg=p["max_weight_kg"],
    )
    boxes = []
    for b in request_dict["boxes"]:
        boxes.append(
            Box(
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
        )
    return request_dict["task_id"], pallet, boxes


def test_block_expansion_preserves_quantity_geometry_rotation():
    state = PalletState(1200, 800, 1800, 1000.0)
    checker = FeasibilityChecker(1200, 800, 1800, 1000.0)
    item = RemainingItem(
        {
            "sku_id": "SKU-A",
            "length_mm": 200,
            "width_mm": 100,
            "height_mm": 150,
            "weight_kg": 2.5,
            "quantity": 10,
            "strict_upright": False,
            "fragile": False,
            "stackable": True,
        }
    )
    spec = _BlockSpec(
        sku_id="SKU-A",
        placed_dims=(200, 100, 150),
        rotation_code="LWH",
        nx=2,
        ny=1,
        nz=2,
        item_count=4,
        total_weight=10.0,
        fragile=False,
        stackable=True,
        strict_upright=False,
        x=0,
        y=0,
        z=0,
        heuristic_score=1.0,
    )

    candidate = _materialize_block_candidate(spec, item, state, checker)
    assert candidate is not None
    assert len(candidate.units) == 4
    assert all(unit.rotation_code == "LWH" for unit in candidate.units)
    assert all(unit.placed_dims == (200, 100, 150) for unit in candidate.units)
    assert (
        candidate.aabb.x_min,
        candidate.aabb.y_min,
        candidate.aabb.z_min,
        candidate.aabb.x_max,
        candidate.aabb.y_max,
        candidate.aabb.z_max,
    ) == (0, 0, 0, 400, 100, 300)


def test_block_feasibility_respects_stackable_false():
    state = PalletState(1200, 800, 1800, 1000.0)
    checker = FeasibilityChecker(1200, 800, 1800, 1000.0)
    base = PlacedBox(
        sku_id="BASE",
        instance_index=0,
        aabb=AABB(0, 0, 0, 400, 300, 200),
        weight=10.0,
        fragile=False,
        stackable=False,
        strict_upright=False,
        rotation_code="LWH",
        placed_dims=(400, 300, 200),
    )
    state.place(base)

    item = RemainingItem(
        {
            "sku_id": "TOP",
            "length_mm": 400,
            "width_mm": 300,
            "height_mm": 200,
            "weight_kg": 5.0,
            "quantity": 2,
            "strict_upright": False,
            "fragile": False,
            "stackable": True,
        }
    )
    spec = _BlockSpec(
        sku_id="TOP",
        placed_dims=(400, 300, 200),
        rotation_code="LWH",
        nx=1,
        ny=1,
        nz=1,
        item_count=1,
        total_weight=5.0,
        fragile=False,
        stackable=True,
        strict_upright=False,
        x=0,
        y=0,
        z=200,
        heuristic_score=1.0,
    )

    candidate = _materialize_block_candidate(spec, item, state, checker)
    assert candidate is None


def test_default_solve_dispatches_to_portfolio(monkeypatch):
    from solver import solver as solver_module

    called = {"portfolio": False, "hybrid": False}

    def fake_portfolio_request(request, model_dir="models", time_budget_ms=900):
        called["portfolio"] = True
        return {
            "task_id": request["task_id"],
            "solver_version": "portfolio-block-v0.1",
            "solve_time_ms": 1,
            "placements": [],
            "unplaced": [],
        }

    def fake_hybrid_request(**kwargs):
        called["hybrid"] = True
        raise AssertionError("legacy_hybrid should not be called by default")

    monkeypatch.setattr(solver_module, "solve_portfolio_request", fake_portfolio_request)
    monkeypatch.setattr(solver_module, "solve_hybrid_request", fake_hybrid_request)

    pallet = Pallet("EUR", 1200, 800, 1800, 1000.0)
    boxes = [
        Box("SKU", "", 200, 200, 200, 1.0, 1),
    ]
    solve("task", pallet, boxes, request_dict=None)
    assert called["portfolio"] is True
    assert called["hybrid"] is False


def test_portfolio_solver_falls_back_without_model_and_stays_valid(tmp_path):
    request_dict = generate_scenario("test_exact_fit", "exact_fit", seed=46)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        strategy="portfolio_block",
        model_dir=str(tmp_path / "missing-model"),
        time_budget_ms=900,
    )
    result = evaluate_solution(request_dict, solution_to_dict(solution))
    assert result["valid"] is True


def test_legacy_greedy_strategy_regression_stays_valid():
    request_dict = generate_scenario("test_legacy_random", "random_mixed", seed=45)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        strategy="legacy_greedy",
        time_budget_ms=900,
    )

    result = evaluate_solution(request_dict, solution_to_dict(solution))
    assert result["valid"] is True
    assert solution.solve_time_ms < 1000


def test_fragile_staged_instances_prefers_sturdy_then_anchors_then_light_fragile():
    pallet = Pallet("EUR", 1200, 800, 1800, 1000.0)
    boxes = [
        Box("STURDY", "", 600, 400, 200, 10.0, 6, fragile=False),
        Box("FRAG_HEAVY", "", 300, 400, 300, 3.5, 4, fragile=True),
        Box("FRAG_LIGHT", "", 200, 400, 300, 1.0, 8, fragile=True),
    ]

    staged = _fragile_staged_instances(boxes, pallet)
    staged_skus = [box.sku_id for box, _ in staged]

    assert staged_skus[:4] == ["STURDY"] * 4
    assert staged_skus[4:6] == ["FRAG_HEAVY"] * 2
    assert staged_skus[6] == "FRAG_LIGHT"
