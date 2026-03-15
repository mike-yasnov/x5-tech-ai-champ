"""Integration tests: generate scenarios, solve, validate."""

import json
import os
import sys

import pytest

# Ensure project root is on path for generator/validator imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_scenario
from solver.heuristics import STRATEGY_CONFIGS
from solver.models import Box, Pallet, load_request, solution_to_dict
from solver.packer import pack_greedy
from solver.solver import solve
from validator import evaluate_solution


SCENARIOS = [
    "heavy_water",
    "fragile_tower",
    "liquid_tetris",
    "random_mixed",
    "exact_fit",
    "fragile_mix",
    "support_tetris",
    "cavity_fill",
]


def _make_request_dict(scenario_type: str, seed: int = 42) -> dict:
    """Generate a scenario and return the raw dict."""
    return generate_scenario(f"test_{scenario_type}", scenario_type, seed=seed)


def _request_to_models(request_dict: dict):
    """Convert request dict to (task_id, Pallet, list[Box])."""
    pallet_data = request_dict["pallet"]
    pallet = Pallet(
        type_id=pallet_data.get("type_id", "unknown"),
        length_mm=pallet_data["length_mm"],
        width_mm=pallet_data["width_mm"],
        max_height_mm=pallet_data["max_height_mm"],
        max_weight_kg=pallet_data["max_weight_kg"],
    )
    boxes = []
    for box_data in request_dict["boxes"]:
        boxes.append(
            Box(
                sku_id=box_data["sku_id"],
                description=box_data.get("description", ""),
                length_mm=box_data["length_mm"],
                width_mm=box_data["width_mm"],
                height_mm=box_data["height_mm"],
                weight_kg=box_data["weight_kg"],
                quantity=box_data["quantity"],
                strict_upright=box_data.get("strict_upright", False),
                fragile=box_data.get("fragile", False),
                stackable=box_data.get("stackable", True),
            )
        )
    return request_dict["task_id"], pallet, boxes


@pytest.mark.parametrize("scenario_type", SCENARIOS)
def test_scenario_produces_valid_solution(scenario_type):
    """Solver should produce a valid solution for each scenario."""
    request_dict = _make_request_dict(scenario_type)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        n_restarts=5,
        time_budget_ms=5000,
    )

    response_dict = solution_to_dict(solution)
    result = evaluate_solution(request_dict, response_dict)

    assert result["valid"] is True, (
        f"Invalid solution for {scenario_type}: {result.get('error')}"
    )


@pytest.mark.parametrize("scenario_type", SCENARIOS)
def test_scenario_score_positive(scenario_type):
    """Each scenario should produce a positive score."""
    request_dict = _make_request_dict(scenario_type)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        n_restarts=5,
        time_budget_ms=5000,
    )

    response_dict = solution_to_dict(solution)
    result = evaluate_solution(request_dict, response_dict)

    assert result["valid"]
    assert result["final_score"] > 0, f"Score should be positive for {scenario_type}"


@pytest.mark.parametrize("scenario_type", SCENARIOS)
def test_scenario_solve_time(scenario_type):
    """Solve time should be under 5 seconds."""
    request_dict = _make_request_dict(scenario_type)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        n_restarts=5,
        time_budget_ms=5000,
    )

    assert solution.solve_time_ms < 5000, (
        f"Solve time {solution.solve_time_ms}ms exceeds 5s for {scenario_type}"
    )


def test_solution_format_complete():
    """Output format should match expected schema."""
    request_dict = _make_request_dict("random_mixed")
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
    )

    data = solution_to_dict(solution)
    assert "task_id" in data
    assert "solver_version" in data
    assert "solve_time_ms" in data
    assert "placements" in data
    assert "unplaced" in data

    if data["placements"]:
        placement = data["placements"][0]
        assert "sku_id" in placement
        assert "instance_index" in placement
        assert "position" in placement
        assert "x_mm" in placement["position"]
        assert "y_mm" in placement["position"]
        assert "z_mm" in placement["position"]
        assert "dimensions_placed" in placement
        assert "rotation_code" in placement


def test_exact_fit_scenario_is_near_perfect():
    """Exact-fit scenario should have no free space and a near-perfect score."""
    request_dict = _make_request_dict("exact_fit")
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        n_restarts=5,
        time_budget_ms=5000,
    )

    result = evaluate_solution(request_dict, solution_to_dict(solution))

    assert result["valid"] is True
    assert len(solution.unplaced) == 0
    assert result["metrics"]["volume_utilization"] == pytest.approx(1.0)
    assert result["metrics"]["item_coverage"] == pytest.approx(1.0)
    assert result["final_score"] >= 0.99


@pytest.mark.parametrize(
    "scenario_type,min_score",
    [
        ("fragile_mix", 0.90),
        ("support_tetris", 0.85),
    ],
)
def test_diagnostic_scenarios_score_well(scenario_type, min_score):
    request_dict = _make_request_dict(scenario_type)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        n_restarts=8,
        time_budget_ms=5000,
    )

    result = evaluate_solution(request_dict, solution_to_dict(solution))

    assert result["valid"] is True
    assert result["final_score"] >= min_score


@pytest.mark.parametrize(
    "scenario_type,min_spread",
    [
        ("fragile_mix", 0.015),
        ("cavity_fill", 0.05),
    ],
)
def test_diagnostic_scenarios_separate_strategies(scenario_type, min_spread):
    request_dict = _make_request_dict(scenario_type)
    task_id, pallet, boxes = _request_to_models(request_dict)

    scores = []
    for strategy in STRATEGY_CONFIGS:
        solution = pack_greedy(
            task_id=task_id,
            pallet=pallet,
            boxes=boxes,
            sort_key_name=strategy.sort_key_name,
            placement_policy=strategy.placement_policy,
            randomized=strategy.randomized,
            noise_factor=strategy.noise_factor,
        )
        result = evaluate_solution(request_dict, solution_to_dict(solution))
        assert result["valid"] is True
        scores.append(result["final_score"])

    assert max(scores) - min(scores) >= min_spread
