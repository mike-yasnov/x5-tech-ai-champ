"""Tests for the simulated annealing solver path."""

from generator import generate_scenario
from solver.models import Box, Pallet, solution_to_dict
from solver.solver import solve
from validator import evaluate_solution


def _request_to_models(request_dict: dict):
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


def test_annealing_solver_returns_valid_solution():
    request_dict = generate_scenario("annealing_fragile_mix", "fragile_mix", seed=47)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        time_budget_ms=2000,
        algorithm="annealing",
    )

    result = evaluate_solution(request_dict, solution_to_dict(solution))
    assert result["valid"] is True


def test_annealing_solver_solves_exact_fit_without_leaving_gaps():
    request_dict = generate_scenario("annealing_exact_fit", "exact_fit", seed=46)
    task_id, pallet, boxes = _request_to_models(request_dict)

    annealing_solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        time_budget_ms=1500,
        algorithm="annealing",
    )

    annealing_result = evaluate_solution(
        request_dict, solution_to_dict(annealing_solution)
    )

    assert annealing_result["valid"] is True
    assert len(annealing_solution.unplaced) == 0
    assert annealing_result["metrics"]["volume_utilization"] == 1.0
    assert annealing_result["metrics"]["item_coverage"] == 1.0
