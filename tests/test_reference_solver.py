"""Regression tests for the offline exact reference path and certificate flow."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_scenario
from scenario_catalog import BENCHMARK_SCENARIOS
from solver.exact_reference import certify_request
from solver.models import Box, Pallet, solution_to_dict
from solver.solver import solve
from validator import evaluate_packing_quality, evaluate_solution


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


SEED_BY_SCENARIO = dict(BENCHMARK_SCENARIOS)


@pytest.mark.parametrize(
    "scenario_type,expected",
    [
        ("exact_fit", {"volume_utilization": 1.0, "item_coverage": 1.0, "fragility_score": 1.0}),
        ("count_preference", {"volume_utilization": 1.0, "item_coverage": 0.6667, "fragility_score": 1.0}),
        ("cavity_fill", {"volume_utilization": 0.6667, "item_coverage": 0.8333, "fragility_score": 1.0}),
        ("fragile_mix", {"volume_utilization": 1.0, "item_coverage": 0.9545, "fragility_score": 1.0}),
        ("support_tetris", {"volume_utilization": 1.0, "item_coverage": 0.8571, "fragility_score": 1.0}),
        ("small_gap_fill", {"volume_utilization": 0.5833, "item_coverage": 1.0, "fragility_score": 1.0}),
        ("non_stackable_caps", {"volume_utilization": 0.75, "item_coverage": 1.0, "fragility_score": 1.0}),
        ("mixed_column_repeat", {"volume_utilization": 0.8019, "item_coverage": 1.0, "fragility_score": 1.0}),
    ],
)
def test_exact_reference_structured_targets(scenario_type, expected):
    request_dict = generate_scenario(
        f"test_{scenario_type}",
        scenario_type,
        seed=SEED_BY_SCENARIO[scenario_type],
    )
    certificate = certify_request(request_dict, time_budget_ms=5000)
    assert certificate["eps_optimal"] is True

    packing = evaluate_packing_quality(request_dict, certificate["response"])
    assert packing["valid"] is True
    for metric_name, expected_value in expected.items():
        assert packing["metrics"][metric_name] == pytest.approx(expected_value, abs=1e-4)


@pytest.mark.parametrize("scenario_type", ["heavy_water", "liquid_tetris"])
def test_exact_reference_certificate_format_and_bounds(scenario_type):
    request_dict = generate_scenario(
        f"test_{scenario_type}",
        scenario_type,
        seed=SEED_BY_SCENARIO[scenario_type],
    )
    certificate = certify_request(request_dict, time_budget_ms=5000)

    assert certificate["task_id"] == request_dict["task_id"]
    assert certificate["strategy"] == "exact_reference"
    assert "request_hash" in certificate
    assert "response" in certificate
    assert certificate["packing_score_ub"] >= certificate["packing_score_lb"]
    assert certificate["gap"] == pytest.approx(
        round(certificate["packing_score_ub"] - certificate["packing_score_lb"], 4),
        abs=1e-4,
    )


@pytest.mark.parametrize("scenario_type", ["liquid_tetris", "cavity_fill"])
def test_certificate_generation_is_reproducible(scenario_type):
    request_dict = generate_scenario(
        f"test_{scenario_type}",
        scenario_type,
        seed=SEED_BY_SCENARIO[scenario_type],
    )
    certificate_a = certify_request(request_dict, time_budget_ms=5000)
    certificate_b = certify_request(request_dict, time_budget_ms=5000)

    assert certificate_a["request_hash"] == certificate_b["request_hash"]
    assert certificate_a["packing_score_lb"] == certificate_b["packing_score_lb"]
    assert certificate_a["packing_score_ub"] == certificate_b["packing_score_ub"]
    assert certificate_a["gap"] == certificate_b["gap"]


@pytest.mark.parametrize("scenario_type", ["cavity_fill", "fragile_mix", "support_tetris"])
def test_fast_solver_gap_against_reference_is_non_negative(scenario_type):
    request_dict = generate_scenario(
        f"test_{scenario_type}",
        scenario_type,
        seed=SEED_BY_SCENARIO[scenario_type],
    )
    certificate = certify_request(request_dict, time_budget_ms=5000)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        strategy="portfolio_block",
        time_budget_ms=5000,
    )
    packing = evaluate_packing_quality(request_dict, solution_to_dict(solution))
    assert packing["valid"] is True
    assert certificate["packing_score_lb"] + 1e-9 >= packing["packing_score"]


@pytest.mark.parametrize("scenario_type", ["heavy_water", "fragile_mix"])
def test_exact_reference_uses_neutral_time_scoring(scenario_type):
    request_dict = generate_scenario(
        f"test_{scenario_type}",
        scenario_type,
        seed=SEED_BY_SCENARIO[scenario_type],
    )
    certificate = certify_request(request_dict, time_budget_ms=5000)

    assert certificate["wall_time_ms"] >= 0
    assert certificate["response"]["wall_time_ms"] == certificate["wall_time_ms"]
    assert certificate["response"]["solve_time_ms"] == 1

    evaluated = evaluate_solution(request_dict, certificate["response"])
    assert evaluated["valid"] is True
    assert evaluated["metrics"]["time_score"] == pytest.approx(1.0, abs=1e-9)
    assert evaluated["final_score"] == pytest.approx(
        certificate["packing_score_lb"] + 0.10,
        abs=1e-4,
    )
