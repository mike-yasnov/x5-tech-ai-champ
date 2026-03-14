"""Unit tests for validator hard constraints."""

from validator import evaluate_solution


def _base_request():
    return {
        "task_id": "t",
        "pallet": {
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 1800,
            "max_weight_kg": 1000.0,
        },
        "boxes": [
            {
                "sku_id": "BASE",
                "description": "Base box",
                "length_mm": 400,
                "width_mm": 400,
                "height_mm": 200,
                "weight_kg": 10.0,
                "quantity": 1,
                "strict_upright": False,
                "fragile": False,
                "stackable": False,
            },
            {
                "sku_id": "TOP",
                "description": "Top box",
                "length_mm": 200,
                "width_mm": 200,
                "height_mm": 200,
                "weight_kg": 5.0,
                "quantity": 1,
                "strict_upright": False,
                "fragile": False,
                "stackable": True,
            },
        ],
    }


def test_non_stackable_box_rejects_top_placement():
    request = _base_request()
    response = {
        "task_id": "t",
        "solver_version": "test",
        "solve_time_ms": 100,
        "placements": [
            {
                "sku_id": "BASE",
                "instance_index": 0,
                "position": {"x_mm": 0, "y_mm": 0, "z_mm": 0},
                "dimensions_placed": {"length_mm": 400, "width_mm": 400, "height_mm": 200},
                "rotation_code": "LWH",
            },
            {
                "sku_id": "TOP",
                "instance_index": 0,
                "position": {"x_mm": 100, "y_mm": 100, "z_mm": 200},
                "dimensions_placed": {"length_mm": 200, "width_mm": 200, "height_mm": 200},
                "rotation_code": "LWH",
            },
        ],
        "unplaced": [],
    }

    result = evaluate_solution(request, response)

    assert result["valid"] is False
    assert "non-stackable" in result["error"]


def test_non_stackable_not_triggered_without_top_overlap():
    request = _base_request()
    response = {
        "task_id": "t",
        "solver_version": "test",
        "solve_time_ms": 100,
        "placements": [
            {
                "sku_id": "BASE",
                "instance_index": 0,
                "position": {"x_mm": 0, "y_mm": 0, "z_mm": 0},
                "dimensions_placed": {"length_mm": 400, "width_mm": 400, "height_mm": 200},
                "rotation_code": "LWH",
            },
            {
                "sku_id": "TOP",
                "instance_index": 0,
                "position": {"x_mm": 500, "y_mm": 0, "z_mm": 0},
                "dimensions_placed": {"length_mm": 200, "width_mm": 200, "height_mm": 200},
                "rotation_code": "LWH",
            },
        ],
        "unplaced": [],
    }

    result = evaluate_solution(request, response)

    assert result["valid"] is True


def test_fragile_heavy_top_does_not_count_as_fragility_violation():
    request = {
        "task_id": "fragile-rule",
        "pallet": {
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 1800,
            "max_weight_kg": 1000.0,
        },
        "boxes": [
            {
                "sku_id": "BOTTOM",
                "description": "Fragile base",
                "length_mm": 400,
                "width_mm": 400,
                "height_mm": 200,
                "weight_kg": 1.0,
                "quantity": 1,
                "strict_upright": False,
                "fragile": True,
                "stackable": True,
            },
            {
                "sku_id": "TOP_FRAGILE",
                "description": "Heavy but fragile top",
                "length_mm": 400,
                "width_mm": 400,
                "height_mm": 200,
                "weight_kg": 5.0,
                "quantity": 1,
                "strict_upright": False,
                "fragile": True,
                "stackable": True,
            },
            {
                "sku_id": "TOP_STURDY",
                "description": "Heavy sturdy top",
                "length_mm": 400,
                "width_mm": 400,
                "height_mm": 200,
                "weight_kg": 5.0,
                "quantity": 1,
                "strict_upright": False,
                "fragile": False,
                "stackable": True,
            },
        ],
    }

    fragile_top_response = {
        "task_id": "fragile-rule",
        "solver_version": "test",
        "solve_time_ms": 100,
        "placements": [
            {
                "sku_id": "BOTTOM",
                "instance_index": 0,
                "position": {"x_mm": 0, "y_mm": 0, "z_mm": 0},
                "dimensions_placed": {"length_mm": 400, "width_mm": 400, "height_mm": 200},
                "rotation_code": "LWH",
            },
            {
                "sku_id": "TOP_FRAGILE",
                "instance_index": 0,
                "position": {"x_mm": 0, "y_mm": 0, "z_mm": 200},
                "dimensions_placed": {"length_mm": 400, "width_mm": 400, "height_mm": 200},
                "rotation_code": "LWH",
            },
        ],
        "unplaced": [],
    }
    sturdy_top_response = {
        "task_id": "fragile-rule",
        "solver_version": "test",
        "solve_time_ms": 100,
        "placements": [
            {
                "sku_id": "BOTTOM",
                "instance_index": 0,
                "position": {"x_mm": 0, "y_mm": 0, "z_mm": 0},
                "dimensions_placed": {"length_mm": 400, "width_mm": 400, "height_mm": 200},
                "rotation_code": "LWH",
            },
            {
                "sku_id": "TOP_STURDY",
                "instance_index": 0,
                "position": {"x_mm": 0, "y_mm": 0, "z_mm": 200},
                "dimensions_placed": {"length_mm": 400, "width_mm": 400, "height_mm": 200},
                "rotation_code": "LWH",
            },
        ],
        "unplaced": [],
    }

    fragile_result = evaluate_solution(request, fragile_top_response)
    sturdy_result = evaluate_solution(request, sturdy_top_response)

    assert fragile_result["valid"] is True
    assert sturdy_result["valid"] is True
    assert fragile_result["metrics"]["fragility_score"] == 1.0
    assert sturdy_result["metrics"]["fragility_score"] == 0.95
