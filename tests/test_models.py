"""Tests for solver.models."""

import json
import os
import tempfile

import pytest

from alternative_solver.models import Box, Pallet, Placement, Solution, UnplacedItem, load_request, solution_to_dict


@pytest.fixture
def sample_request_path():
    """Create a temporary request JSON file."""
    data = {
        "task_id": "test_001",
        "pallet": {
            "type_id": "EUR_1200x800",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 1800,
            "max_weight_kg": 1000.0,
        },
        "boxes": [
            {
                "sku_id": "SKU-A",
                "description": "Test Box A",
                "length_mm": 400,
                "width_mm": 300,
                "height_mm": 200,
                "weight_kg": 5.0,
                "quantity": 3,
                "strict_upright": False,
                "fragile": False,
                "stackable": True,
            },
            {
                "sku_id": "SKU-B",
                "description": "Test Box B",
                "length_mm": 200,
                "width_mm": 200,
                "height_mm": 300,
                "weight_kg": 8.0,
                "quantity": 2,
                "strict_upright": True,
                "fragile": True,
                "stackable": True,
            },
        ],
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    yield path
    os.unlink(path)


def test_load_request(sample_request_path):
    task_id, pallet, boxes = load_request(sample_request_path)
    assert task_id == "test_001"
    assert pallet.length_mm == 1200
    assert pallet.width_mm == 800
    assert pallet.max_weight_kg == 1000.0
    assert len(boxes) == 2
    assert boxes[0].sku_id == "SKU-A"
    assert boxes[0].quantity == 3
    assert boxes[1].strict_upright is True
    assert boxes[1].fragile is True


def test_box_properties():
    box = Box("SKU", "desc", 400, 300, 200, 5.0, 1)
    assert box.volume == 400 * 300 * 200
    assert box.base_area == 400 * 300


def test_placement_properties():
    p = Placement("SKU", 0, 10, 20, 30, 400, 300, 200, "LWH")
    assert p.x_max == 410
    assert p.y_max == 320
    assert p.z_max == 230
    assert p.volume == 400 * 300 * 200
    assert p.base_area == 400 * 300


def test_solution_to_dict():
    sol = Solution(
        task_id="test",
        solver_version="1.0.0",
        solve_time_ms=100,
        placements=[
            Placement("SKU-A", 0, 0, 0, 0, 400, 300, 200, "LWH"),
        ],
        unplaced=[
            UnplacedItem("SKU-B", 1, "no_space"),
        ],
    )
    d = solution_to_dict(sol)
    assert d["task_id"] == "test"
    assert d["solve_time_ms"] == 100
    assert len(d["placements"]) == 1
    assert d["placements"][0]["position"]["x_mm"] == 0
    assert d["placements"][0]["dimensions_placed"]["length_mm"] == 400
    assert d["placements"][0]["rotation_code"] == "LWH"
    assert len(d["unplaced"]) == 1
    assert d["unplaced"][0]["reason"] == "no_space"
