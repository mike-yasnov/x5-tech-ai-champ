"""Tests for solver.packer."""

from solver.models import Box, Pallet
from solver.packer import pack_greedy, SORT_KEYS


def _simple_pallet():
    return Pallet("EUR", 1200, 800, 1800, 1000.0)


def _simple_boxes():
    return [
        Box("SKU-A", "Box A", 400, 300, 200, 5.0, 3),
    ]


def test_pack_simple_boxes():
    """3 identical boxes should all be placed."""
    pallet = _simple_pallet()
    boxes = _simple_boxes()
    sol = pack_greedy("test", pallet, boxes)
    assert len(sol.placements) == 3
    assert len(sol.unplaced) == 0


def test_pack_produces_valid_positions():
    """All placements should have non-negative coordinates within bounds."""
    pallet = _simple_pallet()
    boxes = _simple_boxes()
    sol = pack_greedy("test", pallet, boxes)
    for p in sol.placements:
        assert p.x_mm >= 0
        assert p.y_mm >= 0
        assert p.z_mm >= 0
        assert p.x_mm + p.length_mm <= pallet.length_mm
        assert p.y_mm + p.width_mm <= pallet.width_mm
        assert p.z_mm + p.height_mm <= pallet.max_height_mm


def test_pack_no_collisions():
    """Placed boxes should not overlap."""
    pallet = _simple_pallet()
    boxes = _simple_boxes()
    sol = pack_greedy("test", pallet, boxes)
    for i, a in enumerate(sol.placements):
        for j, b in enumerate(sol.placements):
            if i >= j:
                continue
            # Check no AABB overlap
            ox = max(0, min(a.x_max, b.x_max) - max(a.x_mm, b.x_mm))
            oy = max(0, min(a.y_max, b.y_max) - max(a.y_mm, b.y_mm))
            oz = max(0, min(a.z_max, b.z_max) - max(a.z_mm, b.z_mm))
            assert ox * oy * oz == 0, f"Collision between placement {i} and {j}"


def test_pack_weight_limit():
    """When weight exceeds limit, excess boxes should be unplaced."""
    pallet = Pallet("EUR", 1200, 800, 1800, 15.0)  # only 15kg capacity
    boxes = [Box("SKU-A", "Heavy", 400, 300, 200, 6.0, 4)]  # 4 × 6kg = 24kg
    sol = pack_greedy("test", pallet, boxes)
    placed_weight = len(sol.placements) * 6.0
    assert placed_weight <= 15.0 + 0.01
    assert len(sol.unplaced) > 0


def test_all_sort_keys_work():
    """All predefined sort keys should produce valid solutions."""
    pallet = _simple_pallet()
    boxes = _simple_boxes()
    for key_name in SORT_KEYS:
        sol = pack_greedy("test", pallet, boxes, sort_key_name=key_name)
        assert sol.task_id == "test"
        assert len(sol.placements) > 0


def test_solve_time_recorded():
    """solve_time_ms should be recorded."""
    pallet = _simple_pallet()
    boxes = _simple_boxes()
    sol = pack_greedy("test", pallet, boxes)
    assert sol.solve_time_ms >= 0
