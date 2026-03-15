"""Tests for hybrid postprocess safety guards and fragility counting.

Tests enforce task spec compliance (docs/task.md). DO NOT MODIFY without explicit approval.
"""

from base_solver.hybrid.geometry import AABB
from base_solver.hybrid.pallet_state import PlacedBox
from base_solver.hybrid.postprocess import _supported_safely, _count_fragility_violations


def _placed_box(
    sku_id: str,
    aabb: AABB,
    weight: float,
    fragile: bool = False,
    stackable: bool = True,
) -> PlacedBox:
    return PlacedBox(
        sku_id=sku_id,
        instance_index=0,
        aabb=aabb,
        weight=weight,
        fragile=fragile,
        stackable=stackable,
        strict_upright=False,
        rotation_code="LWH",
        placed_dims=(aabb.length_x(), aabb.width_y(), aabb.height_z()),
    )


# ============================================================
# _supported_safely tests
# ============================================================

class TestSupportedSafely:
    def test_rejects_heavy_box_on_fragile_support(self):
        bottom = _placed_box("BOTTOM", AABB(0, 0, 0, 400, 400, 200), 1.0, fragile=True)
        top = _placed_box("TOP", AABB(0, 0, 200, 400, 400, 400), 5.0, fragile=False)
        placements = [bottom, top]
        assert _supported_safely(top, placements, 1) is False

    def test_rejects_support_from_non_stackable_box(self):
        bottom = _placed_box("BOTTOM", AABB(0, 0, 0, 400, 400, 200), 5.0, stackable=False)
        top = _placed_box("TOP", AABB(0, 0, 200, 400, 400, 400), 1.0)
        placements = [bottom, top]
        assert _supported_safely(top, placements, 1) is False

    def test_accepts_regular_supported_stack(self):
        bottom = _placed_box("BOTTOM", AABB(0, 0, 0, 400, 400, 200), 5.0)
        top = _placed_box("TOP", AABB(0, 0, 200, 400, 400, 400), 1.0)
        placements = [bottom, top]
        assert _supported_safely(top, placements, 1) is True

    def test_fragile_heavy_on_fragile_ok(self):
        """Per spec: only NON-FRAGILE heavy triggers fragility rejection."""
        bottom = _placed_box("BOTTOM", AABB(0, 0, 0, 400, 400, 200), 1.0, fragile=True)
        top = _placed_box("TOP", AABB(0, 0, 200, 400, 400, 400), 5.0, fragile=True)
        placements = [bottom, top]
        assert _supported_safely(top, placements, 1) is True

    def test_floor_always_safe(self):
        box = _placed_box("A", AABB(0, 0, 0, 400, 400, 200), 5.0)
        assert _supported_safely(box, [box], 0) is True

    def test_light_box_on_fragile_ok(self):
        """Box ≤ 2kg on fragile should be safe."""
        bottom = _placed_box("BOTTOM", AABB(0, 0, 0, 400, 400, 200), 1.0, fragile=True)
        top = _placed_box("TOP", AABB(0, 0, 200, 400, 400, 400), 1.5, fragile=False)
        placements = [bottom, top]
        assert _supported_safely(top, placements, 1) is True


# ============================================================
# _count_fragility_violations tests
# ============================================================

class TestCountFragilityViolations:
    def test_no_violations(self):
        placements = [
            _placed_box("A", AABB(0, 0, 0, 400, 400, 200), 5.0),
            _placed_box("B", AABB(0, 0, 200, 400, 400, 400), 3.0),
        ]
        assert _count_fragility_violations(placements) == 0

    def test_non_fragile_heavy_on_fragile_counted(self):
        placements = [
            _placed_box("FRAG", AABB(0, 0, 0, 400, 400, 200), 1.0, fragile=True),
            _placed_box("HEAVY", AABB(0, 0, 200, 400, 400, 400), 5.0, fragile=False),
        ]
        assert _count_fragility_violations(placements) == 1

    def test_fragile_on_fragile_not_counted(self):
        """Per spec: fragile-on-fragile is NOT a violation."""
        placements = [
            _placed_box("FRAG1", AABB(0, 0, 0, 400, 400, 200), 1.0, fragile=True),
            _placed_box("FRAG2", AABB(0, 0, 200, 400, 400, 400), 5.0, fragile=True),
        ]
        assert _count_fragility_violations(placements) == 0

    def test_light_on_fragile_not_counted(self):
        """Box ≤ 2kg on fragile is not a violation."""
        placements = [
            _placed_box("FRAG", AABB(0, 0, 0, 400, 400, 200), 1.0, fragile=True),
            _placed_box("LIGHT", AABB(0, 0, 200, 400, 400, 400), 1.5, fragile=False),
        ]
        assert _count_fragility_violations(placements) == 0

    def test_multiple_violations(self):
        placements = [
            _placed_box("FRAG1", AABB(0, 0, 0, 400, 400, 200), 1.0, fragile=True),
            _placed_box("FRAG2", AABB(400, 0, 0, 800, 400, 200), 1.0, fragile=True),
            _placed_box("HEAVY1", AABB(0, 0, 200, 400, 400, 400), 5.0, fragile=False),
            _placed_box("HEAVY2", AABB(400, 0, 200, 800, 400, 400), 5.0, fragile=False),
        ]
        assert _count_fragility_violations(placements) == 2

    def test_cumulative_load_through_fragile_chain(self):
        placements = [
            _placed_box("FRAG1", AABB(0, 0, 0, 400, 400, 200), 1.0, fragile=True),
            _placed_box("FRAG2", AABB(0, 0, 200, 400, 400, 400), 1.0, fragile=True),
            _placed_box("HEAVY", AABB(0, 0, 400, 400, 400, 600), 3.0, fragile=False),
        ]
        assert _count_fragility_violations(placements) == 2

    def test_no_contact_no_violation(self):
        """Heavy non-fragile above fragile but not touching — no violation."""
        placements = [
            _placed_box("FRAG", AABB(0, 0, 0, 400, 400, 200), 1.0, fragile=True),
            _placed_box("HEAVY", AABB(0, 0, 400, 400, 400, 600), 5.0, fragile=False),
        ]
        assert _count_fragility_violations(placements) == 0
