"""Comprehensive tests for solver/hybrid/feasibility.py FeasibilityChecker.

Tests enforce task spec compliance (docs/task.md). DO NOT MODIFY without explicit approval.
"""

from solver.hybrid.feasibility import FeasibilityChecker
from solver.hybrid.geometry import AABB
from solver.hybrid.pallet_state import PalletState, PlacedBox


def _checker(length=1200, width=800, max_height=1800, max_weight=1000.0):
    return FeasibilityChecker(length, width, max_height, max_weight)


def _state(length=1200, width=800, max_height=1800, max_weight=1000.0):
    return PalletState(length, width, max_height, max_weight)


def _placed(sku_id, x1, y1, z1, x2, y2, z2, weight=5.0,
            fragile=False, stackable=True):
    return PlacedBox(
        sku_id=sku_id,
        instance_index=0,
        aabb=AABB(x1, y1, z1, x2, y2, z2),
        weight=weight,
        fragile=fragile,
        stackable=stackable,
        strict_upright=False,
        rotation_code="LWH",
        placed_dims=(x2 - x1, y2 - y1, z2 - z1),
    )


# ============================================================
# Bounds
# ============================================================

class TestCheckBounds:
    def test_valid_inside(self):
        c = _checker()
        assert c.check_bounds(AABB(0, 0, 0, 400, 300, 200)) is True

    def test_valid_at_boundary(self):
        c = _checker()
        assert c.check_bounds(AABB(0, 0, 0, 1200, 800, 1800)) is True

    def test_exceeds_x(self):
        c = _checker()
        assert c.check_bounds(AABB(1000, 0, 0, 1400, 300, 200)) is False

    def test_exceeds_y(self):
        c = _checker()
        assert c.check_bounds(AABB(0, 600, 0, 400, 900, 200)) is False

    def test_exceeds_z(self):
        c = _checker()
        assert c.check_bounds(AABB(0, 0, 1700, 400, 300, 1900)) is False

    def test_negative_origin(self):
        c = _checker()
        assert c.check_bounds(AABB(-1, 0, 0, 400, 300, 200)) is False


# ============================================================
# Weight
# ============================================================

class TestCheckWeight:
    def test_under_limit(self):
        c = _checker(max_weight=1000.0)
        state = _state(max_weight=1000.0)
        assert c.check_weight(500.0, state) is True

    def test_at_limit(self):
        c = _checker(max_weight=1000.0)
        state = _state(max_weight=1000.0)
        state.place(_placed("A", 0, 0, 0, 400, 300, 200, weight=500.0))
        assert c.check_weight(500.0, state) is True

    def test_over_limit(self):
        c = _checker(max_weight=1000.0)
        state = _state(max_weight=1000.0)
        state.place(_placed("A", 0, 0, 0, 400, 300, 200, weight=999.0))
        assert c.check_weight(2.0, state) is False


# ============================================================
# Collision
# ============================================================

class TestCheckCollision:
    def test_no_overlap(self):
        c = _checker()
        state = _state()
        state.place(_placed("A", 0, 0, 0, 400, 300, 200))
        assert c.check_collision(AABB(400, 0, 0, 800, 300, 200), state) is True

    def test_overlap(self):
        c = _checker()
        state = _state()
        state.place(_placed("A", 0, 0, 0, 400, 300, 200))
        assert c.check_collision(AABB(200, 100, 0, 600, 400, 200), state) is False

    def test_empty_state_no_collision(self):
        c = _checker()
        state = _state()
        assert c.check_collision(AABB(0, 0, 0, 400, 300, 200), state) is True


# ============================================================
# Support
# ============================================================

class TestCheckSupport:
    def test_floor_always_supported(self):
        c = _checker()
        state = _state()
        assert c.check_support(AABB(0, 0, 0, 400, 300, 200), state) is True

    def test_60_percent_support(self):
        c = _checker()
        state = _state()
        state.place(_placed("A", 0, 0, 0, 500, 400, 200))
        # Top box shifted by 200 in x: overlap = 300*400 = 120000, base = 500*400 = 200000 → 60%
        assert c.check_support(AABB(200, 0, 200, 700, 400, 400), state) is True

    def test_insufficient_support(self):
        c = _checker()
        state = _state()
        state.place(_placed("A", 0, 0, 0, 400, 300, 200))
        # Floating, mostly unsupported
        assert c.check_support(AABB(300, 0, 200, 700, 300, 400), state) is False


# ============================================================
# Stackable
# ============================================================

class TestCheckStackable:
    def test_on_stackable_ok(self):
        c = _checker()
        state = _state()
        state.place(_placed("A", 0, 0, 0, 400, 400, 200, stackable=True))
        assert c.check_stackable(AABB(0, 0, 200, 400, 400, 400), state) is True

    def test_on_non_stackable_rejected(self):
        c = _checker()
        state = _state()
        state.place(_placed("A", 0, 0, 0, 400, 400, 200, stackable=False))
        assert c.check_stackable(AABB(0, 0, 200, 400, 400, 400), state) is False

    def test_non_stackable_no_xy_overlap_ok(self):
        c = _checker()
        state = _state()
        state.place(_placed("A", 0, 0, 0, 400, 400, 200, stackable=False))
        # Box at same z but different XY — no overlap
        assert c.check_stackable(AABB(500, 0, 200, 900, 400, 400), state) is True


# ============================================================
# Upright
# ============================================================

class TestCheckUpright:
    def test_matching_height(self):
        c = _checker()
        assert c.check_upright(200, 200, True) is True

    def test_mismatched_height(self):
        c = _checker()
        assert c.check_upright(300, 200, True) is False

    def test_non_strict_any_height(self):
        c = _checker()
        assert c.check_upright(300, 200, False) is True


# ============================================================
# is_feasible integration
# ============================================================

class TestIsFeasible:
    def test_all_pass(self):
        c = _checker()
        state = _state()
        ok, reason = c.is_feasible(
            aabb=AABB(0, 0, 0, 400, 300, 200),
            weight=5.0,
            orig_h=200,
            strict_upright=False,
            stackable_below=True,
            state=state,
        )
        assert ok is True
        assert reason == ""

    def test_bounds_failure(self):
        c = _checker()
        state = _state()
        ok, reason = c.is_feasible(
            aabb=AABB(1000, 0, 0, 1400, 300, 200),
            weight=5.0,
            orig_h=200,
            strict_upright=False,
            stackable_below=True,
            state=state,
        )
        assert ok is False
        assert reason == "bounds"

    def test_weight_failure(self):
        c = _checker(max_weight=10.0)
        state = _state(max_weight=10.0)
        ok, reason = c.is_feasible(
            aabb=AABB(0, 0, 0, 400, 300, 200),
            weight=15.0,
            orig_h=200,
            strict_upright=False,
            stackable_below=True,
            state=state,
        )
        assert ok is False
        assert reason == "weight"

    def test_upright_failure(self):
        c = _checker()
        state = _state()
        ok, reason = c.is_feasible(
            aabb=AABB(0, 0, 0, 400, 300, 200),
            weight=5.0,
            orig_h=300,  # doesn't match placed height 200
            strict_upright=True,
            stackable_below=True,
            state=state,
        )
        assert ok is False
        assert reason == "upright"

    def test_stackable_failure(self):
        c = _checker()
        state = _state()
        state.place(_placed("A", 0, 0, 0, 400, 400, 200, stackable=False))
        ok, reason = c.is_feasible(
            aabb=AABB(0, 0, 200, 400, 400, 400),
            weight=5.0,
            orig_h=200,
            strict_upright=False,
            stackable_below=True,
            state=state,
        )
        assert ok is False
        assert reason == "stackable"
