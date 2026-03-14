"""Tests for solver.pallet_state."""

import pytest

from solver.models import Pallet
from solver.pallet_state import PalletState


@pytest.fixture
def pallet():
    return Pallet("EUR", 1200, 800, 1800, 1000.0)


@pytest.fixture
def state(pallet):
    return PalletState(pallet)


def test_place_on_floor(state):
    """Box placed at (0,0,0) should succeed."""
    assert state.can_place(400, 300, 200, 0, 0, 0, 5.0)


def test_out_of_bounds_x(state):
    """Box exceeding pallet length should fail."""
    assert not state.can_place(400, 300, 200, 1000, 0, 0, 5.0)


def test_out_of_bounds_y(state):
    """Box exceeding pallet width should fail."""
    assert not state.can_place(400, 300, 200, 0, 600, 0, 5.0)


def test_out_of_bounds_z(state):
    """Box exceeding max height should fail."""
    assert not state.can_place(400, 300, 1700, 0, 0, 200, 5.0)


def test_collision_rejected(state):
    """Placing overlapping boxes should fail."""
    state.place("A", 400, 300, 200, 0, 0, 0, 5.0)
    # Overlapping box
    assert not state.can_place(400, 300, 200, 100, 100, 0, 5.0)


def test_no_collision_adjacent(state):
    """Adjacent boxes (no overlap) should succeed."""
    state.place("A", 400, 300, 200, 0, 0, 0, 5.0)
    # Right next to it
    assert state.can_place(400, 300, 200, 400, 0, 0, 5.0)


def test_overweight_rejected(state):
    """Total weight exceeding limit should fail."""
    state.place("A", 400, 300, 200, 0, 0, 0, 999.0)
    assert not state.can_place(400, 300, 200, 400, 0, 0, 2.0)


def test_support_60_percent_passes(state):
    """Box with ≥60% support should pass."""
    # Place base box: 400x300 at (0,0,0)
    state.place("A", 400, 300, 200, 0, 0, 0, 5.0)
    # Place on top, shifted slightly but ≥60% overlap
    # 400x300 shifted by 100 in x: overlap = 300*300 = 90000, base = 400*300 = 120000, ratio = 75%
    assert state.can_place(400, 300, 200, 100, 0, 200, 5.0)


def test_support_insufficient_fails(state):
    """Box with <60% support should fail."""
    state.place("A", 400, 300, 200, 0, 0, 0, 5.0)
    # Place on top, heavily shifted: overlap very small
    # Shifted by 300 in x: overlap = 100*300 = 30000, base = 400*300 = 120000, ratio = 25%
    assert not state.can_place(400, 300, 200, 300, 0, 200, 5.0)


def test_floating_box_fails(state):
    """Box in the air with no support should fail."""
    assert not state.can_place(400, 300, 200, 0, 0, 500, 5.0)


def test_stackable_false_blocks(state):
    """Cannot place on top of a non-stackable box."""
    state.place("A", 400, 300, 200, 0, 0, 0, 5.0, stackable=False)
    # Try to place on top
    assert not state.can_place(400, 300, 200, 0, 0, 200, 5.0)


def test_extreme_points_update(state):
    """After placing a box, new extreme points should be generated."""
    initial_eps = len(state.extreme_points)
    assert initial_eps == 1  # Just (0,0,0)

    state.place("A", 400, 300, 200, 0, 0, 0, 5.0)

    # Should have new EPs: right, front, top of the placed box
    assert len(state.extreme_points) >= 3
    eps = state.extreme_points
    assert (400, 0, 0) in eps    # right
    assert (0, 300, 0) in eps    # front
    assert (0, 0, 200) in eps    # top


def test_weight_tracking(state):
    """Current weight should be updated after placement."""
    assert state.current_weight == 0.0
    state.place("A", 400, 300, 200, 0, 0, 0, 5.0)
    assert state.current_weight == 5.0
    state.place("B", 400, 300, 200, 400, 0, 0, 3.0)
    assert state.current_weight == 8.0


def test_max_z_tracking(state):
    """max_z should track the highest point."""
    state.place("A", 400, 300, 200, 0, 0, 0, 5.0)
    assert state.max_z == 200
    state.place("B", 400, 300, 200, 0, 0, 200, 5.0)
    assert state.max_z == 400
