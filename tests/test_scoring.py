"""Tests for solver.scoring."""

from solver.models import Pallet
from solver.pallet_state import PalletState
from solver.scoring import score_placement


def _make_state():
    pallet = Pallet("EUR", 1200, 800, 1800, 1000.0)
    return PalletState(pallet)


def test_floor_placement_high_score():
    """Placing on the floor (z=0) should score well."""
    state = _make_state()
    score = score_placement(state, 400, 300, 200, 0, 0, 0, 5.0, False)
    assert score > 0.5


def test_lower_z_scores_better():
    """Lower z positions should score higher than higher ones."""
    state = _make_state()
    state.place("A", 400, 300, 200, 0, 0, 0, 5.0)

    # Place on floor (adjacent)
    score_low = score_placement(state, 400, 300, 200, 400, 0, 0, 5.0, False)
    # Place on top
    score_high = score_placement(state, 400, 300, 200, 0, 0, 200, 5.0, False)

    assert score_low > score_high


def test_corner_placement_contact_bonus():
    """Placing in corner (touching 2 walls) should score higher than center."""
    state = _make_state()
    # Corner placement (touches x=0 and y=0 walls)
    score_corner = score_placement(state, 400, 300, 200, 0, 0, 0, 5.0, False)
    # More interior placement (touches no walls in y)
    score_mid = score_placement(state, 400, 300, 200, 400, 250, 0, 5.0, False)

    assert score_corner > score_mid


def test_fragility_penalty():
    """Heavy box on fragile should get penalty."""
    state = _make_state()
    state.place("fragile_box", 400, 300, 200, 0, 0, 0, 1.0, fragile=True)

    # Heavy box on top of fragile
    score_heavy = score_placement(state, 400, 300, 200, 0, 0, 200, 5.0, False)
    # Light box on top of fragile
    score_light = score_placement(state, 400, 300, 200, 0, 0, 200, 1.5, False)

    assert score_light > score_heavy


def test_score_is_bounded():
    """Score should be between 0 and 1."""
    state = _make_state()
    score = score_placement(state, 400, 300, 200, 0, 0, 0, 5.0, False)
    assert 0.0 <= score <= 1.0
