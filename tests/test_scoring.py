"""Tests for solver.scoring placement heuristic.

Tests enforce task spec compliance (docs/task.md). DO NOT MODIFY without explicit approval.
"""

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

    score_low = score_placement(state, 400, 300, 200, 400, 0, 0, 5.0, False)
    score_high = score_placement(state, 400, 300, 200, 0, 0, 200, 5.0, False)

    assert score_low > score_high


def test_corner_placement_contact_bonus():
    """Placing in corner (touching 2 walls) should score higher than center."""
    state = _make_state()
    score_corner = score_placement(state, 400, 300, 200, 0, 0, 0, 5.0, False)
    score_mid = score_placement(state, 400, 300, 200, 400, 250, 0, 5.0, False)

    assert score_corner > score_mid


def test_fragility_penalty_non_fragile_on_fragile():
    """Non-fragile heavy box on fragile should get penalty."""
    state = _make_state()
    state.place("fragile_box", 400, 300, 200, 0, 0, 0, 1.0, fragile=True)

    score_heavy = score_placement(state, 400, 300, 200, 0, 0, 200, 5.0, False)
    score_light = score_placement(state, 400, 300, 200, 0, 0, 200, 1.5, False)

    assert score_light > score_heavy


def test_fragile_heavy_box_on_fragile_is_not_penalized():
    """Per spec: heavy fragile boxes should NOT trigger fragility penalty."""
    fragile_state = _make_state()
    fragile_state.place("fragile_box", 400, 300, 200, 0, 0, 0, 1.0, fragile=True)
    score_fragile_top = score_placement(
        fragile_state, 400, 300, 200, 0, 0, 200, 5.0, True,
    )

    sturdy_state = _make_state()
    sturdy_state.place("sturdy_box", 400, 300, 200, 0, 0, 0, 1.0, fragile=False)
    score_on_sturdy = score_placement(
        sturdy_state, 400, 300, 200, 0, 0, 200, 5.0, True,
    )

    assert score_fragile_top == score_on_sturdy


def test_strict_fragility_returns_negative():
    """strict_fragility=True with non-fragile heavy on fragile → -10.0."""
    state = _make_state()
    state.place("fragile_box", 400, 300, 200, 0, 0, 0, 1.0, fragile=True)
    score = score_placement(
        state, 400, 300, 200, 0, 0, 200, 5.0, False, strict_fragility=True,
    )
    assert score == -10.0


def test_strict_fragility_fragile_on_fragile_not_blocked():
    """strict_fragility=True but fragile-on-fragile should NOT return -10."""
    state = _make_state()
    state.place("fragile_box", 400, 300, 200, 0, 0, 0, 1.0, fragile=True)
    score = score_placement(
        state, 400, 300, 200, 0, 0, 200, 5.0, True, strict_fragility=True,
    )
    assert score > -10.0


def test_score_is_bounded():
    """Score should be between 0 and 1."""
    state = _make_state()
    score = score_placement(state, 400, 300, 200, 0, 0, 0, 5.0, False)
    assert 0.0 <= score <= 1.0
