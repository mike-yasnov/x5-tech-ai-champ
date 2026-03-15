"""Tests for solver.orientations."""

from alternative_solver.models import Box
from alternative_solver.orientations import get_orientations


def test_regular_box_6_orientations():
    """Box with all different dimensions should have 6 orientations."""
    box = Box("SKU", "desc", 400, 300, 200, 5.0, 1)
    orients = get_orientations(box)
    assert len(orients) == 6
    # All should be permutations of (400, 300, 200)
    for dx, dy, dz, code in orients:
        assert sorted([dx, dy, dz]) == [200, 300, 400]
        assert len(code) == 3


def test_cubic_box_1_orientation():
    """Cube has only 1 unique orientation."""
    box = Box("SKU", "desc", 300, 300, 300, 5.0, 1)
    orients = get_orientations(box)
    assert len(orients) == 1
    assert orients[0][:3] == (300, 300, 300)


def test_two_equal_dims():
    """Box with two equal dimensions should have 3 unique orientations."""
    box = Box("SKU", "desc", 400, 400, 200, 5.0, 1)
    orients = get_orientations(box)
    assert len(orients) == 3


def test_strict_upright_filters():
    """strict_upright=True allows only orientations where H stays on Z axis."""
    box = Box("SKU", "desc", 400, 300, 200, 5.0, 1, strict_upright=True)
    orients = get_orientations(box)
    # Only LWH and WLH (H=200 stays on Z)
    for dx, dy, dz, code in orients:
        assert dz == 200  # original height must be Z
    assert len(orients) == 2


def test_strict_upright_cube():
    """Cubic box with strict_upright still has 1 orientation."""
    box = Box("SKU", "desc", 300, 300, 300, 5.0, 1, strict_upright=True)
    orients = get_orientations(box)
    assert len(orients) == 1


def test_rotation_codes():
    """Check that rotation codes are valid."""
    box = Box("SKU", "desc", 400, 300, 200, 5.0, 1)
    orients = get_orientations(box)
    valid_codes = {"LWH", "LHW", "WLH", "WHL", "HLW", "HWL"}
    for _, _, _, code in orients:
        assert code in valid_codes
