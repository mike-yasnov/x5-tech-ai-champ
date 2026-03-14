"""Regression tests for generator archetype metadata."""

from generator import FOOD_RETAIL_ARCHETYPES, create_box


def test_food_retail_archetypes_include_stackable_flags():
    assert FOOD_RETAIL_ARCHETYPES["banana"]["stackable"] is True
    assert FOOD_RETAIL_ARCHETYPES["sugar"]["stackable"] is True
    assert FOOD_RETAIL_ARCHETYPES["water"]["stackable"] is True
    assert FOOD_RETAIL_ARCHETYPES["wine"]["stackable"] is False
    assert FOOD_RETAIL_ARCHETYPES["chips"]["stackable"] is True
    assert FOOD_RETAIL_ARCHETYPES["eggs"]["stackable"] is False
    assert FOOD_RETAIL_ARCHETYPES["canned"]["stackable"] is True


def test_create_box_propagates_stackable_from_archetype():
    wine = create_box("wine", 1, 1)
    eggs = create_box("eggs", 1, 1)
    water = create_box("water", 1, 1)

    assert wine["stackable"] is False
    assert eggs["stackable"] is False
    assert water["stackable"] is True
