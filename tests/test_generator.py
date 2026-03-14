"""Comprehensive tests for generator archetypes and scenario generation.

Tests enforce task spec compliance (docs/task.md). DO NOT MODIFY without explicit approval.
"""

from generator import FOOD_RETAIL_ARCHETYPES, create_box, generate_scenario
from scenario_catalog import BENCHMARK_SCENARIOS


# ============================================================
# Archetype tests
# ============================================================

REQUIRED_ARCHETYPE_FIELDS = {"desc", "l", "w", "h", "wt", "upright", "fragile", "stackable"}

REQUIRED_BOX_FIELDS = {
    "sku_id", "description", "length_mm", "width_mm", "height_mm",
    "weight_kg", "quantity", "strict_upright", "fragile", "stackable",
}

REQUIRED_PALLET_FIELDS = {"length_mm", "width_mm", "max_height_mm", "max_weight_kg"}


def test_food_retail_archetypes_include_stackable_flags():
    assert FOOD_RETAIL_ARCHETYPES["banana"]["stackable"] is True
    assert FOOD_RETAIL_ARCHETYPES["sugar"]["stackable"] is True
    assert FOOD_RETAIL_ARCHETYPES["water"]["stackable"] is True
    assert FOOD_RETAIL_ARCHETYPES["wine"]["stackable"] is False
    assert FOOD_RETAIL_ARCHETYPES["chips"]["stackable"] is True
    assert FOOD_RETAIL_ARCHETYPES["eggs"]["stackable"] is True
    assert FOOD_RETAIL_ARCHETYPES["canned"]["stackable"] is True


def test_all_archetypes_have_required_fields():
    for key, arch in FOOD_RETAIL_ARCHETYPES.items():
        missing = REQUIRED_ARCHETYPE_FIELDS - set(arch.keys())
        assert not missing, f"Archetype '{key}' missing fields: {missing}"


def test_create_box_propagates_stackable_from_archetype():
    wine = create_box("wine", 1, 1)
    eggs = create_box("eggs", 1, 1)
    water = create_box("water", 1, 1)

    assert wine["stackable"] is False
    assert eggs["stackable"] is True
    assert water["stackable"] is True


def test_create_box_output_schema():
    for key in FOOD_RETAIL_ARCHETYPES:
        box = create_box(key, 1, 5)
        missing = REQUIRED_BOX_FIELDS - set(box.keys())
        assert not missing, f"create_box('{key}') missing fields: {missing}"


def test_create_box_dimensions_noisy():
    """Dimensions should have ±2% noise from base values."""
    base = FOOD_RETAIL_ARCHETYPES["sugar"]
    samples = [create_box("sugar", 1, 1) for _ in range(50)]
    for box in samples:
        assert abs(box["length_mm"] - base["l"]) <= base["l"] * 0.025
        assert abs(box["width_mm"] - base["w"]) <= base["w"] * 0.025
        assert abs(box["height_mm"] - base["h"]) <= base["h"] * 0.025


def test_create_box_weight_noisy():
    """Weight should have ±2% noise from base value."""
    base = FOOD_RETAIL_ARCHETYPES["sugar"]
    samples = [create_box("sugar", 1, 1) for _ in range(50)]
    for box in samples:
        assert abs(box["weight_kg"] - base["wt"]) <= base["wt"] * 0.025


# ============================================================
# Scenario tests
# ============================================================

def test_all_scenarios_generate_valid_json():
    """Every scenario in BENCHMARK_SCENARIOS produces valid JSON with required fields."""
    for scenario_type, seed in BENCHMARK_SCENARIOS:
        result = generate_scenario(f"test_{scenario_type}", scenario_type, seed=seed)
        assert "task_id" in result
        assert "pallet" in result
        assert "boxes" in result
        assert isinstance(result["boxes"], list)
        assert len(result["boxes"]) > 0, f"Scenario '{scenario_type}' has no boxes"


def test_scenario_pallet_has_required_fields():
    for scenario_type, seed in BENCHMARK_SCENARIOS:
        result = generate_scenario(f"test_{scenario_type}", scenario_type, seed=seed)
        pallet = result["pallet"]
        missing = REQUIRED_PALLET_FIELDS - set(pallet.keys())
        assert not missing, f"Scenario '{scenario_type}' pallet missing: {missing}"
        assert pallet["length_mm"] > 0
        assert pallet["width_mm"] > 0
        assert pallet["max_height_mm"] > 0
        assert pallet["max_weight_kg"] > 0


def test_scenario_boxes_have_required_fields():
    for scenario_type, seed in BENCHMARK_SCENARIOS:
        result = generate_scenario(f"test_{scenario_type}", scenario_type, seed=seed)
        for i, box in enumerate(result["boxes"]):
            missing = REQUIRED_BOX_FIELDS - set(box.keys())
            assert not missing, (
                f"Scenario '{scenario_type}' box[{i}] missing: {missing}"
            )


def test_wine_archetype_non_stackable_in_scenarios():
    """All scenarios using wine via create_box must have stackable=False."""
    wine_scenarios = ["liquid_tetris", "fragile_cap_mix"]
    for sc_type in wine_scenarios:
        result = generate_scenario(f"test_{sc_type}", sc_type, seed=100)
        for box in result["boxes"]:
            if "WINE" in box["sku_id"].upper():
                assert box["stackable"] is False, (
                    f"Wine box in '{sc_type}' has stackable=True"
                )


def test_eggs_archetype_stackable_in_scenarios():
    """Eggs created via create_box should have stackable=True."""
    egg_scenarios = ["fragile_tower", "fragile_cap_mix"]
    for sc_type in egg_scenarios:
        result = generate_scenario(f"test_{sc_type}", sc_type, seed=100)
        for box in result["boxes"]:
            if "EGGS" in box["sku_id"].upper():
                assert box["stackable"] is True, (
                    f"Eggs box in '{sc_type}' has stackable=False"
                )


def test_non_stackable_caps_scenario_has_non_stackable_boxes():
    result = generate_scenario("test_ns", "non_stackable_caps", seed=55)
    non_stackable = [b for b in result["boxes"] if not b["stackable"]]
    assert len(non_stackable) > 0, "non_stackable_caps must have at least one non-stackable box"


def test_scenario_reproducibility():
    """Same seed must produce identical output."""
    r1 = generate_scenario("test", "heavy_water", seed=42)
    r2 = generate_scenario("test", "heavy_water", seed=42)
    assert r1 == r2


def test_scenario_different_seeds_differ():
    """Different seeds should produce different box quantities/dimensions."""
    r1 = generate_scenario("test", "random_mixed", seed=1)
    r2 = generate_scenario("test", "random_mixed", seed=999)
    # At minimum, SKU IDs should differ (random component)
    skus1 = {b["sku_id"] for b in r1["boxes"]}
    skus2 = {b["sku_id"] for b in r2["boxes"]}
    assert skus1 != skus2
