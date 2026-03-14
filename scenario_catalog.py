"""Shared scenario catalog for benchmarks and regression tests."""

from __future__ import annotations

ORGANIZER_SCENARIOS = [
    ("heavy_water", 42),
    ("fragile_tower", 43),
    ("liquid_tetris", 44),
    ("random_mixed", 45),
]

DIAGNOSTIC_SCENARIOS = [
    ("exact_fit", 46),
    ("fragile_mix", 47),
    ("support_tetris", 48),
    ("cavity_fill", 49),
    ("count_preference", 50),
]

EXTENDED_REALISTIC_SCENARIOS = [
    ("weight_limited_repeat", 51),
    ("fragile_cap_mix", 52),
    ("mixed_column_repeat", 53),
    ("small_gap_fill", 54),
    ("non_stackable_caps", 55),
]

PRIVATE_TEST_SCENARIOS = [
    ("private_heavy_eggs_crush", 60),
    ("private_all_upright_tight", 61),
    ("private_fragile_dominant", 62),
    ("private_weight_razor", 63),
    ("private_sugar_flood", 64),
    ("private_wine_eggs_dilemma", 65),
    ("private_canned_wall", 66),
    ("private_chips_mountain", 67),
    ("private_weight_tradeoff", 68),
    ("private_full_catalog", 69),
    ("private_micro_batch", 70),
    ("private_upright_overflow", 71),
    ("private_nostack_fragile_mix", 72),
    ("private_heavy_fragile_sandwich", 73),
    ("private_odd_pallet_stress", 74),
]

BENCHMARK_SCENARIOS = (
    ORGANIZER_SCENARIOS
    + EXTENDED_REALISTIC_SCENARIOS
    + DIAGNOSTIC_SCENARIOS
    + PRIVATE_TEST_SCENARIOS
)

BENCHMARK_SCENARIO_NAMES = [name for name, _ in BENCHMARK_SCENARIOS]
