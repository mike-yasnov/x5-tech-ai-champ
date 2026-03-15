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

BENCHMARK_SCENARIOS = (
    ORGANIZER_SCENARIOS
    + EXTENDED_REALISTIC_SCENARIOS
    + DIAGNOSTIC_SCENARIOS
)

BENCHMARK_SCENARIO_NAMES = [name for name, _ in BENCHMARK_SCENARIOS]
