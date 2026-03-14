# Benchmark Report

**Date:** 2026-03-15
**Branch:** feature/rl-solver
**Solver:** Multi-restart greedy EP + LNS + postprocessing
**Config:** restarts=30, time_budget=900ms, workers=1 (sequential)

---

## Benchmark Results

### Сценарии организаторов

| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |
|----------|-------|--------|----------|-----------|------------|--------|-----------|
| heavy_water | **0.7434** | 0.7281 | 0.5978 | 1.0000 | 1.0000 | 107/179 | 517 |
| fragile_tower | **0.7282** | 0.8000 | 0.4773 | 0.8500 | 1.0000 | 21/44 | 520 |
| liquid_tetris | **0.6996** | 0.3992 | 1.0000 | 1.0000 | 1.0000 | 84/84 | 463 |
| random_mixed | **0.7342** | 0.8456 | 0.5047 | 0.6000 | 1.0000 | 54/107 | 636 |

**Average score: 0.7263**

### Расширенные реалистичные сценарии

| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |
|----------|-------|--------|----------|-----------|------------|--------|-----------|
| weight_limited_repeat | **0.7866** | 0.7983 | 0.6250 | 1.0000 | 1.0000 | 85/136 | 827 |
| fragile_cap_mix | **0.7762** | 0.7002 | 0.7536 | 1.0000 | 1.0000 | 52/69 | 621 |
| mixed_column_repeat | **0.9010** | 0.8019 | 1.0000 | 1.0000 | 1.0000 | 77/77 | 561 |
| small_gap_fill | **0.7917** | 0.5833 | 1.0000 | 1.0000 | 1.0000 | 22/22 | 290 |
| non_stackable_caps | **0.8750** | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 18/18 | 199 |

**Average score: 0.8261**

### Sanity и диагностические сценарии

| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |
|----------|-------|--------|----------|-----------|------------|--------|-----------|
| exact_fit | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4/4 | 9 |
| fragile_mix | **0.9864** | 1.0000 | 0.9545 | 1.0000 | 1.0000 | 21/22 | 197 |
| support_tetris | **0.9571** | 1.0000 | 0.8571 | 1.0000 | 1.0000 | 12/14 | 80 |
| cavity_fill | **0.7111** | 0.5556 | 0.7778 | 1.0000 | 1.0000 | 14/18 | 53 |
| count_preference | **0.9000** | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 2/3 | 4 |

**Average score: 0.9109**

**Overall average: 0.8279**

---

## Constraint Benchmark Results

### 1. Dimensions (LxWxH)

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | dim_exact_fill_1box | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 1/1 | 2 | 1 box = pallet size -> perfect fit |
| 2 | dim_4tile_floor | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 4/4 | 6 | 4 boxes tile 600x400 floor exactly (1 layer) |
| 3 | dim_8tile_2layers | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 8/8 | 23 | 8 boxes tile 600x400x400 exactly (2 layers) |
| 4 | dim_max_height_exact | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 1/1 | 1 | Box height == pallet max_height exactly |
| 5 | dim_tiny_box | **PASS** | valid=OK, exact_coverage=OK | 0.5000 | 1/1 | 1 | Minimum-size box (1x1x1 mm) |
| 6 | dim_rotation_needed | **PASS** | valid=OK, exact_coverage=OK | 0.7500 | 1/1 | 1 | Box must be rotated to fit within height limit |

**6/6 passed**

### 2. No Collisions

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | collision_cubes_27 | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 27/27 | 223 | 27 cubes must fill 300^3 without any overlap |
| 2 | collision_two_sizes | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 3/3 | 4 | Two sizes share floor: 400x400 + 2x200x200 fill 600x400 |
| 3 | collision_cubes_64 | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 64/64 | 369 | 64 cubes fill 400^3 collision-free grid |

**3/3 passed**

### 3. Support From Below (>=60%)

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | support_full | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 2/2 | 3 | Two stacked boxes: upper has 100% support |
| 2 | support_borderline_ok | **PASS** | valid=OK, exact_coverage=OK | 0.9167 | 2/2 | 3 | Upper box fits entirely on base -> 100% support easy |
| 3 | support_tower_5 | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 5/5 | 7 | Tower of 5 identical boxes, each layer supported by one below |
| 4 | support_bridging | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 3/3 | 5 | Upper box bridges two base boxes for combined support |

**4/4 passed**

### 4. Strict Upright (No Rotation)

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | upright_native_only | **PASS** | valid=OK, exact_coverage=OK | 0.5625 | 1/1 | 1 | strict_upright box placed in native H orientation |
| 2 | upright_all_boxes | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 1.0000 | 4/4 | 6 | All 4 boxes strict_upright, must keep H on Z axis |
| 3 | upright_mixed | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 0.9167 | 8/8 | 22 | Mix of strict_upright and freely rotatable boxes |
| 4 | upright_no_extra_rotation | **PASS** | valid=OK, exact_coverage=OK | 0.7500 | 8/8 | 25 | Upright boxes where rotation is tempting but forbidden |

**4/4 passed**

### 5. Volume Utilization

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | volume_perfect | **PASS** | valid=OK, exact_volume=OK | 1.0000 | 1/1 | 1 | Volume utilization = 1.0 (perfect fill) |
| 2 | volume_50pct | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 0.7500 | 1/1 | 1 | Volume = 0.5 (box fills half of pallet height) |
| 3 | volume_25pct | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 0.6250 | 1/1 | 1 | Volume = 0.25 (one box is 1/4 of pallet volume) |
| 4 | volume_75pct | **PASS** | valid=OK, exact_coverage=OK, exact_volume=OK | 0.8750 | 3/3 | 5 | Volume = 0.75 (3 tall boxes fill 3/4 of pallet) |

**4/4 passed**

### 6. Item Coverage

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | coverage_full | **PASS** | valid=OK, exact_coverage=OK | 1.0000 | 4/4 | 6 | All items fit -> coverage = 1.0 |
| 2 | coverage_weight_limited | **PASS** | valid=OK, min_coverage=OK | 0.4750 | 1/2 | 2 | Weight limit allows only 1 of 2 boxes -> coverage >= 0.5 |
| 3 | coverage_height_limited | **PASS** | valid=OK, exact_coverage=OK | 0.8500 | 1/2 | 2 | Only 1 of 2 full-floor boxes fits (height limit) -> coverage = 0.5 |
| 4 | coverage_space_limited | **PASS** | valid=OK, exact_coverage=OK | 0.9400 | 4/5 | 6 | 5 boxes requested, only 4 fit on floor -> coverage = 0.8 |
| 5 | coverage_mixed_partial | **PASS** | valid=OK, min_coverage=OK | 0.9400 | 4/5 | 6 | Mixed SKUs: large + small; 4 of 5 items fit -> coverage >= 0.8 |
| 6 | coverage_zero_oversize | **PASS** | valid=OK, exact_coverage=OK | 0.2000 | 0/1 | 0 | Box larger than pallet in all dims -> coverage = 0.0 |

**6/6 passed**

### 7. Fragility (Heavy-on-Fragile)

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | fragile_none | **PASS** | valid=OK, min_fragility=OK | 0.7500 | 4/4 | 10 | No fragile boxes -> fragility_score = 1.0 |
| 2 | fragile_all_light | **PASS** | valid=OK, min_fragility=OK | 0.7500 | 4/4 | 10 | All fragile but light (<=2kg) -> no violations |
| 3 | fragile_heavy_on_fragile | **PASS** | valid=OK, exact_coverage=OK, min_fragility=OK | 1.0000 | 2/2 | 3 | Heavy + fragile: solver places heavy below fragile |
| 4 | fragile_multi_stress | **PASS** | valid=OK, min_fragility=OK | 0.8333 | 8/8 | 32 | 4 fragile + 4 heavy boxes: minimize fragility violations |
| 5 | fragile_order_matters | **PASS** | valid=OK, exact_coverage=OK, min_fragility=OK | 1.0000 | 2/2 | 2 | 2 full-floor boxes: must put heavy below fragile |
| 6 | fragile_boundary_2kg | **PASS** | valid=OK, exact_coverage=OK, min_fragility=OK | 1.0000 | 2/2 | 2 | 2.0 kg box on fragile -> no violation (threshold is >2.0) |
| 7 | fragile_boundary_2_01kg | **PASS** | valid=OK, exact_coverage=OK, min_fragility=OK | 1.0000 | 2/2 | 2 | 2.01 kg on fragile: solver puts heavy below -> fragility=1.0 |

**7/7 passed**

### 8. Execution Time

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | time_trivial | **PASS** | valid=OK, max_time_ms=OK | 1.0000 | 1/1 | 1 | Single box -> solve < 1s |
| 2 | time_medium_20 | **PASS** | valid=OK, max_time_ms=OK | 0.5694 | 20/20 | 274 | 20 items -> solve < 5s |
| 3 | time_large_100 | **PASS** | valid=OK, max_time_ms=OK | 0.5868 | 100/100 | 670 | 100 items -> solve < 30s |
| 4 | time_stress_200 | **PASS** | valid=OK, max_time_ms=OK | 0.8060 | 200/200 | 698 | 200 items, 4 SKUs -> solve < 30s |

**4/4 passed**

### 9. Weight Constraint

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | weight_exact_limit | **PASS** | valid=OK, exact_coverage=OK | 1.0000 | 4/4 | 6 | Total weight (40kg) = max_weight_kg exactly -> all fit |
| 2 | weight_partial | **PASS** | valid=OK, min_coverage=OK | 0.4556 | 2/3 | 3 | Weight limit 25kg, 3x10kg -> only 2 fit |
| 3 | weight_single_overweight | **PASS** | valid=OK, exact_coverage=OK | 0.2000 | 0/1 | 0 | Single box exceeds max_weight -> coverage = 0.0 |

**3/3 passed**

### 10. Non-Stackable

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | stackable_single | **PASS** | valid=OK, exact_coverage=OK | 0.7500 | 1/1 | 1 | Single non-stackable box -> valid |
| 2 | stackable_beside | **PASS** | valid=OK, exact_coverage=OK | 1.0000 | 2/2 | 3 | Non-stackable + stackable side by side -> both fit |
| 3 | stackable_caps_mixed | **PASS** | valid=OK, min_coverage=OK | 0.7500 | 6/6 | 16 | 4 stackable + 2 non-stackable caps -> constraint respected |

**3/3 passed**

### 11. Combined Constraints

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | combined_upright_fragile_weight | **PASS** | valid=OK, min_coverage=OK, min_fragility=OK | 0.7500 | 6/6 | 16 | Upright fragile boxes + heavy boxes + weight limit |
| 2 | combined_ns_fragile_upright | **PASS** | valid=OK, min_coverage=OK, min_fragility=OK | 0.8750 | 6/6 | 13 | Non-stackable + fragile + upright caps on stackable base |

**2/2 passed**

---

### H1. Weight Limit Edge Cases

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | hard_weight_exact | **PASS** | valid=OK, exact_coverage=OK | 0.5370 | 4/4 | 15 | 4x25kg = 100kg = max_weight exactly |
| 2 | hard_weight_overflow_1 | **PASS** | valid=OK, min_coverage=OK | 0.4616 | 5/6 | 12 | 6x10kg=60kg > 50kg limit -> at least 5 placed |
| 3 | hard_weight_one_heavy | **PASS** | valid=OK, min_coverage=OK | 0.5086 | 50/51 | 320 | 1 anchor (180kg) + 50 light (0.5kg), limit 200kg |
| 4 | hard_weight_uniform_cutoff | **PASS** | valid=OK, min_coverage=OK | 0.4833 | 8/12 | 74 | 12x15kg=180kg > 120kg -> max 8 boxes |
| 5 | hard_weight_greedy_trap | **PASS** | valid=OK, min_coverage=OK | 0.5830 | 8/9 | 55 | Greedy trap: 4 big light + 5 small heavy, limit 80kg |

**5/5 passed**

### H2. Support 60% Stress

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | hard_support_borderline | **PASS** | valid=OK, min_coverage=OK | 0.8333 | 6/6 | 18 | Top bridgers need >=60% support from base grid |
| 2 | hard_support_staircase | **PASS** | valid=OK, min_coverage=OK | 0.7000 | 12/12 | 85 | 12 step blocks - staircase packing challenge |
| 3 | hard_support_platform | **PASS** | valid=OK, min_coverage=OK | 0.7344 | 14/14 | 119 | Large top boxes need platform of small bases |
| 4 | hard_support_height_mismatch | **PASS** | valid=OK, min_coverage=OK | 0.7812 | 9/9 | 42 | Mixed base heights - top can't lay flat easily |
| 5 | hard_support_mosaic_floor | **PASS** | valid=OK, min_coverage=OK | 0.6667 | 66/66 | 442 | 64 small tiles must form solid floor for 2 heavy slabs |

**5/5 passed**

### H3. Fragility Challenges

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | hard_fragile_all | **PASS** | valid=OK, min_coverage=OK | 0.6883 | 20/20 | 291 | All 20 boxes fragile+heavy(5kg) -> every stack = violation |
| 2 | hard_fragile_big_bottom_dilemma | **PASS** | valid=OK, min_coverage=OK, min_fragility=OK | 0.6875 | 12/12 | 104 | Big fragile best on bottom by volume, but must go on top |
| 3 | hard_fragile_interleave | **PASS** | valid=OK, min_coverage=OK, min_fragility=OK | 0.9500 | 18/18 | 148 | Heavy-fragile-heavy sandwich -> naive layering creates violations |
| 4 | hard_fragile_singleton | **PASS** | valid=OK, min_coverage=OK, min_fragility=OK | 0.6276 | 9/9 | 80 | 1 precious fragile among 8 heavy -> must go on top |
| 5 | hard_fragile_nonstackable | **PASS** | valid=OK, min_coverage=OK, min_fragility=OK | 0.7344 | 18/18 | 189 | Fragile + non-stackable: double constraint |

**5/5 passed**

### H4. Upright + Non-Stackable

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | hard_upright_all | **PASS** | valid=OK, min_coverage=OK | 0.7734 | 12/12 | 80 | All 12 boxes upright-only (350x250x400) |
| 2 | hard_upright_gap_fill | **PASS** | valid=OK, min_coverage=OK | 0.7734 | 26/26 | 365 | Upright tall (500mm) + flexible fillers |
| 3 | hard_nostack_tall | **PASS** | valid=OK, min_coverage=OK | 0.7000 | 35/36 | 330 | 6 tall non-stackable (600mm) + 30 short fillers |
| 4 | hard_triple_constraint | **PASS** | valid=OK, min_coverage=OK, min_fragility=OK | 0.6312 | 18/18 | 247 | Triple: upright + fragile non-stackable + normal |
| 5 | hard_nostack_flat | **PASS** | valid=OK, min_coverage=OK | 0.5917 | 18/18 | 165 | 8 flat non-stackable (100mm) waste ceiling + 10 cubes |

**5/5 passed**

### H5. Tight Fit / Near-Impossible

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | hard_tetris_exact | **PASS** | valid=OK, min_coverage=OK | 0.8875 | 5/8 | 12 | Tetris exact: 4 half-slabs + 4 quarter-blocks |
| 2 | hard_oversized | **PASS** | valid=OK, min_coverage=OK | 0.6019 | 12/13 | 119 | 2 too-wide + 1 too-tall upright can't fit |
| 3 | hard_diverse_sizes | **PASS** | valid=OK, min_coverage=OK | 0.9042 | 73/73 | 476 | 5 size tiers (100mm-800mm), 73 items |
| 4 | hard_low_ceiling | **PASS** | valid=OK, min_coverage=OK | 0.9159 | 20/24 | 127 | Low ceiling 220mm - single layer |
| 5 | hard_near_full | **PASS** | valid=OK, min_coverage=OK | 0.8600 | 8/15 | 56 | Near full pallet with odd fillers |

**5/5 passed**

### H6. Mixed Constraint Chaos

| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |
|---|-----------|---------|--------|-------|--------|---------|-------------|
| 1 | hard_chaos_retail | **FAIL** | valid=OK, min_coverage=OK, min_fragility=FAIL(0.0!=0.5) | 0.8298 | 58/65 | 525 | Retail nightmare: 5 SKU, all constraints, 800kg limit |
| 2 | hard_chaos_heavy_fragile | **PASS** | valid=OK, min_coverage=OK, min_fragility=OK | 0.8224 | 22/22 | 287 | Heavy fragile eggs (22kg, upright) + support + medium |
| 3 | hard_chaos_all_nostack | **PASS** | valid=OK, min_coverage=OK | 0.5672 | 17/18 | 134 | All non-stackable: only one layer |
| 4 | hard_chaos_squeeze | **PASS** | valid=OK, min_coverage=OK, min_fragility=OK | 0.8583 | 24/24 | 214 | Low ceiling + weight limit + upright + fragile |
| 5 | hard_chaos_maximum | **FAIL** | valid=OK, min_coverage=OK, min_fragility=FAIL(0.0!=0.4) | 0.7935 | 78/81 | 843 | FINAL BOSS: 8 SKU, all archetypes, all constraints |

**3/5 passed**

---

## Summary

### Constraint Tests: 74/76 passed (97.4%)

| Constraint Group | Passed | Total | Rate |
|-----------------|--------|-------|------|
| dimensions | 6 | 6 | 100% |
| collisions | 3 | 3 | 100% |
| support | 4 | 4 | 100% |
| upright | 4 | 4 | 100% |
| volume | 4 | 4 | 100% |
| coverage | 6 | 6 | 100% |
| fragility | 7 | 7 | 100% |
| time | 4 | 4 | 100% |
| weight | 3 | 3 | 100% |
| stackable | 3 | 3 | 100% |
| combined | 2 | 2 | 100% |
| hard_weight | 5 | 5 | 100% |
| hard_support | 5 | 5 | 100% |
| hard_fragile | 5 | 5 | 100% |
| hard_upright | 5 | 5 | 100% |
| hard_tightfit | 5 | 5 | 100% |
| hard_chaos | 3 | 5 | 60% |

### Failed Tests Analysis

| Test | Issue | Root Cause |
|------|-------|------------|
| hard_chaos_retail | fragility=0.0 (need >=0.5) | 5 SKU all-constraint scenario; LNS/postprocess doesn't fully resolve heavy-on-fragile |
| hard_chaos_maximum | fragility=0.0 (need >=0.4) | 8 SKU all-archetype scenario; too many fragility violations in dense mixed packing |
