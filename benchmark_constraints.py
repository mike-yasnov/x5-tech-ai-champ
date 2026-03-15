"""Constraint stress-tests: edge cases and boundary values for every hard/soft constraint.

Each scenario is hand-crafted with a known optimal or expected outcome so that
we can assert the solver handles the constraint correctly.

Usage:
    python benchmark_constraints.py [--strategy portfolio_block] [--output results.json]
"""

import argparse
import json
import time
from typing import Any, Dict, List, Optional

from solver.models import Pallet, Box, solution_to_dict
from solver.solver import solve
from validator import evaluate_solution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    task_id: str,
    pallet: Dict[str, Any],
    boxes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {"task_id": task_id, "pallet": pallet, "boxes": boxes}


def _pallet(l: int, w: int, h: int, wt: float = 10000.0) -> Dict[str, Any]:
    return {
        "type_id": "TEST",
        "length_mm": l,
        "width_mm": w,
        "max_height_mm": h,
        "max_weight_kg": wt,
    }


def _box(
    sku: str,
    l: int, w: int, h: int,
    wt: float = 1.0,
    qty: int = 1,
    upright: bool = False,
    fragile: bool = False,
    stackable: bool = True,
) -> Dict[str, Any]:
    return {
        "sku_id": sku,
        "description": sku,
        "length_mm": l,
        "width_mm": w,
        "height_mm": h,
        "weight_kg": wt,
        "quantity": qty,
        "strict_upright": upright,
        "fragile": fragile,
        "stackable": stackable,
    }


def _request_to_models(request_dict: dict):
    p = request_dict["pallet"]
    pallet = Pallet(
        type_id=p.get("type_id", "TEST"),
        length_mm=p["length_mm"],
        width_mm=p["width_mm"],
        max_height_mm=p["max_height_mm"],
        max_weight_kg=p["max_weight_kg"],
    )
    boxes = []
    for b in request_dict["boxes"]:
        boxes.append(Box(
            sku_id=b["sku_id"],
            description=b.get("description", ""),
            length_mm=b["length_mm"],
            width_mm=b["width_mm"],
            height_mm=b["height_mm"],
            weight_kg=b["weight_kg"],
            quantity=b["quantity"],
            strict_upright=b.get("strict_upright", False),
            fragile=b.get("fragile", False),
            stackable=b.get("stackable", True),
        ))
    return request_dict["task_id"], pallet, boxes


# ===========================================================================
# CONSTRAINT SCENARIOS
# ===========================================================================
# Each entry: (request_dict, expectations_dict)
#   expectations_dict keys:
#     valid            _ must the solution be valid?
#     min_coverage     _ minimum item_coverage expected
#     exact_coverage   _ exact item_coverage expected (for deterministic cases)
#     min_volume       _ minimum volume_utilization expected
#     exact_volume     _ exact volume_utilization expected
#     min_fragility    _ minimum fragility_score expected
#     max_time_ms      _ maximum solve_time_ms allowed
#     description      _ human-readable description
#     constraint_group _ which constraint is being tested


def build_constraint_scenarios() -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    # ===================================================================
    # 1. DIMENSIONS (Length / Width / Height) _ boundary values
    # ===================================================================

    # 1.1 Single box exactly fills pallet -> volume = 1.0, coverage = 1.0
    scenarios.append({
        "request": _make_request(
            "dim_exact_fill_1box",
            _pallet(600, 400, 200),
            [_box("SKU-EXACT", 600, 400, 200, qty=1)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "dimensions",
            "description": "1 box = pallet size -> perfect fit",
        },
    })

    # 1.2 Four boxes tile the floor in one layer -> volume = 1.0
    scenarios.append({
        "request": _make_request(
            "dim_4tile_floor",
            _pallet(600, 400, 200),
            [_box("SKU-TILE", 300, 200, 200, qty=4)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "dimensions",
            "description": "4 boxes tile 600x400 floor exactly (1 layer)",
        },
    })

    # 1.3 Two layers, 8 boxes -> volume = 1.0
    scenarios.append({
        "request": _make_request(
            "dim_8tile_2layers",
            _pallet(600, 400, 400),
            [_box("SKU-TILE8", 300, 200, 200, qty=8)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "dimensions",
            "description": "8 boxes tile 600x400x400 exactly (2 layers)",
        },
    })

    # 1.4 Box barely fits height _ single box h = max_height
    scenarios.append({
        "request": _make_request(
            "dim_max_height_exact",
            _pallet(300, 200, 500),
            [_box("SKU-TALL", 300, 200, 500, qty=1)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "dimensions",
            "description": "Box height == pallet max_height exactly",
        },
    })

    # 1.5 Minimum-size box (1x1x1)
    scenarios.append({
        "request": _make_request(
            "dim_tiny_box",
            _pallet(100, 100, 100),
            [_box("SKU-TINY", 1, 1, 1, wt=0.001, qty=1)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "dimensions",
            "description": "Minimum-size box (1x1x1 mm)",
        },
    })

    # 1.6 Box needs rotation to fit (L>pallet_L but W<pallet_L)
    # Pallet 600x400, box 500x300x200 _ fits as-is.
    # But box 700x300x200 won't fit in 600-wide pallet unless rotated...
    # Actually the box must be a permutation. Let's use 400x300x600 box in 600x400x300 pallet.
    # The box (L=400, W=300, H=600) can only fit if placed as (600, 400, 300) -> but 600>400(W).
    # Better: pallet 600x400x300, box L=300 W=200 H=600 -> need rotation HLW -> (600,300,200) fits.
    scenarios.append({
        "request": _make_request(
            "dim_rotation_needed",
            _pallet(600, 400, 300),
            [_box("SKU-ROTATE", 300, 200, 600, wt=5.0, qty=1, upright=False)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "dimensions",
            "description": "Box must be rotated to fit within height limit",
        },
    })

    # ===================================================================
    # 2. NO COLLISIONS _ tight packing stress
    # ===================================================================

    # 2.1 Many identical cubes filling a volume exactly
    scenarios.append({
        "request": _make_request(
            "collision_cubes_27",
            _pallet(300, 300, 300),
            [_box("SKU-CUBE", 100, 100, 100, qty=27)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "collisions",
            "description": "27 cubes must fill 300^3 without any overlap",
        },
    })

    # 2.2 Two different sizes sharing floor _ no gaps possible between them
    scenarios.append({
        "request": _make_request(
            "collision_two_sizes",
            _pallet(600, 400, 200),
            [
                _box("SKU-BIG", 400, 400, 200, qty=1),
                _box("SKU-SML", 200, 200, 200, qty=2),
            ],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "collisions",
            "description": "Two sizes share floor: 400x400 + 2x200x200 fill 600x400",
        },
    })

    # 2.3 Stress _ 64 small cubes in 4x4x4 grid
    scenarios.append({
        "request": _make_request(
            "collision_cubes_64",
            _pallet(400, 400, 400),
            [_box("SKU-C64", 100, 100, 100, qty=64)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "collisions",
            "description": "64 cubes fill 400^3 _ collision-free grid",
        },
    })

    # ===================================================================
    # 3. SUPPORT FROM BELOW (>= 60% base area)
    # ===================================================================

    # 3.1 Second-layer box has exactly 100% support
    scenarios.append({
        "request": _make_request(
            "support_full",
            _pallet(600, 400, 400),
            [_box("SKU-SUP", 600, 400, 200, qty=2)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "support",
            "description": "Two stacked boxes: upper has 100% support",
        },
    })

    # 3.2 Upper box overhangs but still >= 60% support
    # Base: 600x400 at z=0. Upper: 600x400 shifted by 160mm in X -> overlap = 440/600 = 73%
    # But solver can't place with shift _ we design so solver CAN achieve full support.
    # Better: base 400x400, upper 600x400 -> base covers 400/600 = 66.7% of upper base area.
    scenarios.append({
        "request": _make_request(
            "support_borderline_ok",
            _pallet(600, 400, 400),
            [
                _box("SKU-BASE", 600, 400, 200, qty=1),
                _box("SKU-TOP", 400, 400, 200, qty=1),
            ],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "support",
            "description": "Upper box fits entirely on base -> 100% support easy",
        },
    })

    # 3.3 Tower of 5 identical boxes _ each level needs support check
    scenarios.append({
        "request": _make_request(
            "support_tower_5",
            _pallet(200, 200, 1000),
            [_box("SKU-TOWER", 200, 200, 200, qty=5)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "support",
            "description": "Tower of 5 identical boxes _ each layer supported by one below",
        },
    })

    # 3.4 Upper box needs support from TWO base boxes (bridging)
    # Base: two 300x400 side by side -> cover 600x400.
    # Upper: one 600x400 on top -> fully supported by two bases combined.
    scenarios.append({
        "request": _make_request(
            "support_bridging",
            _pallet(600, 400, 400),
            [
                _box("SKU-HALF", 300, 400, 200, qty=2),
                _box("SKU-BRIDGE", 600, 400, 200, qty=1),
            ],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "support",
            "description": "Upper box bridges two base boxes for combined support",
        },
    })

    # ===================================================================
    # 4. STRICT UPRIGHT _ rotation constraint
    # ===================================================================

    # 4.1 Upright box that fits only in native orientation
    scenarios.append({
        "request": _make_request(
            "upright_native_only",
            _pallet(600, 400, 300),
            [_box("SKU-UPR", 200, 150, 300, qty=1, upright=True)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "upright",
            "description": "strict_upright box placed in native H orientation",
        },
    })

    # 4.2 All boxes strict_upright, non-cube -> solver must respect H axis
    scenarios.append({
        "request": _make_request(
            "upright_all_boxes",
            _pallet(600, 400, 200),
            [_box("SKU-UP-ALL", 300, 200, 200, qty=4, upright=True)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 1.0,
            "constraint_group": "upright",
            "description": "All 4 boxes strict_upright _ must keep H on Z axis",
        },
    })

    # 4.3 Mix: some upright, some free to rotate
    scenarios.append({
        "request": _make_request(
            "upright_mixed",
            _pallet(600, 400, 400),
            [
                _box("SKU-FIXED", 300, 200, 200, qty=4, upright=True),
                _box("SKU-FREE", 200, 200, 200, qty=4, upright=False),
            ],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 5.0 / 6.0,
            "constraint_group": "upright",
            "description": "Mix of strict_upright and freely rotatable boxes",
        },
    })

    # 4.4 Upright box that could fit more if rotated _ but must not be rotated
    # Pallet 600x400x200. Box L=150 W=100 H=200 (upright). qty=8.
    # Upright: LxW on floor = 150x100, so 4x4=16 per layer but only 1 layer -> max 16.
    # But height exactly matches, so 8 boxes easily fit on floor.
    scenarios.append({
        "request": _make_request(
            "upright_no_extra_rotation",
            _pallet(600, 400, 200),
            [_box("SKU-NOROT", 150, 100, 200, qty=8, upright=True)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "upright",
            "description": "Upright boxes where rotation is tempting but forbidden",
        },
    })

    # ===================================================================
    # 5. VOLUME UTILIZATION _ known achievable values
    # ===================================================================

    # 5.1 Perfect volume = 1.0 (redundant with dim tests, but explicit)
    scenarios.append({
        "request": _make_request(
            "volume_perfect",
            _pallet(600, 400, 200),
            [_box("SKU-VPERF", 600, 400, 200, qty=1)],
        ),
        "expect": {
            "valid": True,
            "exact_volume": 1.0,
            "constraint_group": "volume",
            "description": "Volume utilization = 1.0 (perfect fill)",
        },
    })

    # 5.2 Known 50% volume _ 1 box fills half the pallet
    # Pallet 600x400x200 = 48M mm^3. Box 600x400x100 = 24M -> 50%.
    scenarios.append({
        "request": _make_request(
            "volume_50pct",
            _pallet(600, 400, 200),
            [_box("SKU-HALF-V", 600, 400, 100, qty=1)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 0.5,
            "constraint_group": "volume",
            "description": "Volume = 0.5 (box fills half of pallet height)",
        },
    })

    # 5.3 Known 25% volume _ small box in big pallet
    # Pallet 400x400x400 = 64M. Box 200x200x400 = 16M -> 25%.
    scenarios.append({
        "request": _make_request(
            "volume_25pct",
            _pallet(400, 400, 400),
            [_box("SKU-QUARTER", 200, 200, 400, qty=1)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 0.25,
            "constraint_group": "volume",
            "description": "Volume = 0.25 (one box is 1/4 of pallet volume)",
        },
    })

    # 5.4 Known 75% volume _ 6 out of 8 cells filled
    # Pallet 400x400x200. Box 200x200x100 qty=12 -> fills 200x200x100x12 = 48M.
    # Pallet vol = 32M -> 48/32 > 1 won't work.
    # Better: Pallet 400x200x200 = 16M. Box 200x200x200 qty=3 -> 24M > 16M.
    # Pallet 600x400x200=48M. 3 boxes of 300x400x200 = 3x24M = 72M > 48M.
    # Pallet 800x400x200=64M. 3 boxes of 400x400x200 = 3x32M = 96M > 64M.
    # Pallet 400x400x400=64M. 3 cubes 200x200x400 = 3x16M = 48M -> 48/64 = 0.75.
    scenarios.append({
        "request": _make_request(
            "volume_75pct",
            _pallet(400, 400, 400),
            [_box("SKU-3Q", 200, 200, 400, qty=3)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "exact_volume": 0.75,
            "constraint_group": "volume",
            "description": "Volume = 0.75 (3 tall boxes fill 3/4 of pallet)",
        },
    })

    # 5.5 Two SKUs, total volume = 62.5%
    # Pallet 800x400x200 = 64M.
    # SKU-A: 400x400x200 qty=1 -> 32M (50%)
    # SKU-B: 200x200x200 qty=2 -> 2x8M = 16M (12.5% each, but one might not fit on the 400x400 remaining area)
    # Actually remaining space is 400x400x200 -> 2 boxes of 200x200x200 fit on the same layer.
    # Total = 32M + 16M = 48M / 64M = 0.75. Wait, let me recalc.
    # Hmm. Let's keep it simpler.
    # Pallet 400x200x200 = 16M. SKU-A: 200x200x200 qty=1 -> 8M. SKU-B: 200x100x200 qty=2 -> 2x4M=8M.
    # Total = 16M/16M = 1.0. Not what I want.
    # Pallet 400x200x400 = 32M. SKU-A: 200x200x200 qty=2 -> 16M. SKU-B: 200x200x200 qty=1 -> 8M.
    # Total = 24M/32M = 0.75.
    # Let me do exact 5/8 = 0.625:
    # Pallet 400x400x200 = 32M. Box 200x200x200 qty=5 -> 40M > 32M.
    # Only 4 fit on floor. 5th needs 2nd layer but height=200.
    # OK, let's just skip this one. The prior tests cover known volumes well.

    # ===================================================================
    # 6. ITEM COVERAGE _ known achievable values
    # ===================================================================

    # 6.1 All items fit -> coverage = 1.0 (covered above, explicit here)
    scenarios.append({
        "request": _make_request(
            "coverage_full",
            _pallet(600, 400, 200),
            [_box("SKU-COVF", 300, 200, 200, qty=4)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "coverage",
            "description": "All items fit -> coverage = 1.0",
        },
    })

    # 6.2 Only 1 of 2 fits (too heavy) -> coverage = 0.5
    # Pallet max_weight = 5.0 kg. Box weight = 5.0. qty=2. Only 1 fits by weight.
    scenarios.append({
        "request": _make_request(
            "coverage_weight_limited",
            _pallet(600, 400, 200, wt=5.0),
            [_box("SKU-HVY", 300, 200, 200, wt=5.0, qty=2)],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.5,
            "constraint_group": "coverage",
            "description": "Weight limit allows only 1 of 2 boxes -> coverage >= 0.5",
        },
    })

    # 6.3 Height limit: only 1 layer of 2 requested layers fits -> coverage = 0.5
    # Pallet height 200. Box height 200 qty=2 (same footprint). Only 1 fits.
    # Actually if footprint < pallet, 2 can fit side by side. Use full-floor box.
    scenarios.append({
        "request": _make_request(
            "coverage_height_limited",
            _pallet(600, 400, 200),
            [_box("SKU-HLIM", 600, 400, 200, qty=2)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 0.5,
            "constraint_group": "coverage",
            "description": "Only 1 of 2 full-floor boxes fits (height limit) -> coverage = 0.5",
        },
    })

    # 6.4 Space-limited: 5 boxes requested, only 4 fit on floor, no second layer
    # Pallet 400x400x200. Box 200x200x200 qty=5. Floor fits 4. Height=200 -> no second layer.
    scenarios.append({
        "request": _make_request(
            "coverage_space_limited",
            _pallet(400, 400, 200),
            [_box("SKU-SLIM", 200, 200, 200, qty=5)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 0.8,
            "constraint_group": "coverage",
            "description": "5 boxes requested, only 4 fit on floor -> coverage = 0.8",
        },
    })

    # 6.5 Coverage with two SKUs: large takes most space, small partially fits
    # Pallet 600x400x200 = floor 600x400.
    # SKU-A: 600x200x200 qty=1 -> takes half the floor.
    # SKU-B: 200x200x200 qty=4 -> remaining 600x200 fits 3x200x200 = 3 boxes.
    # Total placed: 1+3=4 of 5 total -> 4/5=0.8.
    scenarios.append({
        "request": _make_request(
            "coverage_mixed_partial",
            _pallet(600, 400, 200),
            [
                _box("SKU-WIDE", 600, 200, 200, qty=1),
                _box("SKU-SM", 200, 200, 200, qty=4),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.8,
            "constraint_group": "coverage",
            "description": "Mixed SKUs: large + small; 4 of 5 items fit -> coverage >= 0.8",
        },
    })

    # 6.6 Single oversized box -> coverage = 0.0
    # Box bigger than pallet in every dimension. Solver should place 0 items.
    scenarios.append({
        "request": _make_request(
            "coverage_zero_oversize",
            _pallet(200, 200, 200),
            [_box("SKU-HUGE", 300, 300, 300, qty=1)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 0.0,
            "constraint_group": "coverage",
            "description": "Box larger than pallet in all dims -> coverage = 0.0",
        },
    })

    # ===================================================================
    # 7. FRAGILITY _ heavy-on-fragile penalty
    # ===================================================================

    # 7.1 No fragile items -> fragility_score = 1.0
    scenarios.append({
        "request": _make_request(
            "fragile_none",
            _pallet(600, 400, 400),
            [_box("SKU-NOFR", 300, 200, 200, wt=10.0, qty=4)],
        ),
        "expect": {
            "valid": True,
            "min_fragility": 1.0,
            "constraint_group": "fragility",
            "description": "No fragile boxes -> fragility_score = 1.0",
        },
    })

    # 7.2 All fragile, no heavy (_2kg) on top -> fragility = 1.0
    scenarios.append({
        "request": _make_request(
            "fragile_all_light",
            _pallet(600, 400, 400),
            [_box("SKU-FRLT", 300, 200, 200, wt=1.5, qty=4, fragile=True)],
        ),
        "expect": {
            "valid": True,
            "min_fragility": 1.0,
            "constraint_group": "fragility",
            "description": "All fragile but light (_2kg) -> no violations -> fragility = 1.0",
        },
    })

    # 7.3 Fragile bottom + heavy top _ solver should reorder to avoid penalty
    # 2 fragile light boxes + 2 heavy non-fragile boxes. Ideal: heavy on bottom.
    scenarios.append({
        "request": _make_request(
            "fragile_heavy_on_fragile",
            _pallet(600, 400, 400),
            [
                _box("SKU-FRAG-BOT", 600, 400, 200, wt=1.0, qty=1, fragile=True),
                _box("SKU-HEAVY-TOP", 600, 400, 200, wt=10.0, qty=1),
            ],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "min_fragility": 0.95,
            "constraint_group": "fragility",
            "description": "Heavy + fragile: solver should place heavy below fragile",
        },
    })

    # 7.4 Multiple fragile items on bottom, multiple heavy on top _ worst case
    # If solver places heavy on fragile: each pair = 1 violation. 4 heavy x 2 fragile (with overlap) -> up to 8.
    # Solver should reorder to minimize violations.
    scenarios.append({
        "request": _make_request(
            "fragile_multi_stress",
            _pallet(600, 400, 600),
            [
                _box("SKU-FR-BASE", 300, 200, 200, wt=0.5, qty=4, fragile=True),
                _box("SKU-HV-STACK", 300, 200, 200, wt=5.0, qty=4),
            ],
        ),
        "expect": {
            "valid": True,
            "min_fragility": 0.8,
            "constraint_group": "fragility",
            "description": "4 fragile + 4 heavy boxes: solver should minimize fragility violations",
        },
    })

    # 7.5 Heavy on fragile unavoidable (fragile must be on bottom due to space)
    # Pallet 300x200x400. 1 fragile 300x200x200 + 1 heavy 300x200x200.
    # Both fill the full floor. One must be on bottom. If fragile on bottom -> 1 violation.
    # Solver should place heavy on bottom, fragile on top -> 0 violations.
    scenarios.append({
        "request": _make_request(
            "fragile_order_matters",
            _pallet(300, 200, 400),
            [
                _box("SKU-FR", 300, 200, 200, wt=1.0, qty=1, fragile=True),
                _box("SKU-HV", 300, 200, 200, wt=8.0, qty=1),
            ],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "min_fragility": 1.0,
            "constraint_group": "fragility",
            "description": "2 full-floor boxes: solver must put heavy below fragile",
        },
    })

    # 7.6 Fragile boundary: weight exactly 2.0 kg on fragile -> no violation (>2.0 required)
    scenarios.append({
        "request": _make_request(
            "fragile_boundary_2kg",
            _pallet(300, 200, 400),
            [
                _box("SKU-FR-BOT", 300, 200, 200, wt=0.5, qty=1, fragile=True),
                _box("SKU-2KG", 300, 200, 200, wt=2.0, qty=1),
            ],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "min_fragility": 1.0,
            "constraint_group": "fragility",
            "description": "2.0 kg box on fragile -> no violation (threshold is >2.0)",
        },
    })

    # 7.7 Fragile boundary: weight 2.01 kg on fragile -> 1 violation if placed on top
    scenarios.append({
        "request": _make_request(
            "fragile_boundary_2_01kg",
            _pallet(300, 200, 400),
            [
                _box("SKU-FR-B2", 300, 200, 200, wt=0.5, qty=1, fragile=True),
                _box("SKU-201", 300, 200, 200, wt=2.01, qty=1),
            ],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "min_fragility": 0.95,
            "constraint_group": "fragility",
            "description": "2.01 kg on fragile: solver should put heavy below -> fragility=1.0",
        },
    })

    # ===================================================================
    # 8. EXECUTION TIME _ performance boundaries
    # ===================================================================

    # 8.1 Trivial case -> must be < 1 second
    scenarios.append({
        "request": _make_request(
            "time_trivial",
            _pallet(600, 400, 200),
            [_box("SKU-TRIV", 600, 400, 200, qty=1)],
        ),
        "expect": {
            "valid": True,
            "max_time_ms": 1000,
            "constraint_group": "time",
            "description": "Single box -> solve < 1s",
        },
    })

    # 8.2 Medium load (20 items) -> must be < 5 seconds
    scenarios.append({
        "request": _make_request(
            "time_medium_20",
            _pallet(1200, 800, 1800),
            [_box("SKU-MED", 300, 200, 200, wt=3.0, qty=20)],
        ),
        "expect": {
            "valid": True,
            "max_time_ms": 5000,
            "constraint_group": "time",
            "description": "20 items -> solve < 5s",
        },
    })

    # 8.3 Large load (100 items) -> must be < 30 seconds
    scenarios.append({
        "request": _make_request(
            "time_large_100",
            _pallet(1200, 800, 1800, wt=5000.0),
            [_box("SKU-LRG", 200, 150, 100, wt=1.0, qty=100)],
        ),
        "expect": {
            "valid": True,
            "max_time_ms": 30000,
            "constraint_group": "time",
            "description": "100 items -> solve < 30s",
        },
    })

    # 8.4 Stress: 200 items with mixed SKUs
    scenarios.append({
        "request": _make_request(
            "time_stress_200",
            _pallet(1200, 800, 2000, wt=10000.0),
            [
                _box("SKU-S1", 300, 200, 150, wt=2.0, qty=50),
                _box("SKU-S2", 200, 150, 100, wt=1.5, qty=50),
                _box("SKU-S3", 250, 200, 200, wt=3.0, qty=50),
                _box("SKU-S4", 150, 100, 100, wt=0.8, qty=50),
            ],
        ),
        "expect": {
            "valid": True,
            "max_time_ms": 30000,
            "constraint_group": "time",
            "description": "200 items, 4 SKUs -> solve < 30s",
        },
    })

    # ===================================================================
    # 9. WEIGHT CONSTRAINT _ edge cases
    # ===================================================================

    # 9.1 Total weight exactly at limit
    scenarios.append({
        "request": _make_request(
            "weight_exact_limit",
            _pallet(600, 400, 200, wt=40.0),
            [_box("SKU-WEX", 300, 200, 200, wt=10.0, qty=4)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "weight",
            "description": "Total weight (40kg) = max_weight_kg exactly -> all fit",
        },
    })

    # 9.2 Weight limit forces partial placement
    # 3 boxes x 10kg = 30kg, limit = 25kg -> only 2 fit
    scenarios.append({
        "request": _make_request(
            "weight_partial",
            _pallet(600, 400, 600, wt=25.0),
            [_box("SKU-WPT", 200, 200, 200, wt=10.0, qty=3)],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.66,
            "constraint_group": "weight",
            "description": "Weight limit 25kg, 3x10kg boxes -> only 2 fit -> coverage >= 0.66",
        },
    })

    # 9.3 Single box exceeds weight limit -> cannot place
    scenarios.append({
        "request": _make_request(
            "weight_single_overweight",
            _pallet(600, 400, 200, wt=5.0),
            [_box("SKU-OW", 300, 200, 200, wt=10.0, qty=1)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 0.0,
            "constraint_group": "weight",
            "description": "Single box exceeds max_weight -> coverage = 0.0",
        },
    })

    # ===================================================================
    # 10. NON-STACKABLE _ nothing on top
    # ===================================================================

    # 10.1 Single non-stackable box _ nothing can be on top
    scenarios.append({
        "request": _make_request(
            "stackable_single",
            _pallet(600, 400, 400),
            [_box("SKU-NS", 600, 400, 200, wt=5.0, qty=1, stackable=False)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "stackable",
            "description": "Single non-stackable box -> valid (nothing on top)",
        },
    })

    # 10.2 Non-stackable on floor + stackable beside it (not on top)
    scenarios.append({
        "request": _make_request(
            "stackable_beside",
            _pallet(600, 400, 200),
            [
                _box("SKU-NS2", 300, 400, 200, wt=5.0, qty=1, stackable=False),
                _box("SKU-STK", 300, 400, 200, wt=5.0, qty=1),
            ],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "stackable",
            "description": "Non-stackable + stackable side by side -> both fit",
        },
    })

    # 10.3 Non-stackable caps scenario: some boxes can't have anything on top
    scenarios.append({
        "request": _make_request(
            "stackable_caps_mixed",
            _pallet(600, 400, 600),
            [
                _box("SKU-BASE-S", 300, 200, 200, wt=5.0, qty=4),
                _box("SKU-CAP", 300, 200, 200, wt=2.0, qty=2, stackable=False),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.83,
            "constraint_group": "stackable",
            "description": "4 stackable + 2 non-stackable caps -> solver respects constraint",
        },
    })

    # ===================================================================
    # 11. COMBINED CONSTRAINTS _ multiple constraints active simultaneously
    # ===================================================================

    # 11.1 Upright + fragile + weight limit
    scenarios.append({
        "request": _make_request(
            "combined_upright_fragile_weight",
            _pallet(600, 400, 400, wt=30.0),
            [
                _box("SKU-UF", 200, 150, 200, wt=3.0, qty=4, upright=True, fragile=True),
                _box("SKU-HB", 300, 200, 200, wt=8.0, qty=2),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.5,
            "min_fragility": 0.8,
            "constraint_group": "combined",
            "description": "Upright fragile boxes + heavy boxes + weight limit",
        },
    })

    # 11.2 Non-stackable + fragile + upright
    scenarios.append({
        "request": _make_request(
            "combined_ns_fragile_upright",
            _pallet(600, 400, 400),
            [
                _box("SKU-BASE-C", 300, 200, 200, wt=8.0, qty=4),
                _box("SKU-CAP-FU", 300, 200, 200, wt=1.0, qty=2,
                     upright=True, fragile=True, stackable=False),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.83,
            "min_fragility": 1.0,
            "constraint_group": "combined",
            "description": "Non-stackable + fragile + upright caps on stackable base",
        },
    })

    return scenarios


def build_hard_scenarios() -> List[Dict[str, Any]]:
    """30 hard edge-case stress tests."""
    scenarios: List[Dict[str, Any]] = []

    # ===================================================================
    # Cat 1: Weight Limit Edge Cases (1-5)
    # ===================================================================

    # 1. hard_weight_exact: 4x25kg = 100kg = max_weight
    scenarios.append({
        "request": _make_request(
            "hard_weight_exact",
            _pallet(1200, 800, 1800, wt=100.0),
            [_box("SKU-HEAVYBLK", 400, 400, 200, wt=25.0, qty=4)],
        ),
        "expect": {
            "valid": True,
            "exact_coverage": 1.0,
            "constraint_group": "hard_weight",
            "description": "4x25kg = 100kg = max_weight exactly -> all must fit",
        },
    })

    # 2. hard_weight_overflow_1: 6x10kg = 60kg > 50kg -> 1 unplaced
    scenarios.append({
        "request": _make_request(
            "hard_weight_overflow_1",
            _pallet(1200, 800, 1800, wt=50.0),
            [_box("SKU-DENSE", 200, 200, 200, wt=10.0, qty=6)],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.83,  # 5/6
            "constraint_group": "hard_weight",
            "description": "6x10kg=60kg > 50kg limit -> at least 5 placed",
        },
    })

    # 3. hard_weight_one_heavy: 1 anchor (180kg) + 50 light (0.5kg)
    # Anchor uses 180/200 = 90% weight budget. Remaining 20kg -> 40 light boxes.
    scenarios.append({
        "request": _make_request(
            "hard_weight_one_heavy",
            _pallet(1200, 800, 1800, wt=200.0),
            [
                _box("SKU-LIGHT", 100, 100, 100, wt=0.5, qty=50),
                _box("SKU-ANCHOR", 600, 400, 300, wt=180.0, qty=1),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.8,  # anchor + 40 light = 41/51
            "constraint_group": "hard_weight",
            "description": "1 anchor (180kg) + 50 light (0.5kg), limit 200kg",
        },
    })

    # 4. hard_weight_uniform_cutoff: 12x15kg = 180kg > 120kg -> max 8
    scenarios.append({
        "request": _make_request(
            "hard_weight_uniform_cutoff",
            _pallet(1200, 800, 600, wt=120.0),
            [_box("SKU-UNIFORM", 300, 200, 200, wt=15.0, qty=12)],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.66,  # 8/12
            "constraint_group": "hard_weight",
            "description": "12x15kg=180kg > 120kg -> max 8 boxes",
        },
    })

    # 5. hard_weight_greedy_trap: big light vs small heavy
    scenarios.append({
        "request": _make_request(
            "hard_weight_greedy_trap",
            _pallet(1200, 800, 1800, wt=80.0),
            [
                _box("SKU-BIG-LT", 600, 400, 400, wt=2.0, qty=4),
                _box("SKU-SM-HV", 200, 150, 150, wt=18.0, qty=5),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.55,  # at least 5/9
            "constraint_group": "hard_weight",
            "description": "Greedy trap: 4 big light (2kg) + 5 small heavy (18kg), limit 80kg",
        },
    })

    # ===================================================================
    # Cat 2: Support 60% Rule Stress Tests (6-10)
    # ===================================================================

    # 6. hard_support_borderline: 2x2 base + 2 top bridgers
    # Base: 4x 600x400x200. Top: 2x 600x800x200 bridging pairs.
    # Actually let's be more concrete:
    # Base: 4x 300x400x200 filling floor 1200x400. Wait, pallet is 1200x800.
    # Base: 4x 600x400x200 tiling 1200x800 floor. Top: 2x 600x400x200 on top.
    scenarios.append({
        "request": _make_request(
            "hard_support_borderline",
            _pallet(1200, 800, 600, wt=1000.0),
            [
                _box("SKU-BASE4", 600, 400, 200, wt=10.0, qty=4),
                _box("SKU-BRIDGE", 800, 600, 200, wt=8.0, qty=2),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.83,  # at least 5/6
            "constraint_group": "hard_support",
            "description": "Top bridgers need >=60% support from base grid",
        },
    })

    # 7. hard_support_staircase: 12 step blocks
    scenarios.append({
        "request": _make_request(
            "hard_support_staircase",
            _pallet(1200, 800, 1000, wt=1000.0),
            [_box("SKU-STEP", 400, 400, 200, wt=8.0, qty=12)],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.75,  # 9/12
            "constraint_group": "hard_support",
            "description": "12 step blocks (400x400x200) - staircase packing challenge",
        },
    })

    # 8. hard_support_platform: small base + large top
    scenarios.append({
        "request": _make_request(
            "hard_support_platform",
            _pallet(1200, 800, 800, wt=1000.0),
            [
                _box("SKU-SMBASE", 300, 200, 200, wt=4.0, qty=12),
                _box("SKU-BIGTOP", 900, 600, 200, wt=15.0, qty=2),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.85,  # 12/14
            "constraint_group": "hard_support",
            "description": "Large top boxes need platform of small bases for 60% support",
        },
    })

    # 9. hard_support_height_mismatch: mixed heights make support hard
    scenarios.append({
        "request": _make_request(
            "hard_support_height_mismatch",
            _pallet(1200, 800, 800, wt=1000.0),
            [
                _box("SKU-TALL", 400, 400, 300, wt=8.0, qty=3),
                _box("SKU-SHORT", 400, 400, 200, wt=6.0, qty=3),
                _box("SKU-WIDETOP", 800, 400, 200, wt=10.0, qty=3),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.66,  # 6/9
            "constraint_group": "hard_support",
            "description": "Mixed base heights (300mm vs 200mm) - top can't lay flat easily",
        },
    })

    # 10. hard_support_mosaic_floor: 64 tiles + 2 heavy slabs
    scenarios.append({
        "request": _make_request(
            "hard_support_mosaic_floor",
            _pallet(1200, 800, 600, wt=1000.0),
            [
                _box("SKU-TILE", 150, 100, 100, wt=0.5, qty=64),
                _box("SKU-SLAB", 600, 400, 200, wt=30.0, qty=2),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.90,  # 60/66
            "constraint_group": "hard_support",
            "description": "64 small tiles must form solid floor for 2 heavy slabs",
        },
    })

    # ===================================================================
    # Cat 3: Fragility Constraint Challenges (11-15)
    # ===================================================================

    # 11. hard_fragile_all: 20 fragile heavy boxes
    scenarios.append({
        "request": _make_request(
            "hard_fragile_all",
            _pallet(1200, 800, 600, wt=1000.0),
            [_box("SKU-ALLFR", 300, 200, 200, wt=5.0, qty=20, fragile=True)],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.6,  # at least 12/20
            "constraint_group": "hard_fragile",
            "description": "All 20 boxes fragile+heavy(5kg) -> every stack = violation",
        },
    })

    # 12. hard_fragile_big_bottom_dilemma
    scenarios.append({
        "request": _make_request(
            "hard_fragile_big_bottom_dilemma",
            _pallet(1200, 800, 800, wt=1000.0),
            [
                _box("SKU-BIGFR", 600, 400, 200, wt=3.0, qty=4, fragile=True),
                _box("SKU-SMHVY", 300, 200, 200, wt=12.0, qty=8),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.75,  # 9/12
            "min_fragility": 0.7,
            "constraint_group": "hard_fragile",
            "description": "Big fragile (3kg) best on bottom by volume, but must go on top",
        },
    })

    # 13. hard_fragile_interleave
    scenarios.append({
        "request": _make_request(
            "hard_fragile_interleave",
            _pallet(1200, 800, 1000, wt=1000.0),
            [
                _box("SKU-HVBASE", 600, 400, 200, wt=15.0, qty=6),
                _box("SKU-FRMID", 600, 400, 200, wt=1.5, qty=6, fragile=True),
                _box("SKU-HVTOP", 600, 400, 200, wt=15.0, qty=6),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.55,  # 10/18
            "min_fragility": 0.7,
            "constraint_group": "hard_fragile",
            "description": "Heavy-fragile-heavy sandwich -> naive layering creates violations",
        },
    })

    # 14. hard_fragile_singleton
    scenarios.append({
        "request": _make_request(
            "hard_fragile_singleton",
            _pallet(1200, 800, 800, wt=1000.0),
            [
                _box("SKU-HVSTD", 400, 300, 200, wt=15.0, qty=8),
                _box("SKU-PREC", 200, 200, 100, wt=0.5, qty=1, fragile=True),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.88,  # 8/9
            "min_fragility": 1.0,
            "constraint_group": "hard_fragile",
            "description": "1 precious fragile among 8 heavy -> must go on top",
        },
    })

    # 15. hard_fragile_nonstackable
    scenarios.append({
        "request": _make_request(
            "hard_fragile_nonstackable",
            _pallet(1200, 800, 800, wt=1000.0),
            [
                _box("SKU-FBASE", 600, 400, 200, wt=10.0, qty=4),
                _box("SKU-FRNS", 300, 200, 200, wt=2.0, qty=6,
                     fragile=True, stackable=False),
                _box("SKU-FILL", 300, 200, 200, wt=3.0, qty=8),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.66,  # 12/18
            "min_fragility": 0.8,
            "constraint_group": "hard_fragile",
            "description": "Fragile + non-stackable: double constraint",
        },
    })

    # ===================================================================
    # Cat 4: Strict Upright + Non-Stackable (16-20)
    # ===================================================================

    # 16. hard_upright_all: 12 upright-only boxes
    scenarios.append({
        "request": _make_request(
            "hard_upright_all",
            _pallet(1200, 800, 800, wt=1000.0),
            [_box("SKU-UPONLY", 350, 250, 400, wt=8.0, qty=12, upright=True)],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.75,  # 9/12
            "constraint_group": "hard_upright",
            "description": "All 12 boxes upright-only (350x250x400) - fixed H=400mm",
        },
    })

    # 17. hard_upright_gap_fill
    scenarios.append({
        "request": _make_request(
            "hard_upright_gap_fill",
            _pallet(1200, 800, 800, wt=1000.0),
            [
                _box("SKU-UPTALL", 400, 300, 500, wt=10.0, qty=6, upright=True),
                _box("SKU-FLEXFL", 150, 100, 200, wt=1.5, qty=20),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.5,  # 13/26
            "constraint_group": "hard_upright",
            "description": "Upright tall (500mm) + flexible fillers for gaps",
        },
    })

    # 18. hard_nostack_tall
    scenarios.append({
        "request": _make_request(
            "hard_nostack_tall",
            _pallet(1200, 800, 1800, wt=1000.0),
            [
                _box("SKU-TALLNS", 400, 400, 600, wt=12.0, qty=6, stackable=False),
                _box("SKU-SHFILL", 200, 200, 200, wt=2.0, qty=30),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.5,  # 18/36
            "constraint_group": "hard_upright",
            "description": "6 tall non-stackable (600mm) waste vertical + 30 short fillers",
        },
    })

    # 19. hard_triple_constraint
    scenarios.append({
        "request": _make_request(
            "hard_triple_constraint",
            _pallet(1200, 800, 1000, wt=1000.0),
            [
                _box("SKU-UPRT", 300, 200, 300, wt=5.0, qty=6, upright=True),
                _box("SKU-FRNS2", 300, 200, 200, wt=1.5, qty=4,
                     fragile=True, stackable=False),
                _box("SKU-NORM", 300, 200, 200, wt=8.0, qty=8),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.66,  # 12/18
            "min_fragility": 0.8,
            "constraint_group": "hard_upright",
            "description": "Triple: upright + fragile non-stackable + normal",
        },
    })

    # 20. hard_nostack_flat
    scenarios.append({
        "request": _make_request(
            "hard_nostack_flat",
            _pallet(1200, 800, 1000, wt=1000.0),
            [
                _box("SKU-FLATNS", 400, 300, 100, wt=3.0, qty=8, stackable=False),
                _box("SKU-CUBE", 200, 200, 200, wt=4.0, qty=10),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.55,  # 10/18
            "constraint_group": "hard_upright",
            "description": "8 flat non-stackable (100mm) waste ceiling + 10 cubes",
        },
    })

    # ===================================================================
    # Cat 5: Tight Fit / Near-Impossible Packing (21-25)
    # ===================================================================

    # 21. hard_tetris_exact
    scenarios.append({
        "request": _make_request(
            "hard_tetris_exact",
            _pallet(600, 400, 400, wt=1000.0),
            [
                _box("SKU-HALF", 600, 400, 200, wt=8.0, qty=4),
                _box("SKU-QTR", 300, 200, 200, wt=4.0, qty=4),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.5,  # at least 4/8
            "constraint_group": "hard_tightfit",
            "description": "Tetris exact: 4 half-slabs + 4 quarter-blocks in 600x400x400",
        },
    })

    # 22. hard_oversized: some boxes can't fit
    scenarios.append({
        "request": _make_request(
            "hard_oversized",
            _pallet(1200, 800, 1800, wt=1000.0),
            [
                _box("SKU-NORMAL", 300, 200, 200, wt=5.0, qty=10),
                _box("SKU-TOOWIDE", 1300, 400, 300, wt=20.0, qty=2),
                _box("SKU-TOOTALL", 300, 200, 1900, wt=10.0, qty=1, upright=True),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.76,  # 10/13 (2 too-wide + 1 too-tall unplaceable)
            "constraint_group": "hard_tightfit",
            "description": "2 too-wide (1300mm) + 1 too-tall upright (1900mm) can't fit",
        },
    })

    # 23. hard_diverse_sizes: 5 types, 73 items
    scenarios.append({
        "request": _make_request(
            "hard_diverse_sizes",
            _pallet(1200, 800, 1200, wt=1000.0),
            [
                _box("SKU-TINY", 100, 80, 60, wt=0.3, qty=40),
                _box("SKU-SMALL", 200, 150, 120, wt=1.5, qty=20),
                _box("SKU-MEDIUM", 400, 300, 250, wt=6.0, qty=8),
                _box("SKU-LARGE", 600, 400, 300, wt=12.0, qty=3),
                _box("SKU-XL", 800, 600, 400, wt=25.0, qty=2),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.5,  # many items, limited space
            "constraint_group": "hard_tightfit",
            "description": "5 size tiers (100mm-800mm), 73 items - bin packing diversity",
        },
    })

    # 24. hard_low_ceiling: max_height = 220mm
    scenarios.append({
        "request": _make_request(
            "hard_low_ceiling",
            _pallet(1200, 800, 220, wt=1000.0),
            [
                _box("SKU-FLAT", 300, 200, 200, wt=5.0, qty=16),
                _box("SKU-ROTONLY", 300, 220, 100, wt=3.0, qty=8),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.5,  # 12/24
            "constraint_group": "hard_tightfit",
            "description": "Low ceiling 220mm - single layer, every mm counts",
        },
    })

    # 25. hard_near_full: floor+ceiling tiles + odd fillers
    scenarios.append({
        "request": _make_request(
            "hard_near_full",
            _pallet(1200, 800, 400, wt=1000.0),
            [
                _box("SKU-FLOOR", 600, 400, 200, wt=10.0, qty=6),
                _box("SKU-CEIL", 600, 400, 200, wt=8.0, qty=5),
                _box("SKU-ODD", 350, 350, 180, wt=6.0, qty=4),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.46,  # 7/15
            "constraint_group": "hard_tightfit",
            "description": "6 floor + 5 ceiling tiles + 4 odd (350x350) fillers - near full",
        },
    })

    # ===================================================================
    # Cat 6: Mixed Constraint Chaos (26-30)
    # ===================================================================

    # 26. hard_chaos_retail: 5 SKU, all constraint types, reduced weight
    scenarios.append({
        "request": _make_request(
            "hard_chaos_retail",
            _pallet(1200, 800, 1800, wt=800.0),
            [
                _box("SKU-WATER", 280, 190, 330, wt=9.2, qty=15, upright=True),
                _box("SKU-WINE", 250, 170, 320, wt=8.0, qty=8,
                     upright=True, fragile=True, stackable=False),
                _box("SKU-CHIPS", 600, 400, 400, wt=1.8, qty=10, fragile=True),
                _box("SKU-CANNED", 300, 200, 120, wt=6.0, qty=20, upright=True),
                _box("SKU-SUGAR", 400, 300, 150, wt=10.0, qty=12),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.4,
            "min_fragility": 0.5,
            "constraint_group": "hard_chaos",
            "description": "Retail nightmare: 5 SKU, all constraints, 800kg limit",
        },
    })

    # 27. hard_chaos_heavy_fragile
    scenarios.append({
        "request": _make_request(
            "hard_chaos_heavy_fragile",
            _pallet(1200, 800, 1000, wt=500.0),
            [
                _box("SKU-HVEGGS", 630, 320, 350, wt=22.0, qty=6,
                     upright=True, fragile=True),
                _box("SKU-LTSUP", 300, 200, 150, wt=2.0, qty=10),
                _box("SKU-MEDBOX", 400, 300, 200, wt=8.0, qty=6),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.45,  # weight limited: 6x22=132 + budget for others
            "min_fragility": 0.5,
            "constraint_group": "hard_chaos",
            "description": "Heavy fragile eggs (22kg, upright) + support + medium, 500kg limit",
        },
    })

    # 28. hard_chaos_all_nostack: everything non-stackable -> single layer
    scenarios.append({
        "request": _make_request(
            "hard_chaos_all_nostack",
            _pallet(1200, 800, 1800, wt=1000.0),
            [
                _box("SKU-NSA", 400, 300, 300, wt=8.0, qty=4, stackable=False),
                _box("SKU-NSB", 300, 200, 250, wt=5.0, qty=6, stackable=False),
                _box("SKU-NSC", 200, 200, 200, wt=4.0, qty=8, stackable=False),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.55,  # single layer constraint
            "constraint_group": "hard_chaos",
            "description": "All non-stackable: only one layer, 1800mm height wasted",
        },
    })

    # 29. hard_chaos_squeeze: low ceiling + low weight + upright + fragile
    scenarios.append({
        "request": _make_request(
            "hard_chaos_squeeze",
            _pallet(1200, 800, 500, wt=150.0),
            [
                _box("SKU-UPHVY", 300, 200, 400, wt=12.0, qty=8, upright=True),
                _box("SKU-FLATLT", 400, 300, 100, wt=1.0, qty=6),
                _box("SKU-CUBEFR", 200, 200, 200, wt=2.5, qty=10, fragile=True),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.33,  # heavily constrained
            "min_fragility": 0.6,
            "constraint_group": "hard_chaos",
            "description": "Low ceiling(500mm) + 150kg limit + upright(400mm) + fragile",
        },
    })

    # 30. hard_chaos_maximum: all archetypes, all constraints, low weight
    scenarios.append({
        "request": _make_request(
            "hard_chaos_maximum",
            _pallet(1200, 1000, 2000, wt=600.0),
            [
                _box("SKU-BANANA", 502, 394, 239, wt=19.0, qty=8, upright=True),
                _box("SKU-WINE2", 250, 170, 320, wt=8.0, qty=6,
                     upright=True, fragile=True, stackable=False),
                _box("SKU-CHIPS2", 600, 400, 400, wt=1.8, qty=8, fragile=True),
                _box("SKU-WATER2", 280, 190, 330, wt=9.2, qty=10, upright=True),
                _box("SKU-SUGAR2", 400, 300, 150, wt=10.0, qty=10),
                _box("SKU-EGGS2", 630, 320, 350, wt=22.0, qty=4,
                     upright=True, fragile=True),
                _box("SKU-CAN2", 300, 200, 120, wt=6.0, qty=15, upright=True),
                _box("SKU-TINYFILL", 100, 80, 60, wt=0.3, qty=20),
            ],
        ),
        "expect": {
            "valid": True,
            "min_coverage": 0.3,
            "min_fragility": 0.4,
            "constraint_group": "hard_chaos",
            "description": "FINAL BOSS: 8 SKU, all archetypes, all constraints, 600kg limit",
        },
    })

    return scenarios


# ===========================================================================
# Runner
# ===========================================================================

CONSTRAINT_GROUPS_ORDER = [
    "dimensions",
    "collisions",
    "support",
    "upright",
    "volume",
    "coverage",
    "fragility",
    "time",
    "weight",
    "stackable",
    "combined",
    "hard_weight",
    "hard_support",
    "hard_fragile",
    "hard_upright",
    "hard_tightfit",
    "hard_chaos",
]

PASS = "PASS"
FAIL = "FAIL"


def run_constraint_benchmarks(
    strategy: str = "portfolio_block",
    time_budget_ms: int = 5000,
    n_restarts: int = 10,
) -> List[Dict[str, Any]]:
    scenarios = build_constraint_scenarios() + build_hard_scenarios()
    results = []

    for sc in scenarios:
        req = sc["request"]
        exp = sc["expect"]
        task_id = req["task_id"]

        task_id_m, pallet, boxes = _request_to_models(req)

        t0 = time.perf_counter()
        solution = solve(
            task_id=task_id_m,
            pallet=pallet,
            boxes=boxes,
            request_dict=req,
            n_restarts=n_restarts,
            time_budget_ms=time_budget_ms,
            strategy=strategy,
        )
        wall_ms = int((time.perf_counter() - t0) * 1000)

        resp = solution_to_dict(solution)
        ev = evaluate_solution(req, resp)

        # --- Check expectations ---
        checks: List[Dict[str, Any]] = []

        # Valid
        if "valid" in exp:
            ok = ev.get("valid", False) == exp["valid"]
            checks.append({"check": "valid", "expected": exp["valid"],
                           "actual": ev.get("valid"), "pass": ok})

        metrics = ev.get("metrics", {})

        # Coverage
        if "exact_coverage" in exp:
            actual_cov = metrics.get("item_coverage", 0.0)
            ok = abs(actual_cov - exp["exact_coverage"]) < 0.01
            checks.append({"check": "exact_coverage", "expected": exp["exact_coverage"],
                           "actual": round(actual_cov, 4), "pass": ok})
        if "min_coverage" in exp:
            actual_cov = metrics.get("item_coverage", 0.0)
            ok = actual_cov >= exp["min_coverage"] - 0.01
            checks.append({"check": "min_coverage", "expected": exp["min_coverage"],
                           "actual": round(actual_cov, 4), "pass": ok})

        # Volume
        if "exact_volume" in exp:
            actual_vol = metrics.get("volume_utilization", 0.0)
            ok = abs(actual_vol - exp["exact_volume"]) < 0.01
            checks.append({"check": "exact_volume", "expected": exp["exact_volume"],
                           "actual": round(actual_vol, 4), "pass": ok})
        if "min_volume" in exp:
            actual_vol = metrics.get("volume_utilization", 0.0)
            ok = actual_vol >= exp["min_volume"] - 0.01
            checks.append({"check": "min_volume", "expected": exp["min_volume"],
                           "actual": round(actual_vol, 4), "pass": ok})

        # Fragility
        if "min_fragility" in exp:
            actual_frag = metrics.get("fragility_score", 0.0)
            ok = actual_frag >= exp["min_fragility"] - 0.01
            checks.append({"check": "min_fragility", "expected": exp["min_fragility"],
                           "actual": round(actual_frag, 4), "pass": ok})

        # Time
        if "max_time_ms" in exp:
            ok = solution.solve_time_ms <= exp["max_time_ms"]
            checks.append({"check": "max_time_ms", "expected": exp["max_time_ms"],
                           "actual": solution.solve_time_ms, "pass": ok})

        all_pass = all(c["pass"] for c in checks)

        results.append({
            "task_id": task_id,
            "constraint_group": exp["constraint_group"],
            "description": exp["description"],
            "verdict": PASS if all_pass else FAIL,
            "checks": checks,
            "final_score": ev.get("final_score", 0.0),
            "metrics": metrics,
            "placed": len(solution.placements),
            "total_items": sum(b["quantity"] for b in req["boxes"]),
            "solve_time_ms": solution.solve_time_ms,
            "wall_time_ms": wall_ms,
            "error": ev.get("error"),
        })

    return results


def format_constraint_table(results: List[Dict[str, Any]]) -> str:
    lines = ["## Constraint Benchmark Results", ""]

    for group in CONSTRAINT_GROUPS_ORDER:
        group_results = [r for r in results if r["constraint_group"] == group]
        if not group_results:
            continue

        group_title = {
            "dimensions": "1. Dimensions (LxWxH)",
            "collisions": "2. No Collisions",
            "support": "3. Support From Below (>=60%)",
            "upright": "4. Strict Upright (No Rotation)",
            "volume": "5. Volume Utilization",
            "coverage": "6. Item Coverage",
            "fragility": "7. Fragility (Heavy-on-Fragile)",
            "time": "8. Execution Time",
            "weight": "9. Weight Constraint",
            "stackable": "10. Non-Stackable",
            "combined": "11. Combined Constraints",
            "hard_weight": "H1. Weight Limit Edge Cases",
            "hard_support": "H2. Support 60% Stress",
            "hard_fragile": "H3. Fragility Challenges",
            "hard_upright": "H4. Upright + Non-Stackable",
            "hard_tightfit": "H5. Tight Fit / Near-Impossible",
            "hard_chaos": "H6. Mixed Constraint Chaos",
        }.get(group, group)

        lines.append(f"### {group_title}")
        lines.append("")
        lines.append(
            "| N | Test Case | Verdict | Checks | Score | Placed | Time ms | Description |"
        )
        lines.append(
            "|---|-----------|---------|--------|-------|--------|-----------|-------------|"
        )

        for i, r in enumerate(group_results, 1):
            verdict_str = f"**{r['verdict']}**"
            check_parts = []
            for c in r["checks"]:
                if c["pass"]:
                    check_parts.append(f"{c['check']}=OK")
                else:
                    check_parts.append(f"{c['check']}=FAIL({c['actual']}!={c['expected']})")
            checks_str = ", ".join(check_parts)
            lines.append(
                f"| {i} "
                f"| {r['task_id']} "
                f"| {verdict_str} "
                f"| {checks_str} "
                f"| {r['final_score']:.4f} "
                f"| {r['placed']}/{r['total_items']} "
                f"| {r['solve_time_ms']} "
                f"| {r['description']} |"
            )

        passed = sum(1 for r in group_results if r["verdict"] == PASS)
        total = len(group_results)
        lines.append("")
        lines.append(f"**{group_title}: {passed}/{total} passed**")
        lines.append("")

    # Summary
    total_pass = sum(1 for r in results if r["verdict"] == PASS)
    total = len(results)
    lines.append("---")
    lines.append(f"### Summary: {total_pass}/{total} constraint tests passed")
    lines.append("")

    # Per-group summary table
    lines.append("| Constraint Group | Passed | Total | Rate |")
    lines.append("|-----------------|--------|-------|------|")
    for group in CONSTRAINT_GROUPS_ORDER:
        group_results = [r for r in results if r["constraint_group"] == group]
        if not group_results:
            continue
        passed = sum(1 for r in group_results if r["verdict"] == PASS)
        total_g = len(group_results)
        rate = f"{passed/total_g*100:.0f}%"
        lines.append(f"| {group} | {passed} | {total_g} | {rate} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Constraint Stress-Test Benchmark")
    parser.add_argument(
        "--strategy", default="portfolio_block",
        choices=["portfolio_block", "legacy_hybrid", "legacy_greedy"],
    )
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--time-budget", type=int, default=5000)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    results = run_constraint_benchmarks(
        strategy=args.strategy,
        time_budget_ms=args.time_budget,
        n_restarts=args.restarts,
    )

    md = format_constraint_table(results)
    print(md)

    if args.output:
        slim = [
            {k: v for k, v in r.items() if k != "response"}
            for r in results
        ]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
