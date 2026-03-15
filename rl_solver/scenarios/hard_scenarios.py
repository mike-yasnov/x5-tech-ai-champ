"""30 hard test scenarios targeting constraint edge-cases.

Categories:
  1-5:   Weight limit edge cases
  6-10:  Support (60% rule) stress tests
  11-15: Fragility constraint challenges
  16-20: Strict upright + non-stackable combos
  21-25: Tight-fit / near-impossible packing
  26-30: Mixed constraint chaos
"""

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))

SCENARIOS: List[Dict[str, Any]] = []


def _box(sku: str, desc: str, l: int, w: int, h: int, wt: float, qty: int,
         upright: bool = False, fragile: bool = False, stackable: bool = True) -> Dict[str, Any]:
    return {
        "sku_id": sku, "description": desc,
        "length_mm": l, "width_mm": w, "height_mm": h,
        "weight_kg": wt, "quantity": qty,
        "strict_upright": upright, "fragile": fragile, "stackable": stackable,
    }


def _pallet(l: int = 1200, w: int = 800, h: int = 1800, max_w: float = 1000.0) -> Dict[str, Any]:
    return {"length_mm": l, "width_mm": w, "max_height_mm": h, "max_weight_kg": max_w}


def _scenario(task_id: str, pallet: Dict, boxes: List[Dict]) -> Dict[str, Any]:
    return {"task_id": task_id, "pallet": pallet, "boxes": boxes}


# ============================================================
# 1-5: WEIGHT LIMIT EDGE CASES
# ============================================================

# 1. Exact weight limit — all boxes together = max_weight exactly
SCENARIOS.append(_scenario("hard_weight_exact", _pallet(max_w=100.0), [
    _box("SKU-W1", "Heavy block A", 400, 400, 200, 25.0, 4),
]))

# 2. Weight overflow by 1 box — must leave 1 item unplaced
SCENARIOS.append(_scenario("hard_weight_overflow_1", _pallet(max_w=50.0), [
    _box("SKU-W2", "Dense cube", 300, 300, 300, 10.0, 6),
]))

# 3. Many light boxes but one is extremely heavy
SCENARIOS.append(_scenario("hard_weight_one_heavy", _pallet(max_w=200.0), [
    _box("SKU-W3A", "Light filler", 200, 200, 100, 0.5, 50),
    _box("SKU-W3B", "Anchor block", 600, 400, 300, 180.0, 1),
]))

# 4. All boxes identical, weight forces partial placement
SCENARIOS.append(_scenario("hard_weight_uniform_cutoff", _pallet(1200, 800, 600, 120.0), [
    _box("SKU-W4", "Uniform heavy", 400, 400, 200, 15.0, 12),
]))

# 5. Mix of light/heavy where greedy by volume wastes weight budget
SCENARIOS.append(_scenario("hard_weight_greedy_trap", _pallet(max_w=80.0), [
    _box("SKU-W5A", "Big light", 600, 400, 400, 2.0, 4),
    _box("SKU-W5B", "Small heavy", 200, 200, 200, 18.0, 5),
]))

# ============================================================
# 6-10: SUPPORT 60% RULE STRESS TESTS
# ============================================================

# 6. Small box on top of two half-overlapping boxes (borderline 60%)
SCENARIOS.append(_scenario("hard_support_borderline", _pallet(1200, 800, 600, 1000.0), [
    _box("SKU-S6A", "Base left", 500, 400, 200, 5.0, 2),
    _box("SKU-S6B", "Base right", 500, 400, 200, 5.0, 2),
    _box("SKU-S6C", "Top bridger", 800, 400, 200, 3.0, 2),
]))

# 7. Staircase pattern — each level offset, support barely works
SCENARIOS.append(_scenario("hard_support_staircase", _pallet(1200, 800, 1000, 1000.0), [
    _box("SKU-S7A", "Step block", 400, 400, 200, 8.0, 12),
]))

# 8. Large box needs multi-box support platform
SCENARIOS.append(_scenario("hard_support_platform", _pallet(1200, 800, 800, 1000.0), [
    _box("SKU-S8A", "Small base", 300, 200, 200, 4.0, 12),
    _box("SKU-S8B", "Large top", 900, 600, 300, 10.0, 2),
]))

# 9. Boxes with non-matching heights making support impossible without rotation
SCENARIOS.append(_scenario("hard_support_height_mismatch", _pallet(1200, 800, 800, 1000.0), [
    _box("SKU-S9A", "Tall base", 400, 400, 300, 6.0, 3),
    _box("SKU-S9B", "Short base", 400, 400, 200, 6.0, 3),
    _box("SKU-S9C", "Wide top", 800, 400, 200, 4.0, 3),
]))

# 10. Many tiny boxes — must form a stable floor for a heavy top layer
SCENARIOS.append(_scenario("hard_support_mosaic_floor", _pallet(1200, 800, 600, 1000.0), [
    _box("SKU-S10A", "Tile", 150, 100, 100, 1.0, 64),
    _box("SKU-S10B", "Heavy slab", 600, 400, 200, 30.0, 2),
]))

# ============================================================
# 11-15: FRAGILITY CONSTRAINT CHALLENGES
# ============================================================

# 11. All boxes fragile — nothing heavy (>2kg) can go on top of anything
SCENARIOS.append(_scenario("hard_fragile_all", _pallet(1200, 800, 600, 1000.0), [
    _box("SKU-F11", "All fragile", 300, 200, 200, 5.0, 20, fragile=True),
]))

# 12. Fragile boxes are the biggest, must go at bottom for volume, but rules say top
SCENARIOS.append(_scenario("hard_fragile_big_bottom_dilemma", _pallet(1200, 800, 800, 1000.0), [
    _box("SKU-F12A", "Big fragile", 600, 400, 300, 3.0, 4, fragile=True),
    _box("SKU-F12B", "Small heavy", 300, 200, 200, 12.0, 8),
]))

# 13. Alternating fragile/non-fragile where naive layer approach fails
SCENARIOS.append(_scenario("hard_fragile_interleave", _pallet(1200, 800, 1000, 1000.0), [
    _box("SKU-F13A", "Heavy base", 400, 400, 200, 10.0, 6),
    _box("SKU-F13B", "Fragile mid", 400, 400, 200, 1.5, 6, fragile=True),
    _box("SKU-F13C", "Heavy top", 400, 400, 200, 8.0, 6),
]))

# 14. Single very fragile item among many heavy ones
SCENARIOS.append(_scenario("hard_fragile_singleton", _pallet(1200, 800, 800, 1000.0), [
    _box("SKU-F14A", "Heavy standard", 400, 300, 300, 15.0, 8),
    _box("SKU-F14B", "Precious fragile", 200, 200, 100, 0.5, 1, fragile=True),
]))

# 15. Fragile + non-stackable combo — fragile items that also can't bear weight
SCENARIOS.append(_scenario("hard_fragile_nonstackable", _pallet(1200, 800, 800, 1000.0), [
    _box("SKU-F15A", "Base support", 600, 400, 200, 8.0, 4),
    _box("SKU-F15B", "Fragile no-stack", 400, 300, 200, 2.0, 6, fragile=True, stackable=False),
    _box("SKU-F15C", "Filler", 300, 200, 200, 3.0, 8),
]))

# ============================================================
# 16-20: STRICT UPRIGHT + NON-STACKABLE COMBOS
# ============================================================

# 16. All upright — no rotation allowed, must tile perfectly
SCENARIOS.append(_scenario("hard_upright_all", _pallet(1200, 800, 800, 1000.0), [
    _box("SKU-U16", "Upright only", 350, 250, 400, 5.0, 12, upright=True),
]))

# 17. Upright boxes leave gaps that only rotatable fillers can fill
SCENARIOS.append(_scenario("hard_upright_gap_fill", _pallet(1200, 800, 800, 1000.0), [
    _box("SKU-U17A", "Upright tall", 400, 300, 500, 7.0, 6, upright=True),
    _box("SKU-U17B", "Flexible filler", 200, 300, 100, 2.0, 20),
]))

# 18. Non-stackable boxes that are tall — waste vertical space
SCENARIOS.append(_scenario("hard_nostack_tall", _pallet(1200, 800, 1800, 1000.0), [
    _box("SKU-NS18A", "Tall no-stack", 400, 400, 600, 10.0, 6, stackable=False),
    _box("SKU-NS18B", "Short filler", 200, 200, 100, 1.0, 30),
]))

# 19. Mix: upright + non-stackable + fragile on same pallet
SCENARIOS.append(_scenario("hard_triple_constraint", _pallet(1200, 800, 1000, 1000.0), [
    _box("SKU-TC19A", "Upright liquid", 300, 200, 400, 8.0, 6, upright=True),
    _box("SKU-TC19B", "Fragile glass", 250, 250, 300, 3.0, 4, fragile=True, stackable=False),
    _box("SKU-TC19C", "Normal box", 400, 300, 200, 5.0, 8),
]))

# 20. Non-stackable boxes that are flat (low height) — ceiling wasted
SCENARIOS.append(_scenario("hard_nostack_flat", _pallet(1200, 800, 1000, 1000.0), [
    _box("SKU-NS20A", "Flat no-stack", 600, 400, 100, 4.0, 8, stackable=False),
    _box("SKU-NS20B", "Regular cube", 300, 300, 300, 6.0, 10),
]))

# ============================================================
# 21-25: TIGHT FIT / NEAR-IMPOSSIBLE PACKING
# ============================================================

# 21. Perfect Tetris — boxes designed to tile exactly but only one way
SCENARIOS.append(_scenario("hard_tetris_exact", _pallet(600, 400, 400, 1000.0), [
    _box("SKU-T21A", "Half slab", 600, 200, 200, 5.0, 4),
    _box("SKU-T21B", "Quarter block", 300, 200, 200, 3.0, 4),
]))

# 22. Oversized boxes — some literally don't fit in any orientation
SCENARIOS.append(_scenario("hard_oversized", _pallet(1200, 800, 1800, 1000.0), [
    _box("SKU-O22A", "Normal", 400, 300, 200, 5.0, 10),
    _box("SKU-O22B", "Too wide", 1300, 400, 200, 8.0, 2),
    _box("SKU-O22C", "Too tall upright", 400, 400, 1900, 12.0, 1, upright=True),
]))

# 23. Many different sizes — bin packing diversity stress
SCENARIOS.append(_scenario("hard_diverse_sizes", _pallet(1200, 800, 1200, 1000.0), [
    _box("SKU-D23A", "Tiny", 100, 80, 60, 0.3, 40),
    _box("SKU-D23B", "Small", 200, 150, 120, 1.0, 20),
    _box("SKU-D23C", "Medium", 400, 300, 250, 4.0, 8),
    _box("SKU-D23D", "Large", 600, 500, 400, 10.0, 3),
    _box("SKU-D23E", "XL", 800, 600, 300, 15.0, 2),
]))

# 24. Very low ceiling — only one layer possible
SCENARIOS.append(_scenario("hard_low_ceiling", _pallet(1200, 800, 220, 1000.0), [
    _box("SKU-LC24A", "Fits flat", 300, 200, 200, 3.0, 16),
    _box("SKU-LC24B", "Fits only rotated", 200, 250, 150, 2.0, 8),
]))

# 25. Near-full pallet — last few boxes create unsolvable remainder
SCENARIOS.append(_scenario("hard_near_full", _pallet(1200, 800, 400, 1000.0), [
    _box("SKU-NF25A", "Floor tile", 400, 400, 200, 5.0, 6),
    _box("SKU-NF25B", "Ceiling tile", 400, 400, 200, 5.0, 5),
    _box("SKU-NF25C", "Odd filler", 350, 350, 180, 3.0, 4),
]))

# ============================================================
# 26-30: MIXED CONSTRAINT CHAOS
# ============================================================

# 26. Retail nightmare — many SKUs, all constraint types active
SCENARIOS.append(_scenario("hard_chaos_retail", _pallet(1200, 800, 1800, 800.0), [
    _box("SKU-CH26A", "Water pack", 280, 190, 330, 9.2, 15, upright=True),
    _box("SKU-CH26B", "Wine case", 250, 170, 320, 8.0, 8, upright=True, fragile=True, stackable=False),
    _box("SKU-CH26C", "Chips", 600, 400, 400, 1.8, 10, fragile=True),
    _box("SKU-CH26D", "Canned goods", 300, 200, 120, 6.0, 20, upright=True),
    _box("SKU-CH26E", "Sugar bags", 400, 300, 150, 10.0, 12),
]))

# 27. Weight + fragility conflict — heavy fragile items
SCENARIOS.append(_scenario("hard_chaos_heavy_fragile", _pallet(1200, 800, 1000, 500.0), [
    _box("SKU-CH27A", "Heavy fragile eggs", 630, 320, 350, 22.0, 6, upright=True, fragile=True),
    _box("SKU-CH27B", "Light support", 400, 300, 200, 3.0, 10),
    _box("SKU-CH27C", "Medium standard", 300, 300, 300, 8.0, 6),
]))

# 28. Everything non-stackable — forced single layer
SCENARIOS.append(_scenario("hard_chaos_all_nostack", _pallet(1200, 800, 1800, 1000.0), [
    _box("SKU-CH28A", "NoStack A", 400, 300, 200, 5.0, 4, stackable=False),
    _box("SKU-CH28B", "NoStack B", 300, 200, 300, 4.0, 6, stackable=False),
    _box("SKU-CH28C", "NoStack C", 250, 250, 250, 3.0, 8, stackable=False),
]))

# 29. Upright + weight limit + tight height — multi-constraint squeeze
SCENARIOS.append(_scenario("hard_chaos_squeeze", _pallet(1200, 800, 500, 150.0), [
    _box("SKU-CH29A", "Upright heavy", 300, 200, 400, 12.0, 8, upright=True),
    _box("SKU-CH29B", "Flat light", 600, 400, 100, 1.0, 6),
    _box("SKU-CH29C", "Cube fragile", 200, 200, 200, 2.5, 10, fragile=True),
]))

# 30. Maximum diversity stress — 8 SKUs, all constraints, tight pallet
SCENARIOS.append(_scenario("hard_chaos_maximum", _pallet(1200, 1000, 2000, 600.0), [
    _box("SKU-MAX-A", "Banana box", 502, 394, 239, 19.0, 6, upright=True),
    _box("SKU-MAX-B", "Wine case", 250, 170, 320, 8.0, 5, upright=True, fragile=True, stackable=False),
    _box("SKU-MAX-C", "Chip carton", 600, 400, 400, 1.8, 8, fragile=True),
    _box("SKU-MAX-D", "Water pack", 280, 190, 330, 9.2, 10, upright=True),
    _box("SKU-MAX-E", "Sugar 10kg", 400, 300, 150, 10.0, 8),
    _box("SKU-MAX-F", "Egg crate", 630, 320, 350, 22.0, 3, upright=True, fragile=True),
    _box("SKU-MAX-G", "Canned peas", 300, 200, 120, 6.0, 15, upright=True),
    _box("SKU-MAX-H", "Tiny filler", 100, 80, 60, 0.3, 30),
]))


def get_all_scenarios() -> List[Dict[str, Any]]:
    """Return all 30 hard scenarios."""
    logger.debug("Loading %d hard test scenarios", len(SCENARIOS))
    return SCENARIOS


def save_all(output_dir: str = "rl_solver/scenarios/data"):
    """Save all scenarios as individual JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    for sc in SCENARIOS:
        path = os.path.join(output_dir, f"{sc['task_id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sc, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s", path)
    logger.info("Total: %d scenarios saved to %s", len(SCENARIOS), output_dir)


if __name__ == "__main__":
    save_all()
    print(f"Generated {len(SCENARIOS)} hard scenarios")
