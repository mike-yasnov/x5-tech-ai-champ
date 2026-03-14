import json
import random
from typing import Dict, Any, List

# ==============================
# Настройки случайности
# ==============================

GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)


# ==============================
# Паллеты (реальные типы)
# ==============================

PALLETS = [
    {
        "id": "EUR_1200x800",
        "length_mm": 1200,
        "width_mm": 800,
        "max_height_mm": 1800,  # включая груз, типичный лимит погрузки [web:40]
        "max_weight_kg": 1000.0,  # EPAL / CHEP номинальная нагрузка [web:34][web:35]
    },
    {
        "id": "EUR_1200x1000",
        "length_mm": 1200,
        "width_mm": 1000,
        "max_height_mm": 2000,
        "max_weight_kg": 1000.0,  # типичная safe working load [web:38]
    },
    {
        "id": "US_48x40",
        "length_mm": 1219,  # 48"
        "width_mm": 1016,  # 40"
        "max_height_mm": 2000,
        "max_weight_kg": 1000.0,
    },
]

# ==============================
# Архетипы SKU для фуд‑ритейла
# ==============================

FOOD_RETAIL_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "banana": {
        "desc": "Bananas Box",
        "l": 502,
        "w": 394,
        "h": 239,
        "wt": 19.0,
        "upright": True,
        "fragile": False,
    },
    "sugar": {
        "desc": "Sugar 10kg",
        "l": 400,
        "w": 300,
        "h": 150,
        "wt": 10.0,
        "upright": False,
        "fragile": False,
    },
    "water": {
        "desc": "Water Pack",
        "l": 280,
        "w": 190,
        "h": 330,
        "wt": 9.2,
        "upright": True,
        "fragile": False,
    },
    "wine": {
        "desc": "Wine Case",
        "l": 250,
        "w": 170,
        "h": 320,
        "wt": 8.0,
        "upright": True,
        "fragile": True,
    },
    "chips": {
        "desc": "Chips Carton",
        "l": 600,
        "w": 400,
        "h": 400,
        "wt": 1.8,
        "upright": False,
        "fragile": True,
    },
    "eggs": {
        "desc": "Eggs 360pcs",
        "l": 630,
        "w": 320,
        "h": 350,
        "wt": 22.0,
        "upright": True,
        "fragile": True,
    },
    "canned": {
        "desc": "Canned Peas",
        "l": 300,
        "w": 200,
        "h": 120,
        "wt": 6.0,
        "upright": True,
        "fragile": False,
    },
}


def _noise_int(x: int, rel: float = 0.02) -> int:
    """Небольшой шум размеров (±2%)"""
    return int(round(x * random.uniform(1 - rel, 1 + rel)))


def create_box(archetype_key: str, qty_min: int, qty_max: int) -> Dict[str, Any]:
    base = FOOD_RETAIL_ARCHETYPES[archetype_key]
    return {
        "sku_id": f"SKU-{archetype_key.upper()}-{random.randint(1000, 9999)}",
        "description": base["desc"],
        "length_mm": _noise_int(base["l"]),
        "width_mm": _noise_int(base["w"]),
        "height_mm": _noise_int(base["h"]),
        "weight_kg": round(base["wt"] * random.uniform(0.98, 1.02), 2),
        "quantity": random.randint(qty_min, qty_max),
        "strict_upright": base["upright"],
        "fragile": base["fragile"],
        "stackable": True,
    }


# ==============================
# Сценарии задач
# ==============================


def generate_scenario(
    task_id: str, scenario_type: str, seed: int = GLOBAL_SEED
) -> Dict[str, Any]:
    set_seed(seed)  # воспроизводимость

    pallet = random.choice(PALLETS)
    boxes: List[Dict[str, Any]] = []

    if scenario_type == "heavy_water":
        boxes.append(create_box("water", 80, 140))
        boxes.append(create_box("sugar", 40, 60))

    elif scenario_type == "fragile_tower":
        boxes.append(create_box("banana", 10, 20))
        boxes.append(create_box("chips", 15, 30))
        boxes.append(create_box("eggs", 5, 10))

    elif scenario_type == "liquid_tetris":
        boxes.append(create_box("water", 20, 40))
        boxes.append(create_box("wine", 15, 30))
        boxes.append(create_box("canned", 30, 50))

    elif scenario_type == "random_mixed":
        keys = list(FOOD_RETAIL_ARCHETYPES.keys())
        k = random.randint(4, min(7, len(keys)))
        for key in random.sample(keys, k=k):
            boxes.append(create_box(key, 5, 25))

    elif scenario_type == "exact_fit":
        pallet = {
            "id": "TEST_1200x800x200",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 200,
            "max_weight_kg": 1000.0,
        }
        boxes.append(
            {
                "sku_id": "SKU-EXACT-600x400",
                "description": "Exact fit carton",
                "length_mm": 600,
                "width_mm": 400,
                "height_mm": 200,
                "weight_kg": 10.0,
                "quantity": 4,
                "strict_upright": False,
                "fragile": False,
                "stackable": True,
            }
        )

    elif scenario_type == "fragile_mix":
        pallet = {
            "id": "TEST_1200x800x400",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 400,
            "max_weight_kg": 1000.0,
        }
        boxes.extend(
            [
                {
                    "sku_id": "SKU-BASE-HEAVY",
                    "description": "Heavy support box",
                    "length_mm": 600,
                    "width_mm": 400,
                    "height_mm": 200,
                    "weight_kg": 15.0,
                    "quantity": 2,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
                {
                    "sku_id": "SKU-FRAGILE-MID",
                    "description": "Fragile medium carton",
                    "length_mm": 300,
                    "width_mm": 400,
                    "height_mm": 200,
                    "weight_kg": 3.0,
                    "quantity": 8,
                    "strict_upright": False,
                    "fragile": True,
                    "stackable": True,
                },
                {
                    "sku_id": "SKU-FILL",
                    "description": "Small filler carton",
                    "length_mm": 300,
                    "width_mm": 200,
                    "height_mm": 200,
                    "weight_kg": 2.0,
                    "quantity": 12,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
            ]
        )

    elif scenario_type == "support_tetris":
        pallet = {
            "id": "TEST_1200x800x400",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 400,
            "max_weight_kg": 1000.0,
        }
        boxes.extend(
            [
                {
                    "sku_id": "SKU-LARGE",
                    "description": "Large base block",
                    "length_mm": 800,
                    "width_mm": 400,
                    "height_mm": 200,
                    "weight_kg": 12.0,
                    "quantity": 4,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
                {
                    "sku_id": "SKU-MEDIUM",
                    "description": "Medium filler block",
                    "length_mm": 400,
                    "width_mm": 200,
                    "height_mm": 200,
                    "weight_kg": 4.0,
                    "quantity": 10,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
            ]
        )

    elif scenario_type == "cavity_fill":
        pallet = {
            "id": "TEST_1200x800x300",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 300,
            "max_weight_kg": 1000.0,
        }
        boxes.extend(
            [
                {
                    "sku_id": "SKU-Upright",
                    "description": "Upright carton",
                    "length_mm": 400,
                    "width_mm": 400,
                    "height_mm": 200,
                    "weight_kg": 6.0,
                    "quantity": 6,
                    "strict_upright": True,
                    "fragile": False,
                    "stackable": True,
                },
                {
                    "sku_id": "SKU-FILLER",
                    "description": "Residual-space filler",
                    "length_mm": 200,
                    "width_mm": 200,
                    "height_mm": 200,
                    "weight_kg": 1.5,
                    "quantity": 12,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
            ]
        )

    elif scenario_type == "count_preference":
        pallet = {
            "id": "TEST_600x400x200",
            "length_mm": 600,
            "width_mm": 400,
            "max_height_mm": 200,
            "max_weight_kg": 1000.0,
        }
        boxes.extend(
            [
                {
                    "sku_id": "SKU-LARGE-600x400",
                    "description": "Large carton",
                    "length_mm": 600,
                    "width_mm": 400,
                    "height_mm": 200,
                    "weight_kg": 8.0,
                    "quantity": 1,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
                {
                    "sku_id": "SKU-SMALL-300x400",
                    "description": "Small carton",
                    "length_mm": 300,
                    "width_mm": 400,
                    "height_mm": 200,
                    "weight_kg": 4.0,
                    "quantity": 2,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
            ]
        )

    elif scenario_type == "weight_limited_repeat":
        pallet = {
            "id": "TRAIN_1219x1016x1800",
            "length_mm": 1219,
            "width_mm": 1016,
            "max_height_mm": 1800,
            "max_weight_kg": 900.0,
        }
        boxes.extend(
            [
                create_box("water", 60, 120),
                create_box("sugar", 30, 70),
                create_box("banana", 8, 18),
            ]
        )

    elif scenario_type == "fragile_cap_mix":
        pallet = {
            "id": "TRAIN_1200x1000x1800",
            "length_mm": 1200,
            "width_mm": 1000,
            "max_height_mm": 1800,
            "max_weight_kg": 1000.0,
        }
        boxes.extend(
            [
                create_box("chips", 12, 28),
                create_box("wine", 12, 24),
                create_box("eggs", 6, 10),
                create_box("canned", 20, 40),
            ]
        )

    elif scenario_type == "mixed_column_repeat":
        pallet = {
            "id": "TRAIN_1200x800x2000",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 2000,
            "max_weight_kg": 1000.0,
        }
        boxes.extend(
            [
                create_box("banana", 10, 20),
                create_box("water", 25, 45),
                create_box("canned", 25, 50),
            ]
        )

    elif scenario_type == "small_gap_fill":
        pallet = {
            "id": "TRAIN_1200x800x600",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 600,
            "max_weight_kg": 1000.0,
        }
        boxes.extend(
            [
                {
                    "sku_id": f"SKU-BLOCK-{random.randint(1000, 9999)}",
                    "description": "Gap maker block",
                    "length_mm": 600,
                    "width_mm": 400,
                    "height_mm": 200,
                    "weight_kg": 12.0,
                    "quantity": 4,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
                {
                    "sku_id": f"SKU-FILL-{random.randint(1000, 9999)}",
                    "description": "Gap filler carton",
                    "length_mm": 200,
                    "width_mm": 200,
                    "height_mm": 200,
                    "weight_kg": 1.5,
                    "quantity": 18,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
            ]
        )

    elif scenario_type == "non_stackable_caps":
        pallet = {
            "id": "TEST_1200x800x600",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 600,
            "max_weight_kg": 1000.0,
        }
        boxes.extend(
            [
                {
                    "sku_id": "SKU-BASE-BLOCK",
                    "description": "Stackable base block",
                    "length_mm": 600,
                    "width_mm": 400,
                    "height_mm": 200,
                    "weight_kg": 12.0,
                    "quantity": 4,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
                {
                    "sku_id": "SKU-DISPLAY-CAP",
                    "description": "Fragile do-not-stack display cap",
                    "length_mm": 600,
                    "width_mm": 400,
                    "height_mm": 200,
                    "weight_kg": 4.0,
                    "quantity": 2,
                    "strict_upright": True,
                    "fragile": True,
                    "stackable": False,
                },
                {
                    "sku_id": "SKU-FILLER-200",
                    "description": "Small filler carton",
                    "length_mm": 300,
                    "width_mm": 200,
                    "height_mm": 200,
                    "weight_kg": 1.5,
                    "quantity": 12,
                    "strict_upright": False,
                    "fragile": False,
                    "stackable": True,
                },
            ]
        )

    # ==========================================================
    # PRIVATE TEST scenarios (harder variants of organizer tests)
    # ==========================================================

    elif scenario_type == "private_heavy_eggs_crush":
        # Like heavy_water but with eggs (22kg, fragile, upright) — the hardest archetype.
        # Eggs are simultaneously heavy AND fragile AND upright.
        # Stacking anything >2kg on eggs = fragility violation.
        # Stacking eggs on others wastes 22kg weight budget per box.
        pallet = PALLETS[0]  # EUR 1200x800x1800
        pallet = dict(pallet, max_weight_kg=800.0)
        boxes.extend([
            create_box("eggs", 10, 18),
            create_box("sugar", 25, 50),
            create_box("canned", 20, 40),
        ])

    elif scenario_type == "private_all_upright_tight":
        # Every item is strict_upright on US pallet (odd dimensions 1219x1016).
        # Zero rotation flexibility — solver must tile with fixed orientations.
        pallet = PALLETS[2]  # US 1219x1016x2000
        pallet = dict(pallet, max_weight_kg=900.0)
        boxes.extend([
            create_box("water", 20, 35),
            create_box("wine", 10, 20),
            create_box("eggs", 5, 10),
            create_box("canned", 25, 45),
            create_box("banana", 8, 15),
        ])

    elif scenario_type == "private_fragile_dominant":
        # 3 fragile SKUs with high qty, almost no non-fragile base material.
        # Solver must minimize fragile-on-fragile stacking damage.
        pallet = PALLETS[1]  # EUR 1200x1000x2000
        boxes.extend([
            create_box("wine", 25, 45),
            create_box("chips", 20, 40),
            create_box("eggs", 8, 16),
        ])

    elif scenario_type == "private_weight_razor":
        # All 7 archetypes on tight weight budget (500kg).
        # Total requested weight >> 500kg -> solver must pick wisely.
        # banana(19kg) and eggs(22kg) eat budget fast.
        pallet = PALLETS[0]  # EUR 1200x800x1800
        pallet = dict(pallet, max_weight_kg=500.0)
        for key in FOOD_RETAIL_ARCHETYPES:
            boxes.append(create_box(key, 8, 20))

    elif scenario_type == "private_sugar_flood":
        # Single SKU massive quantity. Sugar is rotation-free.
        # Tests single-SKU columnar optimization at scale.
        pallet = PALLETS[1]  # EUR 1200x1000x2000
        boxes.append(create_box("sugar", 120, 200))

    elif scenario_type == "private_wine_eggs_dilemma":
        # Both wine and eggs are fragile+upright.
        # Wine on eggs: 8kg > 2kg on fragile -> violation.
        # Eggs on wine: 22kg > 2kg on fragile -> violation.
        # They CANNOT be stacked on each other without penalty.
        # Only solution: side-by-side or separate layers with non-fragile between.
        pallet = PALLETS[0]  # EUR 1200x800x1800
        pallet = dict(pallet, max_weight_kg=700.0)
        boxes.extend([
            create_box("wine", 15, 30),
            create_box("eggs", 8, 15),
            create_box("sugar", 10, 20),  # non-fragile separator
        ])

    elif scenario_type == "private_canned_wall":
        # High qty small upright items (canned: 300x200x120, 6kg) + heavy banana.
        # Banana (502x394x239, 19kg) on canned needs solid support platform.
        # Support from tiny canned boxes is fragile (60% rule stress).
        pallet = PALLETS[2]  # US 1219x1016x2000
        boxes.extend([
            create_box("canned", 80, 150),
            create_box("banana", 8, 15),
        ])

    elif scenario_type == "private_chips_mountain":
        # Chips: 600x400x400, only 1.8kg, fragile.
        # Huge boxes, few fit per layer. All fragile but light (<2kg).
        # No fragility violations even when stacked (1.8kg <= 2kg threshold).
        # Test: does solver achieve fragility_score=1.0?
        pallet = PALLETS[0]  # EUR 1200x800x1800
        boxes.append(create_box("chips", 30, 50))

    elif scenario_type == "private_weight_tradeoff":
        # Banana (19kg, big volume) vs sugar (10kg, smaller volume).
        # Weight=600kg. ~30 banana = 570kg fills budget but great volume.
        # ~60 sugar = 600kg also fills budget but less volume per kg.
        # Solver must find optimal mix.
        pallet = PALLETS[0]  # EUR 1200x800x1800
        pallet = dict(pallet, max_weight_kg=600.0)
        boxes.extend([
            create_box("banana", 20, 40),
            create_box("sugar", 30, 60),
        ])

    elif scenario_type == "private_full_catalog":
        # All 7 archetypes, decent quantities, reduced weight.
        # Maximum constraint diversity: upright, fragile, heavy, light.
        # Hardest general-purpose test.
        pallet = PALLETS[1]  # EUR 1200x1000x2000
        pallet = dict(pallet, max_weight_kg=700.0)
        for key in FOOD_RETAIL_ARCHETYPES:
            boxes.append(create_box(key, 10, 30))

    elif scenario_type == "private_micro_batch":
        # All 7 archetypes, tiny quantities (1-3 each).
        # Only ~14 items but maximum constraint diversity.
        # Tests per-SKU decision quality, not volume optimization.
        pallet = PALLETS[0]  # EUR 1200x800x1800
        for key in FOOD_RETAIL_ARCHETYPES:
            boxes.append(create_box(key, 1, 3))

    elif scenario_type == "private_upright_overflow":
        # All upright + high quantities on low ceiling pallet.
        # Water h=330, wine h=320, eggs h=350 — only ~2 layers in 800mm.
        # Many items forced unplaced -> coverage challenge.
        pallet = {
            "id": "LOW_1200x800x800",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 800,
            "max_weight_kg": 1000.0,
        }
        boxes.extend([
            create_box("water", 40, 70),
            create_box("wine", 20, 40),
            create_box("eggs", 10, 20),
        ])

    elif scenario_type == "private_nostack_fragile_mix":
        # Non-stackable items (not in original archetypes!) mixed with fragile.
        # Tests the stackable=false constraint which organizer tests barely cover.
        pallet = PALLETS[1]  # EUR 1200x1000x2000
        boxes.extend([
            create_box("chips", 10, 20),    # fragile, rotatable
            create_box("wine", 10, 20),     # fragile, upright
            create_box("sugar", 15, 30),    # heavy filler
        ])
        # Add non-stackable display items manually
        boxes.append({
            "sku_id": f"SKU-DISPLAY-{random.randint(1000, 9999)}",
            "description": "Display no-stack unit",
            "length_mm": 400,
            "width_mm": 300,
            "height_mm": 250,
            "weight_kg": 3.5,
            "quantity": random.randint(4, 8),
            "strict_upright": True,
            "fragile": True,
            "stackable": False,
        })

    elif scenario_type == "private_heavy_fragile_sandwich":
        # Layer puzzle: heavy base -> fragile mid -> heavy top.
        # Naive solver creates fragility violations in sandwich pattern.
        # Optimal: all heavy below, all fragile on top, no sandwich.
        pallet = PALLETS[0]  # EUR 1200x800x1800
        boxes.extend([
            create_box("banana", 15, 25),   # heavy 19kg, upright, non-fragile
            create_box("chips", 15, 25),    # light 1.8kg, fragile
            create_box("eggs", 5, 10),      # heavy 22kg, fragile, upright
            create_box("water", 15, 25),    # heavy 9.2kg, upright, non-fragile
        ])

    elif scenario_type == "private_odd_pallet_stress":
        # US pallet (1219x1016) with items sized for EUR pallet (1200x800).
        # 19mm and 216mm remainders create awkward gaps.
        # Tests gap-filling and non-standard pallet handling.
        pallet = PALLETS[2]  # US 1219x1016x2000
        pallet = dict(pallet, max_weight_kg=800.0)
        boxes.extend([
            create_box("sugar", 30, 60),    # 400x300 tiles EUR well
            create_box("canned", 30, 60),   # 300x200 tiles EUR well
            create_box("water", 20, 40),    # small upright
        ])

    else:
        raise ValueError(f"Unknown scenario_type: {scenario_type}")

    return {
        "task_id": task_id,
        "pallet": {
            "type_id": pallet["id"],
            "length_mm": pallet["length_mm"],
            "width_mm": pallet["width_mm"],
            "max_height_mm": pallet["max_height_mm"],
            "max_weight_kg": pallet["max_weight_kg"],
        },
        "boxes": boxes,
    }


if __name__ == "__main__":
    set_seed(123)  # фиксируем seed для генерации файлов

    scenarios = [
        "heavy_water",
        "fragile_tower",
        "liquid_tetris",
        "random_mixed",
        "exact_fit",
        "fragile_mix",
        "support_tetris",
        "cavity_fill",
        "count_preference",
        "weight_limited_repeat",
        "fragile_cap_mix",
        "mixed_column_repeat",
        "small_gap_fill",
        "non_stackable_caps",
    ]
    for sc in scenarios:
        task = generate_scenario(f"task_{sc}", sc, seed=123 + scenarios.index(sc))
        filename = f"request_{sc}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)
        print(f"Saved {filename}")
