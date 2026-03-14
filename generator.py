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
        "stackable": False,
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
        "stackable": base.get("stackable", True),
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
    ]
    for sc in scenarios:
        task = generate_scenario(f"task_{sc}", sc, seed=123 + scenarios.index(sc))
        filename = f"request_{sc}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)
        print(f"Saved {filename}")
