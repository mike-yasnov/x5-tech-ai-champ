import json
from typing import Dict, Any, List

def calc_overlap_2d(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
    """Площадь пересечения проекций на XY."""
    dx = max(0, min(b1["x_max"], b2["x_max"]) - max(b1["x_min"], b2["x_min"]))
    dy = max(0, min(b1["y_max"], b2["y_max"]) - max(b1["y_min"], b2["y_min"]))
    return dx * dy

def check_3d_collision(b1: Dict[str, Any], b2: Dict[str, Any]) -> bool:
    """AABB-коллизия: строгое пересечение по всем осям."""
    ox = max(0, min(b1["x_max"], b2["x_max"]) - max(b1["x_min"], b2["x_min"]))
    oy = max(0, min(b1["y_max"], b2["y_max"]) - max(b1["y_min"], b2["y_min"]))
    oz = max(0, min(b1["z_max"], b2["z_max"]) - max(b1["z_min"], b2["z_min"]))
    return ox > 0 and oy > 0 and oz > 0

def evaluate_solution(request: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    pallet = request["pallet"]
    boxes_meta: Dict[str, Any] = {b["sku_id"]: dict(b) for b in request["boxes"]}

    placements: List[Dict[str, Any]] = []
    total_weight_placed = 0.0
    total_requested_items = sum(b["quantity"] for b in request["boxes"])

    # ---------------------------
    # Парсинг и anti-cheat
    # ---------------------------
    for p in response.get("placements", []):
        sku_id = p["sku_id"]
        if sku_id not in boxes_meta:
            return {"valid": False, "error": f"Unknown SKU in response: {sku_id}"}

        sku = boxes_meta[sku_id]
        dim = p["dimensions_placed"]
        pos = p["position"]

        # Валидация rotation_code
        VALID_ROTATION_CODES = {"LWH", "LHW", "WLH", "WHL", "HLW", "HWL"}
        rot_code = p.get("rotation_code", "")
        if rot_code not in VALID_ROTATION_CODES:
            return {
                "valid": False,
                "error": f"Invalid rotation_code '{rot_code}' for {sku_id}. Must be one of: {sorted(VALID_ROTATION_CODES)}",
            }

        # Anti-cheat: проверка, что габариты — это перестановка исходных габаритов
        orig_dims_sorted = sorted([
            sku["length_mm"],
            sku["width_mm"],
            sku["height_mm"]
        ])
        placed_dims_sorted = sorted([
            dim["length_mm"],
            dim["width_mm"],
            dim["height_mm"]
        ])
        if orig_dims_sorted != placed_dims_sorted:
            return {
                "valid": False,
                "error": (
                    f"Cheat detected: dimensions for {sku_id} do not match original. "
                    f"orig={orig_dims_sorted}, placed={placed_dims_sorted}"
                ),
            }

        # Считаем использованное количество по SKU
        used = sku.get("_used_qty", 0) + 1
        sku["_used_qty"] = used
        if used > sku["quantity"]:
            return {
                "valid": False,
                "error": f"Too many items of {sku_id} placed: {used} > {sku['quantity']}",
            }

        x_min = pos["x_mm"]
        y_min = pos["y_mm"]
        z_min = pos["z_mm"]
        x_max = x_min + dim["length_mm"]
        y_max = y_min + dim["width_mm"]
        z_max = z_min + dim["height_mm"]

        box3d = {
            "sku_id": sku_id,
            "weight": sku["weight_kg"],
            "fragile": sku["fragile"],
            "stackable": sku.get("stackable", True),
            "strict_upright": sku["strict_upright"],
            "orig_height": sku["height_mm"],
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max,
            "area": dim["length_mm"] * dim["width_mm"],
            "volume": dim["length_mm"] * dim["width_mm"] * dim["height_mm"],
        }
        placements.append(box3d)
        total_weight_placed += sku["weight_kg"]

    # ---------------------------
    # HARD constraints
    # ---------------------------

    # 1) Перевес
    if total_weight_placed > pallet["max_weight_kg"] + 1e-6:
        return {
            "valid": False,
            "error": f"Overweight: {total_weight_placed:.2f} kg > {pallet['max_weight_kg']:.2f} kg",
        }

    for i, b1 in enumerate(placements):
        # 2) Границы паллеты
        if (
            b1["x_min"] < 0
            or b1["y_min"] < 0
            or b1["z_min"] < 0
            or b1["x_max"] > pallet["length_mm"] + 1e-6
            or b1["y_max"] > pallet["width_mm"] + 1e-6
            or b1["z_max"] > pallet["max_height_mm"] + 1e-6
        ):
            return {
                "valid": False,
                "error": f"Box {b1['sku_id']} is out of pallet bounds.",
            }

        # 3) Upright constraint
        if b1["strict_upright"]:
            height_placed = b1["z_max"] - b1["z_min"]
            if abs(height_placed - b1["orig_height"]) > 1e-6:
                return {
                    "valid": False,
                    "error": f"Box {b1['sku_id']} with strict_upright rotated illegally.",
                }

        # 4) Коллизии
        for j in range(i + 1, len(placements)):
            b2 = placements[j]
            if check_3d_collision(b1, b2):
                return {
                    "valid": False,
                    "error": f"Collision between {b1['sku_id']} and {b2['sku_id']}",
                }

        # 5) Гравитация: опора ≥ 60%
        if b1["z_min"] > 0:
            support_area = 0.0
            for b2 in placements:
                if abs(b2["z_max"] - b1["z_min"]) < 1e-6:
                    support_area += calc_overlap_2d(b1, b2)
            if b1["area"] == 0 or support_area / b1["area"] < 0.6:
                return {
                    "valid": False,
                    "error": f"Box {b1['sku_id']} has insufficient support ({support_area:.1f}/{b1['area']:.1f}).",
                }

    # 6) Stackable: stackable=false → ничего нельзя ставить поверх
    for bottom in placements:
        if bottom["stackable"]:
            continue
        for top in placements:
            if top is bottom:
                continue
            if abs(top["z_min"] - bottom["z_max"]) < 1e-6 and calc_overlap_2d(top, bottom) > 0:
                return {
                    "valid": False,
                    "error": (
                        f"Box {top['sku_id']} placed on non-stackable {bottom['sku_id']}."
                    ),
                }

    # ---------------------------
    # SOFT metrics
    # ---------------------------

    # Volume utilization
    pallet_vol = pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    vol_util = sum(b["volume"] for b in placements) / pallet_vol if pallet_vol > 0 else 0.0

    # Item coverage
    placed_items = len(placements)
    item_coverage = placed_items / total_requested_items if total_requested_items > 0 else 0.0

    # Fragility penalty
    # Per task spec: fragile=true means "нельзя ставить под не-хрупкий груз тяжелее 2 кг"
    # So fragile-on-fragile is NOT a violation, only non-fragile heavy on fragile counts
    fragility_violations = 0
    for top in placements:
        if top["weight"] <= 2.0:
            continue
        if top["fragile"]:
            continue  # fragile-on-fragile is allowed per task definition
        for bottom in placements:
            if not bottom["fragile"]:
                continue
            if abs(top["z_min"] - bottom["z_max"]) < 1e-6 and calc_overlap_2d(top, bottom) > 0:
                fragility_violations += 1

    fragility_score = max(0.0, 1.0 - 0.05 * fragility_violations)

    # Time score
    time_ms = response.get("solve_time_ms", 999999)
    if time_ms <= 1000:
        time_score = 1.0
    elif time_ms <= 5000:
        time_score = 0.7
    elif time_ms <= 30000:
        time_score = 0.3
    else:
        time_score = 0.0

    final_score = (
        0.50 * vol_util
        + 0.30 * item_coverage
        + 0.10 * fragility_score
        + 0.10 * time_score
    )

    # Constraint compliance summary
    has_fragile = any(b["fragile"] for b in placements)
    has_upright = any(b["strict_upright"] for b in placements)
    has_non_stackable = any(not b["stackable"] for b in placements)

    constraint_checks = {
        "bounds": True,
        "no_collision": True,
        "support_60pct": True,
        "weight_limit": True,
        "strict_upright": "pass" if has_upright else "n/a",
        "stackable": "pass" if has_non_stackable else "n/a",
        "fragility_violations": fragility_violations,
    }

    return {
        "valid": True,
        "metrics": {
            "volume_utilization": round(vol_util, 4),
            "item_coverage": round(item_coverage, 4),
            "fragility_score": round(fragility_score, 4),
            "time_score": round(time_score, 4),
        },
        "constraint_checks": constraint_checks,
        "final_score": round(final_score, 4),
    }

# ============================
# Пример использования
# ============================
if __name__ == "__main__":
    # Пример: загрузка из файлов
    with open("request_heavy_water.json", "r", encoding="utf-8") as f_req:
        req = json.load(f_req)
    with open("response_example.json", "r", encoding="utf-8") as f_resp:
        resp = json.load(f_resp)

    result = evaluate_solution(req, resp)
    print(json.dumps(result, indent=2, ensure_ascii=False))