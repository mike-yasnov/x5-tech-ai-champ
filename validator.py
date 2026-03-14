import json
from typing import Any, Dict, List, Tuple


def calc_overlap_2d(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
    """Area of overlap between two boxes projected on the XY plane."""
    dx = max(0, min(b1["x_max"], b2["x_max"]) - max(b1["x_min"], b2["x_min"]))
    dy = max(0, min(b1["y_max"], b2["y_max"]) - max(b1["y_min"], b2["y_min"]))
    return dx * dy


def check_3d_collision(b1: Dict[str, Any], b2: Dict[str, Any]) -> bool:
    """Strict AABB collision test."""
    ox = max(0, min(b1["x_max"], b2["x_max"]) - max(b1["x_min"], b2["x_min"]))
    oy = max(0, min(b1["y_max"], b2["y_max"]) - max(b1["y_min"], b2["y_min"]))
    oz = max(0, min(b1["z_max"], b2["z_max"]) - max(b1["z_min"], b2["z_min"]))
    return ox > 0 and oy > 0 and oz > 0


def _parse_and_validate(
    request: Dict[str, Any],
    response: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    pallet = request["pallet"]
    boxes_meta: Dict[str, Any] = {b["sku_id"]: dict(b) for b in request["boxes"]}

    placements: List[Dict[str, Any]] = []
    total_weight_placed = 0.0
    total_requested_items = sum(b["quantity"] for b in request["boxes"])

    for p in response.get("placements", []):
        sku_id = p["sku_id"]
        if sku_id not in boxes_meta:
            return False, {"error": f"Unknown SKU in response: {sku_id}"}

        sku = boxes_meta[sku_id]
        dim = p["dimensions_placed"]
        pos = p["position"]

        orig_dims_sorted = sorted(
            [sku["length_mm"], sku["width_mm"], sku["height_mm"]]
        )
        placed_dims_sorted = sorted(
            [dim["length_mm"], dim["width_mm"], dim["height_mm"]]
        )
        if orig_dims_sorted != placed_dims_sorted:
            return False, {
                "error": (
                    f"Cheat detected: dimensions for {sku_id} do not match original. "
                    f"orig={orig_dims_sorted}, placed={placed_dims_sorted}"
                )
            }

        used = sku.get("_used_qty", 0) + 1
        sku["_used_qty"] = used
        if used > sku["quantity"]:
            return False, {
                "error": f"Too many items of {sku_id} placed: {used} > {sku['quantity']}"
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
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
            "area": dim["length_mm"] * dim["width_mm"],
            "volume": dim["length_mm"] * dim["width_mm"] * dim["height_mm"],
        }
        placements.append(box3d)
        total_weight_placed += sku["weight_kg"]

    if total_weight_placed > pallet["max_weight_kg"] + 1e-6:
        return False, {
            "error": (
                f"Overweight: {total_weight_placed:.2f} kg > "
                f"{pallet['max_weight_kg']:.2f} kg"
            )
        }

    for i, b1 in enumerate(placements):
        if (
            b1["x_min"] < 0
            or b1["y_min"] < 0
            or b1["z_min"] < 0
            or b1["x_max"] > pallet["length_mm"] + 1e-6
            or b1["y_max"] > pallet["width_mm"] + 1e-6
            or b1["z_max"] > pallet["max_height_mm"] + 1e-6
        ):
            return False, {"error": f"Box {b1['sku_id']} is out of pallet bounds."}

        if b1["strict_upright"]:
            height_placed = b1["z_max"] - b1["z_min"]
            if abs(height_placed - b1["orig_height"]) > 1e-6:
                return False, {
                    "error": f"Box {b1['sku_id']} with strict_upright rotated illegally."
                }

        for j in range(i + 1, len(placements)):
            if check_3d_collision(b1, placements[j]):
                return False, {
                    "error": f"Collision between {b1['sku_id']} and {placements[j]['sku_id']}"
                }

        if b1["z_min"] > 0:
            support_area = 0.0
            for b2 in placements:
                if abs(b2["z_max"] - b1["z_min"]) < 1e-6:
                    support_area += calc_overlap_2d(b1, b2)
            if b1["area"] == 0 or support_area / b1["area"] < 0.6:
                return False, {
                    "error": (
                        f"Box {b1['sku_id']} has insufficient support "
                        f"({support_area:.1f}/{b1['area']:.1f})."
                    )
                }

        if not b1["stackable"]:
            for b2 in placements:
                if b1 is b2:
                    continue
                if abs(b1["z_max"] - b2["z_min"]) < 1e-6 and calc_overlap_2d(b1, b2) > 0:
                    return False, {
                        "error": f"Box {b2['sku_id']} placed on top of non-stackable {b1['sku_id']}."
                    }

    return True, {
        "pallet": pallet,
        "placements": placements,
        "total_requested_items": total_requested_items,
    }


def _count_fragility_violations(placements: List[Dict[str, Any]]) -> int:
    violations = 0
    for top in placements:
        if top["weight"] <= 2.0:
            continue
        for bottom in placements:
            if not bottom["fragile"]:
                continue
            if abs(top["z_min"] - bottom["z_max"]) < 1e-6 and calc_overlap_2d(top, bottom) > 0:
                violations += 1
    return violations


def _packing_metrics(
    pallet: Dict[str, Any],
    placements: List[Dict[str, Any]],
    total_requested_items: int,
) -> Dict[str, float]:
    pallet_vol = (
        pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    )
    vol_util = (
        sum(b["volume"] for b in placements) / pallet_vol if pallet_vol > 0 else 0.0
    )
    placed_items = len(placements)
    item_coverage = (
        placed_items / total_requested_items if total_requested_items > 0 else 0.0
    )
    fragility_violations = _count_fragility_violations(placements)
    fragility_score = max(0.0, 1.0 - 0.05 * fragility_violations)
    packing_score = (
        0.50 * vol_util + 0.30 * item_coverage + 0.10 * fragility_score
    )
    return {
        "volume_utilization": round(vol_util, 4),
        "item_coverage": round(item_coverage, 4),
        "fragility_score": round(fragility_score, 4),
        "packing_score": round(packing_score, 4),
    }


def evaluate_packing_quality(
    request: Dict[str, Any],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    valid, payload = _parse_and_validate(request, response)
    if not valid:
        return {"valid": False, "error": payload["error"]}

    metrics = _packing_metrics(
        pallet=payload["pallet"],
        placements=payload["placements"],
        total_requested_items=payload["total_requested_items"],
    )
    return {
        "valid": True,
        "metrics": metrics,
        "packing_score": metrics["packing_score"],
    }


def evaluate_solution(request: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    packing = evaluate_packing_quality(request, response)
    if not packing.get("valid", False):
        return packing

    metrics = dict(packing["metrics"])
    time_ms = response.get("solve_time_ms", 999999)
    if time_ms <= 1000:
        time_score = 1.0
    elif time_ms <= 5000:
        time_score = 0.7
    elif time_ms <= 30000:
        time_score = 0.3
    else:
        time_score = 0.0

    final_score = packing["packing_score"] + 0.10 * time_score
    metrics["time_score"] = round(time_score, 4)
    return {
        "valid": True,
        "metrics": metrics,
        "packing_score": packing["packing_score"],
        "final_score": round(final_score, 4),
    }


if __name__ == "__main__":
    with open("request_heavy_water.json", "r", encoding="utf-8") as f_req:
        req = json.load(f_req)
    with open("response_example.json", "r", encoding="utf-8") as f_resp:
        resp = json.load(f_resp)

    result = evaluate_solution(req, resp)
    print(json.dumps(result, indent=2, ensure_ascii=False))
