"""Data models for 3D Pallet Packing Solver."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Box:
    """Single SKU type with quantity."""
    sku_id: str
    description: str
    length_mm: int
    width_mm: int
    height_mm: int
    weight_kg: float
    quantity: int
    strict_upright: bool = False
    fragile: bool = False
    stackable: bool = True

    @property
    def volume(self) -> int:
        return self.length_mm * self.width_mm * self.height_mm

    @property
    def base_area(self) -> int:
        return self.length_mm * self.width_mm


@dataclass
class Pallet:
    """Pallet parameters."""
    type_id: str
    length_mm: int
    width_mm: int
    max_height_mm: int
    max_weight_kg: float

    @property
    def volume(self) -> int:
        return self.length_mm * self.width_mm * self.max_height_mm


@dataclass
class Placement:
    """Single box placement in the solution."""
    sku_id: str
    instance_index: int
    x_mm: int
    y_mm: int
    z_mm: int
    length_mm: int  # placed dimension along X
    width_mm: int   # placed dimension along Y
    height_mm: int  # placed dimension along Z
    rotation_code: str

    @property
    def x_max(self) -> int:
        return self.x_mm + self.length_mm

    @property
    def y_max(self) -> int:
        return self.y_mm + self.width_mm

    @property
    def z_max(self) -> int:
        return self.z_mm + self.height_mm

    @property
    def volume(self) -> int:
        return self.length_mm * self.width_mm * self.height_mm

    @property
    def base_area(self) -> int:
        return self.length_mm * self.width_mm


@dataclass
class UnplacedItem:
    """Item that could not be placed."""
    sku_id: str
    quantity_unplaced: int
    reason: str  # weight_limit_exceeded | height_limit_exceeded | no_space


@dataclass
class Solution:
    """Complete packing solution."""
    task_id: str
    solver_version: str
    solve_time_ms: int
    placements: List[Placement] = field(default_factory=list)
    unplaced: List[UnplacedItem] = field(default_factory=list)


def load_request(path: str) -> Tuple[str, Pallet, List[Box]]:
    """Load request JSON and return (task_id, Pallet, list[Box])."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_id = data["task_id"]

    p = data["pallet"]
    pallet = Pallet(
        type_id=p.get("type_id", "unknown"),
        length_mm=p["length_mm"],
        width_mm=p["width_mm"],
        max_height_mm=p["max_height_mm"],
        max_weight_kg=p["max_weight_kg"],
    )

    boxes = []
    for b in data["boxes"]:
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

    total_items = sum(b.quantity for b in boxes)
    total_weight = sum(b.weight_kg * b.quantity for b in boxes)
    total_volume = sum(b.volume * b.quantity for b in boxes)

    logger.info(
        "[load_request] task=%s SKUs=%d items=%d total_weight=%.1fkg total_volume=%.1fm³ pallet=%s(%dx%dx%d)",
        task_id, len(boxes), total_items, total_weight,
        total_volume / 1e9, pallet.type_id,
        pallet.length_mm, pallet.width_mm, pallet.max_height_mm,
    )

    return task_id, pallet, boxes


def solution_to_dict(solution: Solution) -> Dict[str, Any]:
    """Convert Solution to output JSON dict."""
    result: Dict[str, Any] = {
        "task_id": solution.task_id,
        "solver_version": solution.solver_version,
        "solve_time_ms": solution.solve_time_ms,
        "placements": [],
        "unplaced": [],
    }

    for p in solution.placements:
        result["placements"].append({
            "sku_id": p.sku_id,
            "instance_index": p.instance_index,
            "position": {
                "x_mm": p.x_mm,
                "y_mm": p.y_mm,
                "z_mm": p.z_mm,
            },
            "dimensions_placed": {
                "length_mm": p.length_mm,
                "width_mm": p.width_mm,
                "height_mm": p.height_mm,
            },
            "rotation_code": p.rotation_code,
        })

    for u in solution.unplaced:
        result["unplaced"].append({
            "sku_id": u.sku_id,
            "quantity_unplaced": u.quantity_unplaced,
            "reason": u.reason,
        })

    return result
