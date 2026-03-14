from __future__ import annotations

from typing import Dict, List, Tuple

from .geometry import AABB
from .constants import EPSILON, SUPPORT_THRESHOLD


class PlacedBox:
    """An immutable record of a box placed on the pallet."""

    __slots__ = (
        "sku_id",
        "instance_index",
        "aabb",
        "weight",
        "fragile",
        "stackable",
        "strict_upright",
        "rotation_code",
        "placed_dims",
    )

    def __init__(
        self,
        sku_id: str,
        instance_index: int,
        aabb: AABB,
        weight: float,
        fragile: bool,
        stackable: bool,
        strict_upright: bool,
        rotation_code: str,
        placed_dims: Tuple[int, int, int],
    ):
        self.sku_id = sku_id
        self.instance_index = instance_index
        self.aabb = aabb
        self.weight = weight
        self.fragile = fragile
        self.stackable = stackable
        self.strict_upright = strict_upright
        self.rotation_code = rotation_code
        self.placed_dims = placed_dims


class PalletState:
    """Mutable state of one pallet during construction."""

    def __init__(
        self,
        length: int,
        width: int,
        max_height: int,
        max_weight: float,
    ):
        self.length = length
        self.width = width
        self.max_height = max_height
        self.max_weight = max_weight

        self.placed: List[PlacedBox] = []
        self.total_weight: float = 0.0
        self.max_z: int = 0
        self.sku_instance_counters: Dict[str, int] = {}

        self.pallet_volume: int = length * width * max_height
        self.pallet_area: int = length * width
        self.placed_volume: int = 0

    def remaining_weight_capacity(self) -> float:
        return self.max_weight - self.total_weight

    def place(self, box: PlacedBox) -> None:
        self.placed.append(box)
        self.total_weight += box.weight
        self.placed_volume += box.aabb.volume()
        z_top = box.aabb.z_max
        if z_top > self.max_z:
            self.max_z = z_top
        self.sku_instance_counters[box.sku_id] = (
            self.sku_instance_counters.get(box.sku_id, 0) + 1
        )

    def next_instance_index(self, sku_id: str) -> int:
        return self.sku_instance_counters.get(sku_id, 0)

    def copy(self) -> PalletState:
        """Shallow copy for beam search branching."""
        new = PalletState.__new__(PalletState)
        new.length = self.length
        new.width = self.width
        new.max_height = self.max_height
        new.max_weight = self.max_weight
        new.placed = list(self.placed)
        new.total_weight = self.total_weight
        new.max_z = self.max_z
        new.sku_instance_counters = dict(self.sku_instance_counters)
        new.pallet_volume = self.pallet_volume
        new.pallet_area = self.pallet_area
        new.placed_volume = self.placed_volume
        return new

    def get_support_ratio(self, aabb: AABB) -> float:
        """Fraction of the box base area supported from below."""
        if aabb.z_min == 0:
            return 1.0  # on the floor
        base = aabb.base_area()
        if base == 0:
            return 0.0
        support = 0
        for pb in self.placed:
            if abs(pb.aabb.z_max - aabb.z_min) < EPSILON:
                support += aabb.overlap_area_xy(pb.aabb)
        return support / base

    def get_max_z_at(self, x: int, y: int, w: int, d: int) -> int:
        """Highest z_max of any placed box overlapping footprint (x,y,w,d). 0 if floor."""
        test = AABB(x, y, 0, x + w, y + d, 1)
        best = 0
        for pb in self.placed:
            if test.overlap_area_xy(pb.aabb) > 0:
                if pb.aabb.z_max > best:
                    best = pb.aabb.z_max
        return best
