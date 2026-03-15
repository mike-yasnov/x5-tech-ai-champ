from __future__ import annotations

from typing import Tuple

from .geometry import AABB
from .pallet_state import PalletState
from .constants import EPSILON, SUPPORT_THRESHOLD


class FeasibilityChecker:
    """Deterministic hard-constraint checks. Must run BEFORE ML scoring."""

    def __init__(
        self,
        pallet_length: int,
        pallet_width: int,
        max_height: int,
        max_weight: float,
    ):
        self.pallet_length = pallet_length
        self.pallet_width = pallet_width
        self.max_height = max_height
        self.max_weight = max_weight

    def check_bounds(self, aabb: AABB) -> bool:
        return (
            aabb.x_min >= 0
            and aabb.y_min >= 0
            and aabb.z_min >= 0
            and aabb.x_max <= self.pallet_length + EPSILON
            and aabb.y_max <= self.pallet_width + EPSILON
            and aabb.z_max <= self.max_height + EPSILON
        )

    def check_collision(self, aabb: AABB, state: PalletState) -> bool:
        """Return True if NO collision (placement is OK)."""
        for pb in state.placed:
            if aabb.overlaps_3d(pb.aabb):
                return False
        return True

    def check_support(self, aabb: AABB, state: PalletState) -> bool:
        return state.get_support_ratio(aabb) >= SUPPORT_THRESHOLD - EPSILON

    def check_weight(self, add_weight: float, state: PalletState) -> bool:
        return state.total_weight + add_weight <= self.max_weight + EPSILON

    def check_stackable(self, aabb: AABB, state: PalletState) -> bool:
        """Ensure we are not placing on top of a non-stackable box."""
        for pb in state.placed:
            if not pb.stackable:
                if abs(pb.aabb.z_max - aabb.z_min) < EPSILON:
                    if aabb.overlap_area_xy(pb.aabb) > 0:
                        return False
        return True

    def check_upright(self, placed_h: int, orig_h: int, strict_upright: bool) -> bool:
        if not strict_upright:
            return True
        return abs(placed_h - orig_h) < EPSILON

    def is_feasible(
        self,
        aabb: AABB,
        weight: float,
        orig_h: int,
        strict_upright: bool,
        stackable_below: bool,
        state: PalletState,
    ) -> Tuple[bool, str]:
        """Check all hard constraints. Returns (ok, reason)."""
        if not self.check_bounds(aabb):
            return False, "bounds"
        if not self.check_weight(weight, state):
            return False, "weight"
        if not self.check_upright(aabb.height_z(), orig_h, strict_upright):
            return False, "upright"
        if not self.check_collision(aabb, state):
            return False, "collision"
        if not self.check_support(aabb, state):
            return False, "support"
        if not self.check_stackable(aabb, state):
            return False, "stackable"
        return True, ""
