from __future__ import annotations

from typing import List, Set, Tuple

from .geometry import AABB
from .pallet_state import PalletState


class ExtremePointManager:
    """Tracks extreme points (EP) as candidate insertion positions.

    After each placement, generates new EPs at corners of the placed box
    and projects them down to the nearest support surface.
    """

    def __init__(self, pallet_length: int, pallet_width: int):
        self.pallet_length = pallet_length
        self.pallet_width = pallet_width
        self._points: Set[Tuple[int, int, int]] = {(0, 0, 0)}

    def get_points(self) -> List[Tuple[int, int, int]]:
        return sorted(
            self._points,
            key=lambda point: (point[2], point[0] + point[1], point[0], point[1]),
        )

    def update_after_placement(self, aabb: AABB, state: PalletState) -> None:
        """Generate new extreme points from the placed box and project z down."""
        new_eps = [
            # Right of box
            (aabb.x_max, aabb.y_min),
            # Behind box
            (aabb.x_min, aabb.y_max),
        ]

        # Top of box — only if there's room above
        if aabb.z_max < state.max_height:
            self._points.add((aabb.x_min, aabb.y_min, aabb.z_max))

        for (x, y) in new_eps:
            if x >= self.pallet_length or y >= self.pallet_width:
                continue
            # Project z down: find highest z_max among placed boxes whose XY
            # footprint overlaps with a small probe at (x, y)
            z = self._project_z_down(x, y, state)
            self._points.add((x, y, z))

        # Also generate EPs from projections of existing EPs onto the new box
        for ep in list(self._points):
            ex, ey, ez = ep
            # Project existing EP onto right face of new box
            if aabb.y_min <= ey < aabb.y_max and aabb.z_min <= ez < aabb.z_max:
                if aabb.x_max <= self.pallet_length and ex <= aabb.x_max:
                    self._points.add((aabb.x_max, ey, ez))
            # Project existing EP onto back face of new box
            if aabb.x_min <= ex < aabb.x_max and aabb.z_min <= ez < aabb.z_max:
                if aabb.y_max <= self.pallet_width and ey <= aabb.y_max:
                    self._points.add((ex, aabb.y_max, ez))

        # Remove points that are now inside the placed box
        self._points = {
            (px, py, pz)
            for (px, py, pz) in self._points
            if not self._inside_box(px, py, pz, aabb)
        }

    def _project_z_down(self, x: int, y: int, state: PalletState) -> int:
        """Find the support height at point (x, y) — highest z_max of any box below."""
        best_z = 0
        for pb in state.placed:
            a = pb.aabb
            if a.x_min <= x < a.x_max and a.y_min <= y < a.y_max:
                if a.z_max > best_z:
                    best_z = a.z_max
        return best_z

    def _inside_box(self, px: int, py: int, pz: int, aabb: AABB) -> bool:
        return (
            aabb.x_min < px < aabb.x_max
            and aabb.y_min < py < aabb.y_max
            and aabb.z_min < pz < aabb.z_max
        )

    def copy(self) -> ExtremePointManager:
        new = ExtremePointManager.__new__(ExtremePointManager)
        new.pallet_length = self.pallet_length
        new.pallet_width = self.pallet_width
        new._points = set(self._points)
        return new
