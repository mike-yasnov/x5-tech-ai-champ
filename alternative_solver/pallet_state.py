"""Pallet state: tracks placed boxes, extreme points, and validates constraints."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .models import Pallet

logger = logging.getLogger(__name__)

_CELL_MM = 200  # Spatial hash cell size


@dataclass
class PlacedBox:
    """AABB of a placed box with metadata."""
    sku_id: str
    x_min: int
    y_min: int
    z_min: int
    x_max: int
    y_max: int
    z_max: int
    weight_kg: float
    fragile: bool
    stackable: bool

    @property
    def base_area(self) -> int:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


def _overlap_area(
    ax1: int, ay1: int, ax2: int, ay2: int,
    bx1: int, by1: int, bx2: int, by2: int,
) -> int:
    """2D overlap area between two rectangles."""
    dx = max(0, min(ax2, bx2) - max(ax1, bx1))
    dy = max(0, min(ay2, by2) - max(ay1, by1))
    return dx * dy


def _aabb_collision(a: "PlacedBox", bx1: int, by1: int, bz1: int,
                    bx2: int, by2: int, bz2: int) -> bool:
    """Check strict AABB collision. Z checked first for early exit."""
    return (
        a.z_min < bz2 and a.z_max > bz1
        and a.x_min < bx2 and a.x_max > bx1
        and a.y_min < by2 and a.y_max > by1
    )


def _iter_cells(x1: int, y1: int, x2: int, y2: int):
    """Yield (cx, cy) cell coordinates covered by the rectangle."""
    cx1 = x1 // _CELL_MM
    cy1 = y1 // _CELL_MM
    cx2 = (x2 - 1) // _CELL_MM if x2 > x1 else cx1
    cy2 = (y2 - 1) // _CELL_MM if y2 > y1 else cy1
    for cx in range(cx1, cx2 + 1):
        for cy in range(cy1, cy2 + 1):
            yield (cx, cy)


class PalletState:
    """Mutable state of a pallet being packed."""

    def __init__(self, pallet: Pallet):
        self.pallet = pallet
        self.boxes: List[PlacedBox] = []
        self.current_weight: float = 0.0
        self.max_z: int = 0
        # Extreme Points: candidate positions (x, y, z)
        self.extreme_points: List[Tuple[int, int, int]] = [(0, 0, 0)]
        # Spatial indices
        self.boxes_by_top_z: defaultdict = defaultdict(list)
        self.xy_index: defaultdict = defaultdict(list)

    def _candidate_boxes_xy(self, x1: int, y1: int, x2: int, y2: int):
        """Yield unique PlacedBox objects that might overlap the given XY rectangle."""
        seen = set()
        for cell in _iter_cells(x1, y1, x2, y2):
            for idx in self.xy_index.get(cell, ()):
                if idx not in seen:
                    seen.add(idx)
                    yield self.boxes[idx]

    def can_place(
        self,
        dx: int, dy: int, dz: int,
        x: int, y: int, z: int,
        weight_kg: float,
        fragile: bool = False,
        stackable: bool = True,
    ) -> bool:
        """Check if placement is valid against all hard constraints."""
        x2, y2, z2 = x + dx, y + dy, z + dz

        # 1. Bounds check
        if x < 0 or y < 0 or z < 0:
            return False
        if x2 > self.pallet.length_mm or y2 > self.pallet.width_mm or z2 > self.pallet.max_height_mm:
            return False

        # 2. Weight check
        if self.current_weight + weight_kg > self.pallet.max_weight_kg + 1e-6:
            return False

        # 3. Collision check (spatial-indexed)
        for box in self._candidate_boxes_xy(x, y, x2, y2):
            if _aabb_collision(box, x, y, z, x2, y2, z2):
                return False

        # 4. Support check (gravity): ≥60% base area support
        base_area = dx * dy
        if z == 0:
            pass  # Floor provides full support
        else:
            support_area = 0
            for box in self.boxes_by_top_z.get(z, ()):
                # Check XY overlap first
                overlap = _overlap_area(
                    x, y, x2, y2,
                    box.x_min, box.y_min, box.x_max, box.y_max,
                )
                if overlap > 0:
                    # Check stackable constraint
                    if not box.stackable:
                        return False
                    support_area += overlap
            if base_area == 0 or support_area / base_area < 0.6:
                return False

        return True

    def place(
        self,
        sku_id: str,
        dx: int, dy: int, dz: int,
        x: int, y: int, z: int,
        weight_kg: float,
        fragile: bool = False,
        stackable: bool = True,
    ) -> PlacedBox:
        """Place a box and update state. Caller must verify can_place first."""
        x2, y2, z2 = x + dx, y + dy, z + dz

        placed = PlacedBox(
            sku_id=sku_id,
            x_min=x, y_min=y, z_min=z,
            x_max=x2, y_max=y2, z_max=z2,
            weight_kg=weight_kg,
            fragile=fragile,
            stackable=stackable,
        )
        self.boxes.append(placed)
        self.current_weight += weight_kg
        self.max_z = max(self.max_z, z2)

        # Update spatial indices
        idx = len(self.boxes) - 1
        self.boxes_by_top_z[z2].append(placed)
        for cell in _iter_cells(x, y, x2, y2):
            self.xy_index[cell].append(idx)

        # Update extreme points
        self._update_extreme_points(placed)

        logger.debug(
            "[place] sku=%s pos=(%d,%d,%d) dims=(%d,%d,%d) weight=%.1f remaining_capacity=%.1f",
            sku_id, x, y, z, dx, dy, dz, weight_kg,
            self.pallet.max_weight_kg - self.current_weight,
        )
        logger.debug("[place] extreme_points_count=%d", len(self.extreme_points))

        return placed

    def _update_extreme_points(self, placed: PlacedBox) -> None:
        """Generate new extreme points from projections of placed box."""
        new_eps = [
            # Right face projection
            (placed.x_max, placed.y_min, placed.z_min),
            # Front face projection
            (placed.x_min, placed.y_max, placed.z_min),
            # Top face projection
            (placed.x_min, placed.y_min, placed.z_max),
            # Additional top corners
            (placed.x_max, placed.y_min, placed.z_max),
            (placed.x_min, placed.y_max, placed.z_max),
        ]

        # Project placed box edges onto existing box faces
        for box in self.boxes:
            if box is placed:
                continue
            if box.z_max <= placed.z_max:
                new_eps.append((placed.x_min, placed.y_min, box.z_max))
                new_eps.append((placed.x_max, placed.y_min, box.z_max))
                new_eps.append((placed.x_min, placed.y_max, box.z_max))
            if box.x_max <= placed.x_max and box.x_max > placed.x_min:
                new_eps.append((box.x_max, placed.y_min, placed.z_min))
            if box.y_max <= placed.y_max and box.y_max > placed.y_min:
                new_eps.append((placed.x_min, box.y_max, placed.z_min))

        # Remove EPs that are now inside the placed box
        valid_eps = []
        for ep in self.extreme_points:
            ex, ey, ez = ep
            inside = (
                placed.x_min <= ex < placed.x_max
                and placed.y_min <= ey < placed.y_max
                and placed.z_min <= ez < placed.z_max
            )
            if not inside:
                valid_eps.append(ep)

        # Add new EPs that are within pallet bounds and not inside any existing box
        pL, pW, pH = self.pallet.length_mm, self.pallet.width_mm, self.pallet.max_height_mm
        for ep in new_eps:
            ex, ey, ez = ep
            if ex > pL or ey > pW or ez > pH or ex < 0 or ey < 0 or ez < 0:
                continue
            # Use spatial index for inside-box check
            inside_any = False
            for box in self._candidate_boxes_xy(ex, ey, ex + 1, ey + 1):
                if (
                    box.x_min <= ex < box.x_max
                    and box.y_min <= ey < box.y_max
                    and box.z_min <= ez < box.z_max
                ):
                    inside_any = True
                    break
            if not inside_any:
                valid_eps.append(ep)

        # Deduplicate, sort by (z, x, y) to prioritize lower positions, cap count
        self.extreme_points = sorted(set(valid_eps), key=lambda ep: (ep[2], ep[0], ep[1]))[:200]

    def get_fragile_boxes_at_top(self, z: int, x1: int, y1: int, x2: int, y2: int) -> List[PlacedBox]:
        """Find fragile boxes whose top face is at z and overlap with given XY rectangle."""
        result = []
        for box in self.boxes_by_top_z.get(z, ()):
            if box.fragile:
                if _overlap_area(x1, y1, x2, y2, box.x_min, box.y_min, box.x_max, box.y_max) > 0:
                    result.append(box)
        return result

    def contact_area_with_neighbors(self, x: int, y: int, z: int, dx: int, dy: int, dz: int) -> int:
        """Calculate total contact area with walls and adjacent boxes."""
        x2, y2, z2 = x + dx, y + dy, z + dz
        contact = 0

        # Wall contacts (side faces touching pallet edges)
        if x == 0:
            contact += dy * dz
        if y == 0:
            contact += dx * dz
        if x2 == self.pallet.length_mm:
            contact += dy * dz
        if y2 == self.pallet.width_mm:
            contact += dx * dz

        # Adjacent box contacts — use spatial index with 1-cell padding
        for box in self._candidate_boxes_xy(max(0, x - _CELL_MM), max(0, y - _CELL_MM),
                                             x2 + _CELL_MM, y2 + _CELL_MM):
            # Right/left face contact
            if box.x_max == x or box.x_min == x2:
                oy = max(0, min(y2, box.y_max) - max(y, box.y_min))
                oz = max(0, min(z2, box.z_max) - max(z, box.z_min))
                contact += oy * oz
            # Front/back face contact
            if box.y_max == y or box.y_min == y2:
                ox = max(0, min(x2, box.x_max) - max(x, box.x_min))
                oz = max(0, min(z2, box.z_max) - max(z, box.z_min))
                contact += ox * oz

        return contact
