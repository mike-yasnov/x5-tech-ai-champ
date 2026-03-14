"""2D packing algorithms: Maximal Rectangles and Skyline (Best-fit).

Implements the 12 algorithm combinations from Table 1 of Dell'Amico et al. (2026).
Each algorithm packs boxes onto a W×D surface, returning placements.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Rect:
    """A rectangle on the 2D packing surface."""
    x: int
    y: int
    w: int
    h: int


@dataclass
class BoxItem:
    """A box to be packed, with its type info."""
    sku_id: str
    instance_index: int
    width: int   # dimension 1 on the surface
    depth: int   # dimension 2 on the surface
    height: int  # vertical height (becomes layer height)
    weight_kg: float
    fragile: bool
    strict_upright: bool
    allow_rotation: bool = True


@dataclass
class Placement2D:
    """Result of placing a box on a 2D surface."""
    sku_id: str
    instance_index: int
    x: int
    y: int
    dx: int  # placed width
    dy: int  # placed depth
    dz: int  # height (vertical)
    weight_kg: float
    fragile: bool


# ── Sorting strategies ──────────────────────────────────────────────

def sort_by_area_desc(items: List[BoxItem]) -> List[BoxItem]:
    return sorted(items, key=lambda b: -(b.width * b.depth))


def sort_by_diagonal_desc(items: List[BoxItem]) -> List[BoxItem]:
    return sorted(items, key=lambda b: -math.hypot(b.width, b.depth))


def sort_by_longest_side_desc(items: List[BoxItem]) -> List[BoxItem]:
    return sorted(items, key=lambda b: -max(b.width, b.depth))


SORT_STRATEGIES = {
    "area": sort_by_area_desc,
    "diagonal": sort_by_diagonal_desc,
    "longest_side": sort_by_longest_side_desc,
}


# ── Maximal Rectangles Best-Fit ─────────────────────────────────────

class MaximalRectangles:
    """Maximal Rectangles algorithm with Best-fit heuristic."""

    def __init__(self, width: int, height: int):
        self.bin_w = width
        self.bin_h = height
        self.free_rects: List[Rect] = [Rect(0, 0, width, height)]

    def _best_fit(self, w: int, h: int) -> Optional[Tuple[Rect, int, int]]:
        """Find the best free rectangle to fit w×h. Returns (rect, placed_w, placed_h)."""
        best_rect = None
        best_w, best_h = 0, 0
        best_short_side = float("inf")

        for rect in self.free_rects:
            # Try without rotation
            if w <= rect.w and h <= rect.h:
                short = min(rect.w - w, rect.h - h)
                if short < best_short_side:
                    best_short_side = short
                    best_rect = rect
                    best_w, best_h = w, h

            # Try with rotation (swap w and h)
            if h <= rect.w and w <= rect.h:
                short = min(rect.w - h, rect.h - w)
                if short < best_short_side:
                    best_short_side = short
                    best_rect = rect
                    best_w, best_h = h, w

        if best_rect is None:
            return None
        return best_rect, best_w, best_h

    def _best_fit_no_rotate(self, w: int, h: int) -> Optional[Tuple[Rect, int, int]]:
        """Find best fit without rotation."""
        best_rect = None
        best_short_side = float("inf")

        for rect in self.free_rects:
            if w <= rect.w and h <= rect.h:
                short = min(rect.w - w, rect.h - h)
                if short < best_short_side:
                    best_short_side = short
                    best_rect = rect

        if best_rect is None:
            return None
        return best_rect, w, h

    def place(self, w: int, h: int, allow_rotate: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """Place a rectangle. Returns (x, y, placed_w, placed_h) or None."""
        if allow_rotate:
            result = self._best_fit(w, h)
        else:
            result = self._best_fit_no_rotate(w, h)

        if result is None:
            return None

        rect, pw, ph = result
        px, py = rect.x, rect.y

        # Split free rectangles
        self._split_free_rects(px, py, pw, ph)
        self._prune_free_rects()

        return px, py, pw, ph

    def _split_free_rects(self, px: int, py: int, pw: int, ph: int) -> None:
        """Remove overlapping parts and create new free rectangles."""
        new_rects = []
        i = 0
        while i < len(self.free_rects):
            r = self.free_rects[i]
            # Check overlap
            if px >= r.x + r.w or px + pw <= r.x or py >= r.y + r.h or py + ph <= r.y:
                i += 1
                continue

            # There is overlap — split this rect and remove it
            self.free_rects.pop(i)

            # Left
            if px > r.x:
                new_rects.append(Rect(r.x, r.y, px - r.x, r.h))
            # Right
            if px + pw < r.x + r.w:
                new_rects.append(Rect(px + pw, r.y, r.x + r.w - px - pw, r.h))
            # Bottom
            if py > r.y:
                new_rects.append(Rect(r.x, r.y, r.w, py - r.y))
            # Top
            if py + ph < r.y + r.h:
                new_rects.append(Rect(r.x, py + ph, r.w, r.y + r.h - py - ph))

        self.free_rects.extend(new_rects)

    def _prune_free_rects(self) -> None:
        """Remove free rects that are fully contained in another."""
        pruned = []
        for i, a in enumerate(self.free_rects):
            contained = False
            for j, b in enumerate(self.free_rects):
                if i == j:
                    continue
                if (b.x <= a.x and b.y <= a.y
                        and b.x + b.w >= a.x + a.w
                        and b.y + b.h >= a.y + a.h):
                    contained = True
                    break
            if not contained:
                pruned.append(a)
        self.free_rects = pruned


# ── Skyline Best-Fit ────────────────────────────────────────────────

class SkylinePacker:
    """Skyline algorithm with Best-fit heuristic."""

    def __init__(self, width: int, height: int):
        self.bin_w = width
        self.bin_h = height
        # Skyline segments: list of (x, y, width) — y is the top of placed boxes
        self.skyline: List[Tuple[int, int, int]] = [(0, 0, width)]

    def _find_best_fit(self, w: int, h: int) -> Optional[Tuple[int, int, int]]:
        """Find best skyline position. Returns (segment_index, x, y)."""
        best_idx = -1
        best_y = float("inf")
        best_waste = float("inf")

        for i, (sx, sy, sw) in enumerate(self.skyline):
            if sw < w:
                # Try merging with next segments
                merged_w = sw
                max_y = sy
                j = i + 1
                while j < len(self.skyline) and merged_w < w:
                    _, ny, nw = self.skyline[j]
                    merged_w += nw
                    max_y = max(max_y, ny)
                    j += 1
                if merged_w < w:
                    continue
                y = max_y
            else:
                y = sy

            if y + h > self.bin_h:
                continue

            waste = y - min(s[1] for s in self.skyline)
            if y < best_y or (y == best_y and waste < best_waste):
                best_y = y
                best_waste = waste
                best_idx = i

        if best_idx == -1:
            return None
        sx, _, _ = self.skyline[best_idx]
        return best_idx, sx, best_y

    def place(self, w: int, h: int, allow_rotate: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """Place rectangle. Returns (x, y, placed_w, placed_h) or None."""
        results = []

        r1 = self._find_best_fit(w, h)
        if r1:
            results.append((r1, w, h))

        if allow_rotate and w != h:
            r2 = self._find_best_fit(h, w)
            if r2:
                results.append((r2, h, w))

        if not results:
            return None

        # Pick the one with lowest y
        results.sort(key=lambda r: r[0][2])
        (idx, px, py), pw, ph = results[0]

        self._update_skyline(px, py, pw, ph)
        return px, py, pw, ph

    def _update_skyline(self, px: int, py: int, pw: int, ph: int) -> None:
        """Update skyline after placing a rectangle."""
        new_top = py + ph
        new_skyline = []
        placed_left = px
        placed_right = px + pw
        inserted = False

        for sx, sy, sw in self.skyline:
            seg_right = sx + sw

            if seg_right <= placed_left or sx >= placed_right:
                new_skyline.append((sx, sy, sw))
                continue

            # This segment overlaps with the placed rect
            if sx < placed_left:
                new_skyline.append((sx, sy, placed_left - sx))

            if not inserted:
                new_skyline.append((placed_left, new_top, pw))
                inserted = True

            if seg_right > placed_right:
                new_skyline.append((placed_right, sy, seg_right - placed_right))

        if not inserted:
            new_skyline.append((placed_left, new_top, pw))

        # Merge adjacent segments with same y
        merged = [new_skyline[0]]
        for sx, sy, sw in new_skyline[1:]:
            lx, ly, lw = merged[-1]
            if lx + lw == sx and ly == sy:
                merged[-1] = (lx, ly, lw + sw)
            else:
                merged.append((sx, sy, sw))

        self.skyline = merged


# ── Unified packing interface ───────────────────────────────────────

def pack_2d(
    width: int,
    depth: int,
    items: List[BoxItem],
    algorithm: str = "maxrect",
    sort_strategy: str = "area",
    allow_rotation: bool = True,
) -> List[Placement2D]:
    """Pack items onto a W×D surface using specified algorithm and sorting.

    Args:
        width: Pallet width (W)
        depth: Pallet depth (D)
        items: Boxes to pack
        algorithm: "maxrect" or "skyline"
        sort_strategy: "area", "diagonal", or "longest_side"
        allow_rotation: Allow 90° rotation on surface
    """
    sort_fn = SORT_STRATEGIES.get(sort_strategy, sort_by_area_desc)
    sorted_items = sort_fn(list(items))

    if algorithm == "maxrect":
        packer = MaximalRectangles(width, depth)
    else:
        packer = SkylinePacker(width, depth)

    placements = []
    for item in sorted_items:
        result = packer.place(item.width, item.depth, allow_rotate=allow_rotation and item.allow_rotation)
        if result is None:
            continue
        px, py, pw, ph = result
        placements.append(Placement2D(
            sku_id=item.sku_id,
            instance_index=item.instance_index,
            x=px, y=py,
            dx=pw, dy=ph, dz=item.height,
            weight_kg=item.weight_kg,
            fragile=item.fragile,
        ))

    logger.debug(
        "[pack_2d] algo=%s sort=%s rotate=%s placed=%d/%d",
        algorithm, sort_strategy, allow_rotation, len(placements), len(items),
    )
    return placements


# ── All 12 algorithm combinations (Table 1) ─────────────────────────

ALGO_COMBINATIONS = [
    ("maxrect", "area", True),        # 1
    ("maxrect", "area", False),       # 2
    ("maxrect", "diagonal", True),    # 3
    ("maxrect", "diagonal", False),   # 4
    ("maxrect", "longest_side", True),  # 5
    ("maxrect", "longest_side", False), # 6
    ("skyline", "area", True),        # 7
    ("skyline", "area", False),       # 8
    ("skyline", "diagonal", True),    # 9
    ("skyline", "diagonal", False),   # 10
    ("skyline", "longest_side", True),  # 11
    ("skyline", "longest_side", False), # 12
]
