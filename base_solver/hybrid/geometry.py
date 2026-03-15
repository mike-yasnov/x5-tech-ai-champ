from __future__ import annotations


class AABB:
    """Axis-aligned bounding box with integer mm coordinates."""

    __slots__ = ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")

    def __init__(
        self, x_min: int, y_min: int, z_min: int, x_max: int, y_max: int, z_max: int
    ):
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    def volume(self) -> int:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min) * (self.z_max - self.z_min)

    def base_area(self) -> int:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def length_x(self) -> int:
        return self.x_max - self.x_min

    def width_y(self) -> int:
        return self.y_max - self.y_min

    def height_z(self) -> int:
        return self.z_max - self.z_min

    def overlaps_3d(self, other: AABB) -> bool:
        ox = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        oy = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        oz = min(self.z_max, other.z_max) - max(self.z_min, other.z_min)
        return ox > 0 and oy > 0 and oz > 0

    def overlap_area_xy(self, other: AABB) -> int:
        dx = max(0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        dy = max(0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        return dx * dy

    def __repr__(self) -> str:
        return (
            f"AABB(x=[{self.x_min},{self.x_max}], "
            f"y=[{self.y_min},{self.y_max}], "
            f"z=[{self.z_min},{self.z_max}])"
        )


class Rect2D:
    """2D rectangle for free-space tracking on XY plane."""

    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def area(self) -> int:
        return self.w * self.h

    def fits(self, bw: int, bh: int) -> bool:
        return bw <= self.w and bh <= self.h

    def __repr__(self) -> str:
        return f"Rect2D(x={self.x}, y={self.y}, w={self.w}, h={self.h})"
