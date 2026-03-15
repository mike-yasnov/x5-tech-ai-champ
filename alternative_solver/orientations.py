"""Generate valid orientations for boxes considering strict_upright constraint."""

import logging
from typing import List, Tuple

from .models import Box

logger = logging.getLogger(__name__)

# (dx, dy, dz, rotation_code) — which original dimension maps to X, Y, Z
# L=length, W=width, H=height of the original box
ALL_ORIENTATIONS = [
    # (x_dim, y_dim, z_dim, code)
    ("L", "W", "H"),  # LWH — original orientation
    ("L", "H", "W"),  # LHW
    ("W", "L", "H"),  # WLH
    ("W", "H", "L"),  # WHL
    ("H", "L", "W"),  # HLW
    ("H", "W", "L"),  # HWL
]


def _dim_value(box: Box, dim_char: str) -> int:
    if dim_char == "L":
        return box.length_mm
    elif dim_char == "W":
        return box.width_mm
    else:
        return box.height_mm


def get_orientations(box: Box) -> List[Tuple[int, int, int, str]]:
    """Return list of (dx, dy, dz, rotation_code) for valid orientations.

    If strict_upright=True, only orientations where original H is along Z axis.
    """
    results = []
    seen = set()

    for x_char, y_char, z_char in ALL_ORIENTATIONS:
        code = x_char + y_char + z_char

        if box.strict_upright and z_char != "H":
            continue

        dx = _dim_value(box, x_char)
        dy = _dim_value(box, y_char)
        dz = _dim_value(box, z_char)

        dims = (dx, dy, dz)
        if dims in seen:
            continue
        seen.add(dims)

        results.append((dx, dy, dz, code))

    logger.debug(
        "[get_orientations] sku=%s strict_upright=%s orientations=%d",
        box.sku_id, box.strict_upright, len(results),
    )
    return results
