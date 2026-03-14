"""Scoring function for ranking placement candidates."""

import logging
from .pallet_state import PalletState

logger = logging.getLogger(__name__)

# Tunable weights
W_HEIGHT = 0.4
W_CONTACT = 0.3
W_FRAGILITY = 0.2
W_FILL = 0.1


def _center_penalty(state: PalletState, x: int, y: int, dx: int, dy: int) -> float:
    candidate_cx = x + dx / 2.0
    candidate_cy = y + dy / 2.0
    pallet_cx = state.pallet.length_mm / 2.0
    pallet_cy = state.pallet.width_mm / 2.0
    return abs(candidate_cx - pallet_cx) + abs(candidate_cy - pallet_cy)


def score_placement(
    state: PalletState,
    dx: int,
    dy: int,
    dz: int,
    x: int,
    y: int,
    z: int,
    weight_kg: float,
    fragile: bool,
    policy: str = "balanced",
) -> float:
    """Score a candidate placement. Higher = better.

    Components:
    - height_penalty: prefer lower placements (minimize max height growth)
    - contact_bonus: reward contact with walls and neighbors (stability)
    - fragility_penalty: penalize placing heavy items on fragile ones
    - fill_bonus: reward filling lower gaps
    """
    x2, y2, z2 = x + dx, y + dy, z + dz
    max_h = state.pallet.max_height_mm

    # 1. Height penalty: normalized new max height (lower = better → invert)
    new_max_z = max(state.max_z, z2)
    height_score = 1.0 - (new_max_z / max_h)

    # 2. Contact bonus: normalized by surface area of the box
    contact = state.contact_area_with_neighbors(x, y, z, dx, dy, dz)
    surface_area = 2 * (dx * dy + dy * dz + dx * dz)
    contact_score = min(1.0, contact / surface_area) if surface_area > 0 else 0.0

    # 3. Fragility penalty: check if we're placing heavy box on fragile ones
    fragility_score = 1.0
    if weight_kg > 2.0 and z > 0:
        fragile_below = state.get_fragile_boxes_at_top(z, x, y, x2, y2)
        if fragile_below:
            fragility_score = 0.0

    # 4. Fill bonus: prefer positions with lower z (fill gaps first)
    fill_score = 1.0 - (z / max_h) if max_h > 0 else 0.0

    support_area = dx * dy if z == 0 else 0
    if z > 0:
        for box in state.boxes:
            if box.z_max == z:
                ox = max(0, min(x2, box.x_max) - max(x, box.x_min))
                oy = max(0, min(y2, box.y_max) - max(y, box.y_min))
                support_area += ox * oy
    support_score = min(1.0, support_area / max(1, dx * dy))
    fragile_top_score = (z2 / max_h) if (fragile and max_h > 0) else fill_score

    center_score = 1.0 - min(
        1.0,
        _center_penalty(state, x, y, dx, dy)
        / max(1.0, state.pallet.length_mm + state.pallet.width_mm),
    )

    if policy == "dbfl":
        total = 0.6 * fill_score + 0.25 * height_score + 0.15 * contact_score
    elif policy == "max_support":
        total = (
            0.55 * support_score
            + 0.20 * fragility_score
            + 0.15 * fill_score
            + 0.10 * height_score
        )
    elif policy == "max_contact":
        total = (
            0.45 * contact_score
            + 0.25 * support_score
            + 0.15 * fill_score
            + 0.15 * fragility_score
        )
    elif policy == "min_height":
        total = (
            0.50 * height_score
            + 0.25 * fill_score
            + 0.15 * support_score
            + 0.10 * fragility_score
        )
    elif policy == "center_stable":
        total = (
            0.35 * support_score
            + 0.25 * center_score
            + 0.20 * fragility_score
            + 0.20 * height_score
        )
    elif policy == "fragile_safe":
        total = (
            0.35 * fragility_score
            + 0.20 * support_score
            + 0.15 * contact_score
            + 0.15 * fragile_top_score
            + 0.15 * height_score
        )
    else:
        total = (
            W_HEIGHT * height_score
            + W_CONTACT * contact_score
            + W_FRAGILITY * fragility_score
            + W_FILL * fill_score
        )

    return total
