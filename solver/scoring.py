"""Scoring function for ranking placement candidates."""

import logging
from .pallet_state import PalletState

logger = logging.getLogger(__name__)

# Tunable weights
W_HEIGHT = 0.30
W_CONTACT = 0.25
W_FRAGILITY = 0.10
W_FILL = 0.20
W_LAYER = 0.15   # Reward layer-aligned placements

# Alternative weight profiles for multi-restart
# Format: (height, contact, fragility, fill, layer)
WEIGHT_PROFILES = {
    "default": (0.30, 0.25, 0.10, 0.20, 0.15),
    "contact_heavy": (0.25, 0.30, 0.10, 0.20, 0.15),
    "fill_heavy": (0.20, 0.25, 0.10, 0.30, 0.15),
    "layer_heavy": (0.20, 0.20, 0.10, 0.20, 0.30),
    "fragile_avoid": (0.25, 0.20, 0.25, 0.15, 0.15),
    "compact": (0.30, 0.25, 0.05, 0.25, 0.15),
    "wall_hugger": (0.20, 0.35, 0.10, 0.15, 0.20),
    "fragile_strict": (0.15, 0.20, 0.40, 0.15, 0.10),  # Very strong fragile avoidance
}


def score_placement(
    state: PalletState,
    dx: int, dy: int, dz: int,
    x: int, y: int, z: int,
    weight_kg: float,
    fragile: bool,
    weight_profile: str = "default",
) -> float:
    """Score a candidate placement. Higher = better.

    Components:
    - height_penalty: prefer lower placements (minimize max height growth)
    - contact_bonus: reward contact with walls and neighbors (stability)
    - fragility_penalty: hard block placing heavy items on fragile ones
    - fill_bonus: reward filling lower gaps
    - layer_bonus: reward placements aligned with existing layer heights

    Args:
        weight_profile: Name of weight profile from WEIGHT_PROFILES.
    """
    wh, wc, wfr, wfi, wl = WEIGHT_PROFILES.get(weight_profile, WEIGHT_PROFILES["default"])

    x2, y2, z2 = x + dx, y + dy, z + dz
    max_h = state.pallet.max_height_mm

    # 1. Height penalty: normalized new max height (lower = better → invert)
    new_max_z = max(state.max_z, z2)
    height_score = 1.0 - (new_max_z / max_h)

    # 2. Contact bonus: normalized by surface area of the box
    contact = state.contact_area_with_neighbors(x, y, z, dx, dy, dz)
    surface_area = 2 * (dx * dy + dy * dz + dx * dz)
    contact_score = min(1.0, contact / surface_area) if surface_area > 0 else 0.0

    # 3. Fragility: soft penalty for placing heavy box on fragile ones
    # Each violation costs 0.05 in fragility_score (weight 0.10 in final score = 0.005 total)
    # Placing more items often outweighs this penalty
    fragility_component = 1.0
    if weight_kg > 2.0 and z > 0:
        fragile_below = state.get_fragile_boxes_at_top(z, x, y, x2, y2)
        if fragile_below:
            fragility_component = 0.0  # Penalty but allow placement

    # 4. Fill bonus: prefer positions with lower z (fill gaps first)
    fill_score = 1.0 - (z / max_h) if max_h > 0 else 0.0

    # 5. Layer alignment: reward placing at z where box tops align with existing layers
    layer_score = 0.0
    if state.boxes:
        # Check if z matches any existing box top (starting a new layer on top of existing)
        for box in state.boxes:
            if box.z_max == z:
                layer_score = 0.5
                break
        # Extra bonus if z2 matches existing box tops (completing a flat surface)
        for box in state.boxes:
            if abs(box.z_max - z2) < 5:  # within 5mm tolerance
                layer_score = 1.0
                break
    else:
        layer_score = 1.0  # First box on floor is fine

    total = (
        wh * height_score
        + wc * contact_score
        + wfr * fragility_component
        + wfi * fill_score
        + wl * layer_score
    )

    return total
