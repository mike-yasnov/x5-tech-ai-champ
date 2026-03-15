"""Scoring function for ranking placement candidates."""

import logging
from .pallet_state import PalletState

logger = logging.getLogger(__name__)

# Tunable weights
W_HEIGHT = 0.32
W_CONTACT = 0.26
W_FRAGILITY = 0.2
W_FILL = 0.1
W_FOOTPRINT = 0.07
W_EDGE = 0.05


def score_placement(
    state: PalletState,
    dx: int, dy: int, dz: int,
    x: int, y: int, z: int,
    weight_kg: float,
    fragile: bool,
    strict_fragility: bool = False,
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
    if weight_kg > 2.0 and not fragile and z > 0:
        fragile_below = state.get_fragile_boxes_at_top(z, x, y, x2, y2)
        if fragile_below:
            if strict_fragility:
                return -10.0
            fragility_score = 0.0

    # 4. Fill bonus: prefer positions with lower z (fill gaps first)
    fill_score = 1.0 - (z / max_h) if max_h > 0 else 0.0

    # 5. Footprint bonus: larger floor coverage is useful for stable layers
    pallet_area = max(state.pallet.length_mm * state.pallet.width_mm, 1)
    footprint_score = (dx * dy) / pallet_area

    # 6. Edge bonus: touching walls/corners tends to produce cleaner packings
    edge_contacts = 0
    if x == 0:
        edge_contacts += 1
    if y == 0:
        edge_contacts += 1
    if x2 == state.pallet.length_mm:
        edge_contacts += 1
    if y2 == state.pallet.width_mm:
        edge_contacts += 1
    edge_score = min(edge_contacts / 2.0, 1.0)

    total = (
        W_HEIGHT * height_score
        + W_CONTACT * contact_score
        + W_FRAGILITY * fragility_score
        + W_FILL * fill_score
        + W_FOOTPRINT * footprint_score
        + W_EDGE * edge_score
    )

    return total
