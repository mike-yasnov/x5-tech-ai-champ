"""Scoring function for ranking placement candidates."""

import logging
from dataclasses import dataclass
from typing import Optional

from .pallet_state import PalletState

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Tunable weights for scoring function."""
    height: float = 0.25
    contact: float = 0.30
    fragility: float = 0.20
    fill: float = 0.15
    support_quality: float = 0.10


DEFAULT_WEIGHTS = ScoringWeights()


def score_placement(
    state: PalletState,
    dx: int, dy: int, dz: int,
    x: int, y: int, z: int,
    weight_kg: float,
    fragile: bool,
    weights: Optional[ScoringWeights] = None,
) -> float:
    """Score a candidate placement. Higher = better.

    Components:
    - height: prefer lower placements (minimize max height growth)
    - contact: reward contact with walls and neighbors (stability)
    - fragility: penalize placing heavy items on fragile ones
    - fill: reward filling lower gaps
    - support_quality: reward higher support area ratio
    """
    w = weights or DEFAULT_WEIGHTS
    x2, y2, z2 = x + dx, y + dy, z + dz
    max_h = state.pallet.max_height_mm

    # 1. Height: normalized new max height (lower = better)
    new_max_z = max(state.max_z, z2)
    height_score = 1.0 - (new_max_z / max_h)

    # 2. Contact: normalized by surface area
    contact = state.contact_area_with_neighbors(x, y, z, dx, dy, dz)
    surface_area = 2 * (dx * dy + dy * dz + dx * dz)
    contact_score = min(1.0, contact / surface_area) if surface_area > 0 else 0.0

    # 3. Fragility: penalize heavy on fragile
    fragility_score = 1.0
    if weight_kg > 2.0 and z > 0:
        fragile_below = state.get_fragile_boxes_at_top(z, x, y, x2, y2)
        if fragile_below:
            fragility_score = 0.0

    # 4. Fill: prefer lower z positions
    fill_score = 1.0 - (z / max_h) if max_h > 0 else 0.0

    # 5. Support quality: how well-supported is this placement
    base_area = dx * dy
    support_score = 1.0
    if z > 0 and base_area > 0:
        support_area = state.get_support_area(x, y, x2, y2, z)
        support_score = min(1.0, support_area / base_area)

    total = (
        w.height * height_score
        + w.contact * contact_score
        + w.fragility * fragility_score
        + w.fill * fill_score
        + w.support_quality * support_score
    )

    return total
