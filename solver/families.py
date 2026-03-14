"""CreateFamilies: group box types by similar heights (Algorithm 2 from Dell'Amico et al.)."""

import logging
import random
from typing import Dict, List, Tuple

from .models import Box

logger = logging.getLogger(__name__)


def create_families(
    boxes: List[Box],
    num_families: int,
    delta_h: int = 50,
    seed: int = 42,
) -> List[List[Box]]:
    """Partition box types into families with similar heights.

    Algorithm 2 from Dell'Amico et al. (2026):
    1. Sort boxes by height
    2. Divide into f intervals, pick random pivot from each
    3. Assign boxes within ±½(1+γ)Δh of pivot height

    Args:
        boxes: List of box types
        num_families: Number of families to create (f)
        delta_h: Height tolerance parameter (Δh in paper, default 50mm)
        seed: Random seed for pivot selection
    """
    rng = random.Random(seed)

    if not boxes:
        return []

    # Sort by height
    sorted_boxes = sorted(boxes, key=lambda b: b.height_mm)
    n = len(sorted_boxes)

    if num_families >= n:
        # Each box is its own family
        return [[b] for b in sorted_boxes]

    # Select pivot indices from each subinterval
    interval_size = max(1, n // num_families)
    pivots = []
    for j in range(num_families):
        lo = j * interval_size
        hi = min((j + 1) * interval_size, n) - 1
        pivot_idx = rng.randint(lo, hi)
        pivots.append(pivot_idx)

    # γ ∈ [0.3, 0.5]
    gamma = rng.uniform(0.3, 0.5)
    half_range = 0.5 * (1 + gamma) * delta_h

    # Assign boxes to families
    families: List[List[Box]] = [[] for _ in range(num_families)]
    assigned = set()

    for j in range(num_families):
        pivot_h = sorted_boxes[pivots[j]].height_mm
        for i, box in enumerate(sorted_boxes):
            if i in assigned:
                continue
            if abs(box.height_mm - pivot_h) <= half_range:
                families[j].append(box)
                assigned.add(i)

    # Assign remaining boxes to nearest pivot
    for i, box in enumerate(sorted_boxes):
        if i in assigned:
            continue
        best_j = 0
        best_dist = float("inf")
        for j in range(num_families):
            pivot_h = sorted_boxes[pivots[j]].height_mm
            dist = abs(box.height_mm - pivot_h)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        families[best_j].append(box)
        assigned.add(i)

    # Remove empty families
    families = [f for f in families if f]

    logger.info(
        "[create_families] f=%d delta_h=%d gamma=%.2f families=%s",
        num_families, delta_h, gamma,
        [len(f) for f in families],
    )

    return families


def generate_family_configs(
    boxes: List[Box],
    delta_h: int = 50,
    seed: int = 42,
) -> List[List[List[Box]]]:
    """Generate multiple family configurations by varying f from fmin to fmax.

    Returns list of family partitions for each f value.
    """
    n_types = len(set(b.sku_id for b in boxes))
    fmin = max(1, n_types // 3)
    fmax = max(fmin + 1, n_types)

    configs = []
    for f in range(fmin, fmax + 1):
        families = create_families(boxes, f, delta_h=delta_h, seed=seed + f)
        configs.append(families)

    logger.info(
        "[generate_family_configs] fmin=%d fmax=%d configs=%d",
        fmin, fmax, len(configs),
    )
    return configs
