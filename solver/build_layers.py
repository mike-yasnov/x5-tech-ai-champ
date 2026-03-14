"""BuildLayers: generate candidate layers using 12 2D packing algorithms × families.

Algorithm 1 from Dell'Amico et al. (2026).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

from .models import Box, Pallet
from .families import generate_family_configs
from .packing2d import ALGO_COMBINATIONS, BoxItem, Placement2D, pack_2d

logger = logging.getLogger(__name__)


@dataclass
class Layer:
    """A candidate layer: a 2D packing of boxes at a certain height."""
    placements: List[Placement2D]
    height_mm: int        # max height of boxes in this layer
    total_weight_kg: float
    net_area: int         # total area occupied by boxes
    box_counts: Dict[str, int]  # sku_id -> count in this layer
    pallet_area: int      # total pallet area for density calculation

    @property
    def density(self) -> float:
        return self.net_area / self.pallet_area if self.pallet_area > 0 else 0.0

    @property
    def box_count(self) -> int:
        return sum(self.box_counts.values())

    @property
    def signature(self) -> FrozenSet[Tuple[str, int]]:
        """Unique signature for deduplication."""
        return frozenset(self.box_counts.items())

    @property
    def total_volume(self) -> int:
        return sum(p.dx * p.dy * p.dz for p in self.placements)


def _family_to_box_items(family: List[Box]) -> List[BoxItem]:
    """Expand a family of box types into individual BoxItems for 2D packing."""
    items = []
    for box in family:
        allow_rot = not box.strict_upright  # strict_upright only restricts vertical rotation
        for i in range(box.quantity):
            items.append(BoxItem(
                sku_id=box.sku_id,
                instance_index=i,
                width=box.length_mm,
                depth=box.width_mm,
                height=box.height_mm,
                weight_kg=box.weight_kg,
                fragile=box.fragile,
                strict_upright=box.strict_upright,
                allow_rotation=True,  # 2D rotation (90° on surface) is always allowed
            ))
    return items


def _placements_to_layer(placements: List[Placement2D], pallet: Pallet) -> Layer:
    """Convert 2D placements to a Layer object."""
    if not placements:
        return Layer([], 0, 0.0, 0, {}, pallet.length_mm * pallet.width_mm)

    height = max(p.dz for p in placements)
    weight = sum(p.weight_kg for p in placements)
    area = sum(p.dx * p.dy for p in placements)
    counts: Dict[str, int] = {}
    for p in placements:
        counts[p.sku_id] = counts.get(p.sku_id, 0) + 1

    return Layer(
        placements=placements,
        height_mm=height,
        total_weight_kg=weight,
        net_area=area,
        box_counts=counts,
        pallet_area=pallet.length_mm * pallet.width_mm,
    )


def build_layers(
    pallet: Pallet,
    boxes: List[Box],
    delta_h: int = 50,
    seed: int = 42,
) -> List[Layer]:
    """Generate candidate layers using Algorithm 1 from the paper.

    For each family partition × each of 12 2D algorithms → unique layers.

    Args:
        pallet: Pallet dimensions
        boxes: Box types with quantities
        delta_h: Height tolerance for family creation
        seed: Random seed
    """
    family_configs = generate_family_configs(boxes, delta_h=delta_h, seed=seed)

    all_layers: List[Layer] = []
    seen_signatures: set = set()

    for config_idx, families in enumerate(family_configs):
        for family_idx, family in enumerate(families):
            items = _family_to_box_items(family)
            if not items:
                continue

            for algo, sort_strat, allow_rot in ALGO_COMBINATIONS:
                placements = pack_2d(
                    pallet.length_mm, pallet.width_mm,
                    items,
                    algorithm=algo,
                    sort_strategy=sort_strat,
                    allow_rotation=allow_rot,
                )

                if not placements:
                    continue

                layer = _placements_to_layer(placements, pallet)

                # Deduplicate by box composition
                sig = layer.signature
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

                all_layers.append(layer)

    logger.info(
        "[build_layers] configs=%d unique_layers=%d total_boxes_covered=%d",
        len(family_configs), len(all_layers),
        sum(l.box_count for l in all_layers),
    )

    return all_layers
