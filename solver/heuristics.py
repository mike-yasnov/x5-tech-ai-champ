"""Heuristic catalog for comparative CI benchmarking."""

from dataclasses import dataclass
import random
from typing import Callable, Dict, List, Tuple

from .models import Box


SortKey = Callable[[Box], tuple]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    sort_key_name: str
    placement_policy: str
    randomized: bool = False
    noise_factor: float = 0.0


def _sort_volume_desc(box: Box) -> tuple:
    return (-box.volume, -box.base_area, -box.weight_kg)


def _sort_weight_desc(box: Box) -> tuple:
    return (-box.weight_kg, -box.base_area, -box.volume)


def _sort_base_area_desc(box: Box) -> tuple:
    return (-box.base_area, -box.volume, -box.weight_kg)


def _sort_density_desc(box: Box) -> tuple:
    volume = box.volume if box.volume > 0 else 1
    return (-box.weight_kg / volume, -box.weight_kg, -box.base_area)


def _sort_constrained_first(box: Box) -> tuple:
    priority = 0
    if box.strict_upright:
        priority -= 2
    if not box.stackable:
        priority -= 1
    if box.fragile:
        priority += 2
    return (priority, -box.volume, -box.base_area)


def _sort_max_dim_desc(box: Box) -> tuple:
    return (-max(box.length_mm, box.width_mm, box.height_mm), -box.volume)


def _sort_perimeter_desc(box: Box) -> tuple:
    return (-(box.length_mm + box.width_mm), -box.base_area, -box.volume)


def _sort_fragile_last(box: Box) -> tuple:
    return (box.fragile, not box.stackable, -box.weight_kg, -box.volume)


def _sort_stackable_first(box: Box) -> tuple:
    return (not box.stackable, box.fragile, -box.base_area, -box.weight_kg)


def _sort_layer_height_desc(box: Box) -> tuple:
    return (-(box.height_mm // 50), -box.base_area, box.fragile, -box.volume)


def _sort_homogeneous_rank(box: Box) -> tuple:
    return (box.sku_id,)


def _sort_upright_first(box: Box) -> tuple:
    return (not box.strict_upright, box.fragile, -box.base_area, -box.weight_kg)


def _sort_slenderness_desc(box: Box) -> tuple:
    mn = min(box.length_mm, box.width_mm, box.height_mm)
    slenderness = max(box.length_mm, box.width_mm, box.height_mm) / max(1, mn)
    return (-slenderness, -box.volume, -box.base_area)


def _sort_weighted_volume(box: Box) -> tuple:
    return (-(0.65 * box.volume + 35000.0 * box.weight_kg), -box.base_area)


def _sort_smalls_last(box: Box) -> tuple:
    mn = min(box.length_mm, box.width_mm, box.height_mm)
    return (-box.volume, -mn, -box.weight_kg)


SORT_KEYS: Dict[str, SortKey] = {
    "volume_desc": _sort_volume_desc,
    "weight_desc": _sort_weight_desc,
    "base_area_desc": _sort_base_area_desc,
    "density_desc": _sort_density_desc,
    "constrained_first": _sort_constrained_first,
    "max_dim_desc": _sort_max_dim_desc,
    "perimeter_desc": _sort_perimeter_desc,
    "fragile_last": _sort_fragile_last,
    "stackable_first": _sort_stackable_first,
    "layer_height_desc": _sort_layer_height_desc,
    "homogeneous": _sort_homogeneous_rank,
    "upright_first": _sort_upright_first,
    "slenderness_desc": _sort_slenderness_desc,
    "weighted_volume": _sort_weighted_volume,
    "smalls_last": _sort_smalls_last,
}


def sort_boxes(boxes: List[Box], sort_key_name: str) -> List[Box]:
    key = SORT_KEYS.get(sort_key_name, _sort_volume_desc)
    if sort_key_name == "homogeneous":
        groups: Dict[str, List[Box]] = {}
        for box in boxes:
            groups.setdefault(box.sku_id, []).append(box)
        ordered_groups = sorted(
            groups.values(),
            key=lambda group: (
                -(len(group) * group[0].volume),
                -len(group),
                -group[0].base_area,
            ),
        )
        flattened: List[Box] = []
        for group in ordered_groups:
            flattened.extend(group)
        return flattened
    return sorted(boxes, key=key)


def perturb_boxes(boxes: List[Box], noise_factor: float) -> List[Box]:
    items = list(boxes)
    if len(items) < 2 or noise_factor <= 0:
        return items
    swaps = max(1, int(len(items) * noise_factor))
    for _ in range(swaps):
        i = random.randint(0, len(items) - 2)
        j = min(len(items) - 1, i + random.randint(1, min(5, len(items) - i - 1)))
        items[i], items[j] = items[j], items[i]
    return items


STRATEGY_CONFIGS: List[StrategyConfig] = [
    StrategyConfig("volume__balanced", "volume_desc", "balanced"),
    StrategyConfig("volume__dbfl", "volume_desc", "dbfl"),
    StrategyConfig("volume__max_contact", "volume_desc", "max_contact"),
    StrategyConfig("area__balanced", "base_area_desc", "balanced"),
    StrategyConfig("area__max_support", "base_area_desc", "max_support"),
    StrategyConfig("area__max_contact", "base_area_desc", "max_contact"),
    StrategyConfig("weight__balanced", "weight_desc", "balanced"),
    StrategyConfig("weight__center_stable", "weight_desc", "center_stable"),
    StrategyConfig("weight__fragile_last", "fragile_last", "center_stable"),
    StrategyConfig("density__center_stable", "density_desc", "center_stable"),
    StrategyConfig("density__max_contact", "density_desc", "max_contact"),
    StrategyConfig("constrained__max_support", "constrained_first", "max_support"),
    StrategyConfig("max_dim__dbfl", "max_dim_desc", "dbfl"),
    StrategyConfig("max_dim__min_height", "max_dim_desc", "min_height"),
    StrategyConfig("perimeter__max_contact", "perimeter_desc", "max_contact"),
    StrategyConfig("stackable__max_support", "stackable_first", "max_support"),
    StrategyConfig("layer__min_height", "layer_height_desc", "min_height"),
    StrategyConfig("layer__max_support", "layer_height_desc", "max_support"),
    StrategyConfig("homogeneous__dbfl", "homogeneous", "dbfl"),
    StrategyConfig("homogeneous__max_support", "homogeneous", "max_support"),
    StrategyConfig("upright__max_support", "upright_first", "max_support"),
    StrategyConfig("slender__min_height", "slenderness_desc", "min_height"),
    StrategyConfig("slender__max_support", "slenderness_desc", "max_support"),
    StrategyConfig("weighted_volume__center", "weighted_volume", "center_stable"),
    StrategyConfig("smalls_last__max_contact", "smalls_last", "max_contact"),
]

STRATEGY_CONFIGS.extend(
    StrategyConfig(
        name=f"{sort_key_name}__mutated",
        sort_key_name=sort_key_name,
        placement_policy="dbfl",
        randomized=True,
        noise_factor=0.18,
    )
    for sort_key_name in SORT_KEYS
)
