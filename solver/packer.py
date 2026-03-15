"""Greedy packer: places boxes one by one using Extreme Points + scoring."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
import heapq
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .orientations import get_orientations
from .pallet_state import PalletState
from .scoring import score_placement
from . import __version__

logger = logging.getLogger(__name__)


# ── Sort key factories ──────────────────────────────────────────────

def _sort_volume_desc(box: Box) -> tuple:
    return (-box.volume,)


def _sort_weight_desc(box: Box) -> tuple:
    return (-box.weight_kg,)


def _sort_base_area_desc(box: Box) -> tuple:
    return (-box.base_area,)


def _sort_density_desc(box: Box) -> tuple:
    vol = box.volume if box.volume > 0 else 1
    return (-box.weight_kg / vol,)


def _sort_constrained_first(box: Box) -> tuple:
    # Constrained items first, then by volume desc
    priority = 0
    if box.strict_upright:
        priority -= 2
    if not box.stackable:
        priority -= 1
    return (priority, -box.volume)


def _sort_coverage_tie(box: Box) -> tuple:
    return (box.volume, box.base_area, -box.quantity, box.weight_kg)


SORT_KEYS: Dict[str, Callable[[Box], tuple]] = {
    "volume_desc": _sort_volume_desc,
    "weight_desc": _sort_weight_desc,
    "base_area_desc": _sort_base_area_desc,
    "density_desc": _sort_density_desc,
    "constrained_first": _sort_constrained_first,
    "coverage_tie": _sort_coverage_tie,
}


@dataclass(frozen=True)
class _LayerOriented:
    box: Box
    dx: int
    dy: int
    dz: int
    rotation_code: str


@dataclass(frozen=True)
class _LayerPattern:
    sequence: Tuple[Tuple[_LayerOriented, int, int, int], ...]
    dx: int
    dy: int
    height: int
    kind: str
    item_count: int
    constrained_count: int
    volume: int


# ── Expand boxes ────────────────────────────────────────────────────

def _expand_boxes(boxes: List[Box]) -> List[Tuple[Box, int]]:
    """Expand SKUs with quantity > 1 into individual (box, instance_index) pairs."""
    result = []
    for box in boxes:
        for i in range(box.quantity):
            result.append((box, i))
    return result


def order_boxes(boxes: Iterable[Box], sort_key_name: str = "volume_desc") -> List[Box]:
    sort_fn = SORT_KEYS.get(sort_key_name, _sort_volume_desc)
    return sorted(list(boxes), key=sort_fn)


def _pack_instances(
    task_id: str,
    pallet: Pallet,
    instances: List[Tuple[Box, int]],
    sort_label: str,
    strict_fragility: bool = False,
) -> Solution:
    t0 = time.perf_counter()

    state = PalletState(pallet)
    placements: List[Placement] = []
    unplaced_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0})

    logger.info(
        "[pack_greedy] task=%s sort=%s total_instances=%d",
        task_id, sort_label, len(instances),
    )

    for box, inst_idx in instances:
        orientations = get_orientations(box)
        best_score = -1.0
        best_placement: Optional[Tuple[int, int, int, int, int, int, str]] = None

        # Try each EP x orientation
        for ep in list(state.extreme_points):
            ex, ey, ez = ep
            for dx, dy, dz, rot_code in orientations:
                if not state.can_place(
                    dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                ):
                    continue

                sc = score_placement(
                    state, dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile,
                    strict_fragility=strict_fragility,
                )
                if sc > best_score:
                    best_score = sc
                    best_placement = (dx, dy, dz, ex, ey, ez, rot_code)

        if best_placement is not None:
            dx, dy, dz, px, py, pz, rot_code = best_placement
            state.place(
                box.sku_id, dx, dy, dz, px, py, pz,
                box.weight_kg, box.fragile, box.stackable,
            )
            placements.append(Placement(
                sku_id=box.sku_id,
                instance_index=inst_idx,
                x_mm=px, y_mm=py, z_mm=pz,
                length_mm=dx, width_mm=dy, height_mm=dz,
                rotation_code=rot_code,
            ))
        else:
            if state.current_weight + box.weight_kg > pallet.max_weight_kg:
                reason = "weight_limit_exceeded"
            else:
                reason = "no_space"
            unplaced_counts[box.sku_id]["count"] += 1
            unplaced_counts[box.sku_id]["reason"] = reason
            logger.debug(
                "[pack_greedy] unplaced sku=%s instance=%d reason=%s",
                box.sku_id, inst_idx, reason,
            )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    unplaced = [
        UnplacedItem(sku_id=sid, quantity_unplaced=info["count"], reason=info["reason"])
        for sid, info in unplaced_counts.items()
    ]

    logger.info(
        "[pack_greedy] done sort=%s placed=%d unplaced=%d time=%dms",
        sort_label, len(placements), sum(u.quantity_unplaced for u in unplaced), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )


def _repeat_options(max_repeat: int) -> List[int]:
    if max_repeat <= 1:
        return [1]
    values = {1, max_repeat, max(1, int(round(max_repeat * 0.5)))}
    if max_repeat >= 3:
        values.add(max(1, int(round(max_repeat * 0.66))))
    if max_repeat >= 4:
        values.add(max_repeat - 1)
    if max_repeat >= 6:
        values.add(max(2, int(round(max_repeat * 0.33))))
    if max_repeat >= 8:
        values.add(max_repeat - 2)
    return sorted(v for v in values if 1 <= v <= max_repeat)


def _fits_rect(
    rects: Tuple[Tuple[int, int, int, int], ...],
    dx: int,
    dy: int,
    x: int,
    y: int,
    pallet: Pallet,
) -> bool:
    if x + dx > pallet.length_mm or y + dy > pallet.width_mm:
        return False
    for rx, ry, rdx, rdy in rects:
        if x < rx + rdx and x + dx > rx and y < ry + rdy and y + dy > ry:
            return False
    return True


def _inside_rect(rect: Tuple[int, int, int, int], ep: Tuple[int, int]) -> bool:
    rx, ry, rdx, rdy = rect
    x, y = ep
    return rx <= x < rx + rdx and ry <= y < ry + rdy


def _layer_orientations(box: Box) -> List[_LayerOriented]:
    orientations: List[_LayerOriented] = []
    seen = set()
    for dx, dy, dz, rotation_code in get_orientations(box):
        key = (dx, dy, dz)
        if key in seen:
            continue
        seen.add(key)
        orientations.append(
            _LayerOriented(
                box=box,
                dx=dx,
                dy=dy,
                dz=dz,
                rotation_code=rotation_code,
            )
        )
    return orientations


def _build_layer_patterns(
    pallet: Pallet,
    boxes: List[Box],
) -> List[_LayerPattern]:
    orienteds: List[_LayerOriented] = []
    for box in boxes:
        orienteds.extend(_layer_orientations(box))

    patterns: Dict[Tuple[Tuple[Tuple[str, str, int], ...], int, int, int], _LayerPattern] = {}

    def add_pattern(
        sequence: Tuple[Tuple[_LayerOriented, int, int, int], ...],
        dx: int,
        dy: int,
        height: int,
        kind: str,
    ) -> None:
        item_count = sum(count for _, count, _, _ in sequence)
        constrained_count = sum(
            count
            for oriented, count, _, _ in sequence
            if oriented.box.fragile or not oriented.box.stackable
        )
        volume = sum(oriented.box.volume * count for oriented, count, _, _ in sequence)
        pattern = _LayerPattern(
            sequence=sequence,
            dx=dx,
            dy=dy,
            height=height,
            kind=kind,
            item_count=item_count,
            constrained_count=constrained_count,
            volume=volume,
        )
        key = (
            tuple(
                (entry.box.sku_id, entry.rotation_code, count, offset_x, offset_y)
                for entry, count, offset_x, offset_y in pattern.sequence
            ),
            pattern.dx,
            pattern.dy,
            pattern.height,
        )
        patterns[key] = pattern

    def centered_top_layout(
        support_dx: int,
        support_dy: int,
        top_dx: int,
        top_dy: int,
    ) -> Optional[Tuple[int, int, int, int, int, int]]:
        overlap_x = min(support_dx, top_dx)
        overlap_y = min(support_dy, top_dy)
        support_ratio = (overlap_x * overlap_y) / max(top_dx * top_dy, 1)
        if support_ratio < 0.6:
            return None
        raw_top_x = (support_dx - top_dx) // 2
        raw_top_y = (support_dy - top_dy) // 2
        x_min = min(0, raw_top_x)
        y_min = min(0, raw_top_y)
        x_max = max(support_dx, raw_top_x + top_dx)
        y_max = max(support_dy, raw_top_y + top_dy)
        support_offset_x = -x_min
        support_offset_y = -y_min
        top_offset_x = raw_top_x - x_min
        top_offset_y = raw_top_y - y_min
        return (
            support_offset_x,
            support_offset_y,
            top_offset_x,
            top_offset_y,
            x_max - x_min,
            y_max - y_min,
        )

    for oriented in orienteds:
        max_repeat = 1
        if oriented.box.stackable and (
            oriented.box.weight_kg <= 2.0 or not oriented.box.fragile
        ):
            max_repeat = min(
                oriented.box.quantity,
                pallet.max_height_mm // max(oriented.dz, 1),
            )
        for repeat in _repeat_options(max_repeat):
            add_pattern(
                sequence=((oriented, repeat, 0, 0),),
                dx=oriented.dx,
                dy=oriented.dy,
                height=repeat * oriented.dz,
                kind="support",
            )

    tops = [o for o in orienteds if o.box.fragile or not o.box.stackable]
    supports = [o for o in orienteds if o.box.stackable]
    mixed_supports = [o for o in supports if not o.box.fragile]
    for top in tops:
        add_pattern(
            sequence=((top, 1, 0, 0),),
            dx=top.dx,
            dy=top.dy,
            height=top.dz,
            kind="top",
        )

        for support in supports:
            if top.box.weight_kg > 2.0 and not top.box.fragile and support.box.fragile:
                continue
            top_layout = centered_top_layout(support.dx, support.dy, top.dx, top.dy)
            if top_layout is None:
                continue
            (
                support_offset_x,
                support_offset_y,
                top_offset_x,
                top_offset_y,
                pattern_dx,
                pattern_dy,
            ) = top_layout
            max_repeat = min(
                support.box.quantity,
                (pallet.max_height_mm - top.dz) // max(support.dz, 1),
            )
            if support.box.weight_kg > 2.0 and support.box.fragile:
                max_repeat = min(max_repeat, 1)
            for repeat in _repeat_options(max_repeat):
                add_pattern(
                    sequence=(
                        (support, repeat, support_offset_x, support_offset_y),
                        (top, 1, top_offset_x, top_offset_y),
                    ),
                    dx=pattern_dx,
                    dy=pattern_dy,
                    height=repeat * support.dz + top.dz,
                    kind="top_support",
                )

    for lower in mixed_supports:
        max_lower = min(
            lower.box.quantity,
            pallet.max_height_mm // max(lower.dz, 1),
        )
        for lower_repeat in _repeat_options(max_lower):
            lower_height = lower_repeat * lower.dz
            for upper in mixed_supports:
                if (
                    upper.box.sku_id == lower.box.sku_id
                    and upper.rotation_code == lower.rotation_code
                ):
                    continue
                if upper.dx > lower.dx or upper.dy > lower.dy:
                    continue
                if (
                    upper.box.weight_kg > 2.0
                    and not upper.box.fragile
                    and lower.box.fragile
                ):
                    continue
                max_upper = min(
                    upper.box.quantity,
                    (pallet.max_height_mm - lower_height) // max(upper.dz, 1),
                )
                if max_upper <= 0:
                    continue
                for upper_repeat in _repeat_options(max_upper):
                    upper_height = lower_height + upper_repeat * upper.dz
                    add_pattern(
                        sequence=((lower, lower_repeat, 0, 0), (upper, upper_repeat, 0, 0)),
                        dx=lower.dx,
                        dy=lower.dy,
                        height=upper_height,
                        kind="support_mix",
                    )
                    for top in tops:
                        if (
                            top.box.weight_kg > 2.0
                            and not top.box.fragile
                            and upper.box.fragile
                        ):
                            continue
                        top_layout = centered_top_layout(upper.dx, upper.dy, top.dx, top.dy)
                        if top_layout is None:
                            continue
                        (
                            upper_support_offset_x,
                            upper_support_offset_y,
                            top_offset_x,
                            top_offset_y,
                            pattern_dx,
                            pattern_dy,
                        ) = top_layout
                        if upper_height + top.dz > pallet.max_height_mm:
                            continue
                        add_pattern(
                            sequence=(
                                (lower, lower_repeat, 0, 0),
                                (
                                    upper,
                                    upper_repeat,
                                    upper_support_offset_x,
                                    upper_support_offset_y,
                                ),
                                (top, 1, top_offset_x, top_offset_y),
                            ),
                            dx=max(lower.dx, pattern_dx),
                            dy=max(lower.dy, pattern_dy),
                            height=upper_height + top.dz,
                            kind="mixed_top_support",
                        )

    return list(patterns.values())


def _layer_state_value(
    placed_volume: int,
    placed_items: int,
    placed_constrained: int,
    used_area: int,
    pallet: Pallet,
    total_items: int,
    total_constrained: int,
) -> float:
    vol_norm = placed_volume / max(
        pallet.length_mm * pallet.width_mm * pallet.max_height_mm, 1
    )
    cov_norm = placed_items / max(total_items, 1)
    constrained_norm = placed_constrained / max(total_constrained, 1)
    area_norm = used_area / max(pallet.length_mm * pallet.width_mm, 1)
    return (
        vol_norm * 0.52
        + cov_norm * 0.28
        + constrained_norm * 0.14
        + area_norm * 0.06
    )


def _select_layer_branches(
    candidates: List[Tuple[float, _LayerPattern, int, int]],
    branch_width: int,
) -> List[Tuple[float, _LayerPattern, int, int]]:
    if len(candidates) <= branch_width:
        return sorted(candidates, key=lambda item: item[0], reverse=True)

    buckets: Dict[str, List[Tuple[float, _LayerPattern, int, int]]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate[1].kind].append(candidate)
    for kind in buckets:
        buckets[kind].sort(key=lambda item: item[0], reverse=True)

    selected: List[Tuple[float, _LayerPattern, int, int]] = []
    for kind in ("mixed_top_support", "top_support", "top", "support_mix", "support"):
        if kind in buckets and buckets[kind]:
            selected.append(buckets[kind].pop(0))
            if len(selected) >= branch_width:
                return selected[:branch_width]

    leftovers = []
    for bucket in buckets.values():
        leftovers.extend(bucket)
    leftovers.sort(key=lambda item: item[0], reverse=True)
    selected.extend(leftovers[: max(0, branch_width - len(selected))])
    return selected[:branch_width]


def _layer_candidate_score(
    pattern: _LayerPattern,
    x: int,
    y: int,
    rem_constrained: int,
    pallet: Pallet,
    total_items: int,
    total_constrained: int,
) -> float:
    footprint = pattern.dx * pattern.dy
    height_eff = pattern.volume / max(footprint * pattern.height, 1)
    gain_score = _layer_state_value(
        placed_volume=pattern.volume,
        placed_items=pattern.item_count,
        placed_constrained=pattern.constrained_count,
        used_area=footprint,
        pallet=pallet,
        total_items=total_items,
        total_constrained=max(total_constrained, 1),
    )
    score = (
        gain_score * 100.0
        + pattern.constrained_count * 3.5
        + height_eff * 14.0
        + (1.4 if pattern.kind in {"top_support", "mixed_top_support"} else 0.0)
        + (0.5 if pattern.kind == "support_mix" else 0.0)
        + (1.0 if x == 0 else 0.0)
        + (1.0 if y == 0 else 0.0)
        - pattern.height / 20000.0
    )
    if rem_constrained and pattern.constrained_count == 0:
        score -= 5.0 if pattern.item_count <= 2 else 2.0
    return score


def _use_exact_layer_search(
    pallet: Pallet,
    boxes: List[Box],
    patterns: List[_LayerPattern],
) -> bool:
    return False


def _exact_upright_layer_search(
    pallet: Pallet,
    patterns: List[_LayerPattern],
    remaining0: Dict[str, int],
    volume0: Dict[str, int],
    constrained_skus: set[str],
    total_items: int,
    total_constrained: int,
    deadline: float,
) -> Tuple[
    Tuple[Tuple[int, int, int, int], ...],
    Tuple[Tuple[int, int], ...],
    Dict[str, int],
    List[Tuple[_LayerPattern, int, int]],
]:
    root = (tuple(), ((0, 0),), remaining0, [])
    best_state = root
    best_value = -1.0
    seen: Dict[
        Tuple[Tuple[Tuple[int, int, int, int], ...], Tuple[Tuple[str, int], ...], Tuple[Tuple[int, int], ...]],
        float,
    ] = {}
    heap: List[
        Tuple[
            float,
            int,
            Tuple[Tuple[int, int, int, int], ...],
            Tuple[Tuple[int, int], ...],
            Dict[str, int],
            List[Tuple[_LayerPattern, int, int]],
        ]
    ] = []
    counter = 0
    heapq.heappush(heap, (-0.0, counter, root[0], root[1], root[2], root[3]))

    while heap and time.perf_counter() < deadline:
        _, _, rects, eps, remaining, chosen = heapq.heappop(heap)
        placed_items = sum(remaining0[sku] - remaining[sku] for sku in remaining0)
        placed_volume = sum(
            (remaining0[sku] - remaining[sku]) * volume0[sku]
            for sku in remaining0
        )
        rem_constrained = sum(remaining[sku] for sku in constrained_skus)
        placed_constrained = total_constrained - rem_constrained
        used_area = sum(rdx * rdy for _, _, rdx, rdy in rects)
        state_value = _layer_state_value(
            placed_volume=placed_volume,
            placed_items=placed_items,
            placed_constrained=placed_constrained,
            used_area=used_area,
            pallet=pallet,
            total_items=total_items,
            total_constrained=total_constrained,
        )
        state_key = (rects, tuple(sorted(remaining.items())), eps)
        if seen.get(state_key, -1.0) >= state_value - 1e-9:
            continue
        seen[state_key] = state_value
        if state_value > best_value:
            best_value = state_value
            best_state = (rects, eps, remaining, chosen)

        if not eps:
            continue

        ep_x, ep_y = sorted(eps, key=lambda ep: (ep[1], ep[0]))[0]
        candidates: List[Tuple[float, _LayerPattern, int, int]] = []
        for pattern in patterns:
            need: Dict[str, int] = {}
            feasible_pattern = True
            for oriented, count, _, _ in pattern.sequence:
                sku_id = oriented.box.sku_id
                need[sku_id] = need.get(sku_id, 0) + count
                if need[sku_id] > remaining.get(sku_id, 0):
                    feasible_pattern = False
                    break
            if not feasible_pattern:
                continue
            if not _fits_rect(rects, pattern.dx, pattern.dy, ep_x, ep_y, pallet):
                continue
            score = _layer_candidate_score(
                pattern=pattern,
                x=ep_x,
                y=ep_y,
                rem_constrained=rem_constrained,
                pallet=pallet,
                total_items=total_items,
                total_constrained=total_constrained,
            )
            candidates.append((score, pattern, ep_x, ep_y))

        if not candidates:
            new_eps = tuple(sorted(ep for ep in eps if ep != (ep_x, ep_y)))
            counter += 1
            heapq.heappush(
                heap,
                (-state_value, counter, rects, new_eps, dict(remaining), list(chosen)),
            )
            continue

        for _, pattern, x, y in _select_layer_branches(candidates, 12):
            rect = (x, y, pattern.dx, pattern.dy)
            new_rects = tuple(sorted(rects + (rect,)))
            new_remaining = dict(remaining)
            for oriented, count, _, _ in pattern.sequence:
                new_remaining[oriented.box.sku_id] -= count
            new_eps = [ep for ep in eps if not _inside_rect(rect, ep)]
            for ep in ((x + pattern.dx, y), (x, y + pattern.dy)):
                if ep[0] <= pallet.length_mm and ep[1] <= pallet.width_mm:
                    if not any(_inside_rect(existing, ep) for existing in new_rects):
                        if ep not in new_eps:
                            new_eps.append(ep)
            child_rects = new_rects
            child_eps = tuple(sorted(new_eps))
            child_chosen = chosen + [(pattern, x, y)]
            child_items = placed_items + pattern.item_count
            child_volume = placed_volume + pattern.volume
            child_constrained = placed_constrained + pattern.constrained_count
            child_area = used_area + pattern.dx * pattern.dy
            child_value = _layer_state_value(
                placed_volume=child_volume,
                placed_items=child_items,
                placed_constrained=child_constrained,
                used_area=child_area,
                pallet=pallet,
                total_items=total_items,
                total_constrained=total_constrained,
            )
            counter += 1
            heapq.heappush(
                heap,
                (
                    -child_value,
                    counter,
                    child_rects,
                    child_eps,
                    new_remaining,
                    child_chosen,
                ),
            )

        skip_eps = tuple(sorted(ep for ep in eps if ep != (ep_x, ep_y)))
        counter += 1
        heapq.heappush(
            heap,
            (-state_value, counter, rects, skip_eps, dict(remaining), list(chosen)),
        )

    return best_state


def _materialize_layer_solution(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    chosen: List[Tuple[_LayerPattern, int, int]],
    t0: float,
) -> Solution:
    state = PalletState(pallet)
    placements: List[Placement] = []
    next_index: Dict[str, int] = defaultdict(int)

    for pattern, x, y in chosen:
        z = 0
        for oriented, count, offset_x, offset_y in pattern.sequence:
            for _ in range(count):
                px = x + offset_x
                py = y + offset_y
                if not state.can_place(
                    oriented.dx,
                    oriented.dy,
                    oriented.dz,
                    px,
                    py,
                    z,
                    oriented.box.weight_kg,
                    oriented.box.fragile,
                    oriented.box.stackable,
                ):
                    continue
                state.place(
                    oriented.box.sku_id,
                    oriented.dx,
                    oriented.dy,
                    oriented.dz,
                    px,
                    py,
                    z,
                    oriented.box.weight_kg,
                    oriented.box.fragile,
                    oriented.box.stackable,
                )
                placements.append(
                    Placement(
                        sku_id=oriented.box.sku_id,
                        instance_index=next_index[oriented.box.sku_id],
                        x_mm=px,
                        y_mm=py,
                        z_mm=z,
                        length_mm=oriented.dx,
                        width_mm=oriented.dy,
                        height_mm=oriented.dz,
                        rotation_code=oriented.rotation_code,
                    )
                )
                next_index[oriented.box.sku_id] += 1
                z += oriented.dz

    unplaced = []
    for box in boxes:
        leftover = max(0, box.quantity - next_index[box.sku_id])
        if leftover > 0:
            unplaced.append(
                UnplacedItem(
                    sku_id=box.sku_id,
                    quantity_unplaced=leftover,
                    reason="no_space",
                )
            )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )


def pack_small_column_volume_first(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    label: str = "small_column_volume_first",
    time_budget_ms: int = 90,
    max_columns: int = 6,
    branch_width: int = 10,
) -> Solution:
    t0 = time.perf_counter()
    if not boxes:
        return Solution(
            task_id=task_id,
            solver_version=__version__,
            solve_time_ms=0,
            placements=[],
            unplaced=[],
        )

    patterns = _build_layer_patterns(pallet, boxes)
    if not patterns:
        return Solution(
            task_id=task_id,
            solver_version=__version__,
            solve_time_ms=0,
            placements=[],
            unplaced=[UnplacedItem(sku_id=box.sku_id, quantity_unplaced=box.quantity, reason="no_space") for box in boxes],
        )

    remaining0 = {box.sku_id: box.quantity for box in boxes}
    volume0 = {box.sku_id: box.volume for box in boxes}
    constrained_skus = {box.sku_id for box in boxes if box.fragile or not box.stackable}
    total_items = sum(remaining0.values())
    total_constrained = sum(remaining0[sku] for sku in constrained_skus)
    deadline = t0 + max(time_budget_ms, 1) / 1000.0
    min_footprint = min(pattern.dx * pattern.dy for pattern in patterns)
    estimated_slots = (pallet.length_mm * pallet.width_mm) / max(min_footprint, 1)
    if estimated_slots > max_columns + 1:
        return pack_upright_layered(
            task_id,
            pallet,
            boxes,
            label=f"{label}:fallback",
            time_budget_ms=time_budget_ms,
        )

    pattern_ranked = sorted(
        patterns,
        key=lambda pattern: (
            _layer_state_value(
                placed_volume=pattern.volume,
                placed_items=pattern.item_count,
                placed_constrained=pattern.constrained_count,
                used_area=pattern.dx * pattern.dy,
                pallet=pallet,
                total_items=total_items,
                total_constrained=max(total_constrained, 1),
            ),
            pattern.volume,
            pattern.item_count,
        ),
        reverse=True,
    )[:18]

    best_value = -1.0
    best_chosen: List[Tuple[_LayerPattern, int, int]] = []
    seen: Dict[Tuple[Tuple[Tuple[int, int, int, int], ...], Tuple[Tuple[str, int], ...]], float] = {}

    def dfs(
        rects: Tuple[Tuple[int, int, int, int], ...],
        eps: Tuple[Tuple[int, int], ...],
        remaining: Dict[str, int],
        chosen: List[Tuple[_LayerPattern, int, int]],
    ) -> None:
        nonlocal best_value, best_chosen
        if time.perf_counter() >= deadline:
            return
        placed_items = sum(remaining0[sku] - remaining[sku] for sku in remaining0)
        placed_volume = sum(
            (remaining0[sku] - remaining[sku]) * volume0[sku]
            for sku in remaining0
        )
        rem_constrained = sum(remaining[sku] for sku in constrained_skus)
        placed_constrained = total_constrained - rem_constrained
        used_area = sum(rdx * rdy for _, _, rdx, rdy in rects)
        state_value = _layer_state_value(
            placed_volume=placed_volume,
            placed_items=placed_items,
            placed_constrained=placed_constrained,
            used_area=used_area,
            pallet=pallet,
            total_items=total_items,
            total_constrained=total_constrained,
        )
        key = (rects, tuple(sorted(remaining.items())))
        if seen.get(key, -1.0) >= state_value - 1e-9:
            return
        seen[key] = state_value
        if state_value > best_value:
            best_value = state_value
            best_chosen = list(chosen)
        if len(chosen) >= max_columns or not eps:
            return

        candidates: List[Tuple[float, _LayerPattern, int, int]] = []
        for x, y in sorted(eps, key=lambda ep: (ep[1], ep[0])):
            for pattern in pattern_ranked:
                need: Dict[str, int] = {}
                feasible_pattern = True
                for oriented, count, _, _ in pattern.sequence:
                    sku_id = oriented.box.sku_id
                    need[sku_id] = need.get(sku_id, 0) + count
                    if need[sku_id] > remaining.get(sku_id, 0):
                        feasible_pattern = False
                        break
                if not feasible_pattern or not _fits_rect(rects, pattern.dx, pattern.dy, x, y, pallet):
                    continue
                candidates.append(
                    (
                        _layer_candidate_score(
                            pattern=pattern,
                            x=x,
                            y=y,
                            rem_constrained=rem_constrained,
                            pallet=pallet,
                            total_items=total_items,
                            total_constrained=total_constrained,
                        ),
                        pattern,
                        x,
                        y,
                    )
                )
        if not candidates:
            return

        for _, pattern, x, y in sorted(candidates, key=lambda item: item[0], reverse=True)[:branch_width]:
            rect = (x, y, pattern.dx, pattern.dy)
            new_rects = tuple(sorted(rects + (rect,)))
            new_remaining = dict(remaining)
            for oriented, count, _, _ in pattern.sequence:
                new_remaining[oriented.box.sku_id] -= count
            new_eps = [ep for ep in eps if not _inside_rect(rect, ep)]
            for ep in ((x + pattern.dx, y), (x, y + pattern.dy)):
                if ep[0] <= pallet.length_mm and ep[1] <= pallet.width_mm:
                    if not any(_inside_rect(existing, ep) for existing in new_rects):
                        if ep not in new_eps:
                            new_eps.append(ep)
            dfs(new_rects, tuple(sorted(new_eps)), new_remaining, chosen + [(pattern, x, y)])

    dfs(tuple(), ((0, 0),), remaining0, [])
    solution = _materialize_layer_solution(task_id, pallet, boxes, best_chosen, t0)
    logger.info(
        "[pack_small_column_volume_first] task=%s label=%s placed=%d unplaced=%d time=%dms",
        task_id,
        label,
        len(solution.placements),
        sum(item.quantity_unplaced for item in solution.unplaced),
        solution.solve_time_ms,
    )
    return solution


def pack_upright_layered(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    label: str = "upright_layered",
    time_budget_ms: int = 120,
    beam_width: int = 32,
    branch_width: int = 4,
) -> Solution:
    t0 = time.perf_counter()
    if not boxes:
        return Solution(
            task_id=task_id,
            solver_version=__version__,
            solve_time_ms=0,
            placements=[],
            unplaced=[],
        )

    patterns = _build_layer_patterns(pallet, boxes)
    remaining0 = {box.sku_id: box.quantity for box in boxes}
    volume0 = {box.sku_id: box.volume for box in boxes}
    constrained_skus = {
        box.sku_id for box in boxes if box.fragile or not box.stackable
    }
    total_items = sum(remaining0.values())
    total_constrained = sum(remaining0[sku] for sku in constrained_skus)
    deadline = t0 + max(time_budget_ms, 1) / 1000.0

    # (rects, eps, remaining, chosen_patterns)
    if _use_exact_layer_search(pallet, boxes, patterns):
        best_state = _exact_upright_layer_search(
            pallet=pallet,
            patterns=patterns,
            remaining0=remaining0,
            volume0=volume0,
            constrained_skus=constrained_skus,
            total_items=total_items,
            total_constrained=total_constrained,
            deadline=deadline,
        )
    else:
        beam: List[
            Tuple[
                Tuple[Tuple[int, int, int, int], ...],
                Tuple[Tuple[int, int], ...],
                Dict[str, int],
                List[Tuple[_LayerPattern, int, int]],
            ]
        ] = [(tuple(), ((0, 0),), remaining0, [])]
        best_state = beam[0]
        best_value = -1.0

        for _ in range(32):
            if time.perf_counter() >= deadline:
                break
            next_states = []
            for rects, eps, remaining, chosen in beam:
                placed = sum(remaining0[sku] - remaining[sku] for sku in remaining0)
                placed_volume = sum(
                    (remaining0[sku] - remaining[sku]) * volume0[sku]
                    for sku in remaining0
                )
                rem_constrained = sum(remaining[sku] for sku in constrained_skus)
                placed_constrained = total_constrained - rem_constrained
                used_area = sum(rdx * rdy for _, _, rdx, rdy in rects)
                state_value = _layer_state_value(
                    placed_volume=placed_volume,
                    placed_items=placed,
                    placed_constrained=placed_constrained,
                    used_area=used_area,
                    pallet=pallet,
                    total_items=total_items,
                    total_constrained=total_constrained,
                )
                if state_value > best_value:
                    best_value = state_value
                    best_state = (rects, eps, remaining, chosen)

                candidates = []
                for pattern in patterns:
                    need: Dict[str, int] = {}
                    feasible_pattern = True
                    for oriented, count, _, _ in pattern.sequence:
                        sku_id = oriented.box.sku_id
                        need[sku_id] = need.get(sku_id, 0) + count
                        if need[sku_id] > remaining.get(sku_id, 0):
                            feasible_pattern = False
                            break
                    if not feasible_pattern:
                        continue

                    for x, y in eps:
                        if not _fits_rect(rects, pattern.dx, pattern.dy, x, y, pallet):
                            continue
                        score = _layer_candidate_score(
                            pattern=pattern,
                            x=x,
                            y=y,
                            rem_constrained=rem_constrained,
                            pallet=pallet,
                            total_items=total_items,
                            total_constrained=total_constrained,
                        )
                        candidates.append((score, pattern, x, y))

                for _, pattern, x, y in _select_layer_branches(candidates, branch_width):
                    rect = (x, y, pattern.dx, pattern.dy)
                    new_rects = tuple(sorted(rects + (rect,)))
                    new_remaining = dict(remaining)
                    for oriented, count, _, _ in pattern.sequence:
                        new_remaining[oriented.box.sku_id] -= count
                    new_eps = [ep for ep in eps if not _inside_rect(rect, ep)]
                    for ep in ((x + pattern.dx, y), (x, y + pattern.dy)):
                        if ep[0] <= pallet.length_mm and ep[1] <= pallet.width_mm:
                            if not any(_inside_rect(existing, ep) for existing in new_rects):
                                if ep not in new_eps:
                                    new_eps.append(ep)
                    next_states.append(
                        (
                            new_rects,
                            tuple(sorted(new_eps)),
                            new_remaining,
                            chosen + [(pattern, x, y)],
                        )
                    )

            if not next_states:
                break

            deduped: Dict[
                Tuple[Tuple[Tuple[int, int, int, int], ...], Tuple[Tuple[str, int], ...]],
                Tuple[Tuple[int, int, int], Tuple[Any, ...]],
            ] = {}
            for state in next_states:
                rects, eps, remaining, chosen = state
                placed = sum(remaining0[sku] - remaining[sku] for sku in remaining0)
                rem_constrained = sum(remaining[sku] for sku in constrained_skus)
                placed_volume = sum(
                    (remaining0[sku] - remaining[sku]) * volume0[sku]
                    for sku in remaining0
                )
                placed_constrained = total_constrained - rem_constrained
                used_area = sum(rdx * rdy for _, _, rdx, rdy in rects)
                key = (rects, tuple(sorted(remaining.items())))
                value = (
                    (
                        _layer_state_value(
                            placed_volume=placed_volume,
                            placed_items=placed,
                            placed_constrained=placed_constrained,
                            used_area=used_area,
                            pallet=pallet,
                            total_items=total_items,
                            total_constrained=total_constrained,
                        ),
                        placed_volume,
                        placed,
                        -rem_constrained,
                        -used_area,
                    ),
                    state,
                )
                if key not in deduped or value[0] > deduped[key][0]:
                    deduped[key] = value

            pruned = [value[1] for value in deduped.values()]
            pruned.sort(
                key=lambda state: (
                    _layer_state_value(
                        placed_volume=sum(
                            (remaining0[sku] - state[2][sku]) * volume0[sku]
                            for sku in remaining0
                        ),
                        placed_items=sum(remaining0[sku] - state[2][sku] for sku in remaining0),
                        placed_constrained=total_constrained - sum(
                            state[2][sku] for sku in constrained_skus
                        ),
                        used_area=sum(rdx * rdy for _, _, rdx, rdy in state[0]),
                        pallet=pallet,
                        total_items=total_items,
                        total_constrained=total_constrained,
                    ),
                    sum((remaining0[sku] - state[2][sku]) * volume0[sku] for sku in remaining0),
                    sum(remaining0[sku] - state[2][sku] for sku in remaining0),
                ),
                reverse=True,
            )
            beam = pruned[:beam_width]

    _, _, remaining, chosen = best_state
    state = PalletState(pallet)
    placements: List[Placement] = []
    next_index: Dict[str, int] = defaultdict(int)

    for pattern, x, y in chosen:
        z = 0
        for oriented, count, offset_x, offset_y in pattern.sequence:
            for _ in range(count):
                if not state.can_place(
                    oriented.dx,
                    oriented.dy,
                    oriented.dz,
                    x + offset_x,
                    y + offset_y,
                    z,
                    oriented.box.weight_kg,
                    oriented.box.fragile,
                    oriented.box.stackable,
                ):
                    continue
                state.place(
                    oriented.box.sku_id,
                    oriented.dx,
                    oriented.dy,
                    oriented.dz,
                    x + offset_x,
                    y + offset_y,
                    z,
                    oriented.box.weight_kg,
                    oriented.box.fragile,
                    oriented.box.stackable,
                )
                placements.append(
                    Placement(
                        sku_id=oriented.box.sku_id,
                        instance_index=next_index[oriented.box.sku_id],
                        x_mm=x + offset_x,
                        y_mm=y + offset_y,
                        z_mm=z,
                        length_mm=oriented.dx,
                        width_mm=oriented.dy,
                        height_mm=oriented.dz,
                        rotation_code=oriented.rotation_code,
                    )
                )
                next_index[oriented.box.sku_id] += 1
                z += oriented.dz

    unplaced = []
    for box in boxes:
        leftover = max(0, box.quantity - next_index[box.sku_id])
        if leftover > 0:
            unplaced.append(
                UnplacedItem(
                    sku_id=box.sku_id,
                    quantity_unplaced=leftover,
                    reason="no_space",
                )
            )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[pack_upright_layered] task=%s label=%s placed=%d unplaced=%d time=%dms",
        task_id,
        label,
        len(placements),
        sum(item.quantity_unplaced for item in unplaced),
        elapsed_ms,
    )
    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )


def pack_ordered_boxes(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    label: str = "custom_order",
    strict_fragility: bool = False,
) -> Solution:
    instances = _expand_boxes(boxes)
    return _pack_instances(
        task_id,
        pallet,
        instances,
        label,
        strict_fragility=strict_fragility,
    )


def pack_instance_sequence(
    task_id: str,
    pallet: Pallet,
    instances: List[Tuple[Box, int]],
    label: str = "custom_sequence",
    strict_fragility: bool = False,
) -> Solution:
    return _pack_instances(
        task_id,
        pallet,
        instances,
        label,
        strict_fragility=strict_fragility,
    )


# ── Greedy packer ───────────────────────────────────────────────────

def pack_greedy(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str = "volume_desc",
    strict_fragility: bool = False,
) -> Solution:
    """Pack boxes greedily using Extreme Points + scoring function.

    Returns a Solution with placements and unplaced items.
    """
    sorted_boxes = order_boxes(boxes, sort_key_name=sort_key_name)
    return pack_ordered_boxes(
        task_id,
        pallet,
        sorted_boxes,
        label=sort_key_name,
        strict_fragility=strict_fragility,
    )
