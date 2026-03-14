"""Greedy packer: places boxes one by one using Extreme Points + scoring."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from heapq import nlargest
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
    sequence: Tuple[Tuple[_LayerOriented, int], ...]
    dx: int
    dy: int
    height: int
    kind: str


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
    for oriented in orienteds:
        max_repeat = 1
        if oriented.box.stackable and (
            oriented.box.weight_kg <= 2.0 or not oriented.box.fragile
        ):
            max_repeat = min(
                oriented.box.quantity,
                pallet.max_height_mm // max(oriented.dz, 1),
                12,
            )
        for repeat in _repeat_options(max_repeat):
            pattern = _LayerPattern(
                sequence=((oriented, repeat),),
                dx=oriented.dx,
                dy=oriented.dy,
                height=repeat * oriented.dz,
                kind="support",
            )
            key = (
                tuple(
                    (entry.box.sku_id, entry.rotation_code, count)
                    for entry, count in pattern.sequence
                ),
                pattern.dx,
                pattern.dy,
                pattern.height,
            )
            patterns[key] = pattern

    tops = [o for o in orienteds if o.box.fragile or not o.box.stackable]
    supports = [o for o in orienteds if o.box.stackable]
    for top in tops:
        pattern = _LayerPattern(
            sequence=((top, 1),),
            dx=top.dx,
            dy=top.dy,
            height=top.dz,
            kind="top",
        )
        key = (
            ((top.box.sku_id, top.rotation_code, 1),),
            pattern.dx,
            pattern.dy,
            pattern.height,
        )
        patterns[key] = pattern

        for support in supports:
            if top.dx > support.dx or top.dy > support.dy:
                continue
            if top.box.weight_kg > 2.0 and not top.box.fragile and support.box.fragile:
                continue
            max_repeat = min(
                support.box.quantity,
                (pallet.max_height_mm - top.dz) // max(support.dz, 1),
                12,
            )
            if support.box.weight_kg > 2.0 and support.box.fragile:
                max_repeat = min(max_repeat, 1)
            for repeat in _repeat_options(max_repeat):
                pattern = _LayerPattern(
                    sequence=((support, repeat), (top, 1)),
                    dx=support.dx,
                    dy=support.dy,
                    height=repeat * support.dz + top.dz,
                    kind="top_support",
                )
                key = (
                    tuple(
                        (entry.box.sku_id, entry.rotation_code, count)
                        for entry, count in pattern.sequence
                    ),
                    pattern.dx,
                    pattern.dy,
                    pattern.height,
                )
                patterns[key] = pattern

    return list(patterns.values())


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
    constrained_skus = {
        box.sku_id for box in boxes if box.fragile or not box.stackable
    }
    deadline = t0 + max(time_budget_ms, 1) / 1000.0

    # (rects, eps, remaining, chosen_patterns)
    beam: List[
        Tuple[
            Tuple[Tuple[int, int, int, int], ...],
            Tuple[Tuple[int, int], ...],
            Dict[str, int],
            List[Tuple[_LayerPattern, int, int]],
        ]
    ] = [(tuple(), ((0, 0),), remaining0, [])]
    best_state = beam[0]
    best_placed = 0

    for _ in range(32):
        if time.perf_counter() >= deadline:
            break
        next_states = []
        for rects, eps, remaining, chosen in beam:
            placed = sum(remaining0[sku] - remaining[sku] for sku in remaining0)
            if placed > best_placed:
                best_placed = placed
                best_state = (rects, eps, remaining, chosen)

            rem_constrained = sum(remaining[sku] for sku in constrained_skus)
            candidates = []
            for pattern in patterns:
                need: Dict[str, int] = {}
                items = 0
                constrained_items = 0
                volume = 0
                feasible_pattern = True
                for oriented, count in pattern.sequence:
                    sku_id = oriented.box.sku_id
                    need[sku_id] = need.get(sku_id, 0) + count
                    if need[sku_id] > remaining.get(sku_id, 0):
                        feasible_pattern = False
                        break
                    items += count
                    volume += oriented.box.volume * count
                    if sku_id in constrained_skus:
                        constrained_items += count
                if not feasible_pattern:
                    continue

                for x, y in eps:
                    if not _fits_rect(rects, pattern.dx, pattern.dy, x, y, pallet):
                        continue
                    footprint = pattern.dx * pattern.dy
                    height_eff = volume / max(footprint * pattern.height, 1)
                    score = (
                        items * 4.0
                        + constrained_items * 8.0
                        + height_eff * 18.0
                        + (1.0 if pattern.kind == "top_support" else 0.0)
                        + (1.0 if x == 0 else 0.0)
                        + (1.0 if y == 0 else 0.0)
                        - pattern.height / 20000.0
                    )
                    if rem_constrained and constrained_items == 0 and items <= 2:
                        score -= 4.0
                    candidates.append((score, pattern, x, y))

            for _, pattern, x, y in nlargest(branch_width, candidates, key=lambda item: item[0]):
                rect = (x, y, pattern.dx, pattern.dy)
                new_rects = tuple(sorted(rects + (rect,)))
                new_remaining = dict(remaining)
                for oriented, count in pattern.sequence:
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
            used_area = sum(rdx * rdy for _, _, rdx, rdy in rects)
            key = (rects, tuple(sorted(remaining.items())))
            value = ((placed, -rem_constrained, -used_area), state)
            if key not in deduped or value[0] > deduped[key][0]:
                deduped[key] = value

        pruned = [value[1] for value in deduped.values()]
        pruned.sort(
            key=lambda state: (
                sum(remaining0[sku] - state[2][sku] for sku in remaining0),
                -sum(state[2][sku] for sku in constrained_skus),
                -sum(rdx * rdy for _, _, rdx, rdy in state[0]),
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
        for oriented, count in pattern.sequence:
            for _ in range(count):
                if not state.can_place(
                    oriented.dx,
                    oriented.dy,
                    oriented.dz,
                    x,
                    y,
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
                    x,
                    y,
                    z,
                    oriented.box.weight_kg,
                    oriented.box.fragile,
                    oriented.box.stackable,
                )
                placements.append(
                    Placement(
                        sku_id=oriented.box.sku_id,
                        instance_index=next_index[oriented.box.sku_id],
                        x_mm=x,
                        y_mm=y,
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
