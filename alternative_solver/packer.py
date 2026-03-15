"""Greedy packer: places boxes one by one using Extreme Points + scoring."""

import hashlib
import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .orientations import get_orientations
from .pallet_state import PalletState
from .scoring import score_placement
from .ml_ranker import MLPScorer, extract_features
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


def _sort_volume_asc(box: Box) -> tuple:
    return (box.volume,)


def _sort_height_desc(box: Box) -> tuple:
    return (-box.height_mm,)


def _sort_fragile_last(box: Box) -> tuple:
    # Non-fragile stackable first (heavy base), fragile and non-stackable last
    penalty = 0
    if box.fragile:
        penalty += 2
    if not box.stackable:
        penalty += 1
    return (penalty, -box.volume)


def _sort_non_stackable_last(box: Box) -> tuple:
    # Non-stackable items last (they block stacking), then by volume desc
    return (0 if box.stackable else 1, -box.volume)


def _sort_max_dim_desc(box: Box) -> tuple:
    return (-max(box.length_mm, box.width_mm, box.height_mm),)


def _sort_heavy_base_fragile_top(box: Box) -> tuple:
    """Heavy non-fragile first → heavy fragile → light fragile last.

    Ideal for scenarios with mixed fragile/non-fragile items.
    Places heavy stable base, then fragile items by weight desc
    (so heavy fragile are placed while there's still space below lighter fragile).
    """
    if not box.fragile:
        tier = 0  # Non-fragile first (base)
    elif box.weight_kg > 2.0:
        tier = 1  # Heavy fragile next
    else:
        tier = 2  # Light fragile last (top)
    return (tier, -box.weight_kg, -box.volume)


def _sort_stackable_base(box: Box) -> tuple:
    """Stackable non-fragile first, then fragile, non-stackable last.

    Focus on building a solid base with stackable items,
    placing non-stackable on top where they won't block.
    """
    if not box.stackable:
        tier = 3  # Non-stackable last
    elif not box.fragile:
        tier = 0  # Stackable non-fragile first
    elif box.weight_kg > 2.0:
        tier = 1  # Heavy fragile
    else:
        tier = 2  # Light fragile
    return (tier, -box.volume)


def _sort_score_per_kg(box: Box) -> tuple:
    """Sort by marginal score per kg — maximize items placed under weight limit.

    Higher score-per-kg items should be placed first to fit more items
    before hitting the weight limit.
    """
    # volume contributes to vol_util (50% weight), each item contributes to coverage (30%)
    # Use volume/weight as proxy for value-per-kg
    vol = max(box.volume, 1)
    return (-vol / max(box.weight_kg, 0.01),)


def _sort_light_fillers_first(box: Box) -> tuple:
    """Light items first — maximize item count under weight limit.

    Place lightest items first to maximize coverage when weight-limited.
    """
    return (box.weight_kg, -box.volume)


def _sort_coverage_optimal(box: Box) -> tuple:
    """Maximize coverage: lightest and smallest first.

    Each item contributes equally to coverage regardless of size.
    Placing light small items first maximizes total items placed.
    """
    return (box.weight_kg * box.volume,)


def _make_random_sort(seed: int) -> Callable[[Box], tuple]:
    """Create a deterministic pseudo-random sort key using hash."""
    def key(box: Box) -> tuple:
        h = hashlib.md5(f"{box.sku_id}:{seed}".encode()).hexdigest()
        return (int(h[:8], 16),)
    return key


# Ordered by effectiveness across scenarios (best strategies first for adaptive budget)
SORT_KEYS: Dict[str, Callable[[Box], tuple]] = {
    "constrained_first": _sort_constrained_first,   # Best for heavy_water
    "base_area_desc": _sort_base_area_desc,          # Best for liquid_tetris
    "fragile_last": _sort_fragile_last,              # Best for fragile_mix
    "volume_desc": _sort_volume_desc,                # Best for random_mixed
    "volume_asc": _sort_volume_asc,                  # Best for cavity_fill
    "density_desc": _sort_density_desc,              # Good for fragile_mix
    "non_stackable_last": _sort_non_stackable_last,  # Good for random_mixed
    "height_desc": _sort_height_desc,                # Good overall
    "weight_desc": _sort_weight_desc,                # Secondary
    "max_dim_desc": _sort_max_dim_desc,              # Secondary
    "heavy_base_fragile_top": _sort_heavy_base_fragile_top,  # Best for fragile_tower
    "stackable_base": _sort_stackable_base,                  # Good for mixed scenarios
    "score_per_kg": _sort_score_per_kg,                      # Best for weight-limited
    "light_fillers_first": _sort_light_fillers_first,        # Maximize coverage under weight limit
    "coverage_optimal": _sort_coverage_optimal,              # Light+small first
}

# Add randomized sort strategies for exploration
for _seed in range(10):
    SORT_KEYS[f"random_{_seed}"] = _make_random_sort(_seed)


# ── Expand boxes ────────────────────────────────────────────────────

def _expand_boxes(boxes: List[Box]) -> List[Tuple[Box, int]]:
    """Expand SKUs with quantity > 1 into individual (box, instance_index) pairs."""
    result = []
    for box in boxes:
        for i in range(box.quantity):
            result.append((box, i))
    return result


# ── Greedy packer ───────────────────────────────────────────────────

def pack_greedy(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str = "volume_desc",
    time_limit_ms: int = 0,
    weight_profile: str = "default",
) -> Solution:
    """Pack boxes greedily using Extreme Points + scoring function.

    Args:
        time_limit_ms: If > 0, stop packing when this time limit is reached.
        weight_profile: Scoring weight profile name.
    """
    t0 = time.perf_counter()

    sort_fn = SORT_KEYS.get(sort_key_name, _sort_volume_desc)

    # Sort boxes by key, then expand
    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = _expand_boxes(sorted_boxes)

    state = PalletState(pallet)
    placements: List[Placement] = []
    unplaced_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0})

    # Cache orientations per SKU (same SKU = same orientations)
    orientation_cache: Dict[str, list] = {}

    logger.info(
        "[pack_greedy] task=%s sort=%s total_instances=%d",
        task_id, sort_key_name, len(instances),
    )

    check_interval = max(4, len(instances) // 20)  # Check time every ~5% of items

    for item_idx, (box, inst_idx) in enumerate(instances):
        # Periodic time check inside packing loop
        if time_limit_ms > 0 and item_idx % check_interval == 0 and item_idx > 0:
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > time_limit_ms:
                logger.debug("[pack_greedy] time limit reached at item %d/%d", item_idx, len(instances))
                for remaining_box, remaining_idx in instances[item_idx:]:
                    if state.current_weight + remaining_box.weight_kg > pallet.max_weight_kg:
                        reason = "weight_limit_exceeded"
                    else:
                        reason = "no_space"
                    unplaced_counts[remaining_box.sku_id]["count"] += 1
                    unplaced_counts[remaining_box.sku_id]["reason"] = reason
                break

        if box.sku_id not in orientation_cache:
            orientation_cache[box.sku_id] = get_orientations(box)
        orientations = orientation_cache[box.sku_id]
        best_score = -1.0
        best_placement: Optional[Tuple[int, int, int, int, int, int, str]] = None

        # Try each EP × orientation (no list() copy — EPs not modified during scoring)
        for ep in state.extreme_points:
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
                    weight_profile=weight_profile,
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
            # Determine reason
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
        sort_key_name, len(placements), sum(u.quantity_unplaced for u in unplaced), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )


# ── ML-guided greedy packer ───────────────────────────────────────

def pack_greedy_ml(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    scorer: MLPScorer,
    sort_key_name: str = "volume_desc",
    time_limit_ms: int = 0,
) -> Solution:
    """Greedy packer using ML scorer instead of hand-tuned scoring.

    Same algorithm as pack_greedy, but placement scoring uses the MLP.
    """
    t0 = time.perf_counter()

    sort_fn = SORT_KEYS.get(sort_key_name, _sort_volume_desc)
    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = _expand_boxes(sorted_boxes)
    total_items = len(instances)

    # Precompute remaining volumes
    vols = [b.length_mm * b.width_mm * b.height_mm for b, _ in instances]
    cumvol = [0] * (total_items + 1)
    for i in range(total_items - 1, -1, -1):
        cumvol[i] = cumvol[i + 1] + vols[i]

    state = PalletState(pallet)
    placements: List[Placement] = []
    unplaced_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0})

    check_interval = max(10, total_items // 10)

    for item_idx, (box, inst_idx) in enumerate(instances):
        if time_limit_ms > 0 and item_idx % check_interval == 0 and item_idx > 0:
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > time_limit_ms:
                for remaining_box, _ in instances[item_idx:]:
                    reason = "weight_limit_exceeded" if state.current_weight + remaining_box.weight_kg > pallet.max_weight_kg else "no_space"
                    unplaced_counts[remaining_box.sku_id]["count"] += 1
                    unplaced_counts[remaining_box.sku_id]["reason"] = reason
                break

        orientations = get_orientations(box)
        best_score = -float("inf")
        best_placement: Optional[Tuple[int, int, int, int, int, int, str]] = None

        items_remaining = total_items - item_idx
        remaining_vol = cumvol[item_idx]

        for ep in list(state.extreme_points):
            ex, ey, ez = ep
            for dx, dy, dz, rot_code in orientations:
                if not state.can_place(
                    dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                ):
                    continue

                feats = extract_features(
                    state, dx, dy, dz, ex, ey, ez,
                    box.weight_kg, box.fragile, box.stackable,
                    items_remaining, total_items, remaining_vol,
                )
                sc = scorer.predict(feats)

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
                sku_id=box.sku_id, instance_index=inst_idx,
                x_mm=px, y_mm=py, z_mm=pz,
                length_mm=dx, width_mm=dy, height_mm=dz,
                rotation_code=rot_code,
            ))
        else:
            reason = "weight_limit_exceeded" if state.current_weight + box.weight_kg > pallet.max_weight_kg else "no_space"
            unplaced_counts[box.sku_id]["count"] += 1
            unplaced_counts[box.sku_id]["reason"] = reason

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    unplaced = [
        UnplacedItem(sku_id=sid, quantity_unplaced=info["count"], reason=info["reason"])
        for sid, info in unplaced_counts.items()
    ]

    logger.info(
        "[pack_greedy_ml] done sort=%s placed=%d unplaced=%d time=%dms",
        sort_key_name, len(placements),
        sum(u.quantity_unplaced for u in unplaced), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )


# ── Best-fit packer (for small instances) ──────────────────────────

def pack_two_phase(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str = "volume_desc",
    weight_profile: str = "default",
) -> Solution:
    """Two-phase packing: non-fragile items first (base), then fragile on top.

    This avoids the problem where greedy places fragile items early,
    blocking heavy items from being placed on top of them.
    """
    t0 = time.perf_counter()

    sort_fn = SORT_KEYS.get(sort_key_name, _sort_volume_desc)

    # Split into phases
    non_fragile = [b for b in boxes if not b.fragile]
    heavy_fragile = [b for b in boxes if b.fragile and b.weight_kg > 2.0]
    light_fragile = [b for b in boxes if b.fragile and b.weight_kg <= 2.0]

    # Sort each phase
    phase1 = sorted(non_fragile, key=sort_fn)
    phase2 = sorted(heavy_fragile, key=sort_fn)
    phase3 = sorted(light_fragile, key=sort_fn)

    all_phases = [(phase1, "non_fragile"), (phase2, "heavy_fragile"), (phase3, "light_fragile")]

    state = PalletState(pallet)
    placements: List[Placement] = []
    unplaced_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0})
    orientation_cache: Dict[str, list] = {}

    for phase_boxes, phase_name in all_phases:
        instances = _expand_boxes(phase_boxes)
        for box, inst_idx in instances:
            if box.sku_id not in orientation_cache:
                orientation_cache[box.sku_id] = get_orientations(box)
            orientations = orientation_cache[box.sku_id]
            best_score = -1.0
            best_placement = None

            for ep in state.extreme_points:
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
                        weight_profile=weight_profile,
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
                    sku_id=box.sku_id, instance_index=inst_idx,
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

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    unplaced = [
        UnplacedItem(sku_id=sid, quantity_unplaced=info["count"], reason=info["reason"])
        for sid, info in unplaced_counts.items()
    ]

    logger.info(
        "[pack_two_phase] done sort=%s profile=%s placed=%d unplaced=%d time=%dms",
        sort_key_name, weight_profile, len(placements),
        sum(u.quantity_unplaced for u in unplaced), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )


def pack_greedy_with_order(
    task_id: str,
    pallet: Pallet,
    order: List[Tuple[Box, int]],
    weight_profile: str = "default",
) -> Solution:
    """Pack boxes in a given order (list of (box, instance_index) pairs).

    Used by local search to evaluate permutations of item order.
    """
    t0 = time.perf_counter()
    state = PalletState(pallet)
    placements: List[Placement] = []
    unplaced_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0})
    orientation_cache: Dict[str, list] = {}

    for box, inst_idx in order:
        if box.sku_id not in orientation_cache:
            orientation_cache[box.sku_id] = get_orientations(box)
        orientations = orientation_cache[box.sku_id]
        best_score = -1.0
        best_placement = None

        for ep in state.extreme_points:
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
                    weight_profile=weight_profile,
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
                sku_id=box.sku_id, instance_index=inst_idx,
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

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    unplaced = [
        UnplacedItem(sku_id=sid, quantity_unplaced=info["count"], reason=info["reason"])
        for sid, info in unplaced_counts.items()
    ]

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )


def pack_beam_search(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str = "volume_desc",
    weight_profile: str = "default",
    beam_width: int = 4,
    time_limit_ms: int = 500,
) -> Solution:
    """Beam search packer: keep top-K partial solutions at each step.

    More expensive than greedy but explores multiple packing arrangements.
    """
    import copy
    t0 = time.perf_counter()

    sort_fn = SORT_KEYS.get(sort_key_name, _sort_volume_desc)
    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = _expand_boxes(sorted_boxes)

    # Each beam entry: (state, placements, total_score)
    initial_state = PalletState(pallet)
    beam = [(initial_state, [], 0.0)]

    orientation_cache: Dict[str, list] = {}

    for item_idx, (box, inst_idx) in enumerate(instances):
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > time_limit_ms * 0.9:
            break

        if box.sku_id not in orientation_cache:
            orientation_cache[box.sku_id] = get_orientations(box)
        orientations = orientation_cache[box.sku_id]
        candidates = []

        for state, placements, cumulative_score in beam:
            # Try all EPs × orientations for this beam entry
            item_candidates = []
            for ep in state.extreme_points:
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
                        weight_profile=weight_profile,
                    )
                    item_candidates.append((sc, dx, dy, dz, ex, ey, ez, rot_code))

            if not item_candidates:
                # Item can't be placed in this beam, carry forward without it
                candidates.append((cumulative_score - 0.01, state, placements))
                continue

            # Keep top beam_width placements for this beam entry
            item_candidates.sort(key=lambda x: -x[0])
            for sc, dx, dy, dz, px, py, pz, rot_code in item_candidates[:beam_width]:
                new_state = copy.deepcopy(state)
                new_state.place(
                    box.sku_id, dx, dy, dz, px, py, pz,
                    box.weight_kg, box.fragile, box.stackable,
                )
                new_placements = placements + [Placement(
                    sku_id=box.sku_id, instance_index=inst_idx,
                    x_mm=px, y_mm=py, z_mm=pz,
                    length_mm=dx, width_mm=dy, height_mm=dz,
                    rotation_code=rot_code,
                )]
                candidates.append((cumulative_score + sc, new_state, new_placements))

        # Prune beam to top-K
        candidates.sort(key=lambda x: -x[0])
        beam = [(s, p, score) for score, s, p in candidates[:beam_width]]

    # Pick best beam entry by placement count, then cumulative score
    best_placements = []
    best_count = 0
    best_score = -1.0
    for state, placements, cum_score in beam:
        if len(placements) > best_count or (len(placements) == best_count and cum_score > best_score):
            best_count = len(placements)
            best_score = cum_score
            best_placements = placements

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Build unplaced
    placed_counts: Dict[str, int] = defaultdict(int)
    for p in best_placements:
        placed_counts[p.sku_id] += 1
    unplaced = []
    for box in boxes:
        placed = placed_counts.get(box.sku_id, 0)
        if placed < box.quantity:
            unplaced.append(UnplacedItem(
                sku_id=box.sku_id,
                quantity_unplaced=box.quantity - placed,
                reason="no_space",
            ))

    logger.info(
        "[pack_beam] done sort=%s profile=%s beam=%d placed=%d time=%dms",
        sort_key_name, weight_profile, beam_width,
        len(best_placements), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=best_placements,
        unplaced=unplaced,
    )


def pack_layer(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key_name: str = "volume_desc",
    weight_profile: str = "default",
) -> Solution:
    """Layer-based packer: build complete horizontal layers for better density.

    Algorithm:
    1. Group items by compatible heights (within 10% tolerance)
    2. For each layer, fill a 2D strip using shelf-next-fit
    3. Stack layers bottom-up
    """
    t0 = time.perf_counter()

    sort_fn = SORT_KEYS.get(sort_key_name, _sort_volume_desc)
    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = _expand_boxes(sorted_boxes)

    # Group instances by height (try each orientation's dz)
    # For each instance, pick the orientation that creates the best layer height
    orientation_cache: Dict[str, list] = {}
    for box, _ in instances:
        if box.sku_id not in orientation_cache:
            orientation_cache[box.sku_id] = get_orientations(box)

    state = PalletState(pallet)
    placements: List[Placement] = []
    unplaced_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0})

    # Try to place using standard EP greedy but with layer-aware scoring
    # that strongly rewards same-z-level placements
    for box, inst_idx in instances:
        orientations = orientation_cache[box.sku_id]
        best_score = -1.0
        best_placement = None

        for ep in state.extreme_points:
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
                    weight_profile=weight_profile,
                )

                # Layer bonus: strongly prefer staying at current layer z
                if state.boxes:
                    current_layer_z = max(
                        (b.z_min for b in state.boxes),
                        key=lambda z: sum(1 for b in state.boxes if b.z_min == z),
                    )
                    if ez == current_layer_z:
                        sc += 0.15

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

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    unplaced = [
        UnplacedItem(sku_id=sid, quantity_unplaced=info["count"], reason=info["reason"])
        for sid, info in unplaced_counts.items()
    ]

    logger.info(
        "[pack_layer] done sort=%s profile=%s placed=%d unplaced=%d time=%dms",
        sort_key_name, weight_profile, len(placements),
        sum(u.quantity_unplaced for u in unplaced), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )


MAX_BESTFIT_ITEMS = 50  # Only use for small instances


def pack_bestfit(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
) -> Solution:
    """Best-first greedy: at each step, choose the best (item, position, orientation)
    from ALL remaining items. More expensive but finds better solutions for small instances.
    """
    t0 = time.perf_counter()

    remaining = _expand_boxes(boxes)
    total_items = len(remaining)

    if total_items > MAX_BESTFIT_ITEMS:
        logger.info("[pack_bestfit] too many items (%d > %d), skipping", total_items, MAX_BESTFIT_ITEMS)
        return Solution(task_id=task_id, solver_version=__version__, solve_time_ms=0)

    state = PalletState(pallet)
    placements: List[Placement] = []

    logger.info("[pack_bestfit] task=%s total_instances=%d", task_id, total_items)

    orientation_cache: Dict[str, list] = {}

    while remaining:
        best_score = -1.0
        best_idx = -1
        best_placement = None

        for idx, (box, inst_idx) in enumerate(remaining):
            # Skip if weight limit reached
            if state.current_weight + box.weight_kg > pallet.max_weight_kg + 1e-6:
                continue

            if box.sku_id not in orientation_cache:
                orientation_cache[box.sku_id] = get_orientations(box)
            orientations = orientation_cache[box.sku_id]
            for ep in state.extreme_points:
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
                    )
                    if sc > best_score:
                        best_score = sc
                        best_idx = idx
                        best_placement = (dx, dy, dz, ex, ey, ez, rot_code, box, inst_idx)

        if best_placement is None:
            break

        dx, dy, dz, px, py, pz, rot_code, box, inst_idx = best_placement
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
        remaining.pop(best_idx)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Build unplaced summary
    unplaced_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0})
    for box, inst_idx in remaining:
        if state.current_weight + box.weight_kg > pallet.max_weight_kg:
            reason = "weight_limit_exceeded"
        else:
            reason = "no_space"
        unplaced_counts[box.sku_id]["count"] += 1
        unplaced_counts[box.sku_id]["reason"] = reason

    unplaced = [
        UnplacedItem(sku_id=sid, quantity_unplaced=info["count"], reason=info["reason"])
        for sid, info in unplaced_counts.items()
    ]

    logger.info(
        "[pack_bestfit] done placed=%d unplaced=%d time=%dms",
        len(placements), sum(u.quantity_unplaced for u in unplaced), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=placements,
        unplaced=unplaced,
    )
