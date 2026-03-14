"""Beam search packer using ML scorer for candidate ranking."""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .orientations import get_orientations
from .pallet_state import PalletState
from .scoring import score_placement
from .ml_ranker import MLPScorer, extract_features
from . import __version__

logger = logging.getLogger(__name__)


@dataclass
class BeamState:
    """Single beam: pallet state + placements so far + cumulative ML score."""
    pallet_state: PalletState
    placements: List[Placement] = field(default_factory=list)
    cumulative_score: float = 0.0
    placed_volume: int = 0


def pack_beam(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    scorer: MLPScorer,
    beam_width: int = 3,
    sort_key_name: str = "volume_desc",
    time_limit_ms: int = 0,
) -> Solution:
    """Pack boxes using beam search with ML scorer.

    At each step:
    1. For each beam, try all EPs × orientations for the current item
    2. Score each candidate using the MLP
    3. Keep top beam_width beams
    """
    from .packer import SORT_KEYS, _expand_boxes

    t0 = time.perf_counter()

    sort_fn = SORT_KEYS.get(sort_key_name)
    if sort_fn is None:
        from .packer import _sort_volume_desc
        sort_fn = _sort_volume_desc

    sorted_boxes = sorted(boxes, key=sort_fn)
    instances = _expand_boxes(sorted_boxes)
    total_items = len(instances)

    # Compute remaining volumes for each step
    total_remaining_volumes = []
    cum = 0
    vols = [b.length_mm * b.width_mm * b.height_mm for b, _ in instances]
    for i in range(len(instances)):
        total_remaining_volumes.append(sum(vols[i:]))

    # Initialize single beam
    beams: List[BeamState] = [
        BeamState(pallet_state=PalletState(pallet))
    ]

    check_interval = max(5, total_items // 20)

    for item_idx, (box, inst_idx) in enumerate(instances):
        # Time check
        if time_limit_ms > 0 and item_idx % check_interval == 0 and item_idx > 0:
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > time_limit_ms:
                logger.debug("[pack_beam] time limit at item %d/%d", item_idx, total_items)
                break

        orientations = get_orientations(box)
        items_remaining = total_items - item_idx
        remaining_vol = total_remaining_volumes[item_idx]

        candidates: List[Tuple[float, int, Tuple]] = []  # (score, beam_idx, placement_info)

        for beam_idx, beam in enumerate(beams):
            state = beam.pallet_state
            found_any = False

            for ep in list(state.extreme_points):
                ex, ey, ez = ep
                for dx, dy, dz, rot_code in orientations:
                    if not state.can_place(
                        dx, dy, dz, ex, ey, ez,
                        box.weight_kg, box.fragile, box.stackable,
                    ):
                        continue

                    # Use ML scorer
                    feats = extract_features(
                        state, dx, dy, dz, ex, ey, ez,
                        box.weight_kg, box.fragile, box.stackable,
                        items_remaining, total_items, remaining_vol,
                    )
                    ml_score = scorer.predict(feats)

                    # Also compute heuristic score as tiebreaker
                    h_score = score_placement(
                        state, dx, dy, dz, ex, ey, ez,
                        box.weight_kg, box.fragile,
                    )
                    if h_score < 0:
                        continue  # Hard block (fragile violation)

                    # Combined score: ML score + small heuristic bonus
                    combined = ml_score + 0.01 * h_score

                    # Add cumulative score from previous placements
                    total_score = beam.cumulative_score + combined

                    candidates.append((
                        total_score, beam_idx,
                        (dx, dy, dz, ex, ey, ez, rot_code, ml_score)
                    ))
                    found_any = True

            if not found_any:
                # This beam can't place the item - keep it as-is with penalty
                candidates.append((
                    beam.cumulative_score - 0.1,  # penalty for skipping
                    beam_idx,
                    None,  # no placement
                ))

        if not candidates:
            break

        # Sort by total score descending, keep top beam_width
        candidates.sort(key=lambda c: c[0], reverse=True)
        top_candidates = candidates[:beam_width]

        new_beams: List[BeamState] = []
        for total_score, beam_idx, placement_info in top_candidates:
            old_beam = beams[beam_idx]

            if placement_info is None:
                # Keep beam as-is (item skipped)
                new_beam = BeamState(
                    pallet_state=_copy_pallet_state(old_beam.pallet_state, pallet),
                    placements=list(old_beam.placements),
                    cumulative_score=total_score,
                    placed_volume=old_beam.placed_volume,
                )
            else:
                dx, dy, dz, px, py, pz, rot_code, ml_score = placement_info

                # Deep copy the pallet state
                new_state = _copy_pallet_state(old_beam.pallet_state, pallet)
                new_state.place(
                    box.sku_id, dx, dy, dz, px, py, pz,
                    box.weight_kg, box.fragile, box.stackable,
                )

                new_placements = list(old_beam.placements)
                new_placements.append(Placement(
                    sku_id=box.sku_id,
                    instance_index=inst_idx,
                    x_mm=px, y_mm=py, z_mm=pz,
                    length_mm=dx, width_mm=dy, height_mm=dz,
                    rotation_code=rot_code,
                ))

                new_beam = BeamState(
                    pallet_state=new_state,
                    placements=new_placements,
                    cumulative_score=total_score,
                    placed_volume=old_beam.placed_volume + dx * dy * dz,
                )

            new_beams.append(new_beam)

        beams = new_beams

    # Pick best beam by placed volume (not cumulative score)
    best_beam = max(beams, key=lambda b: b.placed_volume)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "[pack_beam] done sort=%s beam_width=%d placed=%d time=%dms",
        sort_key_name, beam_width, len(best_beam.placements), elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=best_beam.placements,
        unplaced=[],  # Simplified: don't track unplaced for beam search
    )


def _copy_pallet_state(src: PalletState, pallet: Pallet) -> PalletState:
    """Create a deep copy of PalletState."""
    new_state = PalletState(pallet)
    new_state.boxes = [copy.copy(b) for b in src.boxes]
    new_state.current_weight = src.current_weight
    new_state.max_z = src.max_z
    new_state.extreme_points = list(src.extreme_points)
    return new_state
