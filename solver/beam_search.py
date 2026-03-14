"""Beam search packer: explores multiple placement paths simultaneously."""

import copy
import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from .models import Box, Pallet, Placement, Solution, UnplacedItem
from .orientations import get_orientations
from .pallet_state import PalletState
from .scoring import score_placement
from . import __version__

logger = logging.getLogger(__name__)


def _expand_boxes(boxes: List[Box]) -> List[Tuple[Box, int]]:
    """Expand SKUs with quantity > 1 into individual (box, instance_index) pairs."""
    result = []
    for box in boxes:
        for i in range(box.quantity):
            result.append((box, i))
    return result


class BeamState:
    """A single beam state: pallet state + placements so far."""
    __slots__ = ("pallet_state", "placements", "cumulative_score")

    def __init__(self, pallet_state: PalletState, placements: List[Placement], cumulative_score: float):
        self.pallet_state = pallet_state
        self.placements = placements
        self.cumulative_score = cumulative_score

    def clone(self) -> "BeamState":
        new_state = PalletState(self.pallet_state.pallet)
        new_state.boxes = list(self.pallet_state.boxes)
        new_state.current_weight = self.pallet_state.current_weight
        new_state.max_z = self.pallet_state.max_z
        new_state.extreme_points = list(self.pallet_state.extreme_points)
        return BeamState(new_state, list(self.placements), self.cumulative_score)


def pack_beam_search(
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    sort_key: Optional[Callable[[Box], tuple]] = None,
    beam_width: int = 3,
    candidates_per_step: int = 5,
) -> Solution:
    """Beam search packer: keeps top-k partial solutions at each step.

    Args:
        beam_width: Number of partial solutions to keep at each step
        candidates_per_step: Number of best placements to consider per beam state
    """
    t0 = time.perf_counter()

    # Sort and expand boxes
    if sort_key:
        sorted_boxes = sorted(boxes, key=sort_key)
    else:
        sorted_boxes = sorted(boxes, key=lambda b: (-b.volume,))
    instances = _expand_boxes(sorted_boxes)

    total_items = len(instances)

    # Initialize beam with single empty state
    initial_state = PalletState(pallet)
    beam: List[BeamState] = [BeamState(initial_state, [], 0.0)]

    logger.info(
        "[beam_search] task=%s beam_width=%d candidates=%d items=%d",
        task_id, beam_width, candidates_per_step, total_items,
    )

    for step, (box, inst_idx) in enumerate(instances):
        orientations = get_orientations(box)
        next_beam: List[BeamState] = []

        for bs in beam:
            candidates: List[Tuple[float, int, int, int, int, int, int, str]] = []

            for ep in bs.pallet_state.extreme_points:
                ex, ey, ez = ep
                for dx, dy, dz, rot_code in orientations:
                    if not bs.pallet_state.can_place(
                        dx, dy, dz, ex, ey, ez,
                        box.weight_kg, box.fragile, box.stackable,
                    ):
                        continue
                    sc = score_placement(
                        bs.pallet_state, dx, dy, dz, ex, ey, ez,
                        box.weight_kg, box.fragile,
                    )
                    candidates.append((sc, dx, dy, dz, ex, ey, ez, rot_code))

            if not candidates:
                # Can't place this box — carry forward as-is
                next_beam.append(bs)
                continue

            # Take top candidates_per_step
            candidates.sort(key=lambda c: -c[0])
            top = candidates[:candidates_per_step]

            for sc, dx, dy, dz, px, py, pz, rot_code in top:
                new_bs = bs.clone()
                new_bs.pallet_state.place(
                    box.sku_id, dx, dy, dz, px, py, pz,
                    box.weight_kg, box.fragile, box.stackable,
                )
                new_bs.placements.append(Placement(
                    sku_id=box.sku_id,
                    instance_index=inst_idx,
                    x_mm=px, y_mm=py, z_mm=pz,
                    length_mm=dx, width_mm=dy, height_mm=dz,
                    rotation_code=rot_code,
                ))
                new_bs.cumulative_score = sc + bs.cumulative_score
                next_beam.append(new_bs)

        # Prune beam to beam_width
        next_beam.sort(key=lambda bs: -bs.cumulative_score)
        beam = next_beam[:beam_width]

        if not beam:
            break

    # Select best beam state
    best = max(beam, key=lambda bs: len(bs.placements) * 1000 + bs.cumulative_score)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Build unplaced list
    placed_counts: Dict[str, int] = defaultdict(int)
    for p in best.placements:
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
        "[beam_search] done placed=%d/%d time=%dms",
        len(best.placements), total_items, elapsed_ms,
    )

    return Solution(
        task_id=task_id,
        solver_version=__version__,
        solve_time_ms=elapsed_ms,
        placements=best.placements,
        unplaced=unplaced,
    )
