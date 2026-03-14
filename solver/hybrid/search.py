from __future__ import annotations

import time
from typing import List, Optional, Tuple

from .candidate_gen import Candidate, CandidateGenerator, RemainingItem
from .feasibility import FeasibilityChecker
from .free_space import ExtremePointManager
from .pallet_state import PalletState, PlacedBox


def _heuristic_score(c: Candidate, state: PalletState) -> float:
    """Fallback ranking without ML.

    Priorities:
    - lower z is better (fill bottom first)
    - larger base area is better (stable foundation)
    - heavier non-fragile items first (heavy bottom)
    - strongly penalize placing heavy items directly on fragile items
    """
    from .constants import EPSILON, FRAGILE_WEIGHT_THRESHOLD

    z_norm = c.aabb.z_min / max(state.max_height, 1)
    area_norm = c.aabb.base_area() / max(state.pallet_area, 1)
    weight_norm = c.weight / max(state.max_weight, 1)

    # Prefer non-fragile items at lower z by giving them a weight bonus
    # Fragile items get no weight bonus, making them naturally deferred
    effective_weight = weight_norm if not c.fragile else weight_norm * 0.3

    # Penalize placing heavy items directly on fragile items below
    fragile_below_penalty = 0.0
    if c.weight > FRAGILE_WEIGHT_THRESHOLD and c.aabb.z_min > 0 and not c.fragile:
        for pb in state.placed:
            if pb.fragile and abs(pb.aabb.z_max - c.aabb.z_min) < EPSILON:
                if c.aabb.overlap_area_xy(pb.aabb) > 0:
                    fragile_below_penalty = 5.0
                    break

    wall_count = 0
    if c.aabb.x_min == 0 or c.aabb.x_max >= state.length:
        wall_count += 1
    if c.aabb.y_min == 0 or c.aabb.y_max >= state.width:
        wall_count += 1
    corner_touch = 1.0 if wall_count == 2 else 0.0

    return (
        -z_norm * 3.0
        + area_norm * 2.0
        + effective_weight * 1.0
        + wall_count * 0.25
        + corner_touch * 0.15
        - fragile_below_penalty
    )


def _make_placed_box(c: Candidate) -> PlacedBox:
    return PlacedBox(
        sku_id=c.sku_id,
        instance_index=c.instance_index,
        aabb=c.aabb,
        weight=c.weight,
        fragile=c.fragile,
        stackable=c.stackable,
        strict_upright=c.strict_upright,
        rotation_code=c.rotation_code,
        placed_dims=c.placed_dims,
    )


def _update_remaining(remaining: List[RemainingItem], sku_id: str) -> List[RemainingItem]:
    """Return a new remaining list with one less of the given SKU."""
    result = []
    for item in remaining:
        if item.sku_id == sku_id and item.remaining_qty > 0:
            new_item = RemainingItem.__new__(RemainingItem)
            new_item.sku_id = item.sku_id
            new_item.length = item.length
            new_item.width = item.width
            new_item.height = item.height
            new_item.weight = item.weight
            new_item.strict_upright = item.strict_upright
            new_item.fragile = item.fragile
            new_item.stackable = item.stackable
            new_item.remaining_qty = item.remaining_qty - 1
            result.append(new_item)
        else:
            result.append(item)
    return result


def greedy_solve(
    remaining: List[RemainingItem],
    state: PalletState,
    ep_manager: ExtremePointManager,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
) -> Tuple[List[PlacedBox], List[RemainingItem]]:
    """Single-pass greedy: generate candidates, rank by heuristic, pick best, repeat."""
    placements: List[PlacedBox] = []

    while True:
        # Check if anything remains
        total_remaining = sum(it.remaining_qty for it in remaining)
        if total_remaining == 0:
            break

        candidates = candidate_gen.generate(remaining, state, ep_manager)
        if not candidates:
            break

        # Rank by heuristic
        scored = [(c, _heuristic_score(c, state)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        best = scored[0][0]

        # Place the best candidate
        placed = _make_placed_box(best)
        state.place(placed)
        ep_manager.update_after_placement(best.aabb, state)
        placements.append(placed)

        # Update remaining
        remaining = _update_remaining(remaining, best.sku_id)

    return placements, remaining


class _BeamNode:
    """A node in the beam search tree."""

    __slots__ = ("state", "ep_manager", "remaining", "placements", "score")

    def __init__(
        self,
        state: PalletState,
        ep_manager: ExtremePointManager,
        remaining: List[RemainingItem],
        placements: List[PlacedBox],
        score: float,
    ):
        self.state = state
        self.ep_manager = ep_manager
        self.remaining = remaining
        self.placements = placements
        self.score = score


def _evaluate_node_score(
    state: PalletState, num_placed: int, total_items: int
) -> float:
    """Quick evaluation of a node for beam ranking.

    Approximates the final composite score: 0.5*vol + 0.3*coverage.
    Fragility is handled by the heuristic/model ranking, not node selection.
    """
    vol_util = state.placed_volume / max(state.pallet_volume, 1)
    coverage = num_placed / max(total_items, 1)
    return 0.50 * vol_util + 0.30 * coverage


def beam_search_solve(
    remaining: List[RemainingItem],
    state: PalletState,
    ep_manager: ExtremePointManager,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    model: Optional[object] = None,
    extractor: Optional[object] = None,
    beam_width: int = 8,
    max_expansions: int = 5,
    time_budget_s: float = 4.0,
) -> Tuple[List[PlacedBox], List[RemainingItem]]:
    """Beam search over placement sequences.

    At each step, for each node:
    1. Generate feasible candidates
    2. Rank by HYB model (or heuristic fallback)
    3. Branch into top-K expansions
    4. Keep global top beam_width nodes
    """
    t0 = time.perf_counter()
    use_model = model is not None and model.is_trained and extractor is not None
    total_items = sum(it.remaining_qty for it in remaining)

    root = _BeamNode(
        state=state.copy(),
        ep_manager=ep_manager.copy(),
        remaining=list(remaining),
        placements=[],
        score=0.0,
    )
    beam = [root]
    best_complete: Optional[_BeamNode] = None

    step = 0
    while beam:
        step += 1

        # Time check
        elapsed = time.perf_counter() - t0
        if elapsed > time_budget_s * 0.85:
            break

        all_children: List[_BeamNode] = []
        any_expanded = False

        for node in beam:
            total_rem = sum(it.remaining_qty for it in node.remaining)
            if total_rem == 0:
                score = _evaluate_node_score(node.state, len(node.placements), total_items)
                if best_complete is None or score > best_complete.score:
                    node.score = score
                    best_complete = node
                continue

            candidates = candidate_gen.generate(
                node.remaining, node.state, node.ep_manager
            )
            if not candidates:
                score = _evaluate_node_score(node.state, len(node.placements), total_items)
                if best_complete is None or score > best_complete.score:
                    node.score = score
                    best_complete = node
                continue

            # Rank candidates
            if use_model:
                X = extractor.extract_batch(node.state, candidates, node.remaining)
                ranked_idx = model.rank(X)
            else:
                scored = [
                    (i, _heuristic_score(c, node.state))
                    for i, c in enumerate(candidates)
                ]
                scored.sort(key=lambda x: -x[1])
                ranked_idx = [i for i, _ in scored]

            # Take top-K expansions
            for ci in ranked_idx[:max_expansions]:
                c = candidates[ci]
                new_state = node.state.copy()
                new_ep = node.ep_manager.copy()
                new_remaining = _update_remaining(node.remaining, c.sku_id)

                placed = _make_placed_box(c)
                new_state.place(placed)
                new_ep.update_after_placement(c.aabb, new_state)

                new_placements = node.placements + [placed]
                child_score = _evaluate_node_score(
                    new_state, len(new_placements), total_items
                )

                child = _BeamNode(
                    state=new_state,
                    ep_manager=new_ep,
                    remaining=new_remaining,
                    placements=new_placements,
                    score=child_score,
                )
                all_children.append(child)
                any_expanded = True

        if not any_expanded:
            break

        # Keep top beam_width children
        all_children.sort(key=lambda n: -n.score)
        beam = all_children[:beam_width]

        # Track best
        for node in beam:
            if best_complete is None or node.score > best_complete.score:
                best_complete = node

    # Return best solution found
    if best_complete is not None:
        return best_complete.placements, best_complete.remaining

    # Fallback: greedy from original state
    return greedy_solve(remaining, state, ep_manager, checker, candidate_gen)
