from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

from .geometry import AABB
from .pallet_state import PalletState
from .free_space import ExtremePointManager
from .feasibility import FeasibilityChecker
from .rotations import get_orientations


class Candidate(NamedTuple):
    sku_id: str
    instance_index: int
    aabb: AABB
    rotation_code: str
    weight: float
    fragile: bool
    stackable: bool
    strict_upright: bool
    placed_dims: Tuple[int, int, int]
    orig_height: int


class RemainingItem:
    """Represents a SKU with remaining quantity to place."""

    __slots__ = (
        "sku_id",
        "length",
        "width",
        "height",
        "weight",
        "strict_upright",
        "fragile",
        "stackable",
        "remaining_qty",
    )

    def __init__(self, box_spec: Dict, placed_count: int = 0):
        self.sku_id: str = box_spec["sku_id"]
        self.length: int = box_spec["length_mm"]
        self.width: int = box_spec["width_mm"]
        self.height: int = box_spec["height_mm"]
        self.weight: float = box_spec["weight_kg"]
        self.strict_upright: bool = box_spec.get("strict_upright", False)
        self.fragile: bool = box_spec.get("fragile", False)
        self.stackable: bool = box_spec.get("stackable", True)
        self.remaining_qty: int = box_spec["quantity"] - placed_count


class CandidateGenerator:
    """Produces feasible candidate placements from remaining items + extreme points."""

    def __init__(self, checker: FeasibilityChecker, max_candidates: int = 200):
        self.checker = checker
        self.max_candidates = max_candidates

    def generate(
        self,
        remaining: List[RemainingItem],
        state: PalletState,
        ep_manager: ExtremePointManager,
    ) -> List[Candidate]:
        points = ep_manager.get_points()

        # Limit extreme points to keep generation fast
        # Scale down with placed count for O(1)-ish generation per step
        n_placed = len(state.placed)
        max_eps = 50 if n_placed < 50 else 30 if n_placed < 100 else 20
        if len(points) > max_eps:
            # Prefer lower z points and corner/edge points
            points.sort(key=lambda p: (p[2], p[0] + p[1]))
            points = points[:max_eps]

        raw: List[Tuple[Candidate, float]] = []

        for item in remaining:
            if item.remaining_qty <= 0:
                continue

            instance_idx = state.next_instance_index(item.sku_id)
            orientations = get_orientations(
                item.length, item.width, item.height, item.strict_upright
            )

            for ep_x, ep_y, ep_z in points:
                for pl, pw, ph, rot_code in orientations:
                    # Quick bounds check before expensive operations
                    if ep_x + pl > self.checker.pallet_length + 1:
                        continue
                    if ep_y + pw > self.checker.pallet_width + 1:
                        continue
                    if ep_z + ph > self.checker.max_height + 1:
                        continue

                    # Project z down to actual support surface
                    z_base = state.get_max_z_at(ep_x, ep_y, pl, pw)
                    z = max(ep_z, z_base)

                    aabb = AABB(ep_x, ep_y, z, ep_x + pl, ep_y + pw, z + ph)

                    ok, _ = self.checker.is_feasible(
                        aabb,
                        item.weight,
                        item.height,
                        item.strict_upright,
                        True,
                        state,
                    )
                    if not ok:
                        continue

                    cand = Candidate(
                        sku_id=item.sku_id,
                        instance_index=instance_idx,
                        aabb=aabb,
                        rotation_code=rot_code,
                        weight=item.weight,
                        fragile=item.fragile,
                        stackable=item.stackable,
                        strict_upright=item.strict_upright,
                        placed_dims=(pl, pw, ph),
                        orig_height=item.height,
                    )

                    # Heuristic priority for diversity selection:
                    # lower z is better, larger base area is better, heavier is better
                    priority = -z * 1e6 + aabb.base_area() * 1e3 + item.weight
                    raw.append((cand, priority))

        if not raw:
            return []

        # Deduplicate by (sku_id, x, y, z, rotation_code)
        seen = set()
        deduped: List[Tuple[Candidate, float]] = []
        for cand, prio in raw:
            key = (
                cand.sku_id,
                cand.aabb.x_min,
                cand.aabb.y_min,
                cand.aabb.z_min,
                cand.rotation_code,
            )
            if key not in seen:
                seen.add(key)
                deduped.append((cand, prio))

        # Diversity cap: sort by priority, take top max_candidates
        deduped.sort(key=lambda x: -x[1])
        return [c for c, _ in deduped[: self.max_candidates]]
