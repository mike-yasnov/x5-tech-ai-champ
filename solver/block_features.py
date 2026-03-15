from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .hybrid.candidate_gen import RemainingItem
from .hybrid.pallet_state import PalletState


FEATURE_DIM = 28
POLICIES = ("foundation", "fragile_last", "coverage_fill", "legacy_portfolio")


@dataclass(frozen=True)
class BlockFeatureView:
    item_count: int
    nx: int
    ny: int
    nz: int
    block_length: int
    block_width: int
    block_height: int
    block_weight: float
    unit_weight: float
    unit_volume: int
    fragile: bool
    strict_upright: bool
    support_ratio: float
    wall_count: int
    corner_touch: bool
    z_min: int
    residual_x: int
    residual_y: int
    remaining_height: int
    heuristic_score: float


class BlockFeatureExtractor:
    """Tabular features for optional block-ranking models."""

    def extract(
        self,
        state: PalletState,
        candidate: BlockFeatureView,
        remaining: List[RemainingItem],
        policy_name: str,
    ) -> np.ndarray:
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        pal_vol = max(state.pallet_volume, 1)
        pal_area = max(state.pallet_area, 1)

        total_remaining = sum(item.remaining_qty for item in remaining)
        fragile_remaining = sum(item.remaining_qty for item in remaining if item.fragile)
        remaining_weight = sum(item.remaining_qty * item.weight for item in remaining)
        remaining_volume = sum(
            item.remaining_qty * item.length * item.width * item.height
            for item in remaining
        )
        floor_coverage = sum(
            pb.aabb.base_area() for pb in state.placed if pb.aabb.z_min == 0
        ) / pal_area
        support_boxes = 0
        if candidate.z_min > 0:
            for pb in state.placed:
                if pb.aabb.z_max == candidate.z_min:
                    support_boxes += 1

        f[0] = state.placed_volume / pal_vol
        f[1] = state.total_weight / max(state.max_weight, 1)
        f[2] = state.max_z / max(state.max_height, 1)
        f[3] = min(len(state.placed) / 200.0, 1.0)
        f[4] = min(total_remaining / 200.0, 1.0)
        f[5] = min(remaining_weight / max(state.max_weight, 1), 2.0) / 2.0
        f[6] = min(remaining_volume / pal_vol, 2.0) / 2.0
        f[7] = fragile_remaining / max(total_remaining, 1)
        f[8] = min(floor_coverage, 1.0)
        f[9] = min(support_boxes / 8.0, 1.0)

        block_volume = candidate.block_length * candidate.block_width * candidate.block_height
        f[10] = block_volume / pal_vol
        f[11] = candidate.block_weight / max(state.max_weight, 1)
        f[12] = candidate.item_count / 64.0
        f[13] = candidate.block_length * candidate.block_width / pal_area
        f[14] = candidate.block_height / max(state.max_height, 1)
        f[15] = candidate.unit_weight / max(state.max_weight, 1)
        f[16] = min(candidate.unit_volume / max(pal_vol, 1), 1.0)
        f[17] = 1.0 if candidate.fragile else 0.0
        f[18] = 1.0 if candidate.strict_upright else 0.0
        f[19] = candidate.support_ratio
        f[20] = candidate.wall_count / 2.0
        f[21] = 1.0 if candidate.corner_touch else 0.0
        f[22] = candidate.z_min / max(state.max_height, 1)
        f[23] = candidate.residual_x / max(state.length, 1)
        f[24] = candidate.residual_y / max(state.width, 1)
        f[25] = candidate.remaining_height / max(state.max_height, 1)
        f[26] = candidate.block_weight / max(state.remaining_weight_capacity(), 1.0)
        f[27] = self._policy_index(policy_name)
        return f

    def extract_batch(
        self,
        state: PalletState,
        candidates: List[BlockFeatureView],
        remaining: List[RemainingItem],
        policy_name: str,
    ) -> np.ndarray:
        X = np.zeros((len(candidates), FEATURE_DIM), dtype=np.float32)
        for idx, candidate in enumerate(candidates):
            X[idx] = self.extract(state, candidate, remaining, policy_name)
        return X

    def _policy_index(self, policy_name: str) -> float:
        try:
            return POLICIES.index(policy_name) / max(len(POLICIES) - 1, 1)
        except ValueError:
            return 0.0
