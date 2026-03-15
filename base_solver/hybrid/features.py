from __future__ import annotations

from typing import List

import numpy as np

from .candidate_gen import Candidate, RemainingItem
from .constants import EPSILON, FRAGILE_WEIGHT_THRESHOLD
from .pallet_state import PalletState


FEATURE_DIM = 28


class FeatureExtractor:
    """Extracts feature vector for HYB model.

    28 features total:
    - State features (10): pallet utilization and remaining item statistics
    - Candidate features (10): properties of the candidate placement
    - Interaction features (8): how candidate fits with current state
    """

    def extract(
        self,
        state: PalletState,
        candidate: Candidate,
        remaining: List[RemainingItem],
    ) -> np.ndarray:
        f = np.zeros(FEATURE_DIM, dtype=np.float32)

        # --- State features (10) ---
        pal_vol = max(state.pallet_volume, 1)
        pal_area = max(state.pallet_area, 1)

        f[0] = state.placed_volume / pal_vol  # volume utilization so far
        f[1] = state.total_weight / max(state.max_weight, 1)  # weight utilization
        f[2] = state.max_z / max(state.max_height, 1)  # height utilization
        f[3] = min(len(state.placed) / 200.0, 1.0)  # num placed (normalized)

        total_remaining = sum(it.remaining_qty for it in remaining)
        f[4] = min(total_remaining / 200.0, 1.0)  # num remaining (normalized)

        remaining_vol = sum(
            it.remaining_qty * it.length * it.width * it.height for it in remaining
        )
        f[5] = min(remaining_vol / pal_vol, 2.0) / 2.0  # remaining volume norm

        remaining_wt = sum(it.remaining_qty * it.weight for it in remaining)
        f[6] = min(remaining_wt / max(state.max_weight, 1), 2.0) / 2.0

        fragile_remaining = sum(
            it.remaining_qty for it in remaining if it.fragile
        )
        f[7] = fragile_remaining / max(total_remaining, 1)  # fragile ratio

        upright_remaining = sum(
            it.remaining_qty for it in remaining if it.strict_upright
        )
        f[8] = upright_remaining / max(total_remaining, 1)  # upright ratio

        f[9] = 0.0  # placeholder for num_extreme_points (set externally if needed)

        # --- Candidate features (10) ---
        c = candidate
        f[10] = c.aabb.volume() / pal_vol  # candidate volume normalized
        f[11] = c.weight / max(state.max_weight, 1)  # candidate weight normalized
        f[12] = c.aabb.z_min / max(state.max_height, 1)  # z_min normalized
        f[13] = c.aabb.base_area() / pal_area  # base area normalized
        f[14] = 1.0 if c.fragile else 0.0
        f[15] = 1.0 if c.strict_upright else 0.0

        dims = sorted(c.placed_dims)
        f[16] = dims[2] / max(dims[0], 1)  # aspect ratio

        f[17] = c.aabb.height_z() / max(state.max_height, 1)  # height normalized

        # touches wall (0, 1, or 2 walls)
        wall_count = 0
        if c.aabb.x_min == 0 or c.aabb.x_max >= state.length:
            wall_count += 1
        if c.aabb.y_min == 0 or c.aabb.y_max >= state.width:
            wall_count += 1
        f[18] = wall_count / 2.0

        # touches corner
        at_x_edge = c.aabb.x_min == 0 or c.aabb.x_max >= state.length
        at_y_edge = c.aabb.y_min == 0 or c.aabb.y_max >= state.width
        f[19] = 1.0 if (at_x_edge and at_y_edge) else 0.0

        # --- Interaction features (8) ---
        f[20] = state.get_support_ratio(c.aabb)  # support ratio

        # num supporting boxes
        num_supporters = 0
        if c.aabb.z_min > 0:
            for pb in state.placed:
                if abs(pb.aabb.z_max - c.aabb.z_min) < EPSILON:
                    if c.aabb.overlap_area_xy(pb.aabb) > 0:
                        num_supporters += 1
        f[21] = min(num_supporters / 5.0, 1.0)

        # z_gap (distance from candidate bottom to highest supporter top)
        f[22] = 0.0  # will be 0 since we project z in candidate_gen

        # xy coverage increase (new footprint area / pallet area)
        f[23] = c.aabb.base_area() / pal_area

        # weight on fragile below
        weight_on_fragile = 0.0
        if c.weight > FRAGILE_WEIGHT_THRESHOLD and not c.fragile and c.aabb.z_min > 0:
            for pb in state.placed:
                if pb.fragile and abs(pb.aabb.z_max - c.aabb.z_min) < EPSILON:
                    if c.aabb.overlap_area_xy(pb.aabb) > 0:
                        weight_on_fragile = 1.0
                        break
        f[24] = weight_on_fragile

        # height after placement normalized
        new_max_z = max(state.max_z, c.aabb.z_max)
        f[25] = new_max_z / max(state.max_height, 1)

        # volume utilization after placement
        new_vol = state.placed_volume + c.aabb.volume()
        f[26] = new_vol / pal_vol

        # wasted space below candidate
        if c.aabb.z_min > 0:
            footprint_height = state.get_max_z_at(
                c.aabb.x_min, c.aabb.y_min,
                c.aabb.length_x(), c.aabb.width_y()
            )
            gap = c.aabb.z_min - footprint_height
            f[27] = min(gap / max(state.max_height, 1), 1.0)
        else:
            f[27] = 0.0

        return f

    def extract_batch(
        self,
        state: PalletState,
        candidates: List[Candidate],
        remaining: List[RemainingItem],
    ) -> np.ndarray:
        """Returns (len(candidates), FEATURE_DIM) array."""
        n = len(candidates)
        X = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        for i, c in enumerate(candidates):
            X[i] = self.extract(state, c, remaining)
        return X
