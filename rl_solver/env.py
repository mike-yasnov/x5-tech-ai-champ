"""Gymnasium environment for 3D Pallet Packing.

Observation space:
  - Pallet heightmap (discretized grid)
  - Current box features (dims, weight, fragile, upright, stackable)
  - Remaining items summary
  - Packing state (fill ratio, weight ratio, max_z)

Action space:
  - Discrete: (grid_position_index * num_orientations + orientation_index)
  - Grid positions are discretized placement points on the pallet

Reward:
  - +volume_placed / pallet_volume for successful placement
  - +bonus for full coverage
  - -penalty for fragility violations
  - -large penalty for invalid placement attempts
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))

# Heightmap resolution
GRID_RES = 50  # mm per cell


class PackingEnv(gym.Env):
    """3D Pallet Packing as a Gymnasium environment."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        grid_res: int = GRID_RES,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.scenarios = scenarios or []
        self.grid_res = grid_res
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._scenario_idx = 0

        # Will be set in reset()
        self.pallet = None
        self.boxes_to_place: List[Dict] = []
        self.current_box_idx = 0
        self.placements: List[Dict] = []
        self.heightmap: Optional[np.ndarray] = None
        self.grid_w = 0
        self.grid_h = 0
        self.current_weight = 0.0
        self.step_count = 0
        self.total_volume = 0
        self.placed_volume = 0
        self._fragility_violations = 0

        # Fixed max dims for normalization (covers all possible pallets)
        self._max_l = 1219  # largest pallet (US 48")
        self._max_w = 1016  # largest pallet (US 40")
        self._max_h = 2000

        # Fixed grid size based on max pallet dimensions — NEVER changes
        self._fixed_gw = max(1, self._max_l // grid_res)  # 24
        self._fixed_gh = max(1, self._max_w // grid_res)   # 20

        # Actual pallet grid (may be smaller, padded to fixed size)
        self.grid_w = self._fixed_gw
        self.grid_h = self._fixed_gh

        # Orientation codes
        self.orientations = [
            "LWH", "LHW", "WLH", "WHL", "HLW", "HWL"
        ]
        self.n_orientations = len(self.orientations)

        # Fixed spaces — NEVER change between resets
        n_positions = self._fixed_gw * self._fixed_gh
        self.n_actions = n_positions * self.n_orientations

        self.action_space = spaces.Discrete(self.n_actions)

        # Observation: fixed-size heightmap + box features + state features
        obs_size = self._fixed_gw * self._fixed_gh + 8 + 5
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

    def _expand_boxes(self, box_specs: List[Dict]) -> List[Dict]:
        """Expand quantity into individual box instances."""
        expanded = []
        for spec in box_specs:
            for i in range(spec["quantity"]):
                expanded.append({
                    "sku_id": spec["sku_id"],
                    "instance_index": i,
                    "length_mm": spec["length_mm"],
                    "width_mm": spec["width_mm"],
                    "height_mm": spec["height_mm"],
                    "weight_kg": spec["weight_kg"],
                    "strict_upright": spec.get("strict_upright", False),
                    "fragile": spec.get("fragile", False),
                    "stackable": spec.get("stackable", True),
                })
        return expanded

    def _get_orientations(self, box: Dict) -> List[Tuple[int, int, int, str]]:
        """Get valid orientations for a box (respecting strict_upright)."""
        l, w, h = box["length_mm"], box["width_mm"], box["height_mm"]
        all_rots = [
            (l, w, h, "LWH"), (l, h, w, "LHW"),
            (w, l, h, "WLH"), (w, h, l, "WHL"),
            (h, l, w, "HLW"), (h, w, l, "HWL"),
        ]
        if box["strict_upright"]:
            # Only rotations where original height stays as Z
            all_rots = [(dx, dy, dz, code) for dx, dy, dz, code in all_rots if dz == h]

        # Deduplicate
        seen = set()
        unique = []
        for dx, dy, dz, code in all_rots:
            key = (dx, dy, dz)
            if key not in seen:
                seen.add(key)
                unique.append((dx, dy, dz, code))
        return unique

    def reset(self, seed=None, options=None):
        """Reset environment with next scenario."""
        super().reset(seed=seed)

        if not self.scenarios:
            raise ValueError("No scenarios loaded")

        scenario = self.scenarios[self._scenario_idx % len(self.scenarios)]
        self._scenario_idx += 1

        self.pallet = scenario["pallet"]
        # Store actual pallet dims for bounds checking
        self._pallet_l = self.pallet["length_mm"]
        self._pallet_w = self.pallet["width_mm"]
        self._pallet_h = self.pallet["max_height_mm"]
        # Actual grid cells used by this pallet (rest is padding)
        self._actual_gw = max(1, self._pallet_l // self.grid_res)
        self._actual_gh = max(1, self._pallet_w // self.grid_res)

        self.boxes_to_place = self._expand_boxes(scenario["boxes"])
        self.total_items = len(self.boxes_to_place)
        self.total_volume = sum(
            b["length_mm"] * b["width_mm"] * b["height_mm"] for b in self.boxes_to_place
        )
        self.current_box_idx = 0
        self.placements = []
        # Fixed-size heightmap — padded with high values outside pallet bounds
        self.heightmap = np.zeros((self._fixed_gw, self._fixed_gh), dtype=np.float32)
        # Mark cells outside actual pallet as "full" (high wall)
        self.heightmap[self._actual_gw:, :] = self._max_h / self.grid_res
        self.heightmap[:, self._actual_gh:] = self._max_h / self.grid_res
        self.current_weight = 0.0
        self.placed_volume = 0
        self.step_count = 0
        self._fragility_violations = 0
        self._placed_boxes_3d = []  # for collision/support checks

        logger.debug(
            "Reset env: scenario=%s, pallet=%dx%dx%d, items=%d",
            scenario["task_id"], self._pallet_l, self._pallet_w, self._pallet_h, self.total_items
        )

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        # Flatten and normalize heightmap
        hm = self.heightmap.flatten() / max(self._max_h, 1)

        # Current box features (or zeros if done)
        if self.current_box_idx < len(self.boxes_to_place):
            box = self.boxes_to_place[self.current_box_idx]
            box_feat = np.array([
                box["length_mm"] / self._max_l,
                box["width_mm"] / self._max_w,
                box["height_mm"] / self._max_h,
                box["weight_kg"] / max(self.pallet["max_weight_kg"], 1),
                float(box["fragile"]),
                float(box["strict_upright"]),
                float(box["stackable"]),
                (box["length_mm"] * box["width_mm"] * box["height_mm"]) /
                (self._max_l * self._max_w * self._max_h),
            ], dtype=np.float32)
        else:
            box_feat = np.zeros(8, dtype=np.float32)

        # State features
        max_z = float(self.heightmap.max()) * self.grid_res if self.heightmap is not None else 0
        state_feat = np.array([
            self.placed_volume / max(self._max_l * self._max_w * self._max_h, 1),
            self.current_weight / max(self.pallet["max_weight_kg"], 1),
            max_z / max(self._max_h, 1),
            (len(self.boxes_to_place) - self.current_box_idx) / max(self.total_items, 1),
            self.step_count / max(self.max_steps, 1),
        ], dtype=np.float32)

        obs = np.concatenate([hm, box_feat, state_feat])
        return np.clip(obs, 0.0, 1.0)

    def _check_collision(self, x, y, z, dx, dy, dz) -> bool:
        """Check if new box collides with any placed box."""
        for pb in self._placed_boxes_3d:
            # AABB overlap check
            if (x < pb["x_max"] and x + dx > pb["x_min"] and
                y < pb["y_max"] and y + dy > pb["y_min"] and
                z < pb["z_max"] and z + dz > pb["z_min"]):
                return True
        return False

    def _check_support(self, x, y, z, dx, dy, dz) -> bool:
        """Check if box has >=60% support at z level."""
        if z == 0:
            return True  # On floor
        base_area = dx * dy
        if base_area == 0:
            return False
        support_area = 0.0
        for pb in self._placed_boxes_3d:
            if abs(pb["z_max"] - z) < 1e-3:
                # Overlap in XY
                ox = max(0, min(x + dx, pb["x_max"]) - max(x, pb["x_min"]))
                oy = max(0, min(y + dy, pb["y_max"]) - max(y, pb["y_min"]))
                support_area += ox * oy
        return support_area / base_area >= 0.6

    def _check_stackable(self, x, y, z, dx, dy) -> bool:
        """Check we're not placing on top of a non-stackable box."""
        if z == 0:
            return True
        for pb in self._placed_boxes_3d:
            if not pb["stackable"] and abs(pb["z_max"] - z) < 1e-3:
                # Check XY overlap
                ox = max(0, min(x + dx, pb["x_max"]) - max(x, pb["x_min"]))
                oy = max(0, min(y + dy, pb["y_max"]) - max(y, pb["y_min"]))
                if ox > 0 and oy > 0:
                    return False
        return True

    def _count_fragility_violations(self, x, y, z, dx, dy, weight) -> int:
        """Count how many fragile boxes this placement violates."""
        if weight <= 2.0:
            return 0
        violations = 0
        for pb in self._placed_boxes_3d:
            if pb["fragile"] and abs(pb["z_max"] - z) < 1e-3:
                ox = max(0, min(x + dx, pb["x_max"]) - max(x, pb["x_min"]))
                oy = max(0, min(y + dy, pb["y_max"]) - max(y, pb["y_min"]))
                if ox > 0 and oy > 0:
                    violations += 1
        return violations

    def _update_heightmap(self, x, y, dx, dy, z_top):
        """Update heightmap with placed box."""
        x1 = max(0, x // self.grid_res)
        y1 = max(0, y // self.grid_res)
        x2 = min(self._fixed_gw, (x + dx + self.grid_res - 1) // self.grid_res)
        y2 = min(self._fixed_gh, (y + dy + self.grid_res - 1) // self.grid_res)
        z_cells = z_top / self.grid_res
        self.heightmap[x1:x2, y1:y2] = np.maximum(
            self.heightmap[x1:x2, y1:y2], z_cells
        )

    def step(self, action: int):
        """Execute placement action."""
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps
        reward = 0.0
        info = {}

        if self.current_box_idx >= len(self.boxes_to_place):
            terminated = True
            return self._get_obs(), 0.0, terminated, truncated, info

        box = self.boxes_to_place[self.current_box_idx]
        valid_orientations = self._get_orientations(box)

        # Decode action
        n_positions = self._fixed_gw * self._fixed_gh
        pos_idx = action // self.n_orientations
        ori_idx = action % self.n_orientations

        # Clamp position
        pos_idx = min(pos_idx, n_positions - 1)
        gx = pos_idx // self._fixed_gh
        gy = pos_idx % self._fixed_gh
        x = gx * self.grid_res
        y = gy * self.grid_res

        # Get orientation (map to valid ones)
        if ori_idx >= len(valid_orientations):
            ori_idx = ori_idx % len(valid_orientations)
        dx, dy, dz, rot_code = valid_orientations[ori_idx]

        # Determine z from heightmap
        x1 = max(0, gx)
        y1 = max(0, gy)
        x2 = min(self._fixed_gw, gx + max(1, dx // self.grid_res))
        y2 = min(self._fixed_gh, gy + max(1, dy // self.grid_res))
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
        z = int(self.heightmap[x1:x2, y1:y2].max()) * self.grid_res

        # Validate placement against ACTUAL pallet bounds
        valid = True
        reason = ""

        # Bounds (use actual pallet dimensions, not fixed max)
        if x + dx > self._pallet_l or y + dy > self._pallet_w or z + dz > self._pallet_h:
            valid = False
            reason = "bounds"

        # Weight
        if valid and self.current_weight + box["weight_kg"] > self.pallet["max_weight_kg"]:
            valid = False
            reason = "weight"

        # Collision
        if valid and self._check_collision(x, y, z, dx, dy, dz):
            valid = False
            reason = "collision"

        # Support
        if valid and not self._check_support(x, y, z, dx, dy, dz):
            valid = False
            reason = "support"

        # Stackable
        if valid and not self._check_stackable(x, y, z, dx, dy):
            valid = False
            reason = "stackable"

        if valid:
            # Place the box
            vol = dx * dy * dz
            pallet_vol = self._pallet_l * self._pallet_w * self._pallet_h

            # Fragility check (soft penalty)
            frag_viol = self._count_fragility_violations(x, y, z, dx, dy, box["weight_kg"])
            self._fragility_violations += frag_viol

            self._placed_boxes_3d.append({
                "x_min": x, "x_max": x + dx,
                "y_min": y, "y_max": y + dy,
                "z_min": z, "z_max": z + dz,
                "fragile": box["fragile"],
                "stackable": box["stackable"],
                "weight": box["weight_kg"],
            })

            self.placements.append({
                "sku_id": box["sku_id"],
                "instance_index": box["instance_index"],
                "position": {"x_mm": x, "y_mm": y, "z_mm": z},
                "dimensions_placed": {"length_mm": dx, "width_mm": dy, "height_mm": dz},
                "rotation_code": rot_code,
            })

            self.current_weight += box["weight_kg"]
            self.placed_volume += vol
            self._update_heightmap(x, y, dx, dy, z + dz)

            # Reward: volume utilization contribution
            reward = vol / pallet_vol
            # Bonus for low placement (encourages bottom-up)
            reward += 0.01 * (1.0 - z / self._pallet_h)
            # Penalty for fragility violation
            reward -= 0.05 * frag_viol

            self.current_box_idx += 1
            info["placed"] = True

            logger.debug(
                "Placed %s at (%d,%d,%d) rot=%s vol=%d",
                box["sku_id"], x, y, z, rot_code, vol
            )
        else:
            # Invalid placement — skip this box
            reward = -0.02  # small penalty
            self.current_box_idx += 1  # move to next box
            info["placed"] = False
            info["skip_reason"] = reason
            logger.debug("Skipped %s: %s", box["sku_id"], reason)

        # Check if all boxes processed
        if self.current_box_idx >= len(self.boxes_to_place):
            terminated = True
            # Final bonus for coverage
            coverage = len(self.placements) / max(self.total_items, 1)
            reward += 0.5 * coverage

        return self._get_obs(), reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Return mask of valid actions for current state."""
        mask = np.zeros(self.n_actions, dtype=np.float32)

        if self.current_box_idx >= len(self.boxes_to_place):
            return mask

        box = self.boxes_to_place[self.current_box_idx]
        valid_orientations = self._get_orientations(box)

        for gx in range(self._actual_gw):
            for gy in range(self._actual_gh):
                x = gx * self.grid_res
                y = gy * self.grid_res
                pos_idx = gx * self._fixed_gh + gy

                for oi, (dx, dy, dz, _) in enumerate(valid_orientations):
                    # Quick bounds check against actual pallet
                    if x + dx > self._pallet_l or y + dy > self._pallet_w:
                        continue

                    # Get z from heightmap
                    x2 = min(self._fixed_gw, gx + max(1, dx // self.grid_res))
                    y2 = min(self._fixed_gh, gy + max(1, dy // self.grid_res))
                    z = int(self.heightmap[gx:x2, gy:y2].max()) * self.grid_res

                    if z + dz > self._pallet_h:
                        continue
                    if self.current_weight + box["weight_kg"] > self.pallet["max_weight_kg"]:
                        continue

                    action_idx = pos_idx * self.n_orientations + oi
                    if action_idx < self.n_actions:
                        mask[action_idx] = 1.0

        return mask

    def get_solution_dict(self, task_id: str, solve_time_ms: int) -> Dict[str, Any]:
        """Convert current placements to submission format."""
        # Build unplaced list
        placed_counts = {}
        for p in self.placements:
            placed_counts[p["sku_id"]] = placed_counts.get(p["sku_id"], 0) + 1

        unplaced = []
        box_specs = {}
        for box in self.boxes_to_place:
            if box["sku_id"] not in box_specs:
                box_specs[box["sku_id"]] = box

        seen_skus = set()
        for box in self.boxes_to_place:
            sid = box["sku_id"]
            if sid in seen_skus:
                continue
            seen_skus.add(sid)
            total_qty = sum(1 for b in self.boxes_to_place if b["sku_id"] == sid)
            placed_qty = placed_counts.get(sid, 0)
            if placed_qty < total_qty:
                unplaced.append({
                    "sku_id": sid,
                    "quantity_unplaced": total_qty - placed_qty,
                    "reason": "no_space",
                })

        return {
            "task_id": task_id,
            "solver_version": "rl-1.0.0",
            "solve_time_ms": solve_time_ms,
            "placements": self.placements,
            "unplaced": unplaced,
        }
