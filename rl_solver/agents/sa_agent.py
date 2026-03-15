"""Simulated Annealing solver for 3D Pallet Packing.

State: permutation of box indices + orientation choices (same as GA chromosome).
Neighborhood: swap two items in order, or change one orientation.
Acceptance: Boltzmann criterion with exponential cooling.
"""

import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))


class SimulatedAnnealingSolver:
    """SA-based 3D packing solver with adaptive cooling."""

    def __init__(
        self,
        initial_temp: float = 10.0,
        cooling_rate: float = 0.9995,
        min_temp: float = 0.001,
        max_iterations: int = 100000,
        time_budget_sec: float = 25.0,
        n_restarts: int = 3,
    ):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.time_budget_sec = time_budget_sec
        self.n_restarts = n_restarts

        logger.info(
            "SA Solver: T0=%.1f, cool=%.4f, min_T=%.4f, max_iter=%d, budget=%.1fs, restarts=%d",
            initial_temp, cooling_rate, min_temp, max_iterations, time_budget_sec, n_restarts,
        )

    def _get_valid_orientations(self, box: Dict) -> List[Tuple[int, int, int, str]]:
        l, w, h = box["length_mm"], box["width_mm"], box["height_mm"]
        all_rots = [
            (l, w, h, "LWH"), (l, h, w, "LHW"),
            (w, l, h, "WLH"), (w, h, l, "WHL"),
            (h, l, w, "HLW"), (h, w, l, "HWL"),
        ]
        if box.get("strict_upright", False):
            all_rots = [(dx, dy, dz, c) for dx, dy, dz, c in all_rots if dz == h]
        seen = set()
        unique = []
        for r in all_rots:
            key = (r[0], r[1], r[2])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def _expand_boxes(self, box_specs: List[Dict]) -> List[Dict]:
        expanded = []
        for spec in box_specs:
            for i in range(spec["quantity"]):
                expanded.append({**spec, "instance_index": i})
        return expanded

    def _greedy_pack(self, order: List[int], orientations: List[int],
                     boxes: List[Dict], pallet: Dict) -> Tuple[List[Dict], int]:
        """Pack boxes in given order with given orientations. Returns (placements, placed_volume)."""
        L = pallet["length_mm"]
        W = pallet["width_mm"]
        H = pallet["max_height_mm"]
        max_weight = pallet["max_weight_kg"]

        placements = []
        placed_3d = []
        current_weight = 0.0
        placed_vol = 0
        eps = [(0, 0, 0)]

        for idx in order:
            if idx >= len(boxes):
                continue
            box = boxes[idx]
            valid_oris = self._get_valid_orientations(box)
            if not valid_oris:
                continue

            if current_weight + box["weight_kg"] > max_weight:
                continue

            # Try preferred orientation first, then others
            ori_idx = orientations[idx] % len(valid_oris)
            ori_order = [ori_idx] + [i for i in range(len(valid_oris)) if i != ori_idx]

            best_pos = None
            best_score = float("inf")
            best_dims = None

            for oi in ori_order:
                dx, dy, dz, rot_code = valid_oris[oi]
                for ex, ey, ez in eps:
                    if ex + dx > L or ey + dy > W or ez + dz > H:
                        continue

                    # Collision
                    collision = False
                    for pb in placed_3d:
                        if (ex < pb[3] and ex + dx > pb[0] and
                            ey < pb[4] and ey + dy > pb[1] and
                            ez < pb[5] and ez + dz > pb[2]):
                            collision = True
                            break
                    if collision:
                        continue

                    # Support
                    if ez > 0:
                        base_area = dx * dy
                        support = 0.0
                        for pb in placed_3d:
                            if abs(pb[5] - ez) < 1e-3:
                                ox = max(0, min(ex + dx, pb[3]) - max(ex, pb[0]))
                                oy = max(0, min(ey + dy, pb[4]) - max(ey, pb[1]))
                                support += ox * oy
                        if base_area > 0 and support / base_area < 0.6:
                            continue

                    # Non-stackable
                    stackable_ok = True
                    if ez > 0:
                        for pb in placed_3d:
                            if not pb[8] and abs(pb[5] - ez) < 1e-3:
                                ox = max(0, min(ex + dx, pb[3]) - max(ex, pb[0]))
                                oy = max(0, min(ey + dy, pb[4]) - max(ey, pb[1]))
                                if ox > 0 and oy > 0:
                                    stackable_ok = False
                                    break
                    if not stackable_ok:
                        continue

                    score = ez * 1000 + ex + ey
                    if score < best_score:
                        best_score = score
                        best_pos = (ex, ey, ez)
                        best_dims = (dx, dy, dz, rot_code)

                if best_pos is not None:
                    break  # Found placement with preferred orientation

            if best_pos is None:
                continue

            px, py, pz = best_pos
            dx, dy, dz, rot_code = best_dims

            placements.append({
                "sku_id": box["sku_id"],
                "instance_index": box["instance_index"],
                "position": {"x_mm": px, "y_mm": py, "z_mm": pz},
                "dimensions_placed": {"length_mm": dx, "width_mm": dy, "height_mm": dz},
                "rotation_code": rot_code,
            })
            placed_3d.append((px, py, pz, px + dx, py + dy, pz + dz,
                              box.get("fragile", False), box["weight_kg"],
                              box.get("stackable", True)))
            current_weight += box["weight_kg"]
            placed_vol += dx * dy * dz

            # Update extreme points
            new_eps = [(px + dx, py, pz), (px, py + dy, pz), (px, py, pz + dz)]
            valid_eps = set(eps)
            for ep in new_eps:
                inside = False
                for pb in placed_3d:
                    if (pb[0] <= ep[0] < pb[3] and pb[1] <= ep[1] < pb[4] and pb[2] <= ep[2] < pb[5]):
                        inside = True
                        break
                if not inside and ep[0] <= L and ep[1] <= W and ep[2] <= H:
                    valid_eps.add(ep)
            eps = sorted(valid_eps, key=lambda e: (e[2], e[0], e[1]))[:200]

        return placements, placed_vol

    def _score(self, placements: List[Dict], placed_vol: int,
               pallet: Dict, total_items: int) -> float:
        """Calculate solution quality score."""
        pallet_vol = pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
        vol_util = placed_vol / pallet_vol if pallet_vol > 0 else 0
        coverage = len(placements) / total_items if total_items > 0 else 0
        return 0.50 * vol_util + 0.30 * coverage + 0.20  # time+fragility bonus approx

    def _neighbor(self, order: List[int], orientations: List[int],
                  n_oris: List[int]) -> Tuple[List[int], List[int]]:
        """Generate neighbor state."""
        new_order = order.copy()
        new_oris = orientations.copy()
        n = len(order)

        r = random.random()
        if r < 0.5:
            # Swap two items
            i, j = random.sample(range(n), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif r < 0.75:
            # Reverse a subsequence
            i, j = sorted(random.sample(range(n), 2))
            new_order[i:j+1] = reversed(new_order[i:j+1])
        else:
            # Change orientation
            idx = random.randrange(n)
            if n_oris[idx] > 1:
                new_oris[idx] = random.randrange(n_oris[idx])

        return new_order, new_oris

    def _single_run(self, boxes: List[Dict], pallet: Dict, total_items: int,
                    n_oris: List[int], time_limit: float) -> Tuple[List[Dict], float]:
        """Single SA run."""
        n = len(boxes)
        t0 = time.perf_counter()

        # Random initial state
        order = list(range(n))
        random.shuffle(order)
        orientations = [random.randrange(max(1, no)) for no in n_oris]

        placements, placed_vol = self._greedy_pack(order, orientations, boxes, pallet)
        current_score = self._score(placements, placed_vol, pallet, total_items)
        best_score = current_score
        best_placements = placements
        best_order = order.copy()
        best_oris = orientations.copy()

        temp = self.initial_temp
        accepted = 0
        improved = 0

        for it in range(self.max_iterations):
            if time.perf_counter() - t0 > time_limit:
                break
            if temp < self.min_temp:
                break

            new_order, new_oris = self._neighbor(order, orientations, n_oris)
            new_placements, new_vol = self._greedy_pack(new_order, new_oris, boxes, pallet)
            new_score = self._score(new_placements, new_vol, pallet, total_items)

            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / temp):
                order = new_order
                orientations = new_oris
                current_score = new_score
                accepted += 1

                if new_score > best_score:
                    best_score = new_score
                    best_placements = new_placements
                    best_order = new_order.copy()
                    best_oris = new_oris.copy()
                    improved += 1

            temp *= self.cooling_rate

        elapsed = time.perf_counter() - t0
        logger.debug(
            "SA run: score=%.4f, placed=%d/%d, accepted=%d, improved=%d, time=%.1fs",
            best_score, len(best_placements), total_items, accepted, improved, elapsed,
        )
        return best_placements, best_score

    def solve(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run SA with restarts and return best solution."""
        t0 = time.perf_counter()
        pallet = scenario["pallet"]
        boxes = self._expand_boxes(scenario["boxes"])
        n_boxes = len(boxes)
        total_items = n_boxes
        n_oris = [len(self._get_valid_orientations(b)) for b in boxes]

        best_placements = []
        best_score = -1.0
        time_per_restart = self.time_budget_sec / max(self.n_restarts, 1)

        for restart in range(self.n_restarts):
            remaining = self.time_budget_sec - (time.perf_counter() - t0)
            if remaining < 0.5:
                break
            limit = min(time_per_restart, remaining)

            placements, score = self._single_run(boxes, pallet, total_items, n_oris, limit)
            if score > best_score:
                best_score = score
                best_placements = placements
                logger.info("SA restart %d: new best=%.4f, placed=%d/%d",
                            restart, best_score, len(best_placements), total_items)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Build unplaced
        placed_skus = {}
        for p in best_placements:
            placed_skus[p["sku_id"]] = placed_skus.get(p["sku_id"], 0) + 1

        unplaced = []
        seen = set()
        for b in boxes:
            sid = b["sku_id"]
            if sid in seen:
                continue
            seen.add(sid)
            total_qty = sum(1 for x in boxes if x["sku_id"] == sid)
            placed_qty = placed_skus.get(sid, 0)
            if placed_qty < total_qty:
                unplaced.append({
                    "sku_id": sid,
                    "quantity_unplaced": total_qty - placed_qty,
                    "reason": "no_space",
                })

        logger.info(
            "SA finished: score=%.4f, placed=%d/%d, time=%dms",
            best_score, len(best_placements), total_items, elapsed_ms,
        )

        return {
            "task_id": scenario["task_id"],
            "solver_version": "sa-1.0.0",
            "solve_time_ms": elapsed_ms,
            "placements": best_placements,
            "unplaced": unplaced,
        }
