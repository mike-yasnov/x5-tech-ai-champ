"""Genetic Algorithm solver for 3D Pallet Packing.

Chromosome: permutation of box indices + orientation choices.
Crossover: Order crossover (OX) for permutation + uniform for orientations.
Mutation: swap mutation for order, random orientation flip.
Fitness: validator score (volume utilization + coverage + fragility + speed).
"""

import logging
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))


class Chromosome:
    """Represents a packing strategy: box order + orientation choices."""

    __slots__ = ["order", "orientations", "fitness", "_hash"]

    def __init__(self, order: List[int], orientations: List[int]):
        self.order = order
        self.orientations = orientations
        self.fitness = 0.0
        self._hash = None

    def copy(self):
        c = Chromosome(self.order.copy(), self.orientations.copy())
        c.fitness = self.fitness
        return c


def _order_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """Order Crossover (OX) for permutation chromosomes."""
    n = len(p1)
    start, end = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[start:end] = p1[start:end]
    used = set(child[start:end])
    pos = end
    for gene in p2[end:] + p2[:end]:
        if gene not in used:
            child[pos % n] = gene
            pos += 1
    return child


def _uniform_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """Uniform crossover for orientation choices."""
    return [p1[i] if random.random() < 0.5 else p2[i] for i in range(len(p1))]


class GeneticSolver:
    """GA-based 3D packing solver."""

    def __init__(
        self,
        pop_size: int = 100,
        n_generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        elite_ratio: float = 0.1,
        tournament_size: int = 5,
        time_budget_sec: float = 25.0,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = max(2, int(pop_size * elite_ratio))
        self.tournament_size = tournament_size
        self.time_budget_sec = time_budget_sec

        logger.info(
            "GA Solver: pop=%d, gens=%d, cx=%.2f, mut=%.2f, elite=%d, budget=%.1fs",
            pop_size, n_generations, crossover_rate, mutation_rate,
            self.elite_count, time_budget_sec,
        )

    def _get_valid_orientations(self, box: Dict) -> List[Tuple[int, int, int, str]]:
        """Return valid orientations for a box."""
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
        """Expand quantities into individual instances."""
        expanded = []
        for spec in box_specs:
            for i in range(spec["quantity"]):
                expanded.append({**spec, "instance_index": i})
        return expanded

    def _greedy_pack(
        self,
        chromosome: Chromosome,
        boxes: List[Dict],
        pallet: Dict,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Greedy packing using chromosome's order and orientation preferences."""
        L = pallet["length_mm"]
        W = pallet["width_mm"]
        H = pallet["max_height_mm"]
        max_weight = pallet["max_weight_kg"]

        placements = []
        unplaced = []
        placed_3d = []
        current_weight = 0.0

        # Simple extreme points
        eps = [(0, 0, 0)]

        for idx in chromosome.order:
            if idx >= len(boxes):
                continue
            box = boxes[idx]
            valid_oris = self._get_valid_orientations(box)
            if not valid_oris:
                unplaced.append(box)
                continue

            ori_idx = chromosome.orientations[idx] % len(valid_oris)
            dx, dy, dz, rot_code = valid_oris[ori_idx]

            if current_weight + box["weight_kg"] > max_weight:
                unplaced.append(box)
                continue

            best_pos = None
            best_score = float("inf")

            for ex, ey, ez in eps:
                if ex + dx > L or ey + dy > W or ez + dz > H:
                    continue

                # Collision check
                collision = False
                for pb in placed_3d:
                    if (ex < pb[3] and ex + dx > pb[0] and
                        ey < pb[4] and ey + dy > pb[1] and
                        ez < pb[5] and ez + dz > pb[2]):
                        collision = True
                        break
                if collision:
                    continue

                # Support check
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

                # Non-stackable check
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

                # Score: prefer lower, back-left positions
                score = ez * 1000 + ex + ey
                if score < best_score:
                    best_score = score
                    best_pos = (ex, ey, ez)

            if best_pos is None:
                # Try all valid orientations
                for oi, (dx2, dy2, dz2, rc2) in enumerate(valid_oris):
                    if oi == ori_idx:
                        continue
                    for ex, ey, ez in eps:
                        if ex + dx2 > L or ey + dy2 > W or ez + dz2 > H:
                            continue
                        collision = False
                        for pb in placed_3d:
                            if (ex < pb[3] and ex + dx2 > pb[0] and
                                ey < pb[4] and ey + dy2 > pb[1] and
                                ez < pb[5] and ez + dz2 > pb[2]):
                                collision = True
                                break
                        if collision:
                            continue
                        if ez > 0:
                            base_area = dx2 * dy2
                            support = 0.0
                            for pb in placed_3d:
                                if abs(pb[5] - ez) < 1e-3:
                                    ox = max(0, min(ex + dx2, pb[3]) - max(ex, pb[0]))
                                    oy = max(0, min(ey + dy2, pb[4]) - max(ey, pb[1]))
                                    support += ox * oy
                            if base_area > 0 and support / base_area < 0.6:
                                continue
                        stackable_ok = True
                        if ez > 0:
                            for pb in placed_3d:
                                if not pb[8] and abs(pb[5] - ez) < 1e-3:
                                    ox = max(0, min(ex + dx2, pb[3]) - max(ex, pb[0]))
                                    oy = max(0, min(ey + dy2, pb[4]) - max(ey, pb[1]))
                                    if ox > 0 and oy > 0:
                                        stackable_ok = False
                                        break
                        if not stackable_ok:
                            continue
                        score = ez * 1000 + ex + ey
                        if best_pos is None or score < best_score:
                            best_score = score
                            best_pos = (ex, ey, ez)
                            dx, dy, dz, rot_code = dx2, dy2, dz2, rc2
                    if best_pos is not None:
                        break

            if best_pos is None:
                unplaced.append(box)
                continue

            px, py, pz = best_pos
            placements.append({
                "sku_id": box["sku_id"],
                "instance_index": box["instance_index"],
                "position": {"x_mm": px, "y_mm": py, "z_mm": pz},
                "dimensions_placed": {"length_mm": dx, "width_mm": dy, "height_mm": dz},
                "rotation_code": rot_code,
            })
            # (x_min, y_min, z_min, x_max, y_max, z_max, fragile, weight, stackable)
            placed_3d.append((px, py, pz, px + dx, py + dy, pz + dz,
                              box.get("fragile", False), box["weight_kg"],
                              box.get("stackable", True)))
            current_weight += box["weight_kg"]

            # Update extreme points
            new_eps = [
                (px + dx, py, pz),
                (px, py + dy, pz),
                (px, py, pz + dz),
            ]
            # Add projections at other box tops
            for pb in placed_3d[:-1]:
                if abs(pb[5] - pz) < 1e-3:  # same z_max as our z_min
                    pass  # already covered
                new_eps.append((px, py, pb[5]))
                new_eps.append((px + dx, py, pb[5]))

            # Filter: remove EPs inside placed boxes
            valid_eps = set(eps)
            for ep in new_eps:
                inside = False
                for pb in placed_3d:
                    if (pb[0] <= ep[0] < pb[3] and
                        pb[1] <= ep[1] < pb[4] and
                        pb[2] <= ep[2] < pb[5]):
                        inside = True
                        break
                if not inside and ep[0] <= L and ep[1] <= W and ep[2] <= H:
                    valid_eps.add(ep)
            eps = sorted(valid_eps, key=lambda e: (e[2], e[0], e[1]))[:200]

        return placements, unplaced

    def _evaluate(self, chromosome: Chromosome, boxes: List[Dict],
                  pallet: Dict, total_items: int) -> float:
        """Evaluate chromosome fitness."""
        placements, unplaced = self._greedy_pack(chromosome, boxes, pallet)

        pallet_vol = pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
        placed_vol = sum(
            p["dimensions_placed"]["length_mm"] *
            p["dimensions_placed"]["width_mm"] *
            p["dimensions_placed"]["height_mm"]
            for p in placements
        )

        vol_util = placed_vol / pallet_vol if pallet_vol > 0 else 0
        coverage = len(placements) / total_items if total_items > 0 else 0

        # Simple fragility penalty (approximate)
        frag_viol = 0  # would need full check, approximate as 0 for speed

        fitness = 0.50 * vol_util + 0.30 * coverage + 0.10 * 1.0 + 0.10 * 1.0
        return fitness

    def solve(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run GA and return best solution."""
        t0 = time.perf_counter()
        pallet = scenario["pallet"]
        boxes = self._expand_boxes(scenario["boxes"])
        n_boxes = len(boxes)
        total_items = n_boxes

        # Number of valid orientations per box
        n_oris = [len(self._get_valid_orientations(b)) for b in boxes]

        # Initialize population
        population = []
        for _ in range(self.pop_size):
            order = list(range(n_boxes))
            random.shuffle(order)
            orientations = [random.randrange(max(1, n)) for n in n_oris]
            population.append(Chromosome(order, orientations))

        # Also add some heuristic orderings
        # Volume descending
        vol_order = sorted(range(n_boxes), key=lambda i: -(
            boxes[i]["length_mm"] * boxes[i]["width_mm"] * boxes[i]["height_mm"]
        ))
        population[0] = Chromosome(vol_order, [0] * n_boxes)

        # Constrained first
        def _constraint_key(i):
            b = boxes[i]
            priority = 0
            if b.get("strict_upright"):
                priority -= 2
            if not b.get("stackable", True):
                priority -= 1
            return (priority, -(b["length_mm"] * b["width_mm"] * b["height_mm"]))

        constr_order = sorted(range(n_boxes), key=_constraint_key)
        if len(population) > 1:
            population[1] = Chromosome(constr_order, [0] * n_boxes)

        # Evaluate initial population
        for c in population:
            c.fitness = self._evaluate(c, boxes, pallet, total_items)

        best = max(population, key=lambda c: c.fitness)
        logger.info("GA gen 0: best=%.4f", best.fitness)

        for gen in range(1, self.n_generations + 1):
            elapsed = time.perf_counter() - t0
            if elapsed > self.time_budget_sec:
                logger.info("GA time budget exceeded at gen %d (%.1fs)", gen, elapsed)
                break

            # Selection + crossover + mutation
            new_pop = []

            # Elitism
            population.sort(key=lambda c: c.fitness, reverse=True)
            for i in range(self.elite_count):
                new_pop.append(population[i].copy())

            while len(new_pop) < self.pop_size:
                # Tournament selection
                parents = []
                for _ in range(2):
                    tournament = random.sample(population, min(self.tournament_size, len(population)))
                    winner = max(tournament, key=lambda c: c.fitness)
                    parents.append(winner)

                if random.random() < self.crossover_rate:
                    child_order = _order_crossover(parents[0].order, parents[1].order)
                    child_oris = _uniform_crossover(parents[0].orientations, parents[1].orientations)
                    child = Chromosome(child_order, child_oris)
                else:
                    child = parents[0].copy()

                # Mutation
                if random.random() < self.mutation_rate:
                    # Swap mutation on order
                    i, j = random.sample(range(n_boxes), 2)
                    child.order[i], child.order[j] = child.order[j], child.order[i]

                if random.random() < self.mutation_rate:
                    # Random orientation flip
                    idx = random.randrange(n_boxes)
                    child.orientations[idx] = random.randrange(max(1, n_oris[idx]))

                child.fitness = self._evaluate(child, boxes, pallet, total_items)
                new_pop.append(child)

            population = new_pop
            gen_best = max(population, key=lambda c: c.fitness)
            if gen_best.fitness > best.fitness:
                best = gen_best.copy()

            if gen % 20 == 0:
                avg_fit = np.mean([c.fitness for c in population])
                logger.debug("GA gen %d: best=%.4f, avg=%.4f", gen, best.fitness, avg_fit)

        # Build final solution from best chromosome
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        placements, unplaced_boxes = self._greedy_pack(best, boxes, pallet)

        # Build unplaced summary
        placed_skus = {}
        for p in placements:
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
            "GA finished: gen=%d, fitness=%.4f, placed=%d/%d, time=%dms",
            min(gen, self.n_generations), best.fitness, len(placements), total_items, elapsed_ms,
        )

        return {
            "task_id": scenario["task_id"],
            "solver_version": "ga-1.0.0",
            "solve_time_ms": elapsed_ms,
            "placements": placements,
            "unplaced": unplaced,
        }
