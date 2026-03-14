"""Simulated annealing over box ordering for a greedy placement kernel."""

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .models import Box, Pallet, Solution
from .packer import SORT_KEYS, expand_boxes, pack_greedy, pack_in_order

logger = logging.getLogger(__name__)

Instance = Tuple[Box, int]
ScoreFn = Callable[[Solution], float]


@dataclass(frozen=True)
class AnnealingResult:
    solution: Solution
    score: float
    iterations: int
    accepted_moves: int
    seed_order: str


def _proxy_score(solution: Solution, pallet: Pallet) -> float:
    placed_volume = sum(placement.volume for placement in solution.placements)
    placed_count = len(solution.placements)
    unplaced_count = sum(item.quantity_unplaced for item in solution.unplaced)
    height_penalty = 0.0
    if solution.placements:
        height_penalty = max(
            placement.z_max for placement in solution.placements
        ) / max(1, pallet.max_height_mm)
    volume_ratio = placed_volume / max(1, pallet.volume)
    total_items = max(1, placed_count + unplaced_count)
    coverage_ratio = placed_count / total_items
    return 0.65 * coverage_ratio + 0.35 * volume_ratio - 0.05 * height_penalty


def _default_seed_orders(boxes: Sequence[Box]) -> List[Tuple[str, List[Instance]]]:
    seeds: List[Tuple[str, List[Instance]]] = []
    for sort_key_name, sort_fn in SORT_KEYS.items():
        sorted_boxes = sorted(boxes, key=sort_fn)
        seeds.append((sort_key_name, expand_boxes(sorted_boxes)))
    return seeds


def _mutate_order(order: Sequence[Instance], rng: random.Random) -> List[Instance]:
    candidate = list(order)
    if len(candidate) < 2:
        return candidate

    if rng.random() < 0.5:
        i, j = sorted(rng.sample(range(len(candidate)), 2))
        candidate[i], candidate[j] = candidate[j], candidate[i]
    else:
        i, j = rng.sample(range(len(candidate)), 2)
        item = candidate.pop(i)
        candidate.insert(j, item)
    return candidate


def anneal_pack(
    task_id: str,
    pallet: Pallet,
    boxes: Sequence[Box],
    score_solution: ScoreFn,
    time_budget_ms: int,
    initial_temperature: float = 0.25,
    cooling_rate: float = 0.992,
    max_plateau_iterations: int = 250,
    rng_seed: int = 0,
) -> AnnealingResult:
    """Search for a better box order using simulated annealing."""
    t0 = time.perf_counter()
    rng = random.Random(rng_seed)

    seeds = _default_seed_orders(boxes)
    if not seeds:
        empty_solution = pack_in_order(
            task_id, pallet, [], order_name="annealing-empty"
        )
        return AnnealingResult(
            solution=empty_solution,
            score=score_solution(empty_solution),
            iterations=0,
            accepted_moves=0,
            seed_order="empty",
        )

    best_solution: Optional[Solution] = None
    best_score = float("-inf")
    best_order: List[Instance] = []
    best_seed = ""

    for seed_name, seed_order in seeds:
        seed_solution = pack_in_order(task_id, pallet, seed_order, order_name=seed_name)
        seed_score = score_solution(seed_solution)
        logger.info(
            "[anneal] seed=%s score=%.4f placed=%d",
            seed_name,
            seed_score,
            len(seed_solution.placements),
        )
        if seed_score > best_score:
            best_score = seed_score
            best_solution = seed_solution
            best_order = list(seed_order)
            best_seed = seed_name

    assert best_solution is not None

    current_order = list(best_order)
    current_solution = best_solution
    current_score = best_score
    temperature = initial_temperature
    iterations = 0
    accepted_moves = 0
    stagnation = 0

    while True:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms >= time_budget_ms:
            break
        if stagnation >= max_plateau_iterations:
            break

        candidate_order = _mutate_order(current_order, rng)
        candidate_solution = pack_in_order(
            task_id, pallet, candidate_order, order_name="annealing-candidate"
        )
        candidate_score = score_solution(candidate_solution)
        delta = candidate_score - current_score
        accept = False
        if delta >= 0:
            accept = True
        elif temperature > 1e-6:
            accept = rng.random() < math.exp(delta / temperature)

        iterations += 1
        temperature *= cooling_rate

        if accept:
            current_order = candidate_order
            current_solution = candidate_solution
            current_score = candidate_score
            accepted_moves += 1
            stagnation = 0
        else:
            stagnation += 1

        if candidate_score > best_score:
            best_score = candidate_score
            best_solution = candidate_solution
            best_order = list(candidate_order)
            stagnation = 0

    total_ms = int((time.perf_counter() - t0) * 1000)
    best_solution.solve_time_ms = total_ms
    logger.info(
        "[anneal] done seed=%s score=%.4f iterations=%d accepted=%d time=%dms",
        best_seed,
        best_score,
        iterations,
        accepted_moves,
        total_ms,
    )
    return AnnealingResult(
        solution=best_solution,
        score=best_score,
        iterations=iterations,
        accepted_moves=accepted_moves,
        seed_order=best_seed,
    )


def solve_with_annealing(
    task_id: str,
    pallet: Pallet,
    boxes: Sequence[Box],
    score_solution: Optional[ScoreFn] = None,
    time_budget_ms: int = 900,
) -> AnnealingResult:
    if score_solution is None:
        score_solution = lambda solution: _proxy_score(solution, pallet)

    return anneal_pack(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        score_solution=score_solution,
        time_budget_ms=time_budget_ms,
    )


def greedy_seed_solution(
    task_id: str, pallet: Pallet, boxes: Sequence[Box]
) -> Solution:
    """Return the current best deterministic greedy baseline."""
    best_solution: Optional[Solution] = None
    best_score = float("-inf")
    for sort_key_name in SORT_KEYS:
        candidate = pack_greedy(
            task_id, pallet, list(boxes), sort_key_name=sort_key_name
        )
        candidate_score = _proxy_score(candidate, pallet)
        if candidate_score > best_score:
            best_score = candidate_score
            best_solution = candidate
    assert best_solution is not None
    return best_solution
