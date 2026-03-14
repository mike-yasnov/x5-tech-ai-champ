"""Simulated annealing over box ordering for a greedy placement kernel."""

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .models import Box, Pallet, Solution
from .packer import SORT_KEYS, expand_boxes, pack_greedy, pack_in_order

logger = logging.getLogger(__name__)

Instance = Tuple[Box, int]
ScoreFn = Callable[[Solution], float]
OrderSignature = Tuple[Tuple[str, int], ...]
EliteEntry = Tuple[float, List[Instance], Solution, str]


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

    constrained_first = sorted(
        boxes,
        key=lambda box: (
            -(1 if box.strict_upright else 0),
            -(1 if box.fragile else 0),
            -(1 if not box.stackable else 0),
            -box.volume,
            -box.weight_kg,
        ),
    )
    seeds.append(("anneal_constraints", expand_boxes(constrained_first)))

    tall_heavy_first = sorted(
        boxes,
        key=lambda box: (-box.height_mm, -box.weight_kg, -box.base_area, -box.volume),
    )
    seeds.append(("anneal_tall_heavy", expand_boxes(tall_heavy_first)))

    footprint_first = sorted(
        boxes,
        key=lambda box: (-box.base_area, -box.height_mm, -box.weight_kg, -box.volume),
    )
    seeds.append(("anneal_footprint", expand_boxes(footprint_first)))
    return seeds


def _order_signature(order: Sequence[Instance]) -> OrderSignature:
    return tuple((box.sku_id, inst_idx) for box, inst_idx in order)


def _volume_priority(instance: Instance) -> Tuple[int, float, int]:
    box, _ = instance
    return (
        box.volume,
        box.weight_kg,
        int(box.strict_upright) + int(box.fragile) + int(not box.stackable),
    )


def _promote_matching_sku(
    order: Sequence[Instance],
    sku_ids: Sequence[str],
    rng: random.Random,
) -> Optional[List[Instance]]:
    candidate = list(order)
    matching = [idx for idx, (box, _) in enumerate(candidate) if box.sku_id in sku_ids]
    if not matching:
        return None

    src_idx = rng.choice(matching)
    dst_hi = max(1, len(candidate) // 3)
    dst_idx = rng.randrange(0, dst_hi)
    item = candidate.pop(src_idx)
    candidate.insert(dst_idx, item)
    return candidate


def _promote_matching_block(
    order: Sequence[Instance],
    sku_ids: Sequence[str],
    rng: random.Random,
) -> Optional[List[Instance]]:
    candidate = list(order)
    promoted = [item for item in candidate if item[0].sku_id in sku_ids]
    if not promoted:
        return None
    remainder = [item for item in candidate if item[0].sku_id not in sku_ids]
    split_idx = rng.randrange(0, max(1, len(remainder) // 3 + 1))
    return remainder[:split_idx] + promoted + remainder[split_idx:]


def _reverse_window(order: Sequence[Instance], rng: random.Random) -> List[Instance]:
    candidate = list(order)
    if len(candidate) < 4:
        return candidate
    i, j = sorted(rng.sample(range(len(candidate)), 2))
    if i == j:
        return candidate
    candidate[i : j + 1] = reversed(candidate[i : j + 1])
    return candidate


def _move_chunk(order: Sequence[Instance], rng: random.Random) -> List[Instance]:
    candidate = list(order)
    if len(candidate) < 5:
        return candidate
    start = rng.randrange(0, len(candidate) - 1)
    max_chunk = min(8, len(candidate) - start)
    chunk_len = rng.randint(2, max_chunk)
    chunk = candidate[start : start + chunk_len]
    del candidate[start : start + chunk_len]
    dst = rng.randrange(0, len(candidate) + 1)
    return candidate[:dst] + chunk + candidate[dst:]


def _sort_front_window(order: Sequence[Instance], window: int) -> List[Instance]:
    candidate = list(order)
    prefix = sorted(candidate[:window], key=_volume_priority, reverse=True)
    return prefix + candidate[window:]


def _build_candidate_orders(
    order: Sequence[Instance],
    current_solution: Solution,
    rng: random.Random,
) -> List[List[Instance]]:
    candidates: List[List[Instance]] = []
    unplaced_skus = [
        item.sku_id for item in current_solution.unplaced if item.quantity_unplaced
    ]
    if unplaced_skus:
        promoted = _promote_matching_sku(order, unplaced_skus, rng)
        if promoted is not None:
            candidates.append(promoted)
        block_promoted = _promote_matching_block(order, unplaced_skus, rng)
        if block_promoted is not None:
            candidates.append(block_promoted)

    constrained_skus = [
        box.sku_id
        for box, _ in order
        if box.strict_upright or box.fragile or not box.stackable
    ]
    if constrained_skus:
        constrained = _promote_matching_block(order, constrained_skus, rng)
        if constrained is not None:
            candidates.append(constrained)

    candidates.append(_mutate_order(order, current_solution, rng))
    candidates.append(_move_chunk(order, rng))
    candidates.append(_reverse_window(order, rng))
    candidates.append(_sort_front_window(order, min(12, len(order))))
    if len(order) >= 24:
        candidates.append(_sort_front_window(order, min(24, len(order))))
    return candidates


def _mutate_order(
    order: Sequence[Instance],
    current_solution: Solution,
    rng: random.Random,
) -> List[Instance]:
    candidate = list(order)
    if len(candidate) < 2:
        return candidate

    unplaced_skus = [
        item.sku_id for item in current_solution.unplaced if item.quantity_unplaced
    ]
    if unplaced_skus and rng.random() < 0.35:
        promoted = _promote_matching_sku(candidate, unplaced_skus, rng)
        if promoted is not None:
            return promoted

    constrained_skus = [
        box.sku_id
        for box, _ in candidate
        if box.strict_upright or box.fragile or not box.stackable
    ]
    if constrained_skus and rng.random() < 0.2:
        promoted = _promote_matching_sku(candidate, constrained_skus, rng)
        if promoted is not None:
            return promoted

    move_type = rng.random()
    if move_type < 0.4:
        i, j = sorted(rng.sample(range(len(candidate)), 2))
        candidate[i], candidate[j] = candidate[j], candidate[i]
        return candidate

    if move_type < 0.8:
        i, j = rng.sample(range(len(candidate)), 2)
        item = candidate.pop(i)
        candidate.insert(j, item)
        return candidate

    return _reverse_window(candidate, rng)


def _evaluate_order(
    task_id: str,
    pallet: Pallet,
    order: Sequence[Instance],
    order_name: str,
    score_solution: ScoreFn,
    cache: Dict[OrderSignature, Tuple[Solution, float]],
) -> Tuple[Solution, float]:
    signature = _order_signature(order)
    cached = cache.get(signature)
    if cached is not None:
        return cached
    solution = pack_in_order(task_id, pallet, order, order_name=order_name)
    scored = (solution, score_solution(solution))
    cache[signature] = scored
    return scored


def _repair_best_order(
    order: Sequence[Instance],
    solution: Solution,
    rng: random.Random,
) -> List[List[Instance]]:
    candidates: List[List[Instance]] = []
    unplaced_skus = [
        item.sku_id for item in solution.unplaced if item.quantity_unplaced
    ]
    if unplaced_skus:
        block_promoted = _promote_matching_block(order, unplaced_skus, rng)
        if block_promoted is not None:
            candidates.append(block_promoted)
        single_promoted = _promote_matching_sku(order, unplaced_skus, rng)
        if single_promoted is not None:
            candidates.append(single_promoted)
    candidates.append(_sort_front_window(order, min(16, len(order))))
    candidates.append(_move_chunk(order, rng))
    return candidates


def _dedupe_elite_pool(
    entries: Sequence[EliteEntry], beam_width: int
) -> List[EliteEntry]:
    by_signature: Dict[OrderSignature, EliteEntry] = {}
    for score, order, solution, label in entries:
        signature = _order_signature(order)
        current = by_signature.get(signature)
        if current is None or score > current[0]:
            by_signature[signature] = (score, list(order), solution, label)
    ranked = sorted(by_signature.values(), key=lambda item: item[0], reverse=True)
    return ranked[:beam_width]


def _beam_refine(
    task_id: str,
    pallet: Pallet,
    elite_pool: Sequence[EliteEntry],
    score_solution: ScoreFn,
    rng: random.Random,
    budget_ms: int,
    cache: Dict[OrderSignature, Tuple[Solution, float]],
    beam_width: int = 4,
    rounds: int = 6,
) -> List[EliteEntry]:
    beam_t0 = time.perf_counter()
    pool = _dedupe_elite_pool(elite_pool, beam_width)
    if not pool:
        return []

    for round_idx in range(rounds):
        elapsed_ms = (time.perf_counter() - beam_t0) * 1000
        if elapsed_ms >= budget_ms:
            break

        expanded: List[EliteEntry] = list(pool)
        for score, order, solution, label in pool:
            for candidate_order in _build_candidate_orders(order, solution, rng):
                candidate_solution, candidate_score = _evaluate_order(
                    task_id,
                    pallet,
                    candidate_order,
                    order_name=f"beam-{round_idx}",
                    score_solution=score_solution,
                    cache=cache,
                )
                expanded.append(
                    (candidate_score, list(candidate_order), candidate_solution, label)
                )

            for repair_order in _repair_best_order(order, solution, rng):
                repair_solution, repair_score = _evaluate_order(
                    task_id,
                    pallet,
                    repair_order,
                    order_name=f"beam-repair-{round_idx}",
                    score_solution=score_solution,
                    cache=cache,
                )
                expanded.append(
                    (repair_score, list(repair_order), repair_solution, label)
                )

        next_pool = _dedupe_elite_pool(expanded, beam_width)
        if next_pool[0][0] <= pool[0][0]:
            pool = next_pool
            continue
        pool = next_pool

    return pool


def _run_annealing_chain(
    task_id: str,
    pallet: Pallet,
    start_order: Sequence[Instance],
    start_solution: Solution,
    start_score: float,
    score_solution: ScoreFn,
    rng: random.Random,
    chain_budget_ms: int,
    initial_temperature: float,
    cooling_rate: float,
    max_plateau_iterations: int,
    cache: Dict[OrderSignature, Tuple[Solution, float]],
) -> Tuple[Solution, float, List[Instance], int, int]:
    chain_t0 = time.perf_counter()
    current_order = list(start_order)
    current_solution = start_solution
    current_score = start_score
    best_order = list(start_order)
    best_solution = start_solution
    best_score = start_score
    temperature = initial_temperature
    iterations = 0
    accepted_moves = 0
    stagnation = 0

    while True:
        elapsed_ms = (time.perf_counter() - chain_t0) * 1000
        if elapsed_ms >= chain_budget_ms:
            break
        if stagnation >= max_plateau_iterations:
            break

        candidate_bundle = []
        seen_signatures = {_order_signature(current_order)}
        for candidate_order in _build_candidate_orders(
            current_order, current_solution, rng
        ):
            signature = _order_signature(candidate_order)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            candidate_solution, candidate_score = _evaluate_order(
                task_id,
                pallet,
                candidate_order,
                order_name="annealing-candidate",
                score_solution=score_solution,
                cache=cache,
            )
            candidate_bundle.append(
                (candidate_score, candidate_order, candidate_solution)
            )

        if not candidate_bundle:
            stagnation += 1
            continue

        candidate_score, candidate_order, candidate_solution = max(
            candidate_bundle, key=lambda item: item[0]
        )
        delta = candidate_score - current_score
        accept = False
        if delta >= 0:
            accept = True
        elif temperature > 1e-6:
            accept = rng.random() < math.exp(delta / max(temperature, 1e-6))

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

    return best_solution, best_score, best_order, iterations, accepted_moves


def anneal_pack(
    task_id: str,
    pallet: Pallet,
    boxes: Sequence[Box],
    score_solution: ScoreFn,
    time_budget_ms: int,
    initial_temperature: float = 0.25,
    cooling_rate: float = 0.996,
    max_plateau_iterations: int = 600,
    rng_seed: int = 0,
) -> AnnealingResult:
    """Search for a better box order using simulated annealing."""
    t0 = time.perf_counter()
    rng = random.Random(rng_seed)
    cache: Dict[OrderSignature, Tuple[Solution, float]] = {}

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

    evaluated_seeds: List[Tuple[float, str, List[Instance], Solution]] = []

    for seed_name, seed_order in seeds:
        seed_solution, seed_score = _evaluate_order(
            task_id,
            pallet,
            seed_order,
            order_name=seed_name,
            score_solution=score_solution,
            cache=cache,
        )
        logger.info(
            "[anneal] seed=%s score=%.4f placed=%d",
            seed_name,
            seed_score,
            len(seed_solution.placements),
        )
        evaluated_seeds.append((seed_score, seed_name, list(seed_order), seed_solution))

    evaluated_seeds.sort(key=lambda item: item[0], reverse=True)
    best_score, best_seed, best_order, best_solution = evaluated_seeds[0]

    elite_count = min(3, len(evaluated_seeds))
    chain_starts = evaluated_seeds[:elite_count]
    elite_pool: List[EliteEntry] = [
        (score, list(order), solution, seed_name)
        for score, seed_name, order, solution in chain_starts
    ]
    iterations = 0
    accepted_moves = 0

    for index, (seed_score, seed_name, seed_order, seed_solution) in enumerate(
        chain_starts
    ):
        elapsed_ms = (time.perf_counter() - t0) * 1000
        remaining_ms = max(0, time_budget_ms - int(elapsed_ms))
        if remaining_ms <= 0:
            break
        chains_left = len(chain_starts) - index
        chain_budget_ms = max(300, remaining_ms // max(1, chains_left))
        chain_solution, chain_score, chain_order, chain_iterations, chain_accepted = (
            _run_annealing_chain(
                task_id=task_id,
                pallet=pallet,
                start_order=seed_order,
                start_solution=seed_solution,
                start_score=seed_score,
                score_solution=score_solution,
                rng=rng,
                chain_budget_ms=chain_budget_ms,
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                max_plateau_iterations=max_plateau_iterations,
                cache=cache,
            )
        )
        iterations += chain_iterations
        accepted_moves += chain_accepted
        logger.info(
            "[anneal] chain seed=%s score=%.4f iterations=%d accepted=%d",
            seed_name,
            chain_score,
            chain_iterations,
            chain_accepted,
        )
        elite_pool.append((chain_score, list(chain_order), chain_solution, seed_name))
        if chain_score > best_score:
            best_score = chain_score
            best_solution = chain_solution
            best_order = chain_order
            best_seed = seed_name

    # Intensify around the best order with several reheated restarts.
    for reheat_index in range(3):
        elapsed_ms = (time.perf_counter() - t0) * 1000
        remaining_ms = max(0, time_budget_ms - int(elapsed_ms))
        if remaining_ms < 300:
            break
        reheated_order = _mutate_order(best_order, best_solution, rng)
        reheated_solution, reheated_score = _evaluate_order(
            task_id,
            pallet,
            reheated_order,
            order_name=f"annealing-reheat-{reheat_index}",
            score_solution=score_solution,
            cache=cache,
        )
        chain_solution, chain_score, chain_order, chain_iterations, chain_accepted = (
            _run_annealing_chain(
                task_id=task_id,
                pallet=pallet,
                start_order=reheated_order,
                start_solution=reheated_solution,
                start_score=reheated_score,
                score_solution=score_solution,
                rng=rng,
                chain_budget_ms=remaining_ms,
                initial_temperature=initial_temperature * 0.8,
                cooling_rate=cooling_rate,
                max_plateau_iterations=max_plateau_iterations // 2,
                cache=cache,
            )
        )
        iterations += chain_iterations
        accepted_moves += chain_accepted
        elite_pool.append(
            (chain_score, list(chain_order), chain_solution, f"reheat-{reheat_index}")
        )
        if chain_score > best_score:
            best_score = chain_score
            best_solution = chain_solution
            best_order = chain_order

    elapsed_ms = (time.perf_counter() - t0) * 1000
    remaining_ms = max(0, time_budget_ms - int(elapsed_ms))
    if remaining_ms >= 250:
        elite_pool = _beam_refine(
            task_id=task_id,
            pallet=pallet,
            elite_pool=elite_pool,
            score_solution=score_solution,
            rng=rng,
            budget_ms=remaining_ms,
            cache=cache,
        )
        if elite_pool and elite_pool[0][0] > best_score:
            best_score, best_order, best_solution, best_seed = elite_pool[0]

    repair_seen = {_order_signature(best_order)}
    for repair_order in _repair_best_order(best_order, best_solution, rng):
        signature = _order_signature(repair_order)
        if signature in repair_seen:
            continue
        repair_seen.add(signature)
        repair_solution, repair_score = _evaluate_order(
            task_id,
            pallet,
            repair_order,
            order_name="annealing-repair",
            score_solution=score_solution,
            cache=cache,
        )
        if repair_score > best_score:
            best_score = repair_score
            best_solution = repair_solution
            best_order = repair_order

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
