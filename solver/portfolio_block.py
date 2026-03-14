from __future__ import annotations

import json
import itertools
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .block_features import BlockFeatureExtractor, BlockFeatureView
from .block_ranker import BlockRanker, normalize_scores
from .hybrid.candidate_gen import CandidateGenerator, RemainingItem
from .hybrid.constants import EPSILON, FRAGILE_WEIGHT_THRESHOLD, UNPLACED_REASONS
from .hybrid.feasibility import FeasibilityChecker
from .hybrid.free_space import ExtremePointManager
from .hybrid.geometry import AABB
from .hybrid.pallet_state import PalletState, PlacedBox
from .hybrid.postprocess import postprocess
from .hybrid.rotations import get_orientations
from .models import Box, Pallet
from .packer import order_boxes, pack_greedy, pack_instance_sequence, pack_ordered_boxes
from .scenario_selector import (
    SEED_FAMILIES,
    ScenarioFingerprint,
    ScenarioSelector,
    compute_request_fingerprint,
)


logger = logging.getLogger(__name__)
_SELECTOR_CACHE: Dict[str, Optional[ScenarioSelector]] = {}
_RANKER_CACHE: Dict[str, Tuple[Optional[BlockRanker], Optional[BlockFeatureExtractor]]] = {}

SOLVER_VERSION = "portfolio-block-v0.2"
CONSTRUCTIVE_POLICIES = ("foundation", "fragile_last", "coverage_fill")
LEGACY_PORTFOLIO_SORTS = (
    "constrained_first",
    "base_area_desc",
    "volume_desc",
    "density_desc",
)
GREEDY_SEED_TO_SORT = {
    "heavy_base": "constrained_first",
    "liquid_fill": "base_area_desc",
    "mixed_volume": "volume_desc",
    "fragile_density": "density_desc",
    "coverage_tie": "coverage_tie",
}
GREEDY_SEED_FAMILIES = tuple(name for name in SEED_FAMILIES if name != "block_structured")


@dataclass(frozen=True)
class BlockCandidate:
    sku_id: str
    rotation_code: str
    placed_dims: Tuple[int, int, int]
    nx: int
    ny: int
    nz: int
    item_count: int
    total_weight: float
    fragile: bool
    stackable: bool
    strict_upright: bool
    aabb: AABB
    units: Tuple[PlacedBox, ...]
    support_ratio: float
    wall_count: int
    corner_touch: bool
    heuristic_score: float

    def feature_view(self, state: PalletState) -> BlockFeatureView:
        unit_volume = (
            self.placed_dims[0] * self.placed_dims[1] * self.placed_dims[2]
        )
        return BlockFeatureView(
            item_count=self.item_count,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            block_length=self.aabb.length_x(),
            block_width=self.aabb.width_y(),
            block_height=self.aabb.height_z(),
            block_weight=self.total_weight,
            unit_weight=self.total_weight / max(self.item_count, 1),
            unit_volume=unit_volume,
            fragile=self.fragile,
            strict_upright=self.strict_upright,
            support_ratio=self.support_ratio,
            wall_count=self.wall_count,
            corner_touch=self.corner_touch,
            z_min=self.aabb.z_min,
            residual_x=state.length - self.aabb.x_max,
            residual_y=state.width - self.aabb.y_max,
            remaining_height=state.max_height - self.aabb.z_max,
            heuristic_score=self.heuristic_score,
        )


@dataclass(frozen=True)
class BlockStep:
    candidate: BlockCandidate
    marginal_gain: float
    frontier_exposure: float


@dataclass
class PolicyRun:
    name: str
    placements: List[PlacedBox]
    remaining: List[RemainingItem]
    block_steps: List[BlockStep]
    score: float
    elapsed_ms: int
    used_model: bool = False
    seed_family: str = ""
    ordered_skus: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FragilityConflict:
    top_signature: Tuple[str, int, int, int, int]
    bottom_signature: Tuple[str, int, int, int, int]
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    z_max: int
    overlap_area: float


@dataclass(frozen=True)
class _BlockSpec:
    sku_id: str
    placed_dims: Tuple[int, int, int]
    rotation_code: str
    nx: int
    ny: int
    nz: int
    item_count: int
    total_weight: float
    fragile: bool
    stackable: bool
    strict_upright: bool
    x: int
    y: int
    z: int
    heuristic_score: float


def solve_request(
    request: Dict[str, Any],
    model_dir: str | Path = "models",
    time_budget_ms: int = 900,
) -> Dict[str, Any]:
    checker, candidate_gen, total_items = _init_common_components(request)
    fingerprint = compute_request_fingerprint(request)
    selector = _load_optional_selector(model_dir) if fingerprint.total_items <= 120 else None
    ranker, extractor = (
        _load_optional_ranker(model_dir)
        if fingerprint.total_items <= 120
        else (None, None)
    )
    t0 = time.perf_counter()
    proxy_upper_bound = _proxy_upper_bound(request)
    repeat_heavy = fingerprint.total_items >= 60 and fingerprint.sku_count <= 4
    prefer_safe_fragility = _should_use_conservative_fragility_selection(fingerprint)

    seed_limit_ms = min(220, max(120, time_budget_ms // 4))
    order_budget_ms = max(80, min(260, time_budget_ms // 4))
    repair_budget_ms = max(80, min(160, time_budget_ms // 6))

    ranked_families = _rank_seed_families(fingerprint, selector)
    runs: List[PolicyRun] = []
    seed_eval_count = 2 if repeat_heavy else 3

    for seed_family in ranked_families[:seed_eval_count]:
        elapsed_so_far = int((time.perf_counter() - t0) * 1000)
        budget_left = max(0, time_budget_ms - elapsed_so_far - repair_budget_ms)
        if budget_left <= 20:
            break
        run = _run_seed_family(
            request=request,
            seed_family=seed_family,
            checker=checker,
            candidate_gen=candidate_gen,
            ranker=ranker,
            extractor=extractor,
            budget_ms=min(seed_limit_ms, budget_left),
        )
        runs.append(run)
        if _is_near_proxy_upper_bound(run.score, proxy_upper_bound):
            break

    if not runs:
        runs.append(
            _run_legacy_portfolio(
                request,
                checker=checker,
                candidate_gen=candidate_gen,
                budget_ms=min(time_budget_ms, 250),
            )
        )

    enable_local_order = fingerprint.total_items <= 120 and fingerprint.sku_count <= 8
    best_greedy = _best_greedy_run(runs)
    if best_greedy is not None and enable_local_order:
        elapsed_so_far = int((time.perf_counter() - t0) * 1000)
        budget_left = max(0, time_budget_ms - elapsed_so_far - repair_budget_ms)
        if budget_left > 25:
            order_search_base = best_greedy
            improved = _local_order_search(
                request=request,
                run=best_greedy,
                checker=checker,
                candidate_gen=candidate_gen,
                budget_ms=min(order_budget_ms, budget_left),
            )
            if _run_sort_key(improved) > _run_sort_key(best_greedy):
                runs.append(improved)
                order_search_base = improved

            elapsed_so_far = int((time.perf_counter() - t0) * 1000)
            strict_budget_left = max(0, time_budget_ms - elapsed_so_far - repair_budget_ms)
            if (
                strict_budget_left > 25
                and _should_try_strict_fragility_search(
                    request=request,
                    fingerprint=fingerprint,
                    run=order_search_base,
                )
            ):
                strict_improved = _local_order_search(
                    request=request,
                    run=order_search_base,
                    checker=checker,
                    candidate_gen=candidate_gen,
                    budget_ms=min(max(80, order_budget_ms), strict_budget_left),
                    strict_fragility=True,
                )
                if _run_sort_key(strict_improved) > _run_sort_key(order_search_base):
                    runs.append(strict_improved)

    enable_repair = fingerprint.total_items <= 120 and not repeat_heavy
    best_two = sorted(
        runs,
        key=lambda candidate: _top_level_run_sort_key(
            candidate,
            request,
            prefer_safe_fragility=prefer_safe_fragility,
        ),
        reverse=True,
    )[:2]
    if best_two and enable_repair:
        per_run_repair_ms = max(35, repair_budget_ms // len(best_two))
        for candidate in best_two:
            elapsed_so_far = int((time.perf_counter() - t0) * 1000)
            budget_left = max(0, time_budget_ms - elapsed_so_far)
            if budget_left <= 20:
                break
            repaired = _repair_candidate_run(
                request=request,
                run=candidate,
                checker=checker,
                candidate_gen=candidate_gen,
                ranker=ranker,
                extractor=extractor,
                budget_ms=min(per_run_repair_ms, budget_left),
            )
            if _run_sort_key(repaired) > _run_sort_key(candidate):
                runs.append(repaired)

    best = max(
        runs,
        key=lambda candidate: _top_level_run_sort_key(
            candidate,
            request,
            prefer_safe_fragility=prefer_safe_fragility,
        ),
    )
    placements, leftover, final_state = _finalize_run(
        request=request,
        run=best,
        checker=checker,
        candidate_gen=candidate_gen,
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return _format_output(
        request["task_id"],
        placements,
        leftover,
        request["boxes"],
        final_state,
        elapsed_ms,
    )


def solve_legacy_greedy_request(
    request: Dict[str, Any],
    time_budget_ms: int = 900,
) -> Dict[str, Any]:
    checker, candidate_gen, _ = _init_common_components(request)
    run = _run_legacy_portfolio(
        request,
        checker=checker,
        candidate_gen=candidate_gen,
        budget_ms=min(time_budget_ms, 250),
    )
    placements, leftover, state = _finalize_run(
        request=request,
        run=run,
        checker=checker,
        candidate_gen=candidate_gen,
    )
    return _format_output(
        request["task_id"],
        placements,
        leftover,
        request["boxes"],
        state,
        run.elapsed_ms,
    )


def collect_ranker_rows(
    request: Dict[str, Any],
    top_k: int = 16,
    finish_budget_ms: int = 80,
) -> List[dict]:
    checker, _, _ = _init_common_components(request)
    state, ep_manager, remaining = _fresh_search_state(request)
    rows: List[dict] = []
    group_id = 0
    policies = ("foundation", "fragile_last", "coverage_fill")
    extractor = BlockFeatureExtractor()
    deadline = time.perf_counter() + 1.5

    while time.perf_counter() < deadline:
        if sum(item.remaining_qty for item in remaining) == 0:
            break
        policy_name = policies[group_id % len(policies)]
        candidates = _generate_block_candidates(
            state=state,
            ep_manager=ep_manager,
            remaining=remaining,
            checker=checker,
            policy_name=policy_name,
            max_block_span=4,
            max_ep=24,
            max_candidates=64,
        )
        if not candidates:
            break

        candidates.sort(key=lambda candidate: candidate.heuristic_score, reverse=True)
        shortlisted = candidates[:top_k]
        views = [candidate.feature_view(state) for candidate in shortlisted]
        X = extractor.extract_batch(state, views, remaining, policy_name)

        for idx, candidate in enumerate(shortlisted):
            forced_state = state.copy()
            forced_ep = ep_manager.copy()
            forced_remaining = _clone_remaining(remaining)
            _apply_block_candidate(candidate, forced_state, forced_ep)
            forced_remaining = _decrement_remaining(
                forced_remaining, candidate.sku_id, candidate.item_count
            )
            best_completion = _best_cheap_completion(
                request=request,
                state=forced_state,
                ep_manager=forced_ep,
                remaining=forced_remaining,
                checker=checker,
                budget_ms=finish_budget_ms,
            )
            rows.append(
                {
                    "group_id": group_id,
                    "policy_name": policy_name,
                    "features": X[idx].tolist(),
                    "target": best_completion.score,
                }
            )

        best_candidate = shortlisted[0]
        _apply_block_candidate(best_candidate, state, ep_manager)
        remaining = _decrement_remaining(
            remaining, best_candidate.sku_id, best_candidate.item_count
        )
        group_id += 1

    return rows


def _init_common_components(
    request: Dict[str, Any]
) -> Tuple[FeasibilityChecker, CandidateGenerator, int]:
    pallet = request["pallet"]
    checker = FeasibilityChecker(
        pallet["length_mm"],
        pallet["width_mm"],
        pallet["max_height_mm"],
        pallet["max_weight_kg"],
    )
    candidate_gen = CandidateGenerator(checker, max_candidates=200)
    total_items = sum(box["quantity"] for box in request["boxes"])
    return checker, candidate_gen, total_items


def _fresh_search_state(
    request: Dict[str, Any]
) -> Tuple[PalletState, ExtremePointManager, List[RemainingItem]]:
    pallet = request["pallet"]
    state = PalletState(
        pallet["length_mm"],
        pallet["width_mm"],
        pallet["max_height_mm"],
        pallet["max_weight_kg"],
    )
    ep_manager = ExtremePointManager(pallet["length_mm"], pallet["width_mm"])
    remaining = [RemainingItem(box) for box in request["boxes"]]
    return state, ep_manager, remaining


def _clone_remaining(remaining: List[RemainingItem]) -> List[RemainingItem]:
    cloned: List[RemainingItem] = []
    for item in remaining:
        new_item = RemainingItem.__new__(RemainingItem)
        new_item.sku_id = item.sku_id
        new_item.length = item.length
        new_item.width = item.width
        new_item.height = item.height
        new_item.weight = item.weight
        new_item.strict_upright = item.strict_upright
        new_item.fragile = item.fragile
        new_item.stackable = item.stackable
        new_item.remaining_qty = item.remaining_qty
        cloned.append(new_item)
    return cloned


def _decrement_remaining(
    remaining: List[RemainingItem],
    sku_id: str,
    count: int,
) -> List[RemainingItem]:
    updated = _clone_remaining(remaining)
    for item in updated:
        if item.sku_id == sku_id:
            item.remaining_qty = max(0, item.remaining_qty - count)
            break
    return updated


def _remaining_from_placements(
    request_boxes: Sequence[Dict[str, Any]],
    placements: Sequence[PlacedBox],
) -> List[RemainingItem]:
    counts: Dict[str, int] = {}
    for placement in placements:
        counts[placement.sku_id] = counts.get(placement.sku_id, 0) + 1
    remaining: List[RemainingItem] = []
    for box in request_boxes:
        remaining.append(
            RemainingItem(box, placed_count=counts.get(box["sku_id"], 0))
        )
    return remaining


def _load_optional_ranker(
    model_dir: str | Path,
) -> Tuple[Optional[BlockRanker], Optional[BlockFeatureExtractor]]:
    cache_key = str(Path(model_dir).resolve())
    if cache_key in _RANKER_CACHE:
        return _RANKER_CACHE[cache_key]
    meta = _read_json(Path(model_dir) / "meta.json")
    if meta and not meta.get("runtime_enabled", False):
        _RANKER_CACHE[cache_key] = (None, None)
        return _RANKER_CACHE[cache_key]
    ranker = BlockRanker(model_dir=model_dir)
    if ranker.load():
        _RANKER_CACHE[cache_key] = (ranker, BlockFeatureExtractor())
        return _RANKER_CACHE[cache_key]
    _RANKER_CACHE[cache_key] = (None, None)
    return _RANKER_CACHE[cache_key]


def _load_optional_selector(model_dir: str | Path) -> Optional[ScenarioSelector]:
    cache_key = str(Path(model_dir).resolve())
    if cache_key in _SELECTOR_CACHE:
        return _SELECTOR_CACHE[cache_key]
    selector = ScenarioSelector(model_dir=model_dir)
    if selector.load():
        _SELECTOR_CACHE[cache_key] = selector
        return selector
    _SELECTOR_CACHE[cache_key] = None
    return None


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def _rank_seed_families(
    fingerprint: ScenarioFingerprint,
    selector: Optional[ScenarioSelector],
) -> List[str]:
    fallback = _fallback_seed_order(fingerprint)
    if fingerprint.total_items > 120:
        return fallback
    if selector is None or not selector.is_trained:
        return fallback

    try:
        learned = selector.rank_families(fingerprint)
    except Exception:
        return fallback

    return _blend_seed_orders(learned, fallback, fingerprint)


def _fallback_seed_order(fingerprint: ScenarioFingerprint) -> List[str]:
    primary: str
    if fingerprint.weight_ratio >= 0.90 and fingerprint.upright_ratio >= 0.50:
        primary = "heavy_base"
    elif fingerprint.fragile_ratio >= 0.30 and fingerprint.total_items < 60:
        primary = "fragile_density"
    elif fingerprint.volume_ratio < 0.60:
        primary = "liquid_fill"
    else:
        primary = "mixed_volume"

    order: List[str] = [primary]
    for seed_family in ("heavy_base", "liquid_fill", "mixed_volume", "fragile_density"):
        if seed_family not in order:
            order.append(seed_family)

    if (
        fingerprint.volume_ratio >= 0.95
        and fingerprint.total_items <= 12
        and "coverage_tie" not in order[:2]
    ):
        order.insert(1, "coverage_tie")
    else:
        order.append("coverage_tie")

    allow_block_top3 = (
        fingerprint.total_items <= 60
        and (
            fingerprint.sku_count <= 4
            or fingerprint.max_sku_share >= 0.45
        )
    )
    if allow_block_top3:
        insert_at = 2 if len(order) >= 2 else len(order)
        if "block_structured" in order:
            order.remove("block_structured")
        order.insert(insert_at, "block_structured")
    else:
        order = [seed_family for seed_family in order if seed_family != "block_structured"]
        order.append("block_structured")

    deduped: List[str] = []
    for seed_family in order:
        if seed_family not in deduped:
            deduped.append(seed_family)
    return deduped


def _blend_seed_orders(
    learned_order: Sequence[str],
    fallback_order: Sequence[str],
    fingerprint: ScenarioFingerprint,
) -> List[str]:
    allow_block_top3 = (
        fingerprint.total_items <= 60
        and (
            fingerprint.sku_count <= 4
            or fingerprint.max_sku_share >= 0.45
        )
    )
    scores: Dict[str, float] = {}
    total = float(len(SEED_FAMILIES))
    for idx, seed_family in enumerate(fallback_order):
        scores[seed_family] = scores.get(seed_family, 0.0) + (total - idx) * 1.0
    for idx, seed_family in enumerate(learned_order):
        scores[seed_family] = scores.get(seed_family, 0.0) + (total - idx) * 0.7

    if not allow_block_top3:
        scores["block_structured"] = scores.get("block_structured", 0.0) - total
    coverage_tie_supported = (
        fingerprint.volume_ratio >= 0.95
        and fingerprint.total_items <= 12
    )
    if coverage_tie_supported:
        scores["coverage_tie"] = scores.get("coverage_tie", 0.0) + 1.5
    else:
        scores["coverage_tie"] = scores.get("coverage_tie", 0.0) - total

    return sorted(SEED_FAMILIES, key=lambda seed_family: scores.get(seed_family, 0.0), reverse=True)


def _proxy_upper_bound(request: Dict[str, Any]) -> float:
    pallet = request["pallet"]
    total_volume = sum(
        box["length_mm"] * box["width_mm"] * box["height_mm"] * box["quantity"]
        for box in request["boxes"]
    )
    pallet_volume = (
        pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    )
    vol_cap = min(1.0, total_volume / max(pallet_volume, 1))
    return 0.50 * vol_cap + 0.30 + 0.10


def _is_near_proxy_upper_bound(score: float, upper_bound: float) -> bool:
    return score >= upper_bound - 0.01


def _run_seed_family(
    request: Dict[str, Any],
    seed_family: str,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    ranker: Optional[BlockRanker],
    extractor: Optional[BlockFeatureExtractor],
    budget_ms: int,
) -> PolicyRun:
    if seed_family == "block_structured":
        return _run_block_structured_seed(
            request=request,
            checker=checker,
            ranker=ranker,
            extractor=extractor,
            budget_ms=budget_ms,
        )
    return _run_greedy_seed_family(
        request=request,
        seed_family=seed_family,
        checker=checker,
        candidate_gen=candidate_gen,
        budget_ms=budget_ms,
    )


def _run_block_structured_seed(
    request: Dict[str, Any],
    checker: FeasibilityChecker,
    ranker: Optional[BlockRanker],
    extractor: Optional[BlockFeatureExtractor],
    budget_ms: int,
) -> PolicyRun:
    if budget_ms <= 0:
        state, _, remaining = _fresh_search_state(request)
        return PolicyRun(
            name="block_structured",
            placements=[],
            remaining=remaining,
            block_steps=[],
            score=_solution_proxy(state, sum(box["quantity"] for box in request["boxes"])),
            elapsed_ms=0,
            seed_family="block_structured",
        )

    budget = {
        "foundation": max(40, int(budget_ms * 0.38)),
        "fragile_last": max(35, int(budget_ms * 0.31)),
        "coverage_fill": max(35, budget_ms - int(budget_ms * 0.69)),
    }
    runs: List[PolicyRun] = []
    for policy_name in CONSTRUCTIVE_POLICIES:
        state, ep_manager, remaining = _fresh_search_state(request)
        deadline = time.perf_counter() + budget[policy_name] / 1000.0
        run = _construct_with_policy(
            request=request,
            policy_name=policy_name,
            state=state,
            ep_manager=ep_manager,
            remaining=remaining,
            checker=checker,
            deadline=deadline,
            ranker=ranker,
            extractor=extractor,
            max_block_span=4,
        )
        run.name = f"block_structured:{policy_name}"
        run.seed_family = "block_structured"
        runs.append(run)
    return _select_best_run(runs)


def _run_greedy_seed_family(
    request: Dict[str, Any],
    seed_family: str,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    budget_ms: int,
    ordered_boxes: Optional[List[Box]] = None,
    apply_postprocess: bool = True,
    strict_fragility: bool = False,
) -> PolicyRun:
    t0 = time.perf_counter()
    task_id, pallet, boxes = _request_to_models(request)
    total_items = sum(box.quantity for box in boxes)
    allow_staged_variant = ordered_boxes is None
    if ordered_boxes is None:
        ordered_boxes = _ordered_boxes_for_seed(boxes, seed_family)
    run_name = f"{seed_family}:strict" if strict_fragility else seed_family
    label = f"{seed_family}:strict" if strict_fragility else seed_family
    base_run = _policy_run_from_solution(
        request=request,
        solution=pack_ordered_boxes(
            task_id,
            pallet,
            ordered_boxes,
            label=label,
            strict_fragility=strict_fragility,
        ),
        total_items=total_items,
        name=run_name,
        seed_family=seed_family,
        ordered_skus=tuple(box.sku_id for box in ordered_boxes),
    )
    candidates = [base_run]
    if allow_staged_variant and seed_family == "fragile_density":
        staged_instances = _fragile_staged_instances(boxes, pallet)
        if staged_instances:
            candidates.append(
                _policy_run_from_solution(
                    request=request,
                    solution=pack_instance_sequence(
                        task_id,
                        pallet,
                        staged_instances,
                        label=f"{label}:staged",
                        strict_fragility=strict_fragility,
                    ),
                    total_items=total_items,
                    name=f"{run_name}:staged",
                    seed_family=seed_family,
                    ordered_skus=(),
                )
            )
    run = _select_best_run(candidates)
    if apply_postprocess:
        post_budget_ms = max(0, budget_ms - int((time.perf_counter() - t0) * 1000))
        run = _maybe_postprocess_run(
            request=request,
            run=run,
            checker=checker,
            candidate_gen=candidate_gen,
            total_items=total_items,
            time_budget_left_ms=post_budget_ms,
        )
        run.seed_family = seed_family
        if run.name == f"{run_name}:staged" or run.name.startswith(f"{run_name}:staged+"):
            run.ordered_skus = ()
        elif not run.ordered_skus:
            run.ordered_skus = tuple(box.sku_id for box in ordered_boxes)
    return run


def _policy_run_from_solution(
    request: Dict[str, Any],
    solution: Any,
    total_items: int,
    name: str,
    seed_family: str,
    ordered_skus: Tuple[str, ...],
) -> PolicyRun:
    placements = _placements_from_public_solution(
        request["boxes"], solution.placements
    )
    remaining = _remaining_from_placements(request["boxes"], placements)
    state, _ = _rebuild_state(request, placements)
    return PolicyRun(
        name=name,
        placements=placements,
        remaining=remaining,
        block_steps=[],
        score=_solution_proxy(state, total_items),
        elapsed_ms=solution.solve_time_ms,
        used_model=False,
        seed_family=seed_family,
        ordered_skus=ordered_skus,
    )


def _fragile_staged_instances(
    boxes: Sequence[Box],
    pallet: Pallet,
) -> List[Tuple[Box, int]]:
    sturdy = sorted(
        [box for box in boxes if not box.fragile],
        key=lambda box: (-box.base_area, -box.volume, -box.weight_kg),
    )
    fragile_heavy = sorted(
        [box for box in boxes if box.fragile and box.weight_kg > FRAGILE_WEIGHT_THRESHOLD],
        key=lambda box: (-box.base_area, -box.volume),
    )
    fragile_light = sorted(
        [box for box in boxes if box.fragile and box.weight_kg <= FRAGILE_WEIGHT_THRESHOLD],
        key=lambda box: (-box.base_area, -box.volume),
    )
    if not sturdy or not fragile_heavy or not fragile_light:
        return []

    pallet_area = pallet.length_mm * pallet.width_mm
    target_area = pallet_area * 0.90
    next_index: Dict[str, int] = {box.sku_id: 0 for box in boxes}
    instances: List[Tuple[Box, int]] = []
    area_acc = 0

    for box in sturdy:
        while next_index[box.sku_id] < box.quantity and area_acc < target_area:
            instances.append((box, next_index[box.sku_id]))
            next_index[box.sku_id] += 1
            area_acc += box.base_area

    for box in fragile_heavy:
        anchor_count = min(2, box.quantity - next_index[box.sku_id])
        for _ in range(anchor_count):
            instances.append((box, next_index[box.sku_id]))
            next_index[box.sku_id] += 1

    for group in (fragile_light, fragile_heavy, sturdy):
        for box in group:
            while next_index[box.sku_id] < box.quantity:
                instances.append((box, next_index[box.sku_id]))
                next_index[box.sku_id] += 1

    return instances


def _ordered_boxes_for_seed(boxes: Sequence[Box], seed_family: str) -> List[Box]:
    if seed_family not in GREEDY_SEED_TO_SORT:
        return list(boxes)
    return order_boxes(boxes, GREEDY_SEED_TO_SORT[seed_family])


def _best_greedy_run(runs: Sequence[PolicyRun]) -> Optional[PolicyRun]:
    greedy_runs = [run for run in runs if run.seed_family in GREEDY_SEED_FAMILIES]
    if not greedy_runs:
        return None
    return _select_best_run(greedy_runs)


def _local_order_search(
    request: Dict[str, Any],
    run: PolicyRun,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    budget_ms: int,
    strict_fragility: bool = False,
) -> PolicyRun:
    if budget_ms <= 0 or not run.ordered_skus:
        return run

    _, _, boxes = _request_to_models(request)
    by_sku = {box.sku_id: box for box in boxes}
    current_order = [by_sku[sku_id] for sku_id in run.ordered_skus if sku_id in by_sku]
    top_n = min(4, len(current_order))
    if top_n < 2:
        return run

    total_items = sum(box["quantity"] for box in request["boxes"])
    deadline = time.perf_counter() + budget_ms / 1000.0
    best_run = run
    best_raw_run = run
    neighbors: List[List[Box]]
    exact_search = len(current_order) <= 3
    if exact_search:
        neighbors = _generate_exact_orderings(current_order)
    else:
        neighbors = _generate_order_neighbors(current_order, top_n)
    raw_runs: List[PolicyRun] = []
    if strict_fragility:
        strict_base = _run_greedy_seed_family(
            request=request,
            seed_family=run.seed_family or "mixed_volume",
            checker=checker,
            candidate_gen=candidate_gen,
            budget_ms=0,
            ordered_boxes=current_order,
            apply_postprocess=False,
            strict_fragility=True,
        )
        raw_runs.append(strict_base)
        best_raw_run = strict_base
        best_run = strict_base
    for neighbor in neighbors:
        if time.perf_counter() >= deadline:
            break
        candidate = _run_greedy_seed_family(
            request=request,
            seed_family=run.seed_family or "mixed_volume",
            checker=checker,
            candidate_gen=candidate_gen,
            budget_ms=0,
            ordered_boxes=neighbor,
            apply_postprocess=False,
            strict_fragility=strict_fragility,
        )
        raw_runs.append(candidate)
        if _run_sort_key(candidate) > _run_sort_key(best_raw_run):
            best_raw_run = candidate

    if not raw_runs:
        return run
    if time.perf_counter() >= deadline:
        return best_raw_run

    raw_runs.sort(key=_run_sort_key, reverse=True)
    if exact_search:
        shortlist_k = len(raw_runs)
    else:
        shortlist_k = 1 if total_items >= 60 and len(current_order) <= 4 else 3
    best_run = run
    for candidate in raw_runs[:shortlist_k]:
        if time.perf_counter() >= deadline:
            break
        ordered_boxes = [by_sku[sku_id] for sku_id in candidate.ordered_skus if sku_id in by_sku]
        candidate = _run_greedy_seed_family(
            request=request,
            seed_family=run.seed_family or "mixed_volume",
            checker=checker,
            candidate_gen=candidate_gen,
            budget_ms=max(0, int((deadline - time.perf_counter()) * 1000)),
            ordered_boxes=ordered_boxes,
            apply_postprocess=True,
            strict_fragility=strict_fragility,
        )
        if _run_sort_key(candidate) > _run_sort_key(best_run):
            best_run = candidate
    if _run_sort_key(best_raw_run) > _run_sort_key(best_run):
        return best_raw_run
    return best_run


def _generate_exact_orderings(
    ordered_boxes: Sequence[Box],
) -> List[List[Box]]:
    seen = set()
    current = tuple(box.sku_id for box in ordered_boxes)
    neighbors: List[List[Box]] = []
    for permutation in itertools.permutations(ordered_boxes):
        key = tuple(box.sku_id for box in permutation)
        if key == current or key in seen:
            continue
        seen.add(key)
        neighbors.append(list(permutation))
    return neighbors


def _generate_order_neighbors(
    ordered_boxes: Sequence[Box],
    top_n: int,
) -> List[List[Box]]:
    neighbors: List[List[Box]] = []
    seen = set()
    prefix = list(ordered_boxes[:top_n])
    suffix = list(ordered_boxes[top_n:])

    for idx in range(1, top_n):
        swap_front = list(prefix)
        swap_front[0], swap_front[idx] = swap_front[idx], swap_front[0]
        key = tuple(box.sku_id for box in swap_front)
        if key not in seen:
            seen.add(key)
            neighbors.append(swap_front + suffix)

        move_front = list(prefix)
        box = move_front.pop(idx)
        move_front.insert(0, box)
        key = tuple(item.sku_id for item in move_front)
        if key not in seen:
            seen.add(key)
            neighbors.append(move_front + suffix)

        adjacent = list(prefix)
        adjacent[idx - 1], adjacent[idx] = adjacent[idx], adjacent[idx - 1]
        key = tuple(box.sku_id for box in adjacent)
        if key not in seen:
            seen.add(key)
            neighbors.append(adjacent + suffix)
    return neighbors


def _should_try_strict_fragility_search(
    request: Dict[str, Any],
    fingerprint: ScenarioFingerprint,
    run: PolicyRun,
) -> bool:
    if not run.ordered_skus:
        return False
    if (
        fingerprint.sku_count > 4
        or fingerprint.total_items > 70
        or fingerprint.volume_ratio > 1.05
    ):
        return False

    total_items = sum(box["quantity"] for box in request["boxes"])
    coverage = len(run.placements) / max(total_items, 1)
    if coverage < 0.85:
        return False

    has_heavy_fragile = any(
        box.get("fragile", False) and box["weight_kg"] > FRAGILE_WEIGHT_THRESHOLD
        for box in request["boxes"]
    )
    has_non_fragile = any(not box.get("fragile", False) for box in request["boxes"])
    return has_heavy_fragile and has_non_fragile


def _repair_candidate_run(
    request: Dict[str, Any],
    run: PolicyRun,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    ranker: Optional[BlockRanker],
    extractor: Optional[BlockFeatureExtractor],
    budget_ms: int,
) -> PolicyRun:
    if budget_ms <= 0 or not run.placements:
        return run

    base_fragility_score = _fragility_score_from_placements(run.placements)
    candidates = [run]
    if base_fragility_score < 0.999:
        candidates.append(
            _repair_fragility_micro_repack(
                request=request,
                run=run,
                checker=checker,
                candidate_gen=candidate_gen,
                budget_ms=max(25, budget_ms // 2),
            )
        )
    if run.block_steps:
        candidates.append(
            _repair_run(
                request=request,
                run=run,
                checker=checker,
                ranker=ranker,
                extractor=extractor,
                budget_ms=max(20, budget_ms // 2),
            )
        )
    candidates.append(
        _repair_remove_and_refill(
            request=request,
            run=run,
            checker=checker,
            candidate_gen=candidate_gen,
            budget_ms=budget_ms,
        )
    )
    best = max(
        candidates,
        key=lambda candidate: _repair_selection_key(candidate, base_fragility_score),
    )
    if (
        _repair_selection_key(best, base_fragility_score)
        <= _repair_selection_key(run, base_fragility_score)
    ):
        return run
    return best


def _repair_selection_key(
    run: PolicyRun,
    base_fragility_score: float,
) -> Tuple[float, int]:
    fragility_score = _fragility_score_from_placements(run.placements)
    fragility_bonus = 0.0
    if fragility_score > base_fragility_score + 1e-9:
        fragility_bonus = 0.30 * (fragility_score - base_fragility_score)
    return (run.score + fragility_bonus, -run.elapsed_ms)


def _repair_fragility_micro_repack(
    request: Dict[str, Any],
    run: PolicyRun,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    budget_ms: int,
) -> PolicyRun:
    if budget_ms <= 0 or len(run.placements) < 3 or len(run.placements) > 70:
        return run

    conflicts = _fragility_conflicts(run.placements)
    if not conflicts:
        return run

    clusters = _fragility_conflict_clusters(run.placements, conflicts)
    if not clusters:
        return run

    total_items = sum(box["quantity"] for box in request["boxes"])
    deadline = time.perf_counter() + budget_ms / 1000.0
    candidate_runs: List[PolicyRun] = [run]
    cluster_sets = [cluster["signatures"] for cluster in clusters[:2]]
    if len(cluster_sets) >= 2:
        combined = set().union(*cluster_sets)
        if combined:
            cluster_sets.append(combined)

    for signatures in cluster_sets:
        if time.perf_counter() >= deadline:
            break
        kept = [
            placement
            for placement in run.placements
            if _placement_signature(placement) not in signatures
        ]
        if len(kept) == len(run.placements):
            continue

        state, ep_manager = _rebuild_state(request, kept)
        remaining = _remaining_from_placements(request["boxes"], kept)
        runs = _local_repack_runs(
            request=request,
            state=state,
            ep_manager=ep_manager,
            remaining=remaining,
            checker=checker,
            candidate_gen=candidate_gen,
            total_items=total_items,
            deadline=deadline,
            base_name=f"{run.name}+micro_repack",
        )
        candidate_runs.extend(runs)

    best = max(
        candidate_runs,
        key=lambda candidate: _repair_selection_key(
            candidate,
            _fragility_score_from_placements(run.placements),
        ),
    )
    if (
        _repair_selection_key(best, _fragility_score_from_placements(run.placements))
        <= _repair_selection_key(run, _fragility_score_from_placements(run.placements))
    ):
        return run
    return best


def _local_repack_runs(
    request: Dict[str, Any],
    state: PalletState,
    ep_manager: ExtremePointManager,
    remaining: List[RemainingItem],
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    total_items: int,
    deadline: float,
    base_name: str,
) -> List[PolicyRun]:
    if time.perf_counter() >= deadline:
        return []

    runs: List[PolicyRun] = []
    rebuild_specs = (
        ("fragile_last", 2),
        ("coverage_fill", 1),
        ("foundation", 1),
    )
    for policy_name, max_block_span in rebuild_specs:
        if time.perf_counter() >= deadline:
            break
        run = _construct_with_policy(
            request=request,
            policy_name=policy_name,
            state=state.copy(),
            ep_manager=ep_manager.copy(),
            remaining=_clone_remaining(remaining),
            checker=checker,
            deadline=deadline,
            ranker=None,
            extractor=None,
            max_block_span=max_block_span,
        )
        run.name = f"{base_name}:{policy_name}"
        post_budget_ms = max(0, int((deadline - time.perf_counter()) * 1000))
        run = _maybe_postprocess_run(
            request=request,
            run=run,
            checker=checker,
            candidate_gen=candidate_gen,
            total_items=total_items,
            time_budget_left_ms=post_budget_ms,
        )
        runs.append(run)
    return runs


def _repair_remove_and_refill(
    request: Dict[str, Any],
    run: PolicyRun,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    budget_ms: int,
) -> PolicyRun:
    if budget_ms <= 0 or len(run.placements) < 3:
        return run

    total_items = sum(box["quantity"] for box in request["boxes"])
    offenders = _fragility_offender_signatures(run.placements)
    remove_count = max(1, math.ceil(len(run.placements) * 0.2))
    ranked = sorted(
        run.placements,
        key=lambda placement: _placement_keep_score(
            placement, request, total_items, offenders
        ),
    )
    removed = {
        _placement_signature(placement)
        for placement in ranked[:remove_count]
    }
    kept = [
        placement
        for placement in run.placements
        if _placement_signature(placement) not in removed
    ]
    state, _ = _rebuild_state(request, kept)
    leftover = _remaining_from_placements(request["boxes"], kept)
    placements, leftover = postprocess(
        _reindex_placements(kept),
        leftover,
        state,
        checker,
        ExtremePointManager(request["pallet"]["length_mm"], request["pallet"]["width_mm"]),
        candidate_gen,
    )
    repaired_state, _ = _rebuild_state(request, placements)
    repaired = PolicyRun(
        name=f"{run.name}+repair_refill",
        placements=placements,
        remaining=leftover,
        block_steps=[],
        score=_solution_proxy(repaired_state, total_items),
        elapsed_ms=run.elapsed_ms + budget_ms,
        used_model=run.used_model,
        seed_family=run.seed_family,
        ordered_skus=run.ordered_skus,
    )
    if _run_sort_key(repaired) <= _run_sort_key(run):
        return run
    return repaired


def _fragility_offender_signatures(
    placements: Sequence[PlacedBox],
) -> set[Tuple[str, int, int, int, int]]:
    offenders: set[Tuple[str, int, int, int, int]] = set()
    for conflict in _fragility_conflicts(placements):
        offenders.add(conflict.top_signature)
        offenders.add(conflict.bottom_signature)
    return offenders


def _fragility_conflicts(
    placements: Sequence[PlacedBox],
) -> List[FragilityConflict]:
    conflicts: List[FragilityConflict] = []
    for top in placements:
        if top.weight <= FRAGILE_WEIGHT_THRESHOLD:
            continue
        for bottom in placements:
            if not bottom.fragile:
                continue
            if abs(top.aabb.z_min - bottom.aabb.z_max) >= EPSILON:
                continue
            overlap_area = top.aabb.overlap_area_xy(bottom.aabb)
            if overlap_area <= 0:
                continue
            conflicts.append(
                FragilityConflict(
                    top_signature=_placement_signature(top),
                    bottom_signature=_placement_signature(bottom),
                    x_min=min(top.aabb.x_min, bottom.aabb.x_min),
                    y_min=min(top.aabb.y_min, bottom.aabb.y_min),
                    x_max=max(top.aabb.x_max, bottom.aabb.x_max),
                    y_max=max(top.aabb.y_max, bottom.aabb.y_max),
                    z_max=max(top.aabb.z_max, bottom.aabb.z_max),
                    overlap_area=overlap_area,
                )
            )
    return conflicts


def _fragility_conflict_clusters(
    placements: Sequence[PlacedBox],
    conflicts: Sequence[FragilityConflict],
) -> List[Dict[str, Any]]:
    if not conflicts:
        return []

    placement_map = {
        _placement_signature(placement): placement for placement in placements
    }
    remaining = list(conflicts)
    clusters: List[Dict[str, Any]] = []
    while remaining:
        current = remaining.pop(0)
        cluster_signatures = {current.top_signature, current.bottom_signature}
        x_min = current.x_min
        y_min = current.y_min
        x_max = current.x_max
        y_max = current.y_max
        overlap_sum = current.overlap_area
        changed = True

        while changed:
            changed = False
            keep: List[FragilityConflict] = []
            for conflict in remaining:
                if (
                    conflict.top_signature in cluster_signatures
                    or conflict.bottom_signature in cluster_signatures
                    or _rectangles_overlap_xy(
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        conflict.x_min,
                        conflict.y_min,
                        conflict.x_max,
                        conflict.y_max,
                    )
                ):
                    cluster_signatures.add(conflict.top_signature)
                    cluster_signatures.add(conflict.bottom_signature)
                    x_min = min(x_min, conflict.x_min)
                    y_min = min(y_min, conflict.y_min)
                    x_max = max(x_max, conflict.x_max)
                    y_max = max(y_max, conflict.y_max)
                    overlap_sum += conflict.overlap_area
                    changed = True
                else:
                    keep.append(conflict)
            remaining = keep

        expanded = set(cluster_signatures)
        for signature, placement in placement_map.items():
            overlap_area = _rectangle_overlap_area_xy(
                x_min,
                y_min,
                x_max,
                y_max,
                placement.aabb.x_min,
                placement.aabb.y_min,
                placement.aabb.x_max,
                placement.aabb.y_max,
            )
            if overlap_area <= 0:
                continue
            if overlap_area >= placement.aabb.base_area() * 0.30:
                expanded.add(signature)

        clusters.append(
            {
                "signatures": expanded,
                "overlap_sum": overlap_sum,
                "size": len(expanded),
            }
        )

    clusters.sort(
        key=lambda cluster: (cluster["overlap_sum"], cluster["size"]),
        reverse=True,
    )
    return clusters


def _rectangles_overlap_xy(
    ax_min: int,
    ay_min: int,
    ax_max: int,
    ay_max: int,
    bx_min: int,
    by_min: int,
    bx_max: int,
    by_max: int,
) -> bool:
    return (
        min(ax_max, bx_max) - max(ax_min, bx_min) > 0
        and min(ay_max, by_max) - max(ay_min, by_min) > 0
    )


def _rectangle_overlap_area_xy(
    ax_min: int,
    ay_min: int,
    ax_max: int,
    ay_max: int,
    bx_min: int,
    by_min: int,
    bx_max: int,
    by_max: int,
) -> int:
    overlap_x = max(0, min(ax_max, bx_max) - max(ax_min, bx_min))
    overlap_y = max(0, min(ay_max, by_max) - max(ay_min, by_min))
    return overlap_x * overlap_y


def _placement_keep_score(
    placement: PlacedBox,
    request: Dict[str, Any],
    total_items: int,
    offenders: set[Tuple[str, int, int, int, int]],
) -> float:
    pallet = request["pallet"]
    pallet_volume = (
        pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    )
    vol_gain = 0.50 * (
        placement.aabb.length_x() * placement.aabb.width_y() * placement.aabb.height_z()
    ) / max(pallet_volume, 1)
    coverage_gain = 0.30 / max(total_items, 1)
    floor_bonus = 0.04 * (1.0 - placement.aabb.z_min / max(pallet["max_height_mm"], 1))
    wall_bonus = 0.02 * int(
        placement.aabb.x_min == 0
        or placement.aabb.y_min == 0
        or placement.aabb.x_max >= pallet["length_mm"]
        or placement.aabb.y_max >= pallet["width_mm"]
    )
    offender_penalty = 0.18 if _placement_signature(placement) in offenders else 0.0
    return vol_gain + coverage_gain + floor_bonus + wall_bonus - offender_penalty


def _placement_signature(placement: PlacedBox) -> Tuple[str, int, int, int, int]:
    return (
        placement.sku_id,
        placement.instance_index,
        placement.aabb.x_min,
        placement.aabb.y_min,
        placement.aabb.z_min,
    )


def _finalize_run(
    request: Dict[str, Any],
    run: PolicyRun,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
) -> Tuple[List[PlacedBox], List[RemainingItem], PalletState]:
    placements = _reindex_placements(run.placements)
    leftover = _clone_remaining(run.remaining)
    final_state, _ = _rebuild_state(request, placements)
    post_placements, post_leftover = postprocess(
        _reindex_placements(placements),
        _clone_remaining(leftover),
        PalletState(
            request["pallet"]["length_mm"],
            request["pallet"]["width_mm"],
            request["pallet"]["max_height_mm"],
            request["pallet"]["max_weight_kg"],
        ),
        checker,
        ExtremePointManager(request["pallet"]["length_mm"], request["pallet"]["width_mm"]),
        candidate_gen,
    )
    post_state, _ = _rebuild_state(request, post_placements)
    total_items = sum(box["quantity"] for box in request["boxes"])
    if _solution_proxy(post_state, total_items) > _solution_proxy(final_state, total_items) + 1e-9:
        return post_placements, post_leftover, post_state
    return placements, leftover, final_state


def _construct_with_policy(
    request: Dict[str, Any],
    policy_name: str,
    state: PalletState,
    ep_manager: ExtremePointManager,
    remaining: List[RemainingItem],
    checker: FeasibilityChecker,
    deadline: float,
    ranker: Optional[BlockRanker],
    extractor: Optional[BlockFeatureExtractor],
    max_block_span: int,
) -> PolicyRun:
    t0 = time.perf_counter()
    total_items = sum(box["quantity"] for box in request["boxes"])
    steps: List[BlockStep] = []
    used_model = False

    while time.perf_counter() < deadline:
        if sum(item.remaining_qty for item in remaining) == 0:
            break
        candidates = _generate_block_candidates(
            state=state,
            ep_manager=ep_manager,
            remaining=remaining,
            checker=checker,
            policy_name=policy_name,
            max_block_span=max_block_span,
            max_ep=24,
            max_candidates=128,
        )
        if not candidates:
            break

        _rank_candidates(
            candidates=candidates,
            state=state,
            remaining=remaining,
            policy_name=policy_name,
            ranker=ranker,
            extractor=extractor,
        )
        used_model = used_model or bool(ranker and extractor)
        best = candidates[0]
        before = _solution_proxy(state, total_items)
        _apply_block_candidate(best, state, ep_manager)
        remaining = _decrement_remaining(remaining, best.sku_id, best.item_count)
        after = _solution_proxy(state, total_items)
        steps.append(
            BlockStep(
                candidate=best,
                marginal_gain=after - before,
                frontier_exposure=_frontier_exposure(best, state),
            )
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return PolicyRun(
        name=policy_name,
        placements=list(state.placed),
        remaining=_clone_remaining(remaining),
        block_steps=steps,
        score=_solution_proxy(state, total_items),
        elapsed_ms=elapsed_ms,
        used_model=used_model,
    )


def _run_legacy_portfolio(
    request: Dict[str, Any],
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    budget_ms: int,
) -> PolicyRun:
    t0 = time.perf_counter()
    task_id, pallet, boxes = _request_to_models(request)
    total_items = sum(box.quantity for box in boxes)
    best: Optional[PolicyRun] = None

    for sort_name in LEGACY_PORTFOLIO_SORTS:
        solution = pack_greedy(task_id, pallet, boxes, sort_key_name=sort_name)
        placements = _placements_from_public_solution(
            request["boxes"], solution.placements
        )
        remaining = _remaining_from_placements(request["boxes"], placements)
        state, _ = _rebuild_state(request, placements)
        run = PolicyRun(
            name=f"legacy_portfolio:{sort_name}",
            placements=placements,
            remaining=remaining,
            block_steps=[],
            score=_solution_proxy(state, total_items),
            elapsed_ms=solution.solve_time_ms,
            used_model=False,
        )
        improved = _maybe_postprocess_run(
            request=request,
            run=run,
            checker=checker,
            candidate_gen=candidate_gen,
            total_items=total_items,
            time_budget_left_ms=max(0, budget_ms - int((time.perf_counter() - t0) * 1000)),
        )
        run = improved
        if best is None or _run_sort_key(run) > _run_sort_key(best):
            best = run
        if budget_ms > 0 and (time.perf_counter() - t0) * 1000 > budget_ms:
            break

    assert best is not None
    return best


def _best_cheap_completion(
    request: Dict[str, Any],
    state: PalletState,
    ep_manager: ExtremePointManager,
    remaining: List[RemainingItem],
    checker: FeasibilityChecker,
    budget_ms: int,
) -> PolicyRun:
    deadline = time.perf_counter() + max(budget_ms, 1) / 1000.0
    per_policy = max(budget_ms // 3, 10)
    runs = []
    for policy_name in ("coverage_fill", "foundation"):
        run = _construct_with_policy(
            request=request,
            policy_name=policy_name,
            state=state.copy(),
            ep_manager=ep_manager.copy(),
            remaining=_clone_remaining(remaining),
            checker=checker,
            deadline=min(deadline, time.perf_counter() + per_policy / 1000.0),
            ranker=None,
            extractor=None,
            max_block_span=4,
        )
        runs.append(run)
    if time.perf_counter() < deadline:
        runs.append(
            _run_legacy_portfolio(
                request,
                checker=checker,
                candidate_gen=CandidateGenerator(checker, max_candidates=200),
                budget_ms=per_policy,
            )
        )
    return _select_best_run(runs)


def _repair_run(
    request: Dict[str, Any],
    run: PolicyRun,
    checker: FeasibilityChecker,
    ranker: Optional[BlockRanker],
    extractor: Optional[BlockFeatureExtractor],
    budget_ms: int,
) -> PolicyRun:
    remove_count = max(1, math.ceil(len(run.block_steps) * 0.2))
    ranked_steps = sorted(
        run.block_steps,
        key=lambda step: (step.marginal_gain, -step.frontier_exposure),
    )
    removed = set(id(step) for step in ranked_steps[:remove_count])
    kept_steps = [step for step in run.block_steps if id(step) not in removed]
    kept_placements = [
        placed
        for step in kept_steps
        for placed in step.candidate.units
    ]

    state, ep_manager = _rebuild_state(request, kept_placements)
    remaining = _remaining_from_placements(request["boxes"], kept_placements)
    deadline = time.perf_counter() + budget_ms / 1000.0

    refill = _construct_with_policy(
        request=request,
        policy_name="coverage_fill",
        state=state.copy(),
        ep_manager=ep_manager.copy(),
        remaining=_clone_remaining(remaining),
        checker=checker,
        deadline=min(deadline, time.perf_counter() + budget_ms * 0.7 / 1000.0),
        ranker=ranker,
        extractor=extractor,
        max_block_span=4,
    )
    runs = [refill]
    if time.perf_counter() < deadline:
        singles = _construct_with_policy(
            request=request,
            policy_name="coverage_fill",
            state=state.copy(),
            ep_manager=ep_manager.copy(),
            remaining=_clone_remaining(remaining),
            checker=checker,
            deadline=deadline,
            ranker=ranker,
            extractor=extractor,
            max_block_span=1,
        )
        runs.append(singles)

    repaired = _select_best_run(runs)
    if _run_sort_key(repaired) <= _run_sort_key(run):
        return run
    repaired.name = f"{run.name}+repair"
    return repaired


def _maybe_postprocess_run(
    request: Dict[str, Any],
    run: PolicyRun,
    checker: FeasibilityChecker,
    candidate_gen: CandidateGenerator,
    total_items: int,
    time_budget_left_ms: int,
) -> PolicyRun:
    if time_budget_left_ms <= 0 or not run.placements:
        return run

    pallet = request["pallet"]
    t0 = time.perf_counter()
    working_state = PalletState(
        pallet["length_mm"],
        pallet["width_mm"],
        pallet["max_height_mm"],
        pallet["max_weight_kg"],
    )
    placements, leftover = postprocess(
        _reindex_placements(run.placements),
        _clone_remaining(run.remaining),
        working_state,
        checker,
        ExtremePointManager(pallet["length_mm"], pallet["width_mm"]),
        candidate_gen,
    )
    post_state, _ = _rebuild_state(request, placements)
    post_score = _solution_proxy(post_state, total_items)
    if post_score <= run.score + 1e-9:
        return run

    return PolicyRun(
        name=f"{run.name}+postprocess",
        placements=placements,
        remaining=leftover,
        block_steps=[],
        score=post_score,
        elapsed_ms=run.elapsed_ms + int((time.perf_counter() - t0) * 1000),
        used_model=run.used_model,
    )


def _generate_block_candidates(
    state: PalletState,
    ep_manager: ExtremePointManager,
    remaining: List[RemainingItem],
    checker: FeasibilityChecker,
    policy_name: str,
    max_block_span: int,
    max_ep: int,
    max_candidates: int,
) -> List[BlockCandidate]:
    points = _rank_extreme_points(ep_manager.get_points(), state)[:max_ep]
    coarse_specs: List[_BlockSpec] = []
    seen_specs = set()

    for item in remaining:
        if item.remaining_qty <= 0:
            continue
        for ep_x, ep_y, ep_z in points:
            for pl, pw, ph, rotation_code in get_orientations(
                item.length, item.width, item.height, item.strict_upright
            ):
                for nx in range(1, max_block_span + 1):
                    for ny in range(1, max_block_span + 1):
                        base_count = nx * ny
                        if base_count > item.remaining_qty:
                            continue
                        for nz in range(1, max_block_span + 1):
                            count = base_count * nz
                            if count > item.remaining_qty:
                                continue
                            block_l = pl * nx
                            block_w = pw * ny
                            block_h = ph * nz
                            z_base = max(
                                ep_z, state.get_max_z_at(ep_x, ep_y, block_l, block_w)
                            )
                            aabb = AABB(
                                ep_x,
                                ep_y,
                                z_base,
                                ep_x + block_l,
                                ep_y + block_w,
                                z_base + block_h,
                            )
                            if not checker.check_bounds(aabb):
                                continue
                            total_weight = count * item.weight
                            if not checker.check_weight(total_weight, state):
                                continue
                            wall_count = int(aabb.x_min == 0 or aabb.x_max >= state.length)
                            wall_count += int(aabb.y_min == 0 or aabb.y_max >= state.width)
                            corner_touch = wall_count == 2
                            support_ratio = state.get_support_ratio(aabb)
                            score = _heuristic_block_score(
                                policy_name=policy_name,
                                state=state,
                                item_count=count,
                                block_l=block_l,
                                block_w=block_w,
                                block_h=block_h,
                                total_weight=total_weight,
                                fragile=item.fragile,
                                z=z_base,
                                support_ratio=support_ratio,
                                wall_count=wall_count,
                                corner_touch=corner_touch,
                            )
                            spec = _BlockSpec(
                                sku_id=item.sku_id,
                                placed_dims=(pl, pw, ph),
                                rotation_code=rotation_code,
                                nx=nx,
                                ny=ny,
                                nz=nz,
                                item_count=count,
                                total_weight=total_weight,
                                fragile=item.fragile,
                                stackable=item.stackable,
                                strict_upright=item.strict_upright,
                                x=ep_x,
                                y=ep_y,
                                z=z_base,
                                heuristic_score=score,
                            )
                            spec_key = (
                                spec.sku_id,
                                spec.rotation_code,
                                spec.nx,
                                spec.ny,
                                spec.nz,
                                spec.x,
                                spec.y,
                                spec.z,
                            )
                            if spec_key in seen_specs:
                                continue
                            seen_specs.add(spec_key)
                            coarse_specs.append(spec)

    coarse_specs.sort(key=lambda spec: spec.heuristic_score, reverse=True)
    coarse_specs = coarse_specs[: max_candidates * 4]
    by_sku = {item.sku_id: item for item in remaining}
    candidates: List[BlockCandidate] = []
    for spec in coarse_specs:
        candidate = _materialize_block_candidate(
            spec,
            by_sku[spec.sku_id],
            state,
            checker,
        )
        if candidate is not None:
            candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _materialize_block_candidate(
    spec: _BlockSpec,
    item: RemainingItem,
    state: PalletState,
    checker: FeasibilityChecker,
) -> Optional[BlockCandidate]:
    sim_state = state.copy()
    units: List[PlacedBox] = []
    instance = state.next_instance_index(item.sku_id)
    pl, pw, ph = spec.placed_dims

    for iz in range(spec.nz):
        for iy in range(spec.ny):
            for ix in range(spec.nx):
                aabb = AABB(
                    spec.x + ix * pl,
                    spec.y + iy * pw,
                    spec.z + iz * ph,
                    spec.x + (ix + 1) * pl,
                    spec.y + (iy + 1) * pw,
                    spec.z + (iz + 1) * ph,
                )
                if not _unit_feasible(aabb, item, sim_state, checker):
                    return None
                placed = PlacedBox(
                    sku_id=item.sku_id,
                    instance_index=instance,
                    aabb=aabb,
                    weight=item.weight,
                    fragile=item.fragile,
                    stackable=item.stackable,
                    strict_upright=item.strict_upright,
                    rotation_code=spec.rotation_code,
                    placed_dims=spec.placed_dims,
                )
                sim_state.place(placed)
                units.append(placed)
                instance += 1

    block_aabb = AABB(
        spec.x,
        spec.y,
        spec.z,
        spec.x + pl * spec.nx,
        spec.y + pw * spec.ny,
        spec.z + ph * spec.nz,
    )
    wall_count = int(block_aabb.x_min == 0 or block_aabb.x_max >= state.length)
    wall_count += int(block_aabb.y_min == 0 or block_aabb.y_max >= state.width)
    return BlockCandidate(
        sku_id=spec.sku_id,
        rotation_code=spec.rotation_code,
        placed_dims=spec.placed_dims,
        nx=spec.nx,
        ny=spec.ny,
        nz=spec.nz,
        item_count=spec.item_count,
        total_weight=spec.total_weight,
        fragile=spec.fragile,
        stackable=spec.stackable,
        strict_upright=spec.strict_upright,
        aabb=block_aabb,
        units=tuple(units),
        support_ratio=state.get_support_ratio(block_aabb),
        wall_count=wall_count,
        corner_touch=wall_count == 2,
        heuristic_score=spec.heuristic_score,
    )


def _unit_feasible(
    aabb: AABB,
    item: RemainingItem,
    state: PalletState,
    checker: FeasibilityChecker,
) -> bool:
    if not checker.check_bounds(aabb):
        return False
    if not checker.check_weight(item.weight, state):
        return False
    if not checker.check_collision(aabb, state):
        return False
    if not checker.check_support(aabb, state):
        return False
    if not checker.check_stackable(aabb, state):
        return False
    if _heavy_non_fragile_on_fragile(aabb, item.weight, item.fragile, state):
        return False
    return True


def _heavy_non_fragile_on_fragile(
    aabb: AABB,
    weight: float,
    fragile: bool,
    state: PalletState,
) -> bool:
    if fragile or weight <= FRAGILE_WEIGHT_THRESHOLD or aabb.z_min == 0:
        return False
    for placed in state.placed:
        if not placed.fragile:
            continue
        if abs(placed.aabb.z_max - aabb.z_min) < EPSILON:
            if aabb.overlap_area_xy(placed.aabb) > 0:
                return True
    return False


def _heuristic_block_score(
    policy_name: str,
    state: PalletState,
    item_count: int,
    block_l: int,
    block_w: int,
    block_h: int,
    total_weight: float,
    fragile: bool,
    z: int,
    support_ratio: float,
    wall_count: int,
    corner_touch: bool,
) -> float:
    vol_norm = (block_l * block_w * block_h) / max(state.pallet_volume, 1)
    area_norm = (block_l * block_w) / max(state.pallet_area, 1)
    z_penalty = z / max(state.max_height, 1)
    weight_norm = total_weight / max(state.max_weight, 1)
    size_norm = item_count / 64.0
    floor_coverage = sum(
        placed.aabb.base_area() for placed in state.placed if placed.aabb.z_min == 0
    ) / max(state.pallet_area, 1)

    if policy_name == "foundation":
        return (
            vol_norm * 4.0
            + area_norm * 2.2
            + weight_norm * 2.2
            + size_norm * 1.5
            + support_ratio * 1.3
            + wall_count * 0.4
            + (0.2 if corner_touch else 0.0)
            - z_penalty * 3.2
            - (0.8 if fragile else 0.0)
        )

    if policy_name == "fragile_last":
        score = (
            vol_norm * 3.0
            + area_norm * 1.8
            + weight_norm * 1.8
            + support_ratio * 1.2
            - z_penalty * 2.5
        )
        if fragile and (
            state.max_z < state.max_height * 0.70
            and state.total_weight < state.max_weight * 0.80
        ):
            score -= 3.5
        elif fragile:
            score += 0.6
        return score

    residual_x = (state.length - block_l) / max(state.length, 1)
    residual_y = (state.width - block_w) / max(state.width, 1)
    fill_bonus = 1.0 - (residual_x + residual_y) * 0.5
    small_bonus = 1.0 - size_norm
    if floor_coverage > 0.60:
        return (
            small_bonus * 2.6
            + fill_bonus * 2.1
            + support_ratio * 1.3
            + wall_count * 0.5
            + (0.25 if corner_touch else 0.0)
            - z_penalty * 1.8
            - (0.4 if fragile and state.max_z < state.max_height * 0.4 else 0.0)
        )
    return (
        vol_norm * 2.0
        + area_norm * 1.4
        + fill_bonus * 1.1
        + support_ratio * 1.0
        - z_penalty * 2.2
    )


def _rank_candidates(
    candidates: List[BlockCandidate],
    state: PalletState,
    remaining: List[RemainingItem],
    policy_name: str,
    ranker: Optional[BlockRanker],
    extractor: Optional[BlockFeatureExtractor],
) -> None:
    heuristic_scores = np.asarray(
        [candidate.heuristic_score for candidate in candidates], dtype=np.float32
    )
    final_scores = heuristic_scores.copy()
    if ranker is not None and extractor is not None and ranker.is_trained:
        views = [candidate.feature_view(state) for candidate in candidates]
        X = extractor.extract_batch(state, views, remaining, policy_name)
        model_scores = ranker.predict(X)
        final_scores = 0.7 * normalize_scores(model_scores) + 0.3 * normalize_scores(
            heuristic_scores
        )
    order = np.argsort(-final_scores)
    candidates[:] = [candidates[idx] for idx in order]


def _apply_block_candidate(
    candidate: BlockCandidate,
    state: PalletState,
    ep_manager: ExtremePointManager,
) -> None:
    for placed in candidate.units:
        state.place(placed)
        ep_manager.update_after_placement(placed.aabb, state)


def _select_best_run(runs: Sequence[PolicyRun]) -> PolicyRun:
    return max(runs, key=_run_sort_key)


def _run_sort_key(run: PolicyRun) -> Tuple[float, int]:
    return (run.score, -run.elapsed_ms)


def _should_use_conservative_fragility_selection(
    fingerprint: ScenarioFingerprint,
) -> bool:
    return (
        fingerprint.sku_count >= 5
        and fingerprint.volume_ratio <= 1.05
        and fingerprint.fragile_ratio >= 0.18
        and fingerprint.upright_ratio >= 0.45
        and (
            fingerprint.non_stackable_ratio >= 0.05
            or fingerprint.weight_ratio >= 0.55
        )
    )


def _top_level_run_sort_key(
    run: PolicyRun,
    request: Dict[str, Any],
    prefer_safe_fragility: bool,
) -> Tuple[float, int]:
    if not prefer_safe_fragility:
        return _run_sort_key(run)

    pallet = request["pallet"]
    pallet_volume = (
        pallet["length_mm"] * pallet["width_mm"] * pallet["max_height_mm"]
    )
    total_items = sum(box["quantity"] for box in request["boxes"])
    placed_volume = sum(
        placement.aabb.length_x() * placement.aabb.width_y() * placement.aabb.height_z()
        for placement in run.placements
    )
    volume_util = placed_volume / max(pallet_volume, 1)
    coverage = len(run.placements) / max(total_items, 1)
    fragility_score = _fragility_score_from_placements(run.placements)
    conservative_score = run.score + 0.40 * fragility_score
    return (conservative_score, -run.elapsed_ms)


def _fragility_score_from_placements(
    placements: Sequence[PlacedBox],
) -> float:
    violations = 0
    for top in placements:
        if top.weight <= FRAGILE_WEIGHT_THRESHOLD:
            continue
        for bottom in placements:
            if not bottom.fragile:
                continue
            if abs(top.aabb.z_min - bottom.aabb.z_max) < EPSILON:
                if top.aabb.overlap_area_xy(bottom.aabb) > 0:
                    violations += 1
    return max(0.0, 1.0 - 0.05 * violations)


def _frontier_exposure(candidate: BlockCandidate, state: PalletState) -> float:
    top_exposure = candidate.aabb.z_max / max(state.max_height, 1)
    side_exposure = 1.0 - (candidate.wall_count / 2.0)
    return top_exposure + side_exposure


def _rank_extreme_points(
    points: Iterable[Tuple[int, int, int]],
    state: PalletState,
) -> List[Tuple[int, int, int]]:
    ranked = list(points)
    ranked.sort(
        key=lambda point: (
            point[2],
            0 if point[0] == 0 or point[0] >= state.length else 1,
            0 if point[1] == 0 or point[1] >= state.width else 1,
            point[0] + point[1],
        )
    )
    return ranked


def _solution_proxy(state: PalletState, total_items: int) -> float:
    placements = state.placed
    vol_util = state.placed_volume / max(state.pallet_volume, 1)
    coverage = len(placements) / max(total_items, 1)
    fragility_violations = 0
    for top in placements:
        if top.weight <= FRAGILE_WEIGHT_THRESHOLD:
            continue
        for bottom in placements:
            if not bottom.fragile:
                continue
            if abs(top.aabb.z_min - bottom.aabb.z_max) < EPSILON:
                if top.aabb.overlap_area_xy(bottom.aabb) > 0:
                    fragility_violations += 1
    fragility_score = max(0.0, 1.0 - 0.05 * fragility_violations)
    return 0.50 * vol_util + 0.30 * coverage + 0.10 * fragility_score


def _rebuild_state(
    request: Dict[str, Any],
    placements: Sequence[PlacedBox],
) -> Tuple[PalletState, ExtremePointManager]:
    pallet = request["pallet"]
    state = PalletState(
        pallet["length_mm"],
        pallet["width_mm"],
        pallet["max_height_mm"],
        pallet["max_weight_kg"],
    )
    ep_manager = ExtremePointManager(pallet["length_mm"], pallet["width_mm"])
    for placement in placements:
        state.place(placement)
        ep_manager.update_after_placement(placement.aabb, state)
    return state, ep_manager


def _request_to_models(request: Dict[str, Any]) -> Tuple[str, Pallet, List[Box]]:
    pallet_data = request["pallet"]
    pallet = Pallet(
        type_id=pallet_data.get("type_id", "unknown"),
        length_mm=pallet_data["length_mm"],
        width_mm=pallet_data["width_mm"],
        max_height_mm=pallet_data["max_height_mm"],
        max_weight_kg=pallet_data["max_weight_kg"],
    )
    boxes = [
        Box(
            sku_id=box["sku_id"],
            description=box.get("description", ""),
            length_mm=box["length_mm"],
            width_mm=box["width_mm"],
            height_mm=box["height_mm"],
            weight_kg=box["weight_kg"],
            quantity=box["quantity"],
            strict_upright=box.get("strict_upright", False),
            fragile=box.get("fragile", False),
            stackable=box.get("stackable", True),
        )
        for box in request["boxes"]
    ]
    return request["task_id"], pallet, boxes


def _placements_from_public_solution(
    request_boxes: Sequence[Dict[str, Any]],
    placements: Sequence[Any],
) -> List[PlacedBox]:
    by_sku = {box["sku_id"]: box for box in request_boxes}
    result: List[PlacedBox] = []
    for placement in placements:
        meta = by_sku[placement.sku_id]
        aabb = AABB(
            placement.x_mm,
            placement.y_mm,
            placement.z_mm,
            placement.x_mm + placement.length_mm,
            placement.y_mm + placement.width_mm,
            placement.z_mm + placement.height_mm,
        )
        result.append(
            PlacedBox(
                sku_id=placement.sku_id,
                instance_index=placement.instance_index,
                aabb=aabb,
                weight=meta["weight_kg"],
                fragile=meta.get("fragile", False),
                stackable=meta.get("stackable", True),
                strict_upright=meta.get("strict_upright", False),
                rotation_code=placement.rotation_code,
                placed_dims=(
                    placement.length_mm,
                    placement.width_mm,
                    placement.height_mm,
                ),
            )
        )
    return result


def _reindex_placements(placements: Sequence[PlacedBox]) -> List[PlacedBox]:
    counters: Dict[str, int] = {}
    indexed: List[PlacedBox] = []
    for placement in placements:
        index = counters.get(placement.sku_id, 0)
        counters[placement.sku_id] = index + 1
        indexed.append(
            PlacedBox(
                sku_id=placement.sku_id,
                instance_index=index,
                aabb=placement.aabb,
                weight=placement.weight,
                fragile=placement.fragile,
                stackable=placement.stackable,
                strict_upright=placement.strict_upright,
                rotation_code=placement.rotation_code,
                placed_dims=placement.placed_dims,
            )
        )
    return indexed


def _format_output(
    task_id: str,
    placements: List[PlacedBox],
    leftover: List[RemainingItem],
    boxes: List[Dict[str, Any]],
    state: PalletState,
    solve_time_ms: int,
) -> Dict[str, Any]:
    box_by_sku = {box["sku_id"]: box for box in boxes}
    placed_list = []
    for placement in _reindex_placements(placements):
        placed_list.append(
            {
                "sku_id": placement.sku_id,
                "instance_index": placement.instance_index,
                "position": {
                    "x_mm": placement.aabb.x_min,
                    "y_mm": placement.aabb.y_min,
                    "z_mm": placement.aabb.z_min,
                },
                "dimensions_placed": {
                    "length_mm": placement.placed_dims[0],
                    "width_mm": placement.placed_dims[1],
                    "height_mm": placement.placed_dims[2],
                },
                "rotation_code": placement.rotation_code,
            }
        )

    unplaced_list = []
    for item in leftover:
        if item.remaining_qty <= 0:
            continue
        meta = box_by_sku[item.sku_id]
        reason = _classify_unplaced_reason(item, state, meta)
        unplaced_list.append(
            {
                "sku_id": item.sku_id,
                "quantity_unplaced": item.remaining_qty,
                "reason": reason,
            }
        )

    return {
        "task_id": task_id,
        "solver_version": SOLVER_VERSION,
        "solve_time_ms": solve_time_ms,
        "placements": placed_list,
        "unplaced": unplaced_list,
    }


def _classify_unplaced_reason(
    item: RemainingItem,
    state: PalletState,
    meta: Dict[str, Any],
) -> str:
    if state.total_weight + item.weight > state.max_weight + EPSILON:
        return UNPLACED_REASONS["weight"]
    min_height = min(meta["length_mm"], meta["width_mm"], meta["height_mm"])
    if meta.get("strict_upright", False):
        min_height = meta["height_mm"]
    if state.max_z + min_height > state.max_height + EPSILON:
        return UNPLACED_REASONS["height"]
    return UNPLACED_REASONS["space"]
