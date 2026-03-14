"""Tests for the portfolio-block solver and strategy dispatch."""

from __future__ import annotations

from generator import generate_scenario
from solver.hybrid.candidate_gen import RemainingItem
from solver.hybrid.feasibility import FeasibilityChecker
from solver.hybrid.geometry import AABB
from solver.hybrid.pallet_state import PalletState, PlacedBox
from solver.models import Box, Pallet, solution_to_dict
from solver.portfolio_block import (
    _BlockSpec,
    _fragility_conflict_clusters,
    _fragility_conflicts,
    _fragility_score_from_placements,
    _generate_exact_orderings,
    _generate_exact_prefix_orderings,
    _fragile_staged_instances,
    _heavy_on_fragile,
    _materialize_block_candidate,
    _should_add_legacy_baseline_candidate,
    _should_try_aggressive_fragile_staging,
    _should_try_upright_prefill_staging,
    _should_use_fast_overload_path,
    _should_try_strict_fragility_search,
    PolicyRun,
)
from solver.scenario_selector import compute_request_fingerprint
from solver.solver import solve
from validator import evaluate_solution


def _request_to_models(request_dict: dict):
    p = request_dict["pallet"]
    pallet = Pallet(
        type_id=p.get("type_id", "unknown"),
        length_mm=p["length_mm"],
        width_mm=p["width_mm"],
        max_height_mm=p["max_height_mm"],
        max_weight_kg=p["max_weight_kg"],
    )
    boxes = []
    for b in request_dict["boxes"]:
        boxes.append(
            Box(
                sku_id=b["sku_id"],
                description=b.get("description", ""),
                length_mm=b["length_mm"],
                width_mm=b["width_mm"],
                height_mm=b["height_mm"],
                weight_kg=b["weight_kg"],
                quantity=b["quantity"],
                strict_upright=b.get("strict_upright", False),
                fragile=b.get("fragile", False),
                stackable=b.get("stackable", True),
            )
        )
    return request_dict["task_id"], pallet, boxes


def test_block_expansion_preserves_quantity_geometry_rotation():
    state = PalletState(1200, 800, 1800, 1000.0)
    checker = FeasibilityChecker(1200, 800, 1800, 1000.0)
    item = RemainingItem(
        {
            "sku_id": "SKU-A",
            "length_mm": 200,
            "width_mm": 100,
            "height_mm": 150,
            "weight_kg": 2.5,
            "quantity": 10,
            "strict_upright": False,
            "fragile": False,
            "stackable": True,
        }
    )
    spec = _BlockSpec(
        sku_id="SKU-A",
        placed_dims=(200, 100, 150),
        rotation_code="LWH",
        nx=2,
        ny=1,
        nz=2,
        item_count=4,
        total_weight=10.0,
        fragile=False,
        stackable=True,
        strict_upright=False,
        x=0,
        y=0,
        z=0,
        heuristic_score=1.0,
    )

    candidate = _materialize_block_candidate(spec, item, state, checker)
    assert candidate is not None
    assert len(candidate.units) == 4
    assert all(unit.rotation_code == "LWH" for unit in candidate.units)
    assert all(unit.placed_dims == (200, 100, 150) for unit in candidate.units)
    assert (
        candidate.aabb.x_min,
        candidate.aabb.y_min,
        candidate.aabb.z_min,
        candidate.aabb.x_max,
        candidate.aabb.y_max,
        candidate.aabb.z_max,
    ) == (0, 0, 0, 400, 100, 300)


def test_block_feasibility_respects_stackable_false():
    state = PalletState(1200, 800, 1800, 1000.0)
    checker = FeasibilityChecker(1200, 800, 1800, 1000.0)
    base = PlacedBox(
        sku_id="BASE",
        instance_index=0,
        aabb=AABB(0, 0, 0, 400, 300, 200),
        weight=10.0,
        fragile=False,
        stackable=False,
        strict_upright=False,
        rotation_code="LWH",
        placed_dims=(400, 300, 200),
    )
    state.place(base)

    item = RemainingItem(
        {
            "sku_id": "TOP",
            "length_mm": 400,
            "width_mm": 300,
            "height_mm": 200,
            "weight_kg": 5.0,
            "quantity": 2,
            "strict_upright": False,
            "fragile": False,
            "stackable": True,
        }
    )
    spec = _BlockSpec(
        sku_id="TOP",
        placed_dims=(400, 300, 200),
        rotation_code="LWH",
        nx=1,
        ny=1,
        nz=1,
        item_count=1,
        total_weight=5.0,
        fragile=False,
        stackable=True,
        strict_upright=False,
        x=0,
        y=0,
        z=200,
        heuristic_score=1.0,
    )

    candidate = _materialize_block_candidate(spec, item, state, checker)
    assert candidate is None


def test_heavy_fragile_is_not_blocked_on_fragile_support():
    state = PalletState(1200, 800, 1800, 1000.0)
    bottom = PlacedBox(
        sku_id="FRAGILE-BASE",
        instance_index=0,
        aabb=AABB(0, 0, 0, 400, 400, 200),
        weight=1.0,
        fragile=True,
        stackable=True,
        strict_upright=False,
        rotation_code="LWH",
        placed_dims=(400, 400, 200),
    )
    state.place(bottom)

    top_aabb = AABB(0, 0, 200, 400, 400, 400)

    assert _heavy_on_fragile(top_aabb, 5.0, True, state) is False
    assert _heavy_on_fragile(top_aabb, 5.0, False, state) is True


def test_default_solve_dispatches_to_portfolio(monkeypatch):
    from solver import solver as solver_module

    called = {"portfolio": False, "hybrid": False}

    def fake_portfolio_request(request, model_dir="models", time_budget_ms=900):
        called["portfolio"] = True
        return {
            "task_id": request["task_id"],
            "solver_version": "portfolio-block-v0.1",
            "solve_time_ms": 1,
            "placements": [],
            "unplaced": [],
        }

    def fake_hybrid_request(**kwargs):
        called["hybrid"] = True
        raise AssertionError("legacy_hybrid should not be called by default")

    monkeypatch.setattr(solver_module, "solve_portfolio_request", fake_portfolio_request)
    monkeypatch.setattr(solver_module, "solve_hybrid_request", fake_hybrid_request)

    pallet = Pallet("EUR", 1200, 800, 1800, 1000.0)
    boxes = [
        Box("SKU", "", 200, 200, 200, 1.0, 1),
    ]
    solve("task", pallet, boxes, request_dict=None)
    assert called["portfolio"] is True
    assert called["hybrid"] is False


def test_portfolio_solver_falls_back_without_model_and_stays_valid(tmp_path):
    request_dict = generate_scenario("test_exact_fit", "exact_fit", seed=46)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        strategy="portfolio_block",
        model_dir=str(tmp_path / "missing-model"),
        time_budget_ms=900,
    )
    result = evaluate_solution(request_dict, solution_to_dict(solution))
    assert result["valid"] is True


def test_legacy_baseline_gate_follows_fast_overload_fingerprint():
    chaotic = compute_request_fingerprint(
        generate_scenario("test_random_mixed", "random_mixed", seed=45)
    )
    compact = compute_request_fingerprint(
        generate_scenario("test_exact_fit", "exact_fit", seed=46)
    )

    assert _should_add_legacy_baseline_candidate(chaotic) is True
    assert _should_add_legacy_baseline_candidate(compact) is False


def test_upright_prefill_gate_targets_low_volume_upright_nonstackable_mix():
    liquid = compute_request_fingerprint(
        generate_scenario("test_liquid_tetris", "liquid_tetris", seed=44)
    )
    upright_tight = compute_request_fingerprint(
        generate_scenario("test_all_upright_tight", "private_all_upright_tight", seed=211)
    )

    assert _should_try_upright_prefill_staging(liquid) is True
    assert _should_try_upright_prefill_staging(upright_tight) is False


def test_legacy_greedy_strategy_regression_stays_valid():
    request_dict = generate_scenario("test_legacy_random", "random_mixed", seed=45)
    task_id, pallet, boxes = _request_to_models(request_dict)

    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        strategy="legacy_greedy",
        time_budget_ms=900,
    )

    result = evaluate_solution(request_dict, solution_to_dict(solution))
    assert result["valid"] is True
    assert solution.solve_time_ms < 1000


def test_fragile_staged_instances_prefers_sturdy_then_anchors_then_light_fragile():
    pallet = Pallet("EUR", 1200, 800, 1800, 1000.0)
    boxes = [
        Box("STURDY", "", 600, 400, 200, 10.0, 6, fragile=False),
        Box("FRAG_HEAVY", "", 300, 400, 300, 3.5, 4, fragile=True),
        Box("FRAG_LIGHT", "", 200, 400, 300, 1.0, 8, fragile=True),
    ]

    staged = _fragile_staged_instances(boxes, pallet)
    staged_skus = [box.sku_id for box, _ in staged]

    assert staged_skus[:4] == ["STURDY"] * 4
    assert staged_skus[4:6] == ["FRAG_HEAVY"] * 2
    assert staged_skus[6] == "FRAG_LIGHT"


def test_aggressive_fragile_staging_gate_targets_mixed_fragile_heavy_layouts():
    pallet = Pallet("EUR", 1200, 800, 1800, 1000.0)
    boxes = [
        Box("STURDY", "", 600, 400, 200, 10.0, 6, fragile=False),
        Box("FRAG_HEAVY", "", 300, 400, 300, 3.5, 4, fragile=True),
        Box("FRAG_LIGHT", "", 200, 400, 300, 1.0, 8, fragile=True),
    ]

    assert _should_try_aggressive_fragile_staging(boxes) is True

    aggressive = _fragile_staged_instances(boxes, pallet, anchor_count=3)
    aggressive_skus = [box.sku_id for box, _ in aggressive]

    assert aggressive_skus[:4] == ["STURDY"] * 4
    assert aggressive_skus[4:7] == ["FRAG_HEAVY"] * 3


def test_generate_exact_orderings_returns_all_unique_permutations_except_current():
    boxes = [
        Box("A", "", 100, 100, 100, 1.0, 1),
        Box("B", "", 100, 100, 100, 1.0, 1),
        Box("C", "", 100, 100, 100, 1.0, 1),
    ]

    orderings = _generate_exact_orderings(boxes)
    keys = {tuple(box.sku_id for box in ordering) for ordering in orderings}

    assert len(orderings) == 5
    assert len(keys) == 5
    assert ("A", "B", "C") not in keys


def test_generate_exact_prefix_orderings_only_permutes_prefix():
    boxes = [
        Box("A", "", 100, 100, 100, 1.0, 1),
        Box("B", "", 100, 100, 100, 1.0, 1),
        Box("C", "", 100, 100, 100, 1.0, 1),
        Box("D", "", 100, 100, 100, 1.0, 1),
    ]

    orderings = _generate_exact_prefix_orderings(boxes, 3)
    keys = {tuple(box.sku_id for box in ordering) for ordering in orderings}

    assert len(orderings) == 5
    assert all(ordering[-1].sku_id == "D" for ordering in orderings)
    assert ("A", "B", "C", "D") not in keys


def test_fragility_conflicts_ignore_heavy_fragile_boxes_on_fragile_support():
    bottom = PlacedBox(
        sku_id="BOTTOM",
        instance_index=0,
        aabb=AABB(0, 0, 0, 400, 400, 200),
        weight=1.0,
        fragile=True,
        stackable=True,
        strict_upright=False,
        rotation_code="LWH",
        placed_dims=(400, 400, 200),
    )
    top_fragile = PlacedBox(
        sku_id="TOP-FRAGILE",
        instance_index=0,
        aabb=AABB(0, 0, 200, 400, 400, 400),
        weight=5.0,
        fragile=True,
        stackable=True,
        strict_upright=False,
        rotation_code="LWH",
        placed_dims=(400, 400, 200),
    )
    top_non_fragile = PlacedBox(
        sku_id="TOP-STURDY",
        instance_index=0,
        aabb=AABB(0, 0, 200, 400, 400, 400),
        weight=5.0,
        fragile=False,
        stackable=True,
        strict_upright=False,
        rotation_code="LWH",
        placed_dims=(400, 400, 200),
    )

    assert _fragility_conflicts([bottom, top_fragile]) == []
    assert _fragility_score_from_placements([bottom, top_fragile]) == 1.0
    assert len(_fragility_conflicts([bottom, top_non_fragile])) == 1
    assert _fragility_score_from_placements([bottom, top_non_fragile]) < 1.0


def test_strict_fragility_search_gates_on_near_fit_heavy_fragile_requests():
    fragile_mix_request = {
        "task_id": "strict-gate",
        "pallet": {
            "type_id": "TEST",
            "length_mm": 1200,
            "width_mm": 800,
            "max_height_mm": 400,
            "max_weight_kg": 1000.0,
        },
        "boxes": [
            {
                "sku_id": "BASE",
                "description": "Base support",
                "length_mm": 600,
                "width_mm": 400,
                "height_mm": 200,
                "weight_kg": 15.0,
                "quantity": 2,
                "strict_upright": False,
                "fragile": False,
                "stackable": True,
            },
            {
                "sku_id": "FRAGILE",
                "description": "Heavy fragile box",
                "length_mm": 300,
                "width_mm": 400,
                "height_mm": 200,
                "weight_kg": 3.0,
                "quantity": 8,
                "strict_upright": False,
                "fragile": True,
                "stackable": True,
            },
            {
                "sku_id": "FILL",
                "description": "Light filler",
                "length_mm": 300,
                "width_mm": 200,
                "height_mm": 200,
                "weight_kg": 1.5,
                "quantity": 4,
                "strict_upright": False,
                "fragile": False,
                "stackable": True,
            },
        ],
    }
    fragile_mix_fp = compute_request_fingerprint(fragile_mix_request)
    fragile_mix_run = PolicyRun(
        name="heavy_base",
        placements=[object()] * 13,
        remaining=[],
        block_steps=[],
        score=0.0,
        elapsed_ms=0,
        seed_family="heavy_base",
        ordered_skus=tuple(box["sku_id"] for box in fragile_mix_request["boxes"]),
    )
    assert (
        _should_try_strict_fragility_search(
            fragile_mix_request,
            fragile_mix_fp,
            fragile_mix_run,
        )
        is True
    )

    tower_request = generate_scenario("test_tower", "fragile_tower", seed=43)
    tower_fp = compute_request_fingerprint(tower_request)
    tower_run = PolicyRun(
        name="fragile_density",
        placements=[object()] * 21,
        remaining=[],
        block_steps=[],
        score=0.0,
        elapsed_ms=0,
        seed_family="fragile_density",
        ordered_skus=tuple(box["sku_id"] for box in tower_request["boxes"]),
    )
    assert _should_try_strict_fragility_search(tower_request, tower_fp, tower_run) is False


def test_fast_overload_path_targets_large_overloaded_high_sku_requests():
    overloaded_request = generate_scenario("test_overloaded", "random_mixed", seed=45)
    overloaded_fp = compute_request_fingerprint(overloaded_request)
    assert _should_use_fast_overload_path(overloaded_fp) is True

    near_fit_request = generate_scenario("test_near_fit", "fragile_cap_mix", seed=48)
    near_fit_fp = compute_request_fingerprint(near_fit_request)
    assert _should_use_fast_overload_path(near_fit_fp) is False


def test_fragility_conflict_clusters_capture_entire_xy_column():
    placements = [
        PlacedBox(
            sku_id="FRAG_BASE",
            instance_index=0,
            aabb=AABB(0, 0, 0, 300, 300, 200),
            weight=3.0,
            fragile=True,
            stackable=True,
            strict_upright=False,
            rotation_code="LWH",
            placed_dims=(300, 300, 200),
        ),
        PlacedBox(
            sku_id="HEAVY_TOP",
            instance_index=0,
            aabb=AABB(0, 0, 200, 300, 300, 400),
            weight=8.0,
            fragile=False,
            stackable=True,
            strict_upright=False,
            rotation_code="LWH",
            placed_dims=(300, 300, 200),
        ),
        PlacedBox(
            sku_id="HEAVY_CAP",
            instance_index=1,
            aabb=AABB(0, 0, 400, 300, 300, 600),
            weight=8.0,
            fragile=False,
            stackable=True,
            strict_upright=False,
            rotation_code="LWH",
            placed_dims=(300, 300, 200),
        ),
        PlacedBox(
            sku_id="SIDE_SAFE",
            instance_index=0,
            aabb=AABB(300, 0, 0, 600, 300, 200),
            weight=8.0,
            fragile=False,
            stackable=True,
            strict_upright=False,
            rotation_code="LWH",
            placed_dims=(300, 300, 200),
        ),
    ]

    conflicts = _fragility_conflicts(placements)
    clusters = _fragility_conflict_clusters(placements, conflicts)

    assert len(conflicts) == 1
    assert len(clusters) == 1
    signatures = clusters[0]["signatures"]
    assert ("FRAG_BASE", 0, 0, 0, 0) in signatures
    assert ("HEAVY_TOP", 0, 0, 0, 200) in signatures
    assert ("HEAVY_CAP", 1, 0, 0, 400) in signatures
    assert ("SIDE_SAFE", 0, 300, 0, 0) not in signatures
