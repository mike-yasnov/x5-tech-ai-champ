"""Shared benchmark helpers for scenarios, methods, and scoring."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from generator import generate_scenario
from solver.models import Box, Pallet, Solution, solution_to_dict
from solver.packer import SORT_KEYS, pack_greedy
from solver.solver import solve
from validator import evaluate_solution

QUALITY_WEIGHTS = {
    "volume_utilization": 0.55,
    "item_coverage": 0.35,
    "fragility_score": 0.10,
}

ORGANIZER_SCENARIOS = [
    ("heavy_water", 42),
    ("fragile_tower", 43),
    ("liquid_tetris", 44),
    ("random_mixed", 45),
]

PROJECT_SCENARIOS = [
    ("exact_fit", 46),
    ("fragile_mix", 47),
    ("support_tetris", 48),
    ("cavity_fill", 49),
    ("count_preference", 50),
]

SCENARIO_GROUPS = {
    "organizers": ORGANIZER_SCENARIOS,
    "project": PROJECT_SCENARIOS,
    "full": ORGANIZER_SCENARIOS + PROJECT_SCENARIOS,
    "smoke": [ORGANIZER_SCENARIOS[0], PROJECT_SCENARIOS[0], PROJECT_SCENARIOS[1]],
}

METHOD_SOLVER = "solver"


def available_methods() -> List[str]:
    return [METHOD_SOLVER] + [
        f"strategy:{strategy_name}" for strategy_name in SORT_KEYS
    ]


def resolve_methods(methods: Sequence[str] | None) -> List[str]:
    if not methods:
        return [METHOD_SOLVER]
    if len(methods) == 1 and methods[0] == "all":
        return available_methods()
    return list(methods)


def get_scenarios(scenario_set: str) -> List[tuple[str, int]]:
    if scenario_set not in SCENARIO_GROUPS:
        raise ValueError(f"Unknown scenario set: {scenario_set}")
    return list(SCENARIO_GROUPS[scenario_set])


def request_to_models(request_dict: dict) -> tuple[str, Pallet, List[Box]]:
    pallet_data = request_dict["pallet"]
    pallet = Pallet(
        type_id=pallet_data.get("type_id", "unknown"),
        length_mm=pallet_data["length_mm"],
        width_mm=pallet_data["width_mm"],
        max_height_mm=pallet_data["max_height_mm"],
        max_weight_kg=pallet_data["max_weight_kg"],
    )
    boxes = []
    for box_data in request_dict["boxes"]:
        boxes.append(
            Box(
                sku_id=box_data["sku_id"],
                description=box_data.get("description", ""),
                length_mm=box_data["length_mm"],
                width_mm=box_data["width_mm"],
                height_mm=box_data["height_mm"],
                weight_kg=box_data["weight_kg"],
                quantity=box_data["quantity"],
                strict_upright=box_data.get("strict_upright", False),
                fragile=box_data.get("fragile", False),
                stackable=box_data.get("stackable", True),
            )
        )
    return request_dict["task_id"], pallet, boxes


def compute_quality_score(metrics: Dict[str, float]) -> float:
    return sum(
        metrics.get(name, 0.0) * weight for name, weight in QUALITY_WEIGHTS.items()
    )


def _strategy_configs() -> Dict[str, Any]:
    return {strategy_name: strategy_name for strategy_name in SORT_KEYS}


def run_method(
    method_name: str,
    task_id: str,
    pallet: Pallet,
    boxes: List[Box],
    request_dict: dict,
    n_restarts: int,
    time_budget_ms: int,
) -> Dict[str, Any]:
    if method_name == METHOD_SOLVER:
        solution = solve(
            task_id=task_id,
            pallet=pallet,
            boxes=boxes,
            request_dict=request_dict,
            n_restarts=n_restarts,
            time_budget_ms=time_budget_ms,
        )
        method_meta: Dict[str, Any] = {
            "type": "solver",
            "n_restarts": n_restarts,
            "time_budget_ms": time_budget_ms,
        }
    elif method_name.startswith("strategy:"):
        strategy_name = method_name.split(":", 1)[1]
        strategy = _strategy_configs().get(strategy_name)
        if strategy is None:
            raise ValueError(f"Unknown method: {method_name}")
        solution = pack_greedy(
            task_id=task_id,
            pallet=pallet,
            boxes=boxes,
            sort_key_name=strategy,
        )
        method_meta = {"type": "strategy", "sort_key_name": strategy}
    else:
        raise ValueError(f"Unknown method: {method_name}")

    response_dict = solution_to_dict(solution)
    evaluation = evaluate_solution(request_dict, response_dict)
    metrics = evaluation.get("metrics", {})

    return {
        "method": method_name,
        "method_meta": method_meta,
        "valid": evaluation.get("valid", False),
        "official_score": evaluation.get("final_score", 0.0),
        "quality_score": compute_quality_score(metrics),
        "metrics": metrics,
        "constraint_checks": evaluation.get("constraint_checks", {}),
        "placed": len(solution.placements),
        "total_items": sum(box["quantity"] for box in request_dict["boxes"]),
        "solve_time_ms": solution.solve_time_ms,
        "error": evaluation.get("error"),
        "response": response_dict,
    }


def run_benchmark_methods(
    methods: Sequence[str],
    scenario_set: str = "full",
    n_restarts: int = len(SORT_KEYS),
    time_budget_ms: int = 5000,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "meta": {
            "methods": list(methods),
            "scenario_set": scenario_set,
            "n_restarts": n_restarts,
            "time_budget_ms": time_budget_ms,
            "quality_weights": QUALITY_WEIGHTS,
        },
        "scenarios": [],
    }

    for scenario_name, seed in get_scenarios(scenario_set):
        request_dict = generate_scenario(
            f"bench_{scenario_name}", scenario_name, seed=seed
        )
        task_id, pallet, boxes = request_to_models(request_dict)
        results = []
        for method_name in methods:
            results.append(
                run_method(
                    method_name=method_name,
                    task_id=task_id,
                    pallet=pallet,
                    boxes=boxes,
                    request_dict=request_dict,
                    n_restarts=n_restarts,
                    time_budget_ms=time_budget_ms,
                )
            )
        report["scenarios"].append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "request_pallet": request_dict["pallet"],
                "request_boxes": request_dict["boxes"],
                "results": results,
            }
        )
    return report


def summarize_methods(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    method_rows: Dict[str, Dict[str, Any]] = {}
    for scenario in report.get("scenarios", []):
        for result in scenario.get("results", []):
            row = method_rows.setdefault(
                result["method"],
                {
                    "method": result["method"],
                    "official_score": 0.0,
                    "quality_score": 0.0,
                    "invalid_count": 0,
                    "placed": 0,
                    "total_items": 0,
                    "solve_time_ms": 0,
                    "scenario_count": 0,
                },
            )
            row["official_score"] += result["official_score"]
            row["quality_score"] += result["quality_score"]
            row["invalid_count"] += 0 if result["valid"] else 1
            row["placed"] += result["placed"]
            row["total_items"] += result["total_items"]
            row["solve_time_ms"] += result["solve_time_ms"]
            row["scenario_count"] += 1

    summary = []
    for row in method_rows.values():
        count = max(1, row.pop("scenario_count"))
        summary.append(
            {
                **row,
                "official_score": row["official_score"] / count,
                "quality_score": row["quality_score"] / count,
                "avg_solve_time_ms": row["solve_time_ms"] / count,
                "item_coverage": row["placed"] / max(1, row["total_items"]),
            }
        )
    summary.sort(
        key=lambda item: (
            item["invalid_count"] == 0,
            item["quality_score"],
            item["official_score"],
        ),
        reverse=True,
    )
    return summary


def write_json(path: str, payload: Dict[str, Any]) -> None:
    import json

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def iter_scenario_results(
    report: Dict[str, Any],
) -> Iterable[tuple[str, Dict[str, Any]]]:
    for scenario in report.get("scenarios", []):
        scenario_name = scenario["scenario"]
        for result in scenario.get("results", []):
            yield scenario_name, result
