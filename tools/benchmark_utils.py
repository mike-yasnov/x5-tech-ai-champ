"""Shared benchmark helpers for scenarios, methods, and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence

from core.generator import generate_scenario
from core.validator import evaluate_solution


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


@dataclass(frozen=True)
class SolverSpec:
    name: str
    box_cls: Any
    pallet_cls: Any
    solve: Callable[..., Any]
    solution_to_dict: Callable[[Any], Dict[str, Any]]
    pack_greedy: Callable[..., Any]
    sort_keys: Sequence[str]


def _solver_specs() -> Dict[str, SolverSpec]:
    from base_solver.models import Box as BaseBox, Pallet as BasePallet
    from base_solver.models import solution_to_dict as base_solution_to_dict
    from base_solver.packer import SORT_KEYS as BASE_SORT_KEYS
    from base_solver.packer import pack_greedy as base_pack_greedy
    from base_solver.solver import solve as base_solve

    from alternative_solver.models import Box as AltBox, Pallet as AltPallet
    from alternative_solver.models import solution_to_dict as alt_solution_to_dict
    from alternative_solver.packer import SORT_KEYS as ALT_SORT_KEYS
    from alternative_solver.packer import pack_greedy as alt_pack_greedy
    from alternative_solver.solver import solve as alt_solve

    return {
        "base": SolverSpec(
            name="base",
            box_cls=BaseBox,
            pallet_cls=BasePallet,
            solve=base_solve,
            solution_to_dict=base_solution_to_dict,
            pack_greedy=base_pack_greedy,
            sort_keys=tuple(BASE_SORT_KEYS.keys()),
        ),
        "alternative": SolverSpec(
            name="alternative",
            box_cls=AltBox,
            pallet_cls=AltPallet,
            solve=alt_solve,
            solution_to_dict=alt_solution_to_dict,
            pack_greedy=alt_pack_greedy,
            sort_keys=tuple(ALT_SORT_KEYS.keys()),
        ),
    }


def available_methods() -> List[str]:
    methods = ["solver:base", "solver:alternative"]
    for solver_name, spec in _solver_specs().items():
        methods.extend(
            f"strategy:{solver_name}:{sort_key_name}"
            for sort_key_name in spec.sort_keys
        )
    return methods


def resolve_methods(methods: Sequence[str] | None) -> List[str]:
    if not methods:
        return ["solver:base", "solver:alternative"]
    if len(methods) == 1 and methods[0] == "all":
        return available_methods()
    return ["solver:base" if method == "solver" else method for method in methods]


def get_scenarios(scenario_set: str) -> List[tuple[str, int]]:
    if scenario_set not in SCENARIO_GROUPS:
        raise ValueError(f"Unknown scenario set: {scenario_set}")
    return list(SCENARIO_GROUPS[scenario_set])


def request_to_models(
    request_dict: dict, solver_name: str
) -> tuple[str, Any, List[Any]]:
    spec = _solver_specs()[solver_name]
    pallet_data = request_dict["pallet"]
    pallet = spec.pallet_cls(
        type_id=pallet_data.get("type_id", "unknown"),
        length_mm=pallet_data["length_mm"],
        width_mm=pallet_data["width_mm"],
        max_height_mm=pallet_data["max_height_mm"],
        max_weight_kg=pallet_data["max_weight_kg"],
    )
    boxes = []
    for box_data in request_dict["boxes"]:
        boxes.append(
            spec.box_cls(
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


def _parse_method(method_name: str) -> tuple[str, str, str | None]:
    parts = method_name.split(":")
    if method_name == "solver":
        return "solver", "base", None
    if len(parts) == 2 and parts[0] == "solver":
        return "solver", parts[1], None
    if len(parts) == 3 and parts[0] == "strategy":
        return "strategy", parts[1], parts[2]
    raise ValueError(f"Unknown method: {method_name}")


def run_method(
    method_name: str,
    task_id: str,
    request_dict: dict,
    n_restarts: int,
    time_budget_ms: int,
) -> Dict[str, Any]:
    method_type, solver_name, strategy_name = _parse_method(method_name)
    spec = _solver_specs().get(solver_name)
    if spec is None:
        raise ValueError(f"Unknown solver: {solver_name}")

    task_id, pallet, boxes = request_to_models(request_dict, solver_name)

    if method_type == "solver":
        solution = spec.solve(
            task_id=task_id,
            pallet=pallet,
            boxes=boxes,
            request_dict=request_dict,
            n_restarts=n_restarts,
            time_budget_ms=time_budget_ms,
        )
        method_meta: Dict[str, Any] = {
            "type": "solver",
            "solver": solver_name,
            "n_restarts": n_restarts,
            "time_budget_ms": time_budget_ms,
        }
    else:
        if strategy_name not in spec.sort_keys:
            raise ValueError(f"Unknown strategy for {solver_name}: {strategy_name}")
        solution = spec.pack_greedy(
            task_id=task_id,
            pallet=pallet,
            boxes=boxes,
            sort_key_name=strategy_name,
        )
        method_meta = {
            "type": "strategy",
            "solver": solver_name,
            "strategy": strategy_name,
        }

    response_dict = spec.solution_to_dict(solution)
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
    n_restarts: int = 10,
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
        results = []
        for method_name in methods:
            results.append(
                run_method(
                    method_name=method_name,
                    task_id=request_dict["task_id"],
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
