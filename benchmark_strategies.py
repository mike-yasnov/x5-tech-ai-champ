"""Compare baseline sort heuristics across generated scenarios."""

import argparse
import json
from typing import Any, Dict, List

from generator import generate_scenario
from solver.models import Box, Pallet, solution_to_dict
from solver.packer import SORT_KEYS, pack_greedy
from validator import evaluate_solution


SCENARIOS = [
    ("heavy_water", 42),
    ("fragile_tower", 43),
    ("liquid_tetris", 44),
    ("random_mixed", 45),
    ("exact_fit", 46),
    ("fragile_mix", 47),
    ("support_tetris", 48),
    ("cavity_fill", 49),
    ("count_preference", 50),
]


def request_to_models(request_dict: dict):
    pallet_data = request_dict["pallet"]
    pallet = Pallet(
        type_id=pallet_data.get("type_id", "unknown"),
        length_mm=pallet_data["length_mm"],
        width_mm=pallet_data["width_mm"],
        max_height_mm=pallet_data["max_height_mm"],
        max_weight_kg=pallet_data["max_weight_kg"],
    )
    boxes: List[Box] = []
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


def benchmark_scenario(scenario_name: str, seed: int) -> Dict[str, Any]:
    request_dict = generate_scenario(f"bench_{scenario_name}", scenario_name, seed=seed)
    task_id, pallet, boxes = request_to_models(request_dict)

    rows: List[Dict[str, Any]] = []
    for strategy_name in SORT_KEYS:
        solution = pack_greedy(task_id, pallet, boxes, sort_key_name=strategy_name)
        result = evaluate_solution(request_dict, solution_to_dict(solution))
        rows.append(
            {
                "strategy": strategy_name,
                "valid": result.get("valid", False),
                "final_score": result.get("final_score", 0.0),
                "metrics": result.get("metrics", {}),
                "placed": len(solution.placements),
                "time_ms": solution.solve_time_ms,
                "error": result.get("error"),
            }
        )

    rows.sort(
        key=lambda row: (row["valid"], row["final_score"], -row["time_ms"]),
        reverse=True,
    )
    return {"scenario": scenario_name, "results": rows}


def render_markdown(report: List[Dict[str, Any]], limit: int) -> str:
    lines = ["## Strategy Benchmark", ""]
    for scenario in report:
        lines.append(f"### {scenario['scenario']}")
        lines.append("")
        lines.append("| Strategy | Score | Valid | Placed | Time (ms) |")
        lines.append("|----------|-------|-------|--------|-----------|")
        for row in scenario["results"][:limit]:
            lines.append(
                f"| {row['strategy']} | {row['final_score']:.4f} | {row['valid']} | {row['placed']} | {row['time_ms']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline sort heuristics")
    parser.add_argument(
        "--scenario", choices=[name for name, _ in SCENARIOS] + ["all"], default="all"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="How many top strategies to print"
    )
    parser.add_argument(
        "--output", default=None, help="Optional path to save JSON report"
    )
    args = parser.parse_args()

    selected = (
        SCENARIOS
        if args.scenario == "all"
        else [item for item in SCENARIOS if item[0] == args.scenario]
    )
    report = [benchmark_scenario(name, seed) for name, seed in selected]
    markdown = render_markdown(report, args.limit)
    print(markdown)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
