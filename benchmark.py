"""Benchmark: run solver on all scenarios and report scores.

Usage:
    python benchmark.py [--restarts N] [--output results.json] [--viz out_dir]

Outputs a markdown table and optionally a JSON file with detailed results.
"""

import argparse
import json
import os
import time

from generator import generate_scenario
from solver.heuristics import STRATEGY_CONFIGS
from solver.models import Box, Pallet, solution_to_dict
from solver.solver import solve
from validator import evaluate_solution


ORGANIZER_SCENARIOS = [
    ("heavy_water", 42),
    ("fragile_tower", 43),
    ("liquid_tetris", 44),
    ("random_mixed", 45),
    ("exact_fit", 46),
    ("fragile_mix", 47),
    ("support_tetris", 48),
    ("cavity_fill", 49),
]

PROJECT_SCENARIOS = [
    ("exact_fit", 46),
    ("fragile_mix", 47),
    ("support_tetris", 48),
    ("cavity_fill", 49),
]

SCENARIOS = ORGANIZER_SCENARIOS + PROJECT_SCENARIOS


def _request_to_models(request_dict: dict):
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


def run_benchmark(n_restarts: int | None = None, time_budget_ms: int = 5000) -> list:
    results = []
    if n_restarts is None:
        n_restarts = len(STRATEGY_CONFIGS)

    for scenario_type, seed in SCENARIOS:
        request_dict = generate_scenario(
            f"bench_{scenario_type}", scenario_type, seed=seed
        )
        task_id, pallet, boxes = _request_to_models(request_dict)

        t0 = time.perf_counter()
        solution = solve(
            task_id=task_id,
            pallet=pallet,
            boxes=boxes,
            request_dict=request_dict,
            n_restarts=n_restarts,
            time_budget_ms=time_budget_ms,
        )
        wall_time_ms = int((time.perf_counter() - t0) * 1000)

        response_dict = solution_to_dict(solution)
        eval_result = evaluate_solution(request_dict, response_dict)
        total_items = sum(box["quantity"] for box in request_dict["boxes"])

        results.append(
            {
                "scenario": scenario_type,
                "valid": eval_result.get("valid", False),
                "final_score": eval_result.get("final_score", 0.0),
                "metrics": eval_result.get("metrics", {}),
                "placed": len(solution.placements),
                "total_items": total_items,
                "solve_time_ms": solution.solve_time_ms,
                "wall_time_ms": wall_time_ms,
                "error": eval_result.get("error"),
                "response": response_dict,
                "request_pallet": request_dict["pallet"],
                "request_boxes": request_dict["boxes"],
            }
        )

    return results


def format_markdown(results: list) -> str:
    lines = ["## Benchmark Results", ""]

    def render_table(title: str, rows: list) -> list:
        section = [f"### {title}", ""]
        section.append(
            "| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |"
        )
        section.append(
            "|----------|-------|--------|----------|-----------|------------|--------|-----------|"
        )

        total_score = 0.0
        for row in rows:
            metrics = row.get("metrics", {})
            if row["valid"]:
                section.append(
                    f"| {row['scenario']} "
                    f"| **{row['final_score']:.4f}** "
                    f"| {metrics.get('volume_utilization', 0):.4f} "
                    f"| {metrics.get('item_coverage', 0):.4f} "
                    f"| {metrics.get('fragility_score', 0):.4f} "
                    f"| {metrics.get('time_score', 0):.4f} "
                    f"| {row['placed']}/{row['total_items']} "
                    f"| {row['solve_time_ms']} |"
                )
                total_score += row["final_score"]
            else:
                section.append(
                    f"| {row['scenario']} "
                    f"| **INVALID** "
                    f"| - | - | - | - "
                    f"| {row['placed']}/{row['total_items']} "
                    f"| {row['solve_time_ms']} |"
                )

        average = total_score / len(rows) if rows else 0
        section.append("")
        section.append(f"**Average score: {average:.4f}**")
        section.append("")
        return section

    organizer_names = {name for name, _ in ORGANIZER_SCENARIOS}
    organizer_results = [row for row in results if row["scenario"] in organizer_names]
    project_results = [row for row in results if row["scenario"] not in organizer_names]

    lines.extend(render_table("Сценарии организаторов", organizer_results))
    lines.extend(render_table("Наши synthetic/diagnostic сценарии", project_results))

    overall_average = (
        sum(row["final_score"] for row in results if row["valid"]) / len(results)
        if results
        else 0
    )
    lines.append(f"**Overall average: {overall_average:.4f}**")
    return "\n".join(lines)


def build_viz_data(results: list) -> list:
    """Extract visualization data from benchmark results."""
    visualizations = []
    for row in results:
        boxes_meta = {box["sku_id"]: box for box in row.get("request_boxes", [])}
        placements = []
        for placement in row.get("response", {}).get("placements", []):
            sku = boxes_meta.get(placement["sku_id"], {})
            dimensions = placement["dimensions_placed"]
            position = placement["position"]
            placements.append(
                {
                    "sku_id": placement["sku_id"],
                    "x_mm": position["x_mm"],
                    "y_mm": position["y_mm"],
                    "z_mm": position["z_mm"],
                    "length_mm": dimensions["length_mm"],
                    "width_mm": dimensions["width_mm"],
                    "height_mm": dimensions["height_mm"],
                    "fragile": sku.get("fragile", False),
                }
            )
        visualizations.append(
            {
                "pallet": row.get("request_pallet", {}),
                "placements": placements,
                "meta": {
                    "scenario": row["scenario"],
                    "score": row.get("final_score", 0),
                    "placed": row.get("placed", 0),
                    "total_items": row.get("total_items", 0),
                },
            }
        )
    return visualizations


def main():
    parser = argparse.ArgumentParser(description="Benchmark 3D Pallet Packing Solver")
    parser.add_argument(
        "--restarts", type=int, default=len(STRATEGY_CONFIGS), help="Number of restarts"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Save detailed results to JSON file"
    )
    parser.add_argument(
        "--viz", default=None, help="Generate 3D visualization HTML files to directory"
    )
    args = parser.parse_args()

    results = run_benchmark(n_restarts=args.restarts)
    markdown = format_markdown(results)
    print(markdown)

    if args.output:
        slim_results = [
            {
                key: value
                for key, value in row.items()
                if key not in ("response", "request_pallet", "request_boxes")
            }
            for row in results
        ]
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(slim_results, file, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {args.output}")

    if args.viz:
        from visualize import generate_html_files

        viz_data = build_viz_data(results)
        viz_json_path = os.path.join(args.viz, "benchmark_viz.json")
        os.makedirs(args.viz, exist_ok=True)
        with open(viz_json_path, "w", encoding="utf-8") as file:
            json.dump(viz_data, file, ensure_ascii=False)
        files = generate_html_files(viz_json_path, args.viz)
        print(f"\nGenerated {len(files)} 3D visualization(s) in {args.viz}/")


if __name__ == "__main__":
    main()
