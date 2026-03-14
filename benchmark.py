"""Benchmark: run solver on all scenarios and report scores.

Usage:
    python benchmark.py [--restarts N] [--output results.json]

Outputs a markdown table and optionally a JSON file with detailed results.
"""

import argparse
import json
import os
import sys
import time

from generator import generate_scenario
from solver.packer import SORT_KEYS
from validator import evaluate_solution
from solver.models import Pallet, Box, solution_to_dict
from solver.solver import solve


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
]

SCENARIOS = ORGANIZER_SCENARIOS + PROJECT_SCENARIOS


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


def run_benchmark(n_restarts: int = 10, time_budget_ms: int = 5000) -> list:
    results = []
    if n_restarts is None:
        n_restarts = len(SORT_KEYS)

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

        total_items = sum(b["quantity"] for b in request_dict["boxes"])

        entry = {
            "scenario": scenario_type,
            "valid": eval_result.get("valid", False),
            "final_score": eval_result.get("final_score", 0.0),
            "metrics": eval_result.get("metrics", {}),
            "constraint_checks": eval_result.get("constraint_checks", {}),
            "placed": len(solution.placements),
            "total_items": total_items,
            "solve_time_ms": solution.solve_time_ms,
            "wall_time_ms": wall_time_ms,
            "error": eval_result.get("error"),
            "response": response_dict,
            "request_pallet": request_dict["pallet"],
            "request_boxes": request_dict["boxes"],
        }
        results.append(entry)

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
        for r in rows:
            m = r.get("metrics", {})
            if r["valid"]:
                section.append(
                    f"| {r['scenario']} "
                    f"| **{r['final_score']:.4f}** "
                    f"| {m.get('volume_utilization', 0):.4f} "
                    f"| {m.get('item_coverage', 0):.4f} "
                    f"| {m.get('fragility_score', 0):.4f} "
                    f"| {m.get('time_score', 0):.4f} "
                    f"| {r['placed']}/{r['total_items']} "
                    f"| {r['solve_time_ms']} |"
                )
                total_score += r["final_score"]
            else:
                section.append(
                    f"| {r['scenario']} "
                    f"| **INVALID** "
                    f"| - | - | - | - "
                    f"| {r['placed']}/{r['total_items']} "
                    f"| {r['solve_time_ms']} |"
                )

        avg = total_score / len(rows) if rows else 0
        section.append("")
        section.append(f"**Average score: {avg:.4f}**")
        section.append("")
        return section

    organizer_names = {name for name, _ in ORGANIZER_SCENARIOS}
    organizer_results = [r for r in results if r["scenario"] in organizer_names]
    project_results = [r for r in results if r["scenario"] not in organizer_names]

    lines.extend(render_table("Сценарии организаторов", organizer_results))
    lines.extend(render_table("Наши synthetic/diagnostic сценарии", project_results))

    overall_avg = (
        sum(r["final_score"] for r in results if r["valid"]) / len(results)
        if results
        else 0
    )
    lines.append(f"**Overall average: {overall_avg:.4f}**")

    # Constraint compliance table
    lines.append("")
    lines.append("### Constraint Compliance")
    lines.append("")
    lines.append("| Scenario | Bounds | Collision | Support 60% | Weight | Upright | Stackable | Fragility Viol. |")
    lines.append("|----------|--------|-----------|-------------|--------|---------|-----------|-----------------|")
    for r in results:
        cc = r.get("constraint_checks", {})
        if not r["valid"]:
            lines.append(f"| {r['scenario']} | FAIL | - | - | - | - | - | - |")
            continue

        def _fmt(v):
            if v is True:
                return "PASS"
            if v == "pass":
                return "PASS"
            if v == "n/a":
                return "n/a"
            return str(v)

        lines.append(
            f"| {r['scenario']} "
            f"| {_fmt(cc.get('bounds', '?'))} "
            f"| {_fmt(cc.get('no_collision', '?'))} "
            f"| {_fmt(cc.get('support_60pct', '?'))} "
            f"| {_fmt(cc.get('weight_limit', '?'))} "
            f"| {_fmt(cc.get('strict_upright', '?'))} "
            f"| {_fmt(cc.get('stackable', '?'))} "
            f"| {cc.get('fragility_violations', '?')} |"
        )
    lines.append("")

    return "\n".join(lines)


def build_viz_data(results: list) -> list:
    """Extract visualization data from benchmark results."""
    viz = []
    for r in results:
        boxes_meta = {b["sku_id"]: b for b in r.get("request_boxes", [])}
        placements = []
        for p in r.get("response", {}).get("placements", []):
            sku = boxes_meta.get(p["sku_id"], {})
            dim = p["dimensions_placed"]
            pos = p["position"]
            placements.append({
                "sku_id": p["sku_id"],
                "x_mm": pos["x_mm"],
                "y_mm": pos["y_mm"],
                "z_mm": pos["z_mm"],
                "length_mm": dim["length_mm"],
                "width_mm": dim["width_mm"],
                "height_mm": dim["height_mm"],
                "fragile": sku.get("fragile", False),
            })
        viz.append({
            "pallet": r.get("request_pallet", {}),
            "placements": placements,
            "meta": {
                "scenario": r["scenario"],
                "score": r.get("final_score", 0),
                "placed": r.get("placed", 0),
                "total_items": r.get("total_items", 0),
            },
        })
    return viz


def main():
    parser = argparse.ArgumentParser(description="Benchmark 3D Pallet Packing Solver")
    parser.add_argument("--restarts", type=int, default=10, help="Number of restarts (default: 10)")
    parser.add_argument("--output", "-o", default=None, help="Save detailed results to JSON file")
    parser.add_argument("--viz", default=None, help="Generate 3D visualization HTML files to directory")
    args = parser.parse_args()

    results = run_benchmark(n_restarts=args.restarts)
    md = format_markdown(results)
    print(md)

    if args.output:
        # Save results without bulky response/request data
        slim_results = [
            {k: v for k, v in r.items() if k not in ("response", "request_pallet", "request_boxes")}
            for r in results
        ]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(slim_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {args.output}")

    if args.viz:
        from visualize import generate_html_files
        viz_data = build_viz_data(results)
        viz_json_path = os.path.join(args.viz, "benchmark_viz.json")
        os.makedirs(args.viz, exist_ok=True)
        with open(viz_json_path, "w", encoding="utf-8") as f:
            json.dump(viz_data, f, ensure_ascii=False)
        files = generate_html_files(viz_json_path, args.viz)
        print(f"\nGenerated {len(files)} 3D visualization(s) in {args.viz}/")


if __name__ == "__main__":
    main()
