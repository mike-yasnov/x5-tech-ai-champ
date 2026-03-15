"""Benchmark: run both solvers on all scenarios and compare scores.

Usage:
    python benchmark.py [--restarts N] [--output results.json] [--viz DIR]
    python benchmark.py --solver base       # run only base_solver
    python benchmark.py --solver alternative # run only alternative_solver
"""

import argparse
import json
import os
import time

from generator import generate_scenario
from validator import evaluate_solution

# Import both solvers under distinct names
from base_solver.models import Pallet as BasePallet, Box as BaseBox, solution_to_dict as base_solution_to_dict
from base_solver.solver import solve as base_solve

from alternative_solver.models import Pallet as AltPallet, Box as AltBox, solution_to_dict as alt_solution_to_dict
from alternative_solver.solver import solve as alt_solve


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

SCENARIOS = ORGANIZER_SCENARIOS + PROJECT_SCENARIOS


def _request_to_base_models(request_dict: dict):
    p = request_dict["pallet"]
    pallet = BasePallet(
        type_id=p.get("type_id", "unknown"),
        length_mm=p["length_mm"],
        width_mm=p["width_mm"],
        max_height_mm=p["max_height_mm"],
        max_weight_kg=p["max_weight_kg"],
    )
    boxes = []
    for b in request_dict["boxes"]:
        boxes.append(
            BaseBox(
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


def _request_to_alt_models(request_dict: dict):
    p = request_dict["pallet"]
    pallet = AltPallet(
        type_id=p.get("type_id", "unknown"),
        length_mm=p["length_mm"],
        width_mm=p["width_mm"],
        max_height_mm=p["max_height_mm"],
        max_weight_kg=p["max_weight_kg"],
    )
    boxes = []
    for b in request_dict["boxes"]:
        boxes.append(
            AltBox(
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


def run_single_solver(solver_name, solve_fn, to_models_fn, to_dict_fn,
                      request_dict, n_restarts, time_budget_ms):
    """Run a single solver on a single scenario and return result dict."""
    task_id, pallet, boxes = to_models_fn(request_dict)

    t0 = time.perf_counter()
    solution = solve_fn(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        n_restarts=n_restarts,
        time_budget_ms=time_budget_ms,
    )
    wall_time_ms = int((time.perf_counter() - t0) * 1000)

    response_dict = to_dict_fn(solution)
    eval_result = evaluate_solution(request_dict, response_dict)

    total_items = sum(b["quantity"] for b in request_dict["boxes"])

    return {
        "solver": solver_name,
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


SOLVERS = {
    "base": (base_solve, _request_to_base_models, base_solution_to_dict),
    "alternative": (alt_solve, _request_to_alt_models, alt_solution_to_dict),
}


def run_benchmark(n_restarts: int = 10, time_budget_ms: int = 5000,
                  solver_filter: str = "both") -> list:
    results = []

    solvers_to_run = []
    if solver_filter in ("both", "base"):
        solvers_to_run.append(("base", *SOLVERS["base"]))
    if solver_filter in ("both", "alternative"):
        solvers_to_run.append(("alternative", *SOLVERS["alternative"]))

    for scenario_type, seed in SCENARIOS:
        request_dict = generate_scenario(
            f"bench_{scenario_type}", scenario_type, seed=seed
        )

        for solver_name, solve_fn, to_models_fn, to_dict_fn in solvers_to_run:
            entry = run_single_solver(
                solver_name, solve_fn, to_models_fn, to_dict_fn,
                request_dict, n_restarts, time_budget_ms,
            )
            entry["scenario"] = scenario_type
            results.append(entry)

    return results


def format_markdown(results: list) -> str:
    lines = ["## Benchmark Results", ""]

    # Determine which solvers are present
    solver_names = sorted(set(r["solver"] for r in results))
    running_both = len(solver_names) > 1

    def render_table(title: str, rows: list) -> list:
        section = [f"### {title}", ""]

        if running_both:
            section.append(
                "| Scenario | Solver | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |"
            )
            section.append(
                "|----------|--------|-------|--------|----------|-----------|------------|--------|-----------|"
            )
        else:
            section.append(
                "| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |"
            )
            section.append(
                "|----------|-------|--------|----------|-----------|------------|--------|-----------|"
            )

        totals = {s: 0.0 for s in solver_names}
        counts = {s: 0 for s in solver_names}

        for r in rows:
            m = r.get("metrics", {})
            solver_col = f"| {r['solver']} " if running_both else ""

            if r["valid"]:
                section.append(
                    f"| {r['scenario']} "
                    f"{solver_col}"
                    f"| **{r['final_score']:.4f}** "
                    f"| {m.get('volume_utilization', 0):.4f} "
                    f"| {m.get('item_coverage', 0):.4f} "
                    f"| {m.get('fragility_score', 0):.4f} "
                    f"| {m.get('time_score', 0):.4f} "
                    f"| {r['placed']}/{r['total_items']} "
                    f"| {r['solve_time_ms']} |"
                )
                totals[r["solver"]] += r["final_score"]
                counts[r["solver"]] += 1
            else:
                section.append(
                    f"| {r['scenario']} "
                    f"{solver_col}"
                    f"| **INVALID** "
                    f"| - | - | - | - "
                    f"| {r['placed']}/{r['total_items']} "
                    f"| {r['solve_time_ms']} |"
                )

        section.append("")
        for s in solver_names:
            avg = totals[s] / counts[s] if counts[s] > 0 else 0
            label = f" ({s})" if running_both else ""
            section.append(f"**Average score{label}: {avg:.4f}**")
        section.append("")
        return section

    organizer_names = {name for name, _ in ORGANIZER_SCENARIOS}
    organizer_results = [r for r in results if r["scenario"] in organizer_names]
    project_results = [r for r in results if r["scenario"] not in organizer_names]

    # Sort: group by scenario, then by solver within scenario
    organizer_results.sort(key=lambda r: (r["scenario"], r["solver"]))
    project_results.sort(key=lambda r: (r["scenario"], r["solver"]))

    lines.extend(render_table("Organizer Scenarios", organizer_results))
    lines.extend(render_table("Diagnostic Scenarios", project_results))

    # Overall comparison
    if running_both:
        lines.append("### Comparison Summary")
        lines.append("")
        lines.append("| Solver | Avg Score (Organizer) | Avg Score (All) |")
        lines.append("|--------|---------------------|-----------------|")
        for s in solver_names:
            org_scores = [r["final_score"] for r in organizer_results
                         if r["solver"] == s and r["valid"]]
            all_scores = [r["final_score"] for r in results
                         if r["solver"] == s and r["valid"]]
            org_avg = sum(org_scores) / len(org_scores) if org_scores else 0
            all_avg = sum(all_scores) / len(all_scores) if all_scores else 0
            lines.append(f"| {s} | {org_avg:.4f} | {all_avg:.4f} |")
        lines.append("")

    # Constraint compliance table
    lines.append("### Constraint Compliance")
    lines.append("")
    solver_col_header = "| Solver " if running_both else ""
    solver_col_sep = "|--------" if running_both else ""
    lines.append(
        f"| Scenario {solver_col_header}| Bounds | Collision | Support 60% | Weight | Upright | Stackable | Fragility Viol. |"
    )
    lines.append(
        f"|----------{solver_col_sep}|--------|-----------|-------------|--------|---------|-----------|-----------------|"
    )
    for r in results:
        cc = r.get("constraint_checks", {})
        solver_col = f"| {r['solver']} " if running_both else ""

        if not r["valid"]:
            lines.append(f"| {r['scenario']} {solver_col}| FAIL | - | - | - | - | - | - |")
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
            f"{solver_col}"
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
            placements.append(
                {
                    "sku_id": p["sku_id"],
                    "x_mm": pos["x_mm"],
                    "y_mm": pos["y_mm"],
                    "z_mm": pos["z_mm"],
                    "length_mm": dim["length_mm"],
                    "width_mm": dim["width_mm"],
                    "height_mm": dim["height_mm"],
                    "fragile": sku.get("fragile", False),
                    "weight_kg": sku.get("weight_kg", 0),
                }
            )
        solver_label = f" [{r['solver']}]" if "solver" in r else ""
        viz.append(
            {
                "pallet": r.get("request_pallet", {}),
                "placements": placements,
                "meta": {
                    "scenario": r["scenario"] + solver_label,
                    "score": r.get("final_score", 0),
                    "placed": r.get("placed", 0),
                    "total_items": r.get("total_items", 0),
                },
            }
        )
    return viz


def main():
    parser = argparse.ArgumentParser(description="Benchmark 3D Pallet Packing Solvers")
    parser.add_argument(
        "--restarts", type=int, default=10, help="Number of restarts (default: 10)"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Save detailed results to JSON file"
    )
    parser.add_argument(
        "--viz", default=None, help="Generate 3D visualization HTML files to directory"
    )
    parser.add_argument(
        "--solver", default="both",
        choices=["both", "base", "alternative"],
        help="Which solver to run (default: both)"
    )
    args = parser.parse_args()

    results = run_benchmark(
        n_restarts=args.restarts,
        solver_filter=args.solver,
    )
    md = format_markdown(results)
    print(md)

    if args.output:
        slim_results = [
            {
                k: v
                for k, v in r.items()
                if k not in ("response", "request_pallet", "request_boxes")
            }
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
