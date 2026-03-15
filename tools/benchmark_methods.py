"""Benchmark multiple solving methods and render leaderboards."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from .benchmark_utils import (
    resolve_methods,
    run_benchmark_methods,
    summarize_methods,
    write_json,
)


def render_markdown(report: Dict[str, Any], limit: int = 0) -> str:
    lines = ["## Method Benchmark", ""]

    summary = summarize_methods(report)
    lines.append("### Overall Leaderboard")
    lines.append("")
    lines.append(
        "| Method | Official Avg | Quality Avg | Invalid | Coverage | Avg Time (ms) |"
    )
    lines.append(
        "|--------|--------------|-------------|---------|----------|---------------|"
    )
    rows = summary if limit <= 0 else summary[:limit]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['official_score']:.4f} | {row['quality_score']:.4f} | "
            f"{row['invalid_count']} | {row['item_coverage']:.4f} | {row['avg_solve_time_ms']:.1f} |"
        )

    for scenario in report.get("scenarios", []):
        lines.extend(["", f"### {scenario['scenario']}", ""])
        lines.append(
            "| Method | Official | Quality | Valid | Placed | Time (ms) | Error |"
        )
        lines.append(
            "|--------|----------|---------|-------|--------|-----------|-------|"
        )
        scenario_rows: List[Dict[str, Any]] = sorted(
            scenario["results"],
            key=lambda item: (
                item["valid"],
                item["quality_score"],
                item["official_score"],
            ),
            reverse=True,
        )
        scenario_rows = scenario_rows if limit <= 0 else scenario_rows[:limit]
        for row in scenario_rows:
            error = row["error"] or ""
            lines.append(
                f"| {row['method']} | {row['official_score']:.4f} | {row['quality_score']:.4f} | "
                f"{row['valid']} | {row['placed']}/{row['total_items']} | {row['solve_time_ms']} | {error} |"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple packing methods")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["solver:base", "solver:alternative"],
        help="Method names to benchmark; use 'all' to include both solvers and every greedy strategy",
    )
    parser.add_argument(
        "--scenario-set",
        choices=["smoke", "organizers", "project", "full"],
        default="full",
        help="Scenario group to benchmark",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=10,
        help="Restart count for solver methods",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=5000,
        help="Per-scenario time budget in ms for solver methods",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit rows in markdown output"
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--markdown-output", default=None, help="Optional markdown output path"
    )
    args = parser.parse_args()

    report = run_benchmark_methods(
        methods=resolve_methods(args.methods),
        scenario_set=args.scenario_set,
        n_restarts=args.restarts,
        time_budget_ms=args.time_budget,
    )
    markdown = render_markdown(report, limit=args.limit)
    print(markdown)

    if args.output:
        write_json(args.output, report)

    if args.markdown_output:
        with open(args.markdown_output, "w", encoding="utf-8") as file:
            file.write(markdown)


if __name__ == "__main__":
    main()
