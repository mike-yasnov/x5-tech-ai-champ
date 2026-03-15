"""Benchmark: compare RL/metaheuristic agents against the original solver.

Runs on hard scenarios for development, but FINAL scoring is on 4 organizer scenarios.

Usage:
    python -m rl_solver.rl_benchmark [--agents ga,sa] [--include-original]
    python -m rl_solver.rl_benchmark --final-only  # only 4 organizer tests
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_solver.scenarios.hard_scenarios import get_all_scenarios

logger = logging.getLogger(__name__)


def _validate_solution(scenario: Dict, response: Dict) -> Dict:
    """Run validator on solution."""
    from validator import evaluate_solution
    return evaluate_solution(scenario, response)


def benchmark_ga(scenarios: List[Dict]) -> List[Dict]:
    """Run GA on all scenarios."""
    from rl_solver.agents.genetic_agent import GeneticSolver
    solver = GeneticSolver(pop_size=100, n_generations=300, time_budget_sec=25.0)
    results = []
    for sc in scenarios:
        t0 = time.perf_counter()
        response = solver.solve(sc)
        wall_ms = int((time.perf_counter() - t0) * 1000)
        eval_result = _validate_solution(sc, response)
        results.append({
            "scenario": sc["task_id"],
            "agent": "ga",
            "valid": eval_result.get("valid", False),
            "score": eval_result.get("final_score", 0.0),
            "metrics": eval_result.get("metrics", {}),
            "placed": len(response["placements"]),
            "total": sum(b["quantity"] for b in sc["boxes"]),
            "wall_ms": wall_ms,
            "error": eval_result.get("error"),
        })
        status = "VALID" if results[-1]["valid"] else f"INVALID: {results[-1]['error']}"
        logger.info("GA %s: %s score=%.4f placed=%d/%d time=%dms",
                     sc["task_id"], status, results[-1]["score"],
                     results[-1]["placed"], results[-1]["total"], wall_ms)
    return results


def benchmark_sa(scenarios: List[Dict]) -> List[Dict]:
    """Run SA on all scenarios."""
    from rl_solver.agents.sa_agent import SimulatedAnnealingSolver
    solver = SimulatedAnnealingSolver(time_budget_sec=25.0, n_restarts=3)
    results = []
    for sc in scenarios:
        t0 = time.perf_counter()
        response = solver.solve(sc)
        wall_ms = int((time.perf_counter() - t0) * 1000)
        eval_result = _validate_solution(sc, response)
        results.append({
            "scenario": sc["task_id"],
            "agent": "sa",
            "valid": eval_result.get("valid", False),
            "score": eval_result.get("final_score", 0.0),
            "metrics": eval_result.get("metrics", {}),
            "placed": len(response["placements"]),
            "total": sum(b["quantity"] for b in sc["boxes"]),
            "wall_ms": wall_ms,
            "error": eval_result.get("error"),
        })
        status = "VALID" if results[-1]["valid"] else f"INVALID: {results[-1]['error']}"
        logger.info("SA %s: %s score=%.4f placed=%d/%d time=%dms",
                     sc["task_id"], status, results[-1]["score"],
                     results[-1]["placed"], results[-1]["total"], wall_ms)
    return results


def benchmark_original(scenarios: List[Dict]) -> List[Dict]:
    """Run original solver on hard scenarios."""
    from solver.models import Pallet, Box, solution_to_dict
    from solver.solver import solve

    results = []
    for sc in scenarios:
        p = sc["pallet"]
        pallet = Pallet(
            type_id=p.get("type_id", "unknown"),
            length_mm=p["length_mm"], width_mm=p["width_mm"],
            max_height_mm=p["max_height_mm"], max_weight_kg=p["max_weight_kg"],
        )
        boxes = []
        for b in sc["boxes"]:
            boxes.append(Box(
                sku_id=b["sku_id"], description=b.get("description", ""),
                length_mm=b["length_mm"], width_mm=b["width_mm"], height_mm=b["height_mm"],
                weight_kg=b["weight_kg"], quantity=b["quantity"],
                strict_upright=b.get("strict_upright", False),
                fragile=b.get("fragile", False), stackable=b.get("stackable", True),
            ))

        t0 = time.perf_counter()
        solution = solve(
            task_id=sc["task_id"], pallet=pallet, boxes=boxes,
            request_dict=sc, n_restarts=30, time_budget_ms=900,
        )
        wall_ms = int((time.perf_counter() - t0) * 1000)

        response = solution_to_dict(solution)
        eval_result = _validate_solution(sc, response)
        results.append({
            "scenario": sc["task_id"],
            "agent": "original",
            "valid": eval_result.get("valid", False),
            "score": eval_result.get("final_score", 0.0),
            "metrics": eval_result.get("metrics", {}),
            "placed": len(solution.placements),
            "total": sum(b["quantity"] for b in sc["boxes"]),
            "wall_ms": wall_ms,
            "error": eval_result.get("error"),
        })
        logger.info("Original %s: score=%.4f placed=%d/%d time=%dms",
                     sc["task_id"], results[-1]["score"],
                     results[-1]["placed"], results[-1]["total"], wall_ms)
    return results


def format_comparison(all_results: Dict[str, List[Dict]]) -> str:
    """Format comparison table."""
    lines = ["## RL Benchmark: Hard Scenarios Comparison", ""]

    # Collect all scenario names
    scenario_names = []
    for results in all_results.values():
        for r in results:
            if r["scenario"] not in scenario_names:
                scenario_names.append(r["scenario"])

    # Header
    agents = list(all_results.keys())
    header = "| Scenario |"
    sep = "|----------|"
    for a in agents:
        header += f" {a} Score | {a} Placed |"
        sep += "----------|----------|"
    lines.append(header)
    lines.append(sep)

    # Rows
    agent_scores = {a: [] for a in agents}
    for sc_name in scenario_names:
        row = f"| {sc_name} |"
        for a in agents:
            r = next((x for x in all_results[a] if x["scenario"] == sc_name), None)
            if r and r["valid"]:
                row += f" **{r['score']:.4f}** | {r['placed']}/{r['total']} |"
                agent_scores[a].append(r["score"])
            elif r:
                row += f" INVALID | {r['placed']}/{r['total']} |"
                agent_scores[a].append(0.0)
            else:
                row += " - | - |"
        lines.append(row)

    # Averages
    lines.append("")
    for a in agents:
        scores = agent_scores[a]
        avg = np.mean(scores) if scores else 0
        lines.append(f"**{a} average: {avg:.4f}** ({len(scores)} scenarios)")

    return "\n".join(lines)


def get_organizer_scenarios() -> List[Dict]:
    """Get the 4 organizer scenarios used for FINAL scoring."""
    from generator import generate_scenario
    ORGANIZER = [
        ("heavy_water", 42),
        ("fragile_tower", 43),
        ("liquid_tetris", 44),
        ("random_mixed", 45),
    ]
    return [
        generate_scenario(f"bench_{name}", name, seed=seed)
        for name, seed in ORGANIZER
    ]


def main():
    parser = argparse.ArgumentParser(description="Benchmark RL agents")
    parser.add_argument("--agents", default="ga,sa", help="Comma-separated agent list")
    parser.add_argument("--include-original", action="store_true", help="Include original solver")
    parser.add_argument("--final-only", action="store_true",
                        help="Only run on 4 organizer scenarios (FINAL scoring)")
    parser.add_argument("--output", "-o", default=None, help="Save results JSON")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.final_only:
        scenarios = get_organizer_scenarios()
        logger.info("FINAL SCORING: running on %d organizer scenarios", len(scenarios))
    else:
        scenarios = get_all_scenarios()
        logger.info("Running benchmark on %d hard scenarios", len(scenarios))

    all_results = {}
    agent_names = [a.strip() for a in args.agents.split(",")]

    if args.include_original:
        agent_names.insert(0, "original")

    for agent_name in agent_names:
        logger.info("=" * 50)
        logger.info("Benchmarking: %s", agent_name.upper())
        logger.info("=" * 50)

        if agent_name == "ga":
            all_results["ga"] = benchmark_ga(scenarios)
        elif agent_name == "sa":
            all_results["sa"] = benchmark_sa(scenarios)
        elif agent_name == "original":
            all_results["original"] = benchmark_original(scenarios)

    md = format_comparison(all_results)
    print(md)

    # If running on hard scenarios, also show final scoring on organizer tests
    if not args.final_only:
        print("\n" + "=" * 60)
        print("FINAL SCORING (4 organizer scenarios)")
        print("=" * 60)
        org_scenarios = get_organizer_scenarios()
        org_results = {}
        for agent_name in agent_names:
            if agent_name == "ga":
                org_results["ga"] = benchmark_ga(org_scenarios)
            elif agent_name == "sa":
                org_results["sa"] = benchmark_sa(org_scenarios)
            elif agent_name == "original":
                org_results["original"] = benchmark_original(org_scenarios)
        print(format_comparison(org_results))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
