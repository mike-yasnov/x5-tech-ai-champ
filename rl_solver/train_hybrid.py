"""Train hybrid RL agent + benchmark GA/SA with postprocessing.

Usage:
    python -m rl_solver.train_hybrid --episodes 300 --device cuda
    python -m rl_solver.train_hybrid --benchmark-only
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_solver.scenarios.hard_scenarios import get_all_scenarios
from rl_solver.agents.hybrid_rl_agent import HybridPackingEnv, HybridPPOAgent
from rl_solver.postprocess_wrapper import apply_postprocessing

logger = logging.getLogger(__name__)


def get_organizer_scenarios():
    from generator import generate_scenario
    ORG = [("heavy_water", 42), ("fragile_tower", 43), ("liquid_tetris", 44), ("random_mixed", 45)]
    return [generate_scenario(f"bench_{n}", n, seed=s) for n, s in ORG]


def train_hybrid(scenarios: List[Dict], episodes: int, device: str, save_dir: str) -> Dict:
    """Train hybrid RL agent."""
    env = HybridPackingEnv(scenarios, max_items=200)

    agent = HybridPPOAgent(
        item_feat_dim=8, context_dim=16, hidden=128,
        lr=3e-4, gamma=0.99, clip_eps=0.2,
        entropy_coef=0.05, value_coef=0.5, n_epochs=4,
        device=device,
    )

    scores_history = []
    best_avg_score = 0.0

    for ep in range(episodes):
        items, ctx, mask = env.reset()
        trajectory = []
        done = False

        while not done:
            action, log_prob, value = agent.select_action(items, ctx, mask)
            trajectory.append({
                "items": items.copy(), "context": ctx.copy(), "mask": mask.copy(),
                "action": action, "log_prob": log_prob, "value": value,
            })
            items, ctx, mask, done = env.step(action)

        # Evaluate: pack in chosen order + postprocess
        _, score = env.evaluate_ordering()
        scores_history.append(score)

        # Train on episode
        metrics = agent.train_on_episode(trajectory, score)

        if (ep + 1) % 5 == 0:
            avg = np.mean(scores_history[-10:])
            logger.info(
                "HybridRL ep %d/%d: score=%.4f, avg10=%.4f, loss=%.4f, ent=%.3f",
                ep + 1, episodes, score, avg,
                metrics.get("loss", 0), metrics.get("entropy", 0),
            )
            if avg > best_avg_score:
                best_avg_score = avg
                agent.save(os.path.join(save_dir, "hybrid_best.pt"))

    agent.save(os.path.join(save_dir, "hybrid_final.pt"))
    return {
        "agent": "hybrid_rl",
        "episodes": episodes,
        "final_avg_score": float(np.mean(scores_history[-20:])),
        "best_avg_score": float(best_avg_score),
    }


def benchmark_with_postprocess(scenarios: List[Dict], label: str = "") -> List[Dict]:
    """Benchmark GA + SA with postprocessing applied."""
    from rl_solver.agents.genetic_agent import GeneticSolver
    from rl_solver.agents.sa_agent import SimulatedAnnealingSolver
    from validator import evaluate_solution

    results = []

    # Original solver
    from solver.models import Pallet, Box, solution_to_dict
    from solver.solver import solve

    for sc in scenarios:
        p = sc["pallet"]
        pallet = Pallet(type_id=p.get("type_id", ""), length_mm=p["length_mm"],
                        width_mm=p["width_mm"], max_height_mm=p["max_height_mm"],
                        max_weight_kg=p["max_weight_kg"])
        boxes = [Box(sku_id=b["sku_id"], description=b.get("description", ""),
                     length_mm=b["length_mm"], width_mm=b["width_mm"], height_mm=b["height_mm"],
                     weight_kg=b["weight_kg"], quantity=b["quantity"],
                     strict_upright=b.get("strict_upright", False),
                     fragile=b.get("fragile", False), stackable=b.get("stackable", True))
                 for b in sc["boxes"]]
        sol = solve(task_id=sc["task_id"], pallet=pallet, boxes=boxes,
                    request_dict=sc, n_restarts=30, time_budget_ms=900)
        r = evaluate_solution(sc, solution_to_dict(sol))
        results.append({"scenario": sc["task_id"], "agent": "original",
                         "score": r.get("final_score", 0), "valid": r.get("valid", False)})

    # GA + postprocess
    ga = GeneticSolver(pop_size=100, n_generations=300, time_budget_sec=25.0)
    for sc in scenarios:
        raw = ga.solve(sc)
        improved = apply_postprocessing(raw, sc, time_budget_ms=5000)
        r = evaluate_solution(sc, improved)
        results.append({"scenario": sc["task_id"], "agent": "ga+post",
                         "score": r.get("final_score", 0), "valid": r.get("valid", False),
                         "error": r.get("error")})

    # SA + postprocess
    sa = SimulatedAnnealingSolver(time_budget_sec=25.0, n_restarts=3)
    for sc in scenarios:
        raw = sa.solve(sc)
        improved = apply_postprocessing(raw, sc, time_budget_ms=5000)
        r = evaluate_solution(sc, improved)
        results.append({"scenario": sc["task_id"], "agent": "sa+post",
                         "score": r.get("final_score", 0), "valid": r.get("valid", False),
                         "error": r.get("error")})

    return results


def benchmark_hybrid(scenarios: List[Dict], model_path: str, device: str) -> List[Dict]:
    """Benchmark trained hybrid RL agent."""
    from validator import evaluate_solution

    env = HybridPackingEnv(scenarios, max_items=200)
    agent = HybridPPOAgent(device=device)

    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        logger.warning("No hybrid model at %s, using random policy", model_path)

    results = []
    for i, sc in enumerate(scenarios):
        items, ctx, mask = env.reset(scenario_idx=i)
        done = False
        while not done:
            action, _, _ = agent.select_action(items, ctx, mask)
            items, ctx, mask, done = env.step(action)

        response, score = env.evaluate_ordering()
        r = evaluate_solution(sc, response)
        results.append({
            "scenario": sc["task_id"],
            "agent": "hybrid_rl",
            "score": r.get("final_score", 0) if r.get("valid") else 0,
            "valid": r.get("valid", False),
            "error": r.get("error"),
        })
        logger.info("HybridRL %s: score=%.4f valid=%s", sc["task_id"], results[-1]["score"], results[-1]["valid"])

    return results


def print_comparison(all_results: List[Dict], scenarios: List[Dict]):
    """Pretty-print comparison table."""
    agents = sorted(set(r["agent"] for r in all_results))
    sc_names = [sc["task_id"] for sc in scenarios]

    print(f"\n{'Scenario':<25}", end="")
    for a in agents:
        print(f" | {a:<12}", end="")
    print()
    print("-" * (25 + 15 * len(agents)))

    agent_totals = {a: [] for a in agents}
    for sc_name in sc_names:
        print(f"{sc_name:<25}", end="")
        for a in agents:
            r = next((x for x in all_results if x["scenario"] == sc_name and x["agent"] == a), None)
            if r and r.get("valid"):
                print(f" | {r['score']:.4f}      ", end="")
                agent_totals[a].append(r["score"])
            elif r:
                print(f" | INVALID     ", end="")
                agent_totals[a].append(0)
            else:
                print(f" | -           ", end="")
        print()

    print("-" * (25 + 15 * len(agents)))
    print(f"{'AVERAGE':<25}", end="")
    for a in agents:
        scores = agent_totals[a]
        avg = np.mean(scores) if scores else 0
        print(f" | {avg:.4f}      ", end="")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="rl_solver/models")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    os.makedirs(args.save_dir, exist_ok=True)
    org_scenarios = get_organizer_scenarios()
    hard_scenarios = get_all_scenarios()

    # Train hybrid RL
    if not args.benchmark_only:
        logger.info("=" * 60)
        logger.info("Training Hybrid RL on %d hard scenarios", len(hard_scenarios))
        logger.info("=" * 60)
        t0 = time.perf_counter()
        result = train_hybrid(hard_scenarios, args.episodes, args.device, args.save_dir)
        result["wall_time_sec"] = round(time.perf_counter() - t0, 1)
        logger.info("Training result: %s", result)
        with open(os.path.join(args.save_dir, "hybrid_training.json"), "w") as f:
            json.dump(result, f, indent=2)

    # Benchmark on organizer scenarios
    logger.info("=" * 60)
    logger.info("FINAL BENCHMARK on 4 organizer scenarios")
    logger.info("=" * 60)

    all_results = benchmark_with_postprocess(org_scenarios)

    # Add hybrid RL
    model_path = os.path.join(args.save_dir, "hybrid_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.save_dir, "hybrid_final.pt")
    hybrid_results = benchmark_hybrid(org_scenarios, model_path, args.device)
    all_results.extend(hybrid_results)

    print_comparison(all_results, org_scenarios)

    # Save results
    with open(os.path.join(args.save_dir, "final_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
