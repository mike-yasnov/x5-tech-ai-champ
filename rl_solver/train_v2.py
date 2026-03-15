"""Train hybrid RL v2 + full benchmark.

Usage:
    python -m rl_solver.train_v2 --episodes 1000 --device cuda
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_solver.scenarios.hard_scenarios import get_all_scenarios
from rl_solver.agents.hybrid_rl_v2 import HybridPackingEnvV2, HybridPPOAgentV2

logger = logging.getLogger(__name__)


def get_organizer_scenarios():
    from generator import generate_scenario
    ORG = [("heavy_water", 42), ("fragile_tower", 43), ("liquid_tetris", 44), ("random_mixed", 45)]
    return [generate_scenario(f"bench_{n}", n, seed=s) for n, s in ORG]


def train_v2(scenarios, episodes, device, save_dir):
    env = HybridPackingEnvV2(scenarios, max_items=200)
    agent = HybridPPOAgentV2(
        item_dim=12, ctx_dim=20, hidden=256, n_heads=4, n_layers=2,
        lr=1e-4, gamma=0.99, clip_eps=0.2,
        entropy_coef=0.02, value_coef=0.5, n_epochs=4, device=device,
    )

    scores = []
    best_avg = 0.0
    n_scenarios = len(scenarios)

    # Curriculum: first 30% episodes use easier scenarios (fewer items)
    easy_indices = sorted(range(n_scenarios),
                          key=lambda i: sum(b["quantity"] for b in scenarios[i]["boxes"]))
    easy_count = max(1, n_scenarios // 3)

    for ep in range(episodes):
        # Curriculum: mix easy/hard
        if ep < episodes * 0.3:
            sc_idx = easy_indices[ep % easy_count]
        else:
            sc_idx = ep % n_scenarios

        traj, score = agent.generate_ordering(env, scenario_idx=sc_idx, greedy=False)
        scores.append(score)
        metrics = agent.train_on_episode(traj, score)

        if (ep + 1) % 10 == 0:
            avg10 = np.mean(scores[-10:])
            logger.info(
                "v2 ep %d/%d: score=%.4f avg10=%.4f loss=%.4f ent=%.3f bl=%.3f lr=%.2e",
                ep + 1, episodes, score, avg10,
                metrics.get("loss", 0), metrics.get("entropy", 0),
                metrics.get("baseline", 0), metrics.get("lr", 0),
            )
            if avg10 > best_avg:
                best_avg = avg10
                agent.save(os.path.join(save_dir, "v2_best.pt"))

        if (ep + 1) % 100 == 0:
            agent.save(os.path.join(save_dir, f"v2_ep{ep+1}.pt"))

    agent.save(os.path.join(save_dir, "v2_final.pt"))
    return {"final_avg": float(np.mean(scores[-20:])), "best_avg": float(best_avg)}


def benchmark_v2(scenarios, model_path, device, n_samples=16):
    """Benchmark v2 with multi-sample evaluation."""
    from validator import evaluate_solution

    env = HybridPackingEnvV2(scenarios, max_items=200)
    agent = HybridPPOAgentV2(device=device)
    if os.path.exists(model_path):
        agent.load(model_path)

    results = []
    for i, sc in enumerate(scenarios):
        response, score = agent.multi_sample_eval(env, scenario_idx=i, n_samples=n_samples)
        r = evaluate_solution(sc, response)
        results.append({
            "scenario": sc["task_id"], "agent": "hybrid-v2",
            "score": r.get("final_score", 0) if r.get("valid") else 0,
            "valid": r.get("valid", False),
            "placed": len(response.get("placements", [])),
            "total": sum(b["quantity"] for b in sc["boxes"]),
        })
        logger.info("v2 %s: score=%.4f placed=%d/%d",
                     sc["task_id"], results[-1]["score"],
                     results[-1]["placed"], results[-1]["total"])
    return results


def benchmark_original(scenarios):
    from solver.models import Pallet, Box, solution_to_dict
    from solver.solver import solve
    from validator import evaluate_solution

    results = []
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
        results.append({
            "scenario": sc["task_id"], "agent": "original",
            "score": r.get("final_score", 0) if r.get("valid") else 0,
            "valid": r.get("valid", False),
            "placed": len(sol.placements),
            "total": sum(b["quantity"] for b in sc["boxes"]),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="rl_solver/models")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.makedirs(args.save_dir, exist_ok=True)

    hard = get_all_scenarios()
    org = get_organizer_scenarios()

    if not args.benchmark_only:
        logger.info("Training v2 on %d hard scenarios, %d episodes", len(hard), args.episodes)
        t0 = time.perf_counter()
        result = train_v2(hard, args.episodes, args.device, args.save_dir)
        result["wall_sec"] = round(time.perf_counter() - t0, 1)
        logger.info("Training done: %s", result)

    # Final benchmark on organizer tests
    logger.info("=" * 60)
    logger.info("FINAL: 4 organizer scenarios, n_samples=%d", args.n_samples)
    logger.info("=" * 60)

    model_path = os.path.join(args.save_dir, "v2_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.save_dir, "v2_final.pt")

    orig = benchmark_original(org)
    v2 = benchmark_v2(org, model_path, args.device, args.n_samples)

    print(f"\n{'Scenario':<25} | {'Original':>10} | {'Hybrid-v2':>10}")
    print("-" * 52)
    orig_scores, v2_scores = [], []
    for o, v in zip(orig, v2):
        print(f"{o['scenario']:<25} | {o['score']:>10.4f} | {v['score']:>10.4f}")
        orig_scores.append(o["score"])
        v2_scores.append(v["score"])
    print("-" * 52)
    print(f"{'AVERAGE':<25} | {np.mean(orig_scores):>10.4f} | {np.mean(v2_scores):>10.4f}")

    with open(os.path.join(args.save_dir, "v2_final_results.json"), "w") as f:
        json.dump({"original": orig, "hybrid_v2": v2}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
