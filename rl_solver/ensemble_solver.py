"""Ensemble solver: combines original solver + hybrid RL v2 + LNS refinement.

Strategy:
  1. Run original solver (multi-restart greedy + LNS + postprocess)
  2. Run hybrid-v2 with N sampled orderings + postprocess
  3. Apply LNS refinement to top-K solutions
  4. Pick the best by validator score

Usage:
    python -m rl_solver.ensemble_solver [--n-samples 64] [--final-only]
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def _validate(scenario: Dict, response: Dict) -> Tuple[float, bool]:
    from validator import evaluate_solution
    r = evaluate_solution(scenario, response)
    if r.get("valid"):
        return r.get("final_score", 0.0), True
    return 0.0, False


def run_original(scenario: Dict, time_budget_ms: int = 900) -> Tuple[Dict, float]:
    """Run original solver."""
    from solver.models import Pallet, Box, solution_to_dict
    from solver.solver import solve

    p = scenario["pallet"]
    pallet = Pallet(type_id=p.get("type_id", ""), length_mm=p["length_mm"],
                    width_mm=p["width_mm"], max_height_mm=p["max_height_mm"],
                    max_weight_kg=p["max_weight_kg"])
    boxes = [Box(sku_id=b["sku_id"], description=b.get("description", ""),
                 length_mm=b["length_mm"], width_mm=b["width_mm"], height_mm=b["height_mm"],
                 weight_kg=b["weight_kg"], quantity=b["quantity"],
                 strict_upright=b.get("strict_upright", False),
                 fragile=b.get("fragile", False), stackable=b.get("stackable", True))
             for b in scenario["boxes"]]

    sol = solve(task_id=scenario["task_id"], pallet=pallet, boxes=boxes,
                request_dict=scenario, n_restarts=30, time_budget_ms=time_budget_ms)
    response = solution_to_dict(sol)
    score, valid = _validate(scenario, response)
    return response, score


def run_hybrid_v2(scenario: Dict, model_path: str, n_samples: int = 64,
                  device: str = "auto") -> Tuple[Dict, float]:
    """Run hybrid-v2 with multi-sample."""
    from rl_solver.agents.hybrid_rl_v2 import HybridPackingEnvV2, HybridPPOAgentV2

    env = HybridPackingEnvV2([scenario], max_items=200)
    agent = HybridPPOAgentV2(device=device)
    if os.path.exists(model_path):
        agent.load(model_path)

    response, score = agent.multi_sample_eval(env, scenario_idx=0, n_samples=n_samples)
    return response, score


def apply_lns_to_response(scenario: Dict, response: Dict, time_budget_ms: int = 3000) -> Tuple[Dict, float]:
    """Apply LNS refinement from original solver to any response."""
    from solver.models import Pallet, Box, Placement, Solution, UnplacedItem, solution_to_dict
    from solver.lns import lns_optimize

    p = scenario["pallet"]
    pallet = Pallet(type_id=p.get("type_id", ""), length_mm=p["length_mm"],
                    width_mm=p["width_mm"], max_height_mm=p["max_height_mm"],
                    max_weight_kg=p["max_weight_kg"])
    boxes_list = []
    for b in scenario["boxes"]:
        boxes_list.append(Box(
            sku_id=b["sku_id"], description=b.get("description", ""),
            length_mm=b["length_mm"], width_mm=b["width_mm"], height_mm=b["height_mm"],
            weight_kg=b["weight_kg"], quantity=b["quantity"],
            strict_upright=b.get("strict_upright", False),
            fragile=b.get("fragile", False), stackable=b.get("stackable", True)))

    # Convert response to Solution
    placements = []
    for pl in response.get("placements", []):
        dim = pl["dimensions_placed"]
        pos = pl["position"]
        placements.append(Placement(
            sku_id=pl["sku_id"], instance_index=pl["instance_index"],
            x_mm=pos["x_mm"], y_mm=pos["y_mm"], z_mm=pos["z_mm"],
            length_mm=dim["length_mm"], width_mm=dim["width_mm"], height_mm=dim["height_mm"],
            rotation_code=pl.get("rotation_code", "LWH")))

    unplaced = []
    for u in response.get("unplaced", []):
        unplaced.append(UnplacedItem(
            sku_id=u["sku_id"], quantity_unplaced=u["quantity_unplaced"],
            reason=u.get("reason", "no_space")))

    initial_solution = Solution(
        task_id=scenario["task_id"],
        solver_version="pre-lns",
        placements=placements,
        unplaced=unplaced,
        solve_time_ms=response.get("solve_time_ms", 0),
    )

    try:
        improved = lns_optimize(
            task_id=scenario["task_id"],
            pallet=pallet,
            boxes=boxes_list,
            initial_solution=initial_solution,
            time_budget_ms=time_budget_ms,
        )
        result = solution_to_dict(improved)
        score, valid = _validate(scenario, result)
        if valid:
            return result, score
    except Exception as e:
        logger.warning("LNS refinement failed: %s", e)

    # Fallback to original response
    score, _ = _validate(scenario, response)
    return response, score


def ensemble_solve(scenario: Dict, model_path: str, n_samples: int = 64,
                   device: str = "auto") -> Tuple[Dict, float, str]:
    """Run ensemble: original + hybrid-v2 + LNS, pick best."""
    candidates = []

    # 1. Original solver
    t0 = time.perf_counter()
    orig_resp, orig_score = run_original(scenario, time_budget_ms=900)
    orig_time = time.perf_counter() - t0
    candidates.append(("original", orig_resp, orig_score))
    logger.info("  original: %.4f (%.1fs)", orig_score, orig_time)

    # 2. Hybrid-v2 multi-sample
    t0 = time.perf_counter()
    v2_resp, v2_score = run_hybrid_v2(scenario, model_path, n_samples, device)
    v2_time = time.perf_counter() - t0
    candidates.append(("hybrid-v2", v2_resp, v2_score))
    logger.info("  hybrid-v2 (x%d): %.4f (%.1fs)", n_samples, v2_score, v2_time)

    # 3. LNS on top of best non-original
    if v2_score > 0:
        t0 = time.perf_counter()
        lns_resp, lns_score = apply_lns_to_response(scenario, v2_resp, time_budget_ms=3000)
        lns_time = time.perf_counter() - t0
        candidates.append(("v2+lns", lns_resp, lns_score))
        logger.info("  v2+lns: %.4f (%.1fs)", lns_score, lns_time)

    # 4. LNS on original too
    t0 = time.perf_counter()
    orig_lns_resp, orig_lns_score = apply_lns_to_response(scenario, orig_resp, time_budget_ms=3000)
    orig_lns_time = time.perf_counter() - t0
    candidates.append(("orig+lns", orig_lns_resp, orig_lns_score))
    logger.info("  orig+lns: %.4f (%.1fs)", orig_lns_score, orig_lns_time)

    # Pick best
    best = max(candidates, key=lambda x: x[2])
    logger.info("  BEST: %s = %.4f", best[0], best[2])
    return best[1], best[2], best[0]


def get_organizer_scenarios():
    from generator import generate_scenario
    ORG = [("heavy_water", 42), ("fragile_tower", 43), ("liquid_tetris", 44), ("random_mixed", 45)]
    return [generate_scenario(f"bench_{n}", n, seed=s) for n, s in ORG]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-dir", default="rl_solver/models")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    scenarios = get_organizer_scenarios()
    # Prefer fine-tuned model
    model_path = os.path.join(args.model_dir, "v2_ft_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, "v2_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, "v2_final.pt")

    print(f"\nEnsemble Solver — 4 organizer tests, n_samples={args.n_samples}")
    print("=" * 65)

    results = []
    for sc in scenarios:
        name = sc["task_id"].replace("bench_", "")
        logger.info("Scenario: %s", name)
        resp, score, winner = ensemble_solve(sc, model_path, args.n_samples, args.device)
        results.append({"scenario": name, "score": score, "winner": winner})

    print(f"\n{'Scenario':<20} | {'Score':>8} | {'Winner':<12}")
    print("-" * 48)
    for r in results:
        print(f"{r['scenario']:<20} | {r['score']:>8.4f} | {r['winner']:<12}")
    avg = np.mean([r["score"] for r in results])
    print("-" * 48)
    print(f"{'AVERAGE':<20} | {avg:>8.4f} |")

    with open(os.path.join(args.model_dir, "ensemble_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
