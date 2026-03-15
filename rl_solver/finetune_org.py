"""Fine-tune hybrid-v2 specifically on organizer scenarios.

Usage:
    python -m rl_solver.finetune_org --episodes 500 --device cuda
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_solver.agents.hybrid_rl_v2 import HybridPackingEnvV2, HybridPPOAgentV2

logger = logging.getLogger(__name__)


def get_organizer_scenarios():
    from generator import generate_scenario
    ORG = [("heavy_water", 42), ("fragile_tower", 43), ("liquid_tetris", 44), ("random_mixed", 45)]
    return [generate_scenario(f"bench_{n}", n, seed=s) for n, s in ORG]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-dir", default="rl_solver/models")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    scenarios = get_organizer_scenarios()
    env = HybridPackingEnvV2(scenarios, max_items=200)

    agent = HybridPPOAgentV2(
        item_dim=12, ctx_dim=20, hidden=256, n_heads=4, n_layers=2,
        lr=5e-5, gamma=0.99, clip_eps=0.15,
        entropy_coef=0.03, value_coef=0.5, n_epochs=6, device=args.device,
    )

    # Load pretrained weights
    base_path = os.path.join(args.model_dir, "v2_best.pt")
    if not os.path.exists(base_path):
        base_path = os.path.join(args.model_dir, "v2_final.pt")
    if os.path.exists(base_path):
        agent.load(base_path)
        # Reset scheduler for fine-tuning
        import torch.optim as optim
        agent.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            agent.optimizer, T_max=args.episodes, eta_min=1e-5)
        logger.info("Loaded pretrained model from %s", base_path)

    scores = {sc["task_id"]: [] for sc in scenarios}
    best_avg = 0.0

    for ep in range(args.episodes):
        sc_idx = ep % len(scenarios)
        traj, score = agent.generate_ordering(env, scenario_idx=sc_idx, greedy=False)
        sc_name = scenarios[sc_idx]["task_id"]
        scores[sc_name].append(score)
        agent.train_on_episode(traj, score)

        if (ep + 1) % 20 == 0:
            avg_per_sc = {k: np.mean(v[-10:]) if v else 0 for k, v in scores.items()}
            overall = np.mean(list(avg_per_sc.values()))
            logger.info("FT ep %d/%d: %s avg=%.4f",
                        ep + 1, args.episodes,
                        " ".join(f"{k.replace('bench_','')}={v:.3f}" for k, v in avg_per_sc.items()),
                        overall)
            if overall > best_avg:
                best_avg = overall
                agent.save(os.path.join(args.model_dir, "v2_ft_best.pt"))

    agent.save(os.path.join(args.model_dir, "v2_ft_final.pt"))

    # Final eval with multi-sample
    logger.info("Final evaluation with multi-sample x32...")
    for i, sc in enumerate(scenarios):
        resp, score = agent.multi_sample_eval(env, scenario_idx=i, n_samples=32)
        logger.info("  %s: %.4f", sc["task_id"], score)


if __name__ == "__main__":
    main()
