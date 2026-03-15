"""Training pipeline for all RL and metaheuristic agents.

Usage:
    python -m rl_solver.train --agent dqn --episodes 500 --device cuda
    python -m rl_solver.train --agent ppo --episodes 1000
    python -m rl_solver.train --agent a2c --episodes 500
    python -m rl_solver.train --agent ga   # no training, direct solve
    python -m rl_solver.train --agent sa   # no training, direct solve
    python -m rl_solver.train --agent all  # run all agents
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_solver.scenarios.hard_scenarios import get_all_scenarios

logger = logging.getLogger(__name__)


def train_dqn(scenarios: List[Dict], episodes: int, device: str, save_dir: str) -> Dict:
    """Train DQN agent."""
    import torch
    from rl_solver.env import PackingEnv
    from rl_solver.agents.dqn_agent import DQNAgent

    env = PackingEnv(scenarios=scenarios, grid_res=50)
    obs, _ = env.reset()
    obs_dim = len(obs)
    n_actions = env.n_actions

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=episodes * 50,
        buffer_size=200000,
        batch_size=128,
        target_update_freq=500,
        hidden_dim=512,
        device=device,
    )

    episode_rewards = []
    episode_placed = []
    best_avg_reward = -float("inf")

    for ep in range(episodes):
        obs, _ = env.reset()
        mask = env.get_action_mask()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_mask = env.get_action_mask() if not done else np.zeros_like(mask)

            agent.store(obs, action, reward, next_obs, float(done), mask, next_mask)
            loss = agent.train_step()

            obs = next_obs
            mask = next_mask
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_placed.append(len(env.placements))

        if (ep + 1) % 10 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_p = np.mean(episode_placed[-10:])
            logger.info(
                "DQN ep %d/%d: reward=%.3f, placed=%.1f, eps=%.3f, items=%d",
                ep + 1, episodes, avg_r, avg_p, agent.epsilon, env.total_items,
            )
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                agent.save(os.path.join(save_dir, "dqn_best.pt"))

    agent.save(os.path.join(save_dir, "dqn_final.pt"))
    return {
        "agent": "dqn",
        "episodes": episodes,
        "final_avg_reward": float(np.mean(episode_rewards[-20:])),
        "final_avg_placed": float(np.mean(episode_placed[-20:])),
        "best_avg_reward": float(best_avg_reward),
    }


def train_ppo(scenarios: List[Dict], episodes: int, device: str, save_dir: str) -> Dict:
    """Train PPO agent."""
    import torch
    from rl_solver.env import PackingEnv
    from rl_solver.agents.ppo_agent import PPOAgent

    env = PackingEnv(scenarios=scenarios, grid_res=50)
    obs, _ = env.reset()
    obs_dim = len(obs)
    n_actions = env.n_actions

    rollout_len = 512

    agent = PPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.02,
        value_coef=0.5,
        n_epochs=4,
        batch_size=64,
        rollout_len=rollout_len,
        hidden_dim=512,
        device=device,
    )

    episode_rewards = []
    episode_placed = []
    best_avg_reward = -float("inf")

    obs, _ = env.reset()
    mask = env.get_action_mask()
    ep_reward = 0.0
    ep_count = 0

    total_steps = episodes * 100  # approx steps per episode
    for step in range(total_steps):
        action, log_prob, value = agent.select_action(obs, mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = env.get_action_mask() if not done else np.zeros_like(mask)

        agent.store(obs, action, reward, done, log_prob, value, mask)
        ep_reward += reward

        if done:
            episode_rewards.append(ep_reward)
            episode_placed.append(len(env.placements))
            ep_count += 1
            ep_reward = 0.0
            obs, _ = env.reset()
            mask = env.get_action_mask()

            if ep_count >= episodes:
                break
        else:
            obs = next_obs
            mask = next_mask

        # PPO update at rollout boundary
        if len(agent.buffer) >= rollout_len:
            with torch.no_grad():
                ns_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                _, last_value = agent.model(ns_t)
                last_value = last_value.item()
            metrics = agent.train_step(last_value)

        if ep_count > 0 and ep_count % 10 == 0 and episode_rewards:
            avg_r = np.mean(episode_rewards[-10:])
            avg_p = np.mean(episode_placed[-10:])
            logger.info(
                "PPO ep %d/%d: reward=%.3f, placed=%.1f",
                ep_count, episodes, avg_r, avg_p,
            )
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                agent.save(os.path.join(save_dir, "ppo_best.pt"))

    agent.save(os.path.join(save_dir, "ppo_final.pt"))
    return {
        "agent": "ppo",
        "episodes": ep_count,
        "final_avg_reward": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0,
        "final_avg_placed": float(np.mean(episode_placed[-20:])) if episode_placed else 0,
        "best_avg_reward": float(best_avg_reward),
    }


def train_a2c(scenarios: List[Dict], episodes: int, device: str, save_dir: str) -> Dict:
    """Train A2C agent."""
    import torch
    from rl_solver.env import PackingEnv
    from rl_solver.agents.a2c_agent import A2CAgent

    env = PackingEnv(scenarios=scenarios, grid_res=50)
    obs, _ = env.reset()
    obs_dim = len(obs)
    n_actions = env.n_actions

    agent = A2CAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        lr=7e-4,
        gamma=0.99,
        n_steps=5,
        entropy_coef=0.01,
        value_coef=0.5,
        hidden_dim=256,
        device=device,
    )

    episode_rewards = []
    episode_placed = []
    best_avg_reward = -float("inf")

    for ep in range(episodes):
        obs, _ = env.reset()
        mask = env.get_action_mask()
        total_reward = 0.0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(obs, mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_mask = env.get_action_mask() if not done else np.zeros_like(mask)

            agent.store(obs, action, reward, done, mask)

            if len(agent._states) >= agent.n_steps or done:
                agent.train_step(next_obs, next_mask)

            obs = next_obs
            mask = next_mask
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_placed.append(len(env.placements))

        if (ep + 1) % 10 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            avg_p = np.mean(episode_placed[-10:])
            logger.info(
                "A2C ep %d/%d: reward=%.3f, placed=%.1f",
                ep + 1, episodes, avg_r, avg_p,
            )
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                agent.save(os.path.join(save_dir, "a2c_best.pt"))

    agent.save(os.path.join(save_dir, "a2c_final.pt"))
    return {
        "agent": "a2c",
        "episodes": episodes,
        "final_avg_reward": float(np.mean(episode_rewards[-20:])),
        "final_avg_placed": float(np.mean(episode_placed[-20:])),
        "best_avg_reward": float(best_avg_reward),
    }


def run_ga(scenarios: List[Dict], save_dir: str) -> Dict:
    """Run GA solver on all scenarios."""
    from rl_solver.agents.genetic_agent import GeneticSolver

    solver = GeneticSolver(
        pop_size=100,
        n_generations=300,
        crossover_rate=0.85,
        mutation_rate=0.15,
        time_budget_sec=25.0,
    )

    results = []
    for sc in scenarios:
        result = solver.solve(sc)
        results.append({
            "task_id": sc["task_id"],
            "placed": len(result["placements"]),
            "total": sum(b["quantity"] for b in sc["boxes"]),
            "time_ms": result["solve_time_ms"],
        })
        logger.info("GA %s: placed=%d/%d time=%dms",
                     sc["task_id"], results[-1]["placed"], results[-1]["total"],
                     result["solve_time_ms"])

    # Save all solutions
    with open(os.path.join(save_dir, "ga_solutions.json"), "w") as f:
        json.dump(results, f, indent=2)

    avg_coverage = np.mean([r["placed"] / max(r["total"], 1) for r in results])
    return {"agent": "ga", "avg_coverage": float(avg_coverage), "n_scenarios": len(results)}


def run_sa(scenarios: List[Dict], save_dir: str) -> Dict:
    """Run SA solver on all scenarios."""
    from rl_solver.agents.sa_agent import SimulatedAnnealingSolver

    solver = SimulatedAnnealingSolver(
        initial_temp=10.0,
        cooling_rate=0.9995,
        max_iterations=100000,
        time_budget_sec=25.0,
        n_restarts=3,
    )

    results = []
    for sc in scenarios:
        result = solver.solve(sc)
        results.append({
            "task_id": sc["task_id"],
            "placed": len(result["placements"]),
            "total": sum(b["quantity"] for b in sc["boxes"]),
            "time_ms": result["solve_time_ms"],
        })
        logger.info("SA %s: placed=%d/%d time=%dms",
                     sc["task_id"], results[-1]["placed"], results[-1]["total"],
                     result["solve_time_ms"])

    with open(os.path.join(save_dir, "sa_solutions.json"), "w") as f:
        json.dump(results, f, indent=2)

    avg_coverage = np.mean([r["placed"] / max(r["total"], 1) for r in results])
    return {"agent": "sa", "avg_coverage": float(avg_coverage), "n_scenarios": len(results)}


def main():
    parser = argparse.ArgumentParser(description="Train RL/metaheuristic agents for 3D packing")
    parser.add_argument("--agent", choices=["dqn", "ppo", "a2c", "ga", "sa", "all"],
                        default="all", help="Which agent to train/run")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes (RL agents)")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--save-dir", default="rl_solver/models", help="Model save directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    os.makedirs(args.save_dir, exist_ok=True)
    scenarios = get_all_scenarios()
    logger.info("Loaded %d scenarios", len(scenarios))

    all_results = {}
    agents_to_run = [args.agent] if args.agent != "all" else ["dqn", "ppo", "a2c", "ga", "sa"]

    for agent_name in agents_to_run:
        logger.info("=" * 60)
        logger.info("Starting %s", agent_name.upper())
        logger.info("=" * 60)
        t0 = time.perf_counter()

        try:
            if agent_name == "dqn":
                result = train_dqn(scenarios, args.episodes, args.device, args.save_dir)
            elif agent_name == "ppo":
                result = train_ppo(scenarios, args.episodes, args.device, args.save_dir)
            elif agent_name == "a2c":
                result = train_a2c(scenarios, args.episodes, args.device, args.save_dir)
            elif agent_name == "ga":
                result = run_ga(scenarios, args.save_dir)
            elif agent_name == "sa":
                result = run_sa(scenarios, args.save_dir)

            elapsed = time.perf_counter() - t0
            result["wall_time_sec"] = round(elapsed, 1)
            all_results[agent_name] = result
            logger.info("%s completed in %.1fs: %s", agent_name.upper(), elapsed, result)

        except Exception as e:
            logger.error("%s FAILED: %s", agent_name.upper(), e, exc_info=True)
            all_results[agent_name] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(args.save_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Print summary table
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:6s}: FAILED - {res['error']}")
        else:
            t = res.get('wall_time_sec', '?')
            if 'final_avg_reward' in res:
                print(f"  {name:6s}: reward={res['final_avg_reward']:.3f}, "
                      f"placed={res['final_avg_placed']:.1f}, time={t}s")
            else:
                print(f"  {name:6s}: coverage={res.get('avg_coverage', 0):.3f}, time={t}s")


if __name__ == "__main__":
    main()
