#!/bin/bash
# Run RL training and benchmarks on the server
# Usage: ssh work "cd ai-chemp && bash rl_solver/run_on_server.sh"

set -e

echo "=== Setting up environment ==="
cd "$(dirname "$0")/.."

pip3 install --quiet torch numpy gymnasium 2>&1 | tail -1

echo ""
echo "=== Running GA + SA benchmarks on hard scenarios ==="
python3 -m rl_solver.rl_benchmark --agents ga,sa --include-original \
    --output rl_solver/models/benchmark_results.json \
    --log-level INFO 2>&1

echo ""
echo "=== Training RL agents (DQN, PPO, A2C) ==="
python3 -m rl_solver.train --agent all --episodes 200 --device auto \
    --save-dir rl_solver/models --log-level INFO 2>&1

echo ""
echo "=== Done ==="
echo "Results saved in rl_solver/models/"
ls -la rl_solver/models/
