"""CLI interface for the 3D Pallet Packing Solver."""

import argparse
import json
import logging
import sys

from .models import load_request, solution_to_dict
from .solver import solve


def main():
    parser = argparse.ArgumentParser(
        description="3D Pallet Packing Solver",
        prog="python -m solver",
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Request JSON file(s)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output JSON file (only for single input; for batch, uses response_<name>.json)",
    )
    parser.add_argument(
        "--restarts", type=int, default=30,
        help="Number of restarts (default: 30)",
    )
    parser.add_argument(
        "--time-budget", type=int, default=900,
        help="Time budget per task in ms (default: 900)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(levelname)s %(name)s] %(message)s",
        stream=sys.stderr,
    )

    for input_path in args.inputs:
        task_id, pallet, boxes = load_request(input_path)

        # Keep original request dict for validator
        with open(input_path, "r", encoding="utf-8") as f:
            request_dict = json.load(f)

        solution = solve(
            task_id=task_id,
            pallet=pallet,
            boxes=boxes,
            request_dict=request_dict,
            n_restarts=args.restarts,
            time_budget_ms=args.time_budget,
        )

        result = solution_to_dict(solution)

        # Determine output path
        if args.output and len(args.inputs) == 1:
            out_path = args.output
        else:
            name = input_path.replace("request_", "response_").replace(".json", ".json")
            if name == input_path:
                name = f"response_{input_path}"
            out_path = out_path = name

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Print score if validator available
        try:
            from core.validator import evaluate_solution
            eval_result = evaluate_solution(request_dict, result)
            if eval_result.get("valid"):
                metrics = eval_result.get("metrics", {})
                print(
                    f"[{task_id}] score={eval_result['final_score']:.4f} "
                    f"vol={metrics.get('volume_utilization', 0):.4f} "
                    f"coverage={metrics.get('item_coverage', 0):.4f} "
                    f"fragility={metrics.get('fragility_score', 0):.4f} "
                    f"time={metrics.get('time_score', 0):.4f} "
                    f"→ {out_path}"
                )
            else:
                print(f"[{task_id}] INVALID: {eval_result.get('error')} → {out_path}")
        except ImportError:
            print(f"[{task_id}] saved → {out_path} (validator not available)")


if __name__ == "__main__":
    main()
