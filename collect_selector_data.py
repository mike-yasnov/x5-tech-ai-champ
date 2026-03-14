"""Collect offline request-level training data for the seed-family selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generator import generate_scenario
from solver.portfolio_block import _finalize_run, _format_output, _init_common_components, _run_seed_family
from solver.scenario_selector import SEED_FAMILIES, compute_request_fingerprint, seed_family_index
from validator import evaluate_solution


DEFAULT_FAMILIES = [
    "heavy_water",
    "fragile_tower",
    "liquid_tetris",
    "random_mixed",
    "exact_fit",
    "fragile_mix",
    "support_tetris",
    "cavity_fill",
    "count_preference",
    "weight_limited_repeat",
    "fragile_cap_mix",
    "mixed_column_repeat",
    "small_gap_fill",
]


def _seed_family_score(request: dict, seed_family: str) -> float:
    checker, candidate_gen, _ = _init_common_components(request)
    run = _run_seed_family(
        request=request,
        seed_family=seed_family,
        checker=checker,
        candidate_gen=candidate_gen,
        ranker=None,
        extractor=None,
        budget_ms=260,
    )
    placements, leftover, state = _finalize_run(
        request=request,
        run=run,
        checker=checker,
        candidate_gen=candidate_gen,
    )
    response = _format_output(
        request["task_id"],
        placements,
        leftover,
        request["boxes"],
        state,
        run.elapsed_ms,
    )
    result = evaluate_solution(request, response)
    return float(result.get("final_score", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect selector training data")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    parser.add_argument(
        "--families",
        nargs="*",
        default=DEFAULT_FAMILIES,
        help="Scenario families to generate",
    )
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-end", type=int, default=1500)
    args = parser.parse_args()

    rows = []
    for family in args.families:
        for seed in range(args.seed_start, args.seed_end):
            request = generate_scenario(f"selector_{family}_{seed}", family, seed=seed)
            fingerprint = compute_request_fingerprint(request)
            seed_scores = [_seed_family_score(request, seed_family) for seed_family in SEED_FAMILIES]
            best_family_idx = int(np.argmax(np.asarray(seed_scores, dtype=np.float32)))
            rows.append(
                {
                    "task_id": request["task_id"],
                    "scenario_family": family,
                    "seed": seed,
                    "features": fingerprint.as_array().tolist(),
                    "target": best_family_idx,
                    "best_seed_family": SEED_FAMILIES[best_family_idx],
                    "scores": seed_scores,
                }
            )

    if not rows:
        raise RuntimeError("No selector rows were collected")

    X = np.asarray([row["features"] for row in rows], dtype=np.float32)
    y = np.asarray([row["target"] for row in rows], dtype=np.int32)
    score_matrix = np.asarray([row["scores"] for row in rows], dtype=np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, scores=score_matrix)

    meta = {
        "families": args.families,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "rows": int(len(rows)),
        "feature_dim": int(X.shape[1]),
        "seed_families": list(SEED_FAMILIES),
        "label_mapping": {name: seed_family_index(name) for name in SEED_FAMILIES},
    }
    with open(output_path.with_suffix(".meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
