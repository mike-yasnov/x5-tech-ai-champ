"""Collect offline training data for the block ranking model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generator import generate_scenario
from base_solver.portfolio_block import collect_ranker_rows


DEFAULT_FAMILIES = [
    "heavy_water",
    "fragile_tower",
    "liquid_tetris",
    "random_mixed",
    "exact_fit",
    "fragile_mix",
    "support_tetris",
    "cavity_fill",
    "weight_limited_repeat",
    "fragile_cap_mix",
    "mixed_column_repeat",
    "small_gap_fill",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect block-ranker training data")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    parser.add_argument(
        "--families",
        nargs="*",
        default=DEFAULT_FAMILIES,
        help="Scenario families to generate",
    )
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-end", type=int, default=1005)
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()

    rows = []
    for family in args.families:
        for seed in range(args.seed_start, args.seed_end):
            request = generate_scenario(f"ranker_{family}_{seed}", family, seed=seed)
            rows.extend(collect_ranker_rows(request, top_k=args.top_k))

    if not rows:
        raise RuntimeError("No ranker rows were collected")

    X = np.asarray([row["features"] for row in rows], dtype=np.float32)
    y = np.asarray([row["target"] for row in rows], dtype=np.float32)

    group_sizes = []
    current_gid = None
    current_size = 0
    for row in rows:
        gid = row["group_id"]
        if current_gid is None:
            current_gid = gid
        if gid != current_gid:
            group_sizes.append(current_size)
            current_gid = gid
            current_size = 0
        current_size += 1
    group_sizes.append(current_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, group=np.asarray(group_sizes))

    meta = {
        "families": args.families,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "rows": int(len(rows)),
        "groups": int(len(group_sizes)),
        "feature_dim": int(X.shape[1]),
    }
    with open(output_path.with_suffix(".meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
