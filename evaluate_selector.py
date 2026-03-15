"""Evaluate the shipped selector on held-out data and the organizer benchmark."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from benchmark import ORGANIZER_SCENARIOS, run_benchmark
from solver.scenario_selector import ScenarioSelector, seed_family_names


def _organizer_average(results: list[dict]) -> float:
    organizer_names = {name for name, _ in ORGANIZER_SCENARIOS}
    rows = [row for row in results if row["scenario"] in organizer_names]
    if not rows:
        return 0.0
    return sum(row["final_score"] for row in rows if row["valid"]) / len(rows)


def _report_dict(
    selector_accuracy: float,
    benchmark_on: list[dict],
    benchmark_off: list[dict],
    block_ranker_artifact: bool,
) -> dict:
    return {
        "selector_top1_accuracy": selector_accuracy,
        "organizer_average_selector_on": _organizer_average(benchmark_on),
        "organizer_average_selector_off": _organizer_average(benchmark_off),
        "block_ranker_artifact_present": block_ranker_artifact,
    }


def _render_markdown(report: dict) -> str:
    return "\n".join(
        [
            "## Selector Evaluation",
            "",
            f"- Held-out top-1 accuracy: `{report['selector_top1_accuracy']:.4f}`",
            f"- Organizer average with selector: `{report['organizer_average_selector_on']:.4f}`",
            f"- Organizer average without selector: `{report['organizer_average_selector_off']:.4f}`",
            f"- Block ranker artifact present: `{report['block_ranker_artifact_present']}`",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the shipped selector")
    parser.add_argument("--dataset", required=True, help="Held-out selector dataset .npz")
    parser.add_argument("--model-dir", default="models", help="Directory with selector artifact")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument("--markdown-output", default=None, help="Optional markdown report path")
    args = parser.parse_args()

    selector = ScenarioSelector(model_dir=args.model_dir)
    if not selector.load():
        raise RuntimeError(f"Selector artifact was not found in {args.model_dir}")

    data = np.load(Path(args.dataset))
    X = data["X"]
    y = data["y"]
    pred = np.argmax(selector.predict_proba(X), axis=1)
    selector_accuracy = float(np.mean(pred == y))

    benchmark_on = run_benchmark(strategy="portfolio_block", model_dir=args.model_dir)
    with tempfile.TemporaryDirectory() as empty_model_dir:
        benchmark_off = run_benchmark(strategy="portfolio_block", model_dir=empty_model_dir)

    report = _report_dict(
        selector_accuracy=selector_accuracy,
        benchmark_on=benchmark_on,
        benchmark_off=benchmark_off,
        block_ranker_artifact=Path(args.model_dir, "xgb_ranker.json").exists(),
    )

    markdown = _render_markdown(report)
    print(markdown)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
    if args.markdown_output:
        with open(args.markdown_output, "w", encoding="utf-8") as file:
            file.write(markdown)


if __name__ == "__main__":
    main()
