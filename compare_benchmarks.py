"""Compare candidate benchmark results against a baseline report."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Tuple

from benchmark_utils import compute_quality_score


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _index_report(report: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for scenario in report.get("scenarios", []):
        for result in scenario.get("results", []):
            index[(scenario["scenario"], result["method"])] = result
    return index


def _normalize_report(report: Any) -> Dict[str, Any]:
    if isinstance(report, dict) and "scenarios" in report:
        return report
    if isinstance(report, list):
        scenarios = []
        for row in report:
            scenarios.append(
                {
                    "scenario": row["scenario"],
                    "results": [
                        {
                            "method": "solver",
                            "official_score": row.get("final_score", 0.0),
                            "quality_score": row.get(
                                "quality_score",
                                compute_quality_score(row.get("metrics", {})),
                            ),
                            "valid": row.get("valid", False),
                        }
                    ],
                }
            )
        return {"scenarios": scenarios}
    raise ValueError("Unsupported benchmark report format")


def _empty_aggregate() -> Dict[str, Any]:
    return {
        "count": 0,
        "official_delta": 0.0,
        "quality_delta": 0.0,
        "candidate_invalid": 0,
        "baseline_invalid": 0,
        "new_invalid": 0,
        "official_regressions": 0,
        "quality_regressions": 0,
    }


def compare_reports(
    baseline_report: Dict[str, Any],
    candidate_report: Dict[str, Any],
    official_threshold: float,
    quality_threshold: float,
) -> Dict[str, Any]:
    baseline_index = _index_report(baseline_report)
    candidate_index = _index_report(candidate_report)
    aggregates: Dict[str, Dict[str, Any]] = {}
    scenario_deltas = []

    for key, candidate in candidate_index.items():
        baseline = baseline_index.get(key)
        if baseline is None:
            continue

        scenario_name, method_name = key
        official_delta = candidate["official_score"] - baseline["official_score"]
        quality_delta = candidate["quality_score"] - baseline["quality_score"]
        new_invalid = int(bool(baseline["valid"]) and not bool(candidate["valid"]))

        aggregate = aggregates.setdefault(method_name, _empty_aggregate())
        aggregate["count"] += 1
        aggregate["official_delta"] += official_delta
        aggregate["quality_delta"] += quality_delta
        aggregate["candidate_invalid"] += 0 if candidate["valid"] else 1
        aggregate["baseline_invalid"] += 0 if baseline["valid"] else 1
        aggregate["new_invalid"] += new_invalid
        aggregate["official_regressions"] += int(official_delta < -official_threshold)
        aggregate["quality_regressions"] += int(quality_delta < -quality_threshold)

        scenario_deltas.append(
            {
                "scenario": scenario_name,
                "method": method_name,
                "baseline_official": baseline["official_score"],
                "candidate_official": candidate["official_score"],
                "official_delta": official_delta,
                "baseline_quality": baseline["quality_score"],
                "candidate_quality": candidate["quality_score"],
                "quality_delta": quality_delta,
                "baseline_valid": baseline["valid"],
                "candidate_valid": candidate["valid"],
                "new_invalid": bool(new_invalid),
            }
        )

    summary = []
    should_fail = False
    for method_name, aggregate in aggregates.items():
        count = max(1, aggregate["count"])
        row = {
            "method": method_name,
            "official_delta_avg": aggregate["official_delta"] / count,
            "quality_delta_avg": aggregate["quality_delta"] / count,
            "new_invalid": aggregate["new_invalid"],
            "official_regressions": aggregate["official_regressions"],
            "quality_regressions": aggregate["quality_regressions"],
            "candidate_invalid": aggregate["candidate_invalid"],
        }
        if (
            row["new_invalid"] > 0
            or row["official_delta_avg"] < -official_threshold
            or row["quality_delta_avg"] < -quality_threshold
        ):
            should_fail = True
        summary.append(row)

    summary.sort(
        key=lambda row: (
            row["new_invalid"] == 0,
            row["quality_delta_avg"],
            row["official_delta_avg"],
        ),
        reverse=True,
    )
    scenario_deltas.sort(
        key=lambda row: (
            row["new_invalid"],
            row["quality_delta"],
            row["official_delta"],
        ),
        reverse=True,
    )

    return {
        "summary": summary,
        "scenario_deltas": scenario_deltas,
        "official_threshold": official_threshold,
        "quality_threshold": quality_threshold,
        "should_fail": should_fail,
    }


def render_markdown(report: Dict[str, Any], limit: int = 20) -> str:
    lines = ["## Baseline Comparison", ""]
    lines.append("### Aggregate Delta by Method")
    lines.append("")
    lines.append(
        "| Method | Official Δ avg | Quality Δ avg | New Invalid | Official Regressions | Quality Regressions |"
    )
    lines.append(
        "|--------|----------------|---------------|-------------|----------------------|---------------------|"
    )
    for row in report["summary"]:
        lines.append(
            f"| {row['method']} | {row['official_delta_avg']:+.4f} | {row['quality_delta_avg']:+.4f} | "
            f"{row['new_invalid']} | {row['official_regressions']} | {row['quality_regressions']} |"
        )

    lines.extend(["", "### Largest Scenario Deltas", ""])
    lines.append(
        "| Scenario | Method | Official Δ | Quality Δ | Baseline Valid | Candidate Valid |"
    )
    lines.append(
        "|----------|--------|------------|-----------|----------------|-----------------|"
    )
    for row in report["scenario_deltas"][:limit]:
        lines.append(
            f"| {row['scenario']} | {row['method']} | {row['official_delta']:+.4f} | {row['quality_delta']:+.4f} | "
            f"{row['baseline_valid']} | {row['candidate_valid']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate and baseline benchmark JSON"
    )
    parser.add_argument(
        "--baseline", required=True, help="Baseline benchmark JSON path"
    )
    parser.add_argument(
        "--candidate", required=True, help="Candidate benchmark JSON path"
    )
    parser.add_argument(
        "--output", default=None, help="Optional JSON comparison output path"
    )
    parser.add_argument(
        "--markdown-output", default=None, help="Optional markdown summary output path"
    )
    parser.add_argument(
        "--official-threshold",
        type=float,
        default=0.01,
        help="Allowed official score regression before failure",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.005,
        help="Allowed quality score regression before failure",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when regression thresholds are exceeded",
    )
    args = parser.parse_args()

    report = compare_reports(
        baseline_report=_normalize_report(_load(args.baseline)),
        candidate_report=_normalize_report(_load(args.candidate)),
        official_threshold=args.official_threshold,
        quality_threshold=args.quality_threshold,
    )
    markdown = render_markdown(report)
    print(markdown)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

    if args.markdown_output:
        with open(args.markdown_output, "w", encoding="utf-8") as file:
            file.write(markdown)

    if args.fail_on_regression and report["should_fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
