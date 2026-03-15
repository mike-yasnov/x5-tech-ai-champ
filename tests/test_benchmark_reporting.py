"""Tests for benchmark comparison utilities."""

import pytest

from tools.benchmark_utils import compute_quality_score
from tools.compare_benchmarks import compare_reports


def test_compute_quality_score_uses_expected_weights():
    metrics = {
        "volume_utilization": 0.8,
        "item_coverage": 0.5,
        "fragility_score": 1.0,
    }
    score = compute_quality_score(metrics)
    assert score == pytest.approx(0.55 * 0.8 + 0.35 * 0.5 + 0.10 * 1.0)


def test_compare_reports_detects_regression_thresholds():
    baseline = {
        "scenarios": [
            {
                "scenario": "fragile_mix",
                "results": [
                    {
                        "method": "solver",
                        "official_score": 0.90,
                        "quality_score": 0.85,
                        "valid": True,
                    }
                ],
            }
        ]
    }
    candidate = {
        "scenarios": [
            {
                "scenario": "fragile_mix",
                "results": [
                    {
                        "method": "solver",
                        "official_score": 0.87,
                        "quality_score": 0.83,
                        "valid": True,
                    }
                ],
            }
        ]
    }

    report = compare_reports(
        baseline, candidate, official_threshold=0.01, quality_threshold=0.005
    )
    assert report["should_fail"] is True
    assert report["summary"][0]["official_regressions"] == 1
    assert report["summary"][0]["quality_regressions"] == 1


def test_compare_reports_flags_new_invalid_solution():
    baseline = {
        "scenarios": [
            {
                "scenario": "exact_fit",
                "results": [
                    {
                        "method": "solver",
                        "official_score": 1.0,
                        "quality_score": 1.0,
                        "valid": True,
                    }
                ],
            }
        ]
    }
    candidate = {
        "scenarios": [
            {
                "scenario": "exact_fit",
                "results": [
                    {
                        "method": "solver",
                        "official_score": 0.0,
                        "quality_score": 0.0,
                        "valid": False,
                    }
                ],
            }
        ]
    }

    report = compare_reports(
        baseline, candidate, official_threshold=0.01, quality_threshold=0.005
    )
    assert report["should_fail"] is True
    assert report["summary"][0]["new_invalid"] == 1
