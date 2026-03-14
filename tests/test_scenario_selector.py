from __future__ import annotations

from generator import generate_scenario
from solver.scenario_selector import ScenarioSelector, compute_request_fingerprint


def test_compute_request_fingerprint_has_bounded_ratios():
    request = generate_scenario("test_selector_fp", "fragile_mix", seed=47)
    fingerprint = compute_request_fingerprint(request)

    assert fingerprint.total_items > 0
    assert fingerprint.sku_count > 0
    assert 0.0 <= fingerprint.max_sku_share <= 1.0
    assert 0.0 <= fingerprint.upright_ratio <= 1.0
    assert 0.0 <= fingerprint.fragile_ratio <= 1.0
    assert 0.0 <= fingerprint.non_stackable_ratio <= 1.0
    assert fingerprint.volume_ratio > 0.0
    assert fingerprint.weight_ratio > 0.0


def test_selector_load_gracefully_fails_without_artifact(tmp_path):
    selector = ScenarioSelector(model_dir=tmp_path / "missing-model")
    assert selector.load() is False
