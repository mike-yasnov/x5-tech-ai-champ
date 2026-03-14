"""Tests for validator: hard constraints and soft metrics."""

import pytest
from validator import evaluate_solution


def _make_request(boxes, pallet=None):
    if pallet is None:
        pallet = {
            "length_mm": 1200, "width_mm": 800,
            "max_height_mm": 1800, "max_weight_kg": 1000.0,
        }
    return {
        "task_id": "test",
        "pallet": pallet,
        "boxes": boxes,
    }


def _make_response(placements, solve_time_ms=50):
    return {
        "task_id": "test",
        "solver_version": "1.0.0",
        "solve_time_ms": solve_time_ms,
        "placements": placements,
        "unplaced": [],
    }


def _placement(sku_id, x, y, z, l, w, h, idx=0, rot="LWH"):
    return {
        "sku_id": sku_id,
        "instance_index": idx,
        "position": {"x_mm": x, "y_mm": y, "z_mm": z},
        "dimensions_placed": {"length_mm": l, "width_mm": w, "height_mm": h},
        "rotation_code": rot,
    }


# ── rotation_code validation ───────────────────────────────────

class TestRotationCodeValidation:
    def test_valid_rotation_codes_accepted(self):
        boxes = [{"sku_id": "A", "description": "", "length_mm": 100, "width_mm": 100,
                  "height_mm": 100, "weight_kg": 1.0, "quantity": 1,
                  "strict_upright": False, "fragile": False, "stackable": True}]
        for code in ["LWH", "LHW", "WLH", "WHL", "HLW", "HWL"]:
            req = _make_request(boxes)
            resp = _make_response([_placement("A", 0, 0, 0, 100, 100, 100, rot=code)])
            result = evaluate_solution(req, resp)
            assert result["valid"], f"rotation_code {code} should be valid"

    def test_invalid_rotation_code_rejected(self):
        boxes = [{"sku_id": "A", "description": "", "length_mm": 100, "width_mm": 100,
                  "height_mm": 100, "weight_kg": 1.0, "quantity": 1,
                  "strict_upright": False, "fragile": False, "stackable": True}]
        req = _make_request(boxes)
        resp = _make_response([_placement("A", 0, 0, 0, 100, 100, 100, rot="INVALID")])
        result = evaluate_solution(req, resp)
        assert not result["valid"]
        assert "rotation_code" in result["error"]

    def test_empty_rotation_code_rejected(self):
        boxes = [{"sku_id": "A", "description": "", "length_mm": 100, "width_mm": 100,
                  "height_mm": 100, "weight_kg": 1.0, "quantity": 1,
                  "strict_upright": False, "fragile": False, "stackable": True}]
        req = _make_request(boxes)
        resp = _make_response([_placement("A", 0, 0, 0, 100, 100, 100, rot="")])
        result = evaluate_solution(req, resp)
        assert not result["valid"]


# ── stackable constraint ────────────────────────────────────────

class TestStackableConstraint:
    def test_box_on_non_stackable_rejected(self):
        """stackable=false: nothing can be placed on top."""
        boxes = [
            {"sku_id": "BOTTOM", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 200, "weight_kg": 5.0, "quantity": 1,
             "strict_upright": False, "fragile": False, "stackable": False},
            {"sku_id": "TOP", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 200, "weight_kg": 3.0, "quantity": 1,
             "strict_upright": False, "fragile": False, "stackable": True},
        ]
        req = _make_request(boxes)
        resp = _make_response([
            _placement("BOTTOM", 0, 0, 0, 400, 400, 200),
            _placement("TOP", 0, 0, 200, 400, 400, 200),
        ])
        result = evaluate_solution(req, resp)
        assert not result["valid"]
        assert "non-stackable" in result["error"]

    def test_non_stackable_alone_valid(self):
        """stackable=false box alone on floor is fine."""
        boxes = [
            {"sku_id": "ALONE", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 200, "weight_kg": 5.0, "quantity": 1,
             "strict_upright": False, "fragile": False, "stackable": False},
        ]
        req = _make_request(boxes)
        resp = _make_response([_placement("ALONE", 0, 0, 0, 400, 400, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"]

    def test_non_stackable_no_overlap_valid(self):
        """Box next to (not on top of) non-stackable is fine."""
        boxes = [
            {"sku_id": "NS", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 200, "weight_kg": 5.0, "quantity": 1,
             "strict_upright": False, "fragile": False, "stackable": False},
            {"sku_id": "SIDE", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 200, "weight_kg": 3.0, "quantity": 1,
             "strict_upright": False, "fragile": False, "stackable": True},
        ]
        req = _make_request(boxes)
        resp = _make_response([
            _placement("NS", 0, 0, 0, 400, 400, 200),
            _placement("SIDE", 400, 0, 0, 400, 400, 200),  # рядом, не сверху
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"]


# ── fragility penalty ───────────────────────────────────────────

class TestFragilityPenalty:
    def test_heavy_on_fragile_penalty(self):
        """Heavy (>2kg) on fragile → fragility_score = 0.95."""
        boxes = [
            {"sku_id": "FRAGILE", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 100, "weight_kg": 2.0, "quantity": 1,
             "strict_upright": False, "fragile": True, "stackable": True},
            {"sku_id": "HEAVY", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 100, "weight_kg": 3.0, "quantity": 1,
             "strict_upright": False, "fragile": False, "stackable": True},
        ]
        req = _make_request(boxes)
        resp = _make_response([
            _placement("FRAGILE", 0, 0, 0, 400, 400, 100),
            _placement("HEAVY", 0, 0, 100, 400, 400, 100),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"]
        assert result["metrics"]["fragility_score"] == pytest.approx(0.95)

    def test_light_on_fragile_no_penalty(self):
        """Light (≤2kg) on fragile → fragility_score = 1.0."""
        boxes = [
            {"sku_id": "FRAGILE", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 100, "weight_kg": 2.0, "quantity": 1,
             "strict_upright": False, "fragile": True, "stackable": True},
            {"sku_id": "LIGHT", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 100, "weight_kg": 2.0, "quantity": 1,
             "strict_upright": False, "fragile": False, "stackable": True},
        ]
        req = _make_request(boxes)
        resp = _make_response([
            _placement("FRAGILE", 0, 0, 0, 400, 400, 100),
            _placement("LIGHT", 0, 0, 100, 400, 400, 100),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"]
        assert result["metrics"]["fragility_score"] == pytest.approx(1.0)

    def test_multiple_fragility_violations(self):
        """Multiple heavy on fragile → cumulative penalty (-0.05 each)."""
        boxes = [
            {"sku_id": "FRAGILE", "description": "", "length_mm": 800, "width_mm": 400,
             "height_mm": 100, "weight_kg": 2.0, "quantity": 1,
             "strict_upright": False, "fragile": True, "stackable": True},
            {"sku_id": "HEAVY", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 100, "weight_kg": 5.0, "quantity": 2,
             "strict_upright": False, "fragile": False, "stackable": True},
        ]
        req = _make_request(boxes)
        resp = _make_response([
            _placement("FRAGILE", 0, 0, 0, 800, 400, 100),
            _placement("HEAVY", 0, 0, 100, 400, 400, 100, idx=0),
            _placement("HEAVY", 400, 0, 100, 400, 400, 100, idx=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"]
        # 2 heavy boxes on 1 fragile = 2 violations → 1.0 - 0.05*2 = 0.90
        assert result["metrics"]["fragility_score"] == pytest.approx(0.90)

    def test_no_fragile_perfect_score(self):
        """No fragile items → fragility_score = 1.0."""
        boxes = [
            {"sku_id": "A", "description": "", "length_mm": 400, "width_mm": 400,
             "height_mm": 200, "weight_kg": 10.0, "quantity": 2,
             "strict_upright": False, "fragile": False, "stackable": True},
        ]
        req = _make_request(boxes)
        resp = _make_response([
            _placement("A", 0, 0, 0, 400, 400, 200, idx=0),
            _placement("A", 0, 0, 200, 400, 400, 200, idx=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"]
        assert result["metrics"]["fragility_score"] == pytest.approx(1.0)


# ── time score thresholds ───────────────────────────────────────

class TestTimeScore:
    @pytest.mark.parametrize("ms,expected", [
        (500, 1.0), (1000, 1.0),   # ≤1s → 1.0
        (1001, 0.7), (5000, 0.7),  # 1-5s → 0.7
        (5001, 0.3), (30000, 0.3), # 5-30s → 0.3
        (30001, 0.0), (99999, 0.0), # >30s → 0.0
    ])
    def test_time_score_thresholds(self, ms, expected):
        boxes = [{"sku_id": "A", "description": "", "length_mm": 100, "width_mm": 100,
                  "height_mm": 100, "weight_kg": 1.0, "quantity": 1,
                  "strict_upright": False, "fragile": False, "stackable": True}]
        req = _make_request(boxes)
        resp = _make_response([_placement("A", 0, 0, 0, 100, 100, 100)], solve_time_ms=ms)
        result = evaluate_solution(req, resp)
        assert result["metrics"]["time_score"] == pytest.approx(expected)


# ── scoring formula ─────────────────────────────────────────────

class TestScoringFormula:
    def test_formula_weights(self):
        """final_score = 0.50*vol + 0.30*cov + 0.10*frag + 0.10*time."""
        boxes = [{"sku_id": "A", "description": "", "length_mm": 100, "width_mm": 100,
                  "height_mm": 100, "weight_kg": 1.0, "quantity": 1,
                  "strict_upright": False, "fragile": False, "stackable": True}]
        pallet = {"length_mm": 1000, "width_mm": 1000, "max_height_mm": 1000, "max_weight_kg": 100.0}
        req = _make_request(boxes, pallet)
        resp = _make_response([_placement("A", 0, 0, 0, 100, 100, 100)])
        result = evaluate_solution(req, resp)
        m = result["metrics"]
        expected = 0.50 * m["volume_utilization"] + 0.30 * m["item_coverage"] + \
                   0.10 * m["fragility_score"] + 0.10 * m["time_score"]
        assert result["final_score"] == pytest.approx(expected, abs=0.001)
