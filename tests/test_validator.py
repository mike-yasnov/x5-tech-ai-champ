"""Comprehensive tests for validator hard constraints and soft metrics.

Tests enforce task spec compliance (docs/task.md). DO NOT MODIFY without explicit approval.
"""

from validator import evaluate_solution


# ============================================================
# Helpers
# ============================================================

def _pallet(length=1200, width=800, max_height=1800, max_weight=1000.0):
    return {
        "length_mm": length,
        "width_mm": width,
        "max_height_mm": max_height,
        "max_weight_kg": max_weight,
    }


def _box_spec(sku_id, length, width, height, weight, quantity=1,
              strict_upright=False, fragile=False, stackable=True):
    return {
        "sku_id": sku_id,
        "description": sku_id,
        "length_mm": length,
        "width_mm": width,
        "height_mm": height,
        "weight_kg": weight,
        "quantity": quantity,
        "strict_upright": strict_upright,
        "fragile": fragile,
        "stackable": stackable,
    }


def _placement(sku_id, x, y, z, length, width, height, instance_index=0,
               rotation_code="LWH"):
    return {
        "sku_id": sku_id,
        "instance_index": instance_index,
        "position": {"x_mm": x, "y_mm": y, "z_mm": z},
        "dimensions_placed": {
            "length_mm": length,
            "width_mm": width,
            "height_mm": height,
        },
        "rotation_code": rotation_code,
    }


def _request(boxes, pallet=None):
    return {
        "task_id": "test",
        "pallet": pallet or _pallet(),
        "boxes": boxes,
    }


def _response(placements, solve_time_ms=100):
    return {
        "task_id": "test",
        "solver_version": "test",
        "solve_time_ms": solve_time_ms,
        "placements": placements,
        "unplaced": [],
    }


# ============================================================
# HARD CONSTRAINTS
# ============================================================

class TestValidSimplePlacement:
    def test_single_box_on_floor(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True
        assert "final_score" in result
        assert result["final_score"] > 0

    def test_two_stacked_boxes(self):
        req = _request([_box_spec("A", 400, 400, 200, 5.0, quantity=2)])
        resp = _response([
            _placement("A", 0, 0, 0, 400, 400, 200, instance_index=0),
            _placement("A", 0, 0, 200, 400, 400, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True


class TestBoundsConstraint:
    def test_out_of_bounds_x(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 1000, 0, 0, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "bounds" in result["error"].lower() or "out" in result["error"].lower()

    def test_out_of_bounds_y(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 600, 0, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False

    def test_out_of_bounds_z(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 1700, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False

    def test_negative_x(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", -1, 0, 0, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False

    def test_negative_y(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, -1, 0, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False

    def test_negative_z(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, -1, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False

    def test_exactly_at_boundary_valid(self):
        req = _request([_box_spec("A", 1200, 800, 1800, 5.0)])
        resp = _response([_placement("A", 0, 0, 0, 1200, 800, 1800)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True


class TestCollisionConstraint:
    def test_overlapping_boxes_invalid(self):
        req = _request([_box_spec("A", 400, 400, 200, 5.0, quantity=2)])
        resp = _response([
            _placement("A", 0, 0, 0, 400, 400, 200, instance_index=0),
            _placement("A", 200, 200, 0, 400, 400, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "collision" in result["error"].lower() or "Collision" in result["error"]

    def test_adjacent_no_collision(self):
        req = _request([_box_spec("A", 400, 400, 200, 5.0, quantity=2)])
        resp = _response([
            _placement("A", 0, 0, 0, 400, 400, 200, instance_index=0),
            _placement("A", 400, 0, 0, 400, 400, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True

    def test_touching_faces_valid(self):
        """Boxes sharing a face (zero-width overlap) are NOT colliding."""
        req = _request([_box_spec("A", 600, 400, 200, 5.0, quantity=2)])
        resp = _response([
            _placement("A", 0, 0, 0, 600, 400, 200, instance_index=0),
            _placement("A", 600, 0, 0, 600, 400, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True


class TestWeightConstraint:
    def test_overweight_rejected(self):
        req = _request(
            [_box_spec("A", 400, 300, 200, 600.0, quantity=2)],
            pallet=_pallet(max_weight=1000.0),
        )
        resp = _response([
            _placement("A", 0, 0, 0, 400, 300, 200, instance_index=0),
            _placement("A", 400, 0, 0, 400, 300, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "overweight" in result["error"].lower() or "Overweight" in result["error"]

    def test_exactly_at_weight_limit_valid(self):
        req = _request(
            [_box_spec("A", 400, 300, 200, 500.0, quantity=2)],
            pallet=_pallet(max_weight=1000.0),
        )
        resp = _response([
            _placement("A", 0, 0, 0, 400, 300, 200, instance_index=0),
            _placement("A", 400, 0, 0, 400, 300, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True


class TestSupportConstraint:
    def test_floor_always_supported(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True

    def test_floating_box_rejected(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 500, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "support" in result["error"].lower()

    def test_60_percent_support_passes(self):
        """Exactly 60% support should be valid."""
        # Base: 500x400 at origin. Top: 500x400 at z=200, shifted 200 in x.
        # Overlap = 300*400 = 120000. Base area top = 500*400 = 200000.
        # Ratio = 120000/200000 = 0.60
        req = _request([
            _box_spec("BASE", 500, 400, 200, 5.0),
            _box_spec("TOP", 500, 400, 200, 5.0),
        ])
        resp = _response([
            _placement("BASE", 0, 0, 0, 500, 400, 200),
            _placement("TOP", 200, 0, 200, 500, 400, 200),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True

    def test_insufficient_support_fails(self):
        """Less than 60% support should fail."""
        # Base: 400x300 at origin. Top: 400x300 at z=200, shifted 300 in x.
        # Overlap = 100*300 = 30000. Base area = 400*300 = 120000.
        # Ratio = 30000/120000 = 0.25 < 0.60
        req = _request([
            _box_spec("BASE", 400, 300, 200, 5.0),
            _box_spec("TOP", 400, 300, 200, 5.0),
        ])
        resp = _response([
            _placement("BASE", 0, 0, 0, 400, 300, 200),
            _placement("TOP", 300, 0, 200, 400, 300, 200),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "support" in result["error"].lower()


class TestStrictUprightConstraint:
    def test_upright_LWH_valid(self):
        """LWH keeps height on Z — valid for strict_upright."""
        req = _request([_box_spec("A", 400, 300, 200, 5.0, strict_upright=True)])
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200, rotation_code="LWH")])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True

    def test_upright_WLH_valid(self):
        """WLH keeps height on Z (90° Z rotation) — valid for strict_upright."""
        req = _request([_box_spec("A", 400, 300, 200, 5.0, strict_upright=True)])
        resp = _response([_placement("A", 0, 0, 0, 300, 400, 200, rotation_code="WLH")])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True

    def test_upright_invalid_rotation(self):
        """HLW puts L on Z — invalid for strict_upright."""
        req = _request([_box_spec("A", 400, 300, 200, 5.0, strict_upright=True)])
        # HLW: H→X, L→Y, W→Z → placed dims (200, 400, 300), height=300 != orig 200
        resp = _response([_placement("A", 0, 0, 0, 200, 400, 300, rotation_code="HLW")])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "upright" in result["error"].lower()


class TestStackableConstraint:
    def test_non_stackable_box_rejects_top_placement(self):
        req = _request([
            _box_spec("BASE", 400, 400, 200, 10.0, stackable=False),
            _box_spec("TOP", 200, 200, 200, 5.0),
        ])
        resp = _response([
            _placement("BASE", 0, 0, 0, 400, 400, 200),
            _placement("TOP", 100, 100, 200, 200, 200, 200),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "non-stackable" in result["error"] or "stackable" in result["error"].lower()

    def test_non_stackable_no_xy_overlap_valid(self):
        """Non-stackable box with no XY overlap above — valid."""
        req = _request([
            _box_spec("BASE", 400, 400, 200, 10.0, stackable=False),
            _box_spec("SIDE", 200, 200, 200, 5.0),
        ])
        resp = _response([
            _placement("BASE", 0, 0, 0, 400, 400, 200),
            _placement("SIDE", 500, 0, 0, 200, 200, 200),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True

    def test_non_stackable_side_by_side_same_z_valid(self):
        """Non-stackable box next to another at same height — valid."""
        req = _request([
            _box_spec("NS", 400, 400, 200, 10.0, stackable=False),
            _box_spec("ADJ", 400, 400, 200, 5.0),
        ])
        resp = _response([
            _placement("NS", 0, 0, 0, 400, 400, 200),
            _placement("ADJ", 400, 0, 0, 400, 400, 200),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True


class TestAntiCheat:
    def test_wrong_dimensions_rejected(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        # Placed dims (500, 300, 200) don't match original (400, 300, 200)
        resp = _response([_placement("A", 0, 0, 0, 500, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "cheat" in result["error"].lower() or "Cheat" in result["error"]

    def test_too_many_instances_rejected(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0, quantity=1)])
        resp = _response([
            _placement("A", 0, 0, 0, 400, 300, 200, instance_index=0),
            _placement("A", 400, 0, 0, 400, 300, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "many" in result["error"].lower() or "Too" in result["error"]

    def test_unknown_sku_rejected(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("UNKNOWN", 0, 0, 0, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is False
        assert "unknown" in result["error"].lower() or "Unknown" in result["error"]


# ============================================================
# SOFT METRICS
# ============================================================

class TestVolumeUtilization:
    def test_calculation(self):
        pallet = _pallet(length=1200, width=800, max_height=1800)
        pallet_vol = 1200 * 800 * 1800
        box_vol = 400 * 300 * 200
        req = _request([_box_spec("A", 400, 300, 200, 5.0)], pallet=pallet)
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200)])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True
        expected = round(box_vol / pallet_vol, 4)
        assert result["metrics"]["volume_utilization"] == expected


class TestItemCoverage:
    def test_full_coverage(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0, quantity=2)])
        resp = _response([
            _placement("A", 0, 0, 0, 400, 300, 200, instance_index=0),
            _placement("A", 400, 0, 0, 400, 300, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["metrics"]["item_coverage"] == 1.0

    def test_partial_coverage(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0, quantity=4)])
        resp = _response([
            _placement("A", 0, 0, 0, 400, 300, 200, instance_index=0),
            _placement("A", 400, 0, 0, 400, 300, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["metrics"]["item_coverage"] == 0.5


class TestFragilityScore:
    def test_fragile_on_fragile_no_penalty(self):
        """Per spec: only NON-FRAGILE heavy (>2kg) on fragile triggers penalty."""
        req = _request([
            _box_spec("BOTTOM", 400, 400, 200, 1.0, fragile=True),
            _box_spec("TOP_FRAGILE", 400, 400, 200, 5.0, fragile=True),
        ])
        resp = _response([
            _placement("BOTTOM", 0, 0, 0, 400, 400, 200),
            _placement("TOP_FRAGILE", 0, 0, 200, 400, 400, 200),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True
        assert result["metrics"]["fragility_score"] == 1.0

    def test_non_fragile_heavy_on_fragile_penalized(self):
        req = _request([
            _box_spec("BOTTOM", 400, 400, 200, 1.0, fragile=True),
            _box_spec("TOP_STURDY", 400, 400, 200, 5.0, fragile=False),
        ])
        resp = _response([
            _placement("BOTTOM", 0, 0, 0, 400, 400, 200),
            _placement("TOP_STURDY", 0, 0, 200, 400, 400, 200),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True
        assert result["metrics"]["fragility_score"] == 0.95  # 1.0 - 0.05*1

    def test_light_box_on_fragile_no_penalty(self):
        """Box ≤ 2kg on fragile should NOT trigger penalty."""
        req = _request([
            _box_spec("BOTTOM", 400, 400, 200, 1.0, fragile=True),
            _box_spec("TOP_LIGHT", 400, 400, 200, 1.5, fragile=False),
        ])
        resp = _response([
            _placement("BOTTOM", 0, 0, 0, 400, 400, 200),
            _placement("TOP_LIGHT", 0, 0, 200, 400, 400, 200),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True
        assert result["metrics"]["fragility_score"] == 1.0

    def test_multiple_violations_accumulate(self):
        req = _request([
            _box_spec("FRAG1", 400, 400, 200, 1.0, fragile=True),
            _box_spec("FRAG2", 400, 400, 200, 1.0, fragile=True),
            _box_spec("HEAVY1", 400, 400, 200, 5.0, quantity=2),
        ])
        resp = _response([
            _placement("FRAG1", 0, 0, 0, 400, 400, 200),
            _placement("FRAG2", 400, 0, 0, 400, 400, 200),
            _placement("HEAVY1", 0, 0, 200, 400, 400, 200, instance_index=0),
            _placement("HEAVY1", 400, 0, 200, 400, 400, 200, instance_index=1),
        ])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True
        # 2 violations: each heavy on each fragile
        assert result["metrics"]["fragility_score"] == 0.90  # 1.0 - 0.05*2


class TestTimeScore:
    def test_under_1s(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200)], solve_time_ms=500)
        result = evaluate_solution(req, resp)
        assert result["metrics"]["time_score"] == 1.0

    def test_1_to_5s(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200)], solve_time_ms=3000)
        result = evaluate_solution(req, resp)
        assert result["metrics"]["time_score"] == 0.7

    def test_5_to_30s(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200)], solve_time_ms=15000)
        result = evaluate_solution(req, resp)
        assert result["metrics"]["time_score"] == 0.3

    def test_over_30s(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200)], solve_time_ms=60000)
        result = evaluate_solution(req, resp)
        assert result["metrics"]["time_score"] == 0.0

    def test_exactly_1s(self):
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200)], solve_time_ms=1000)
        result = evaluate_solution(req, resp)
        assert result["metrics"]["time_score"] == 1.0


class TestFinalScoreFormula:
    def test_weighted_sum(self):
        """Verify final_score = 0.5*vol + 0.3*cov + 0.1*frag + 0.1*time."""
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([_placement("A", 0, 0, 0, 400, 300, 200)], solve_time_ms=100)
        result = evaluate_solution(req, resp)
        m = result["metrics"]
        expected = (
            0.5 * m["volume_utilization"]
            + 0.3 * m["item_coverage"]
            + 0.1 * m["fragility_score"]
            + 0.1 * m["time_score"]
        )
        assert abs(result["final_score"] - expected) < 0.001

    def test_empty_placement_valid(self):
        """No placements should still be valid (score near 0)."""
        req = _request([_box_spec("A", 400, 300, 200, 5.0)])
        resp = _response([])
        result = evaluate_solution(req, resp)
        assert result["valid"] is True
        assert result["final_score"] <= 0.2  # only frag+time contribute
