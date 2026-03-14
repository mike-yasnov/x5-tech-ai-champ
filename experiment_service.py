"""Shared experiment orchestration for the FastAPI + NiceGUI app."""

from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from generator import generate_scenario
from scenario_catalog import BENCHMARK_SCENARIOS, BENCHMARK_SCENARIO_NAMES
from solver.models import Box, Pallet, solution_to_dict
from solver.solver import STRATEGIES, solve
from validator import evaluate_solution
from visualize import build_scenario_viz_data

DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "volume_utilization": 50.0,
    "item_coverage": 30.0,
    "fragility_score": 10.0,
    "time_score": 10.0,
}
SCORE_WEIGHT_KEYS = tuple(DEFAULT_SCORE_WEIGHTS.keys())


def clone_data(value: Any) -> Any:
    return copy.deepcopy(value)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    cleaned = []
    previous_separator = False
    for char in text.strip().lower():
        if char.isalnum():
            cleaned.append(char)
            previous_separator = False
            continue
        if not previous_separator:
            cleaned.append("_")
            previous_separator = True
    return "".join(cleaned).strip("_") or "experiment"


def make_task_id(name: str, scenario_type: str, seed: Optional[int]) -> str:
    parts = [slugify(name)]
    if scenario_type:
        parts.append(slugify(scenario_type))
    if seed is not None:
        parts.append(str(seed))
    return "_".join(part for part in parts if part)


def generate_request_from_scenario(
    scenario_type: str,
    seed: int,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    task_id = make_task_id(name or scenario_type, scenario_type, seed)
    return generate_scenario(task_id, scenario_type, seed=seed)


def request_to_models(request_dict: Dict[str, Any]) -> tuple[str, Pallet, List[Box]]:
    pallet_data = request_dict["pallet"]
    pallet = Pallet(
        type_id=pallet_data.get("type_id", pallet_data.get("id", "unknown")),
        length_mm=pallet_data["length_mm"],
        width_mm=pallet_data["width_mm"],
        max_height_mm=pallet_data["max_height_mm"],
        max_weight_kg=pallet_data["max_weight_kg"],
    )
    boxes = [
        Box(
            sku_id=box["sku_id"],
            description=box.get("description", ""),
            length_mm=box["length_mm"],
            width_mm=box["width_mm"],
            height_mm=box["height_mm"],
            weight_kg=box["weight_kg"],
            quantity=box["quantity"],
            strict_upright=box.get("strict_upright", False),
            fragile=box.get("fragile", False),
            stackable=box.get("stackable", True),
        )
        for box in request_dict.get("boxes", [])
    ]
    return request_dict["task_id"], pallet, boxes


def normalize_score_weights(score_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    raw = {
        key: max(0.0, float((score_weights or {}).get(key, default)))
        for key, default in DEFAULT_SCORE_WEIGHTS.items()
    }
    total = sum(raw.values())
    if total <= 0:
        raw = dict(DEFAULT_SCORE_WEIGHTS)
        total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def compute_weighted_score(
    metrics: Dict[str, float],
    score_weights: Optional[Dict[str, float]],
) -> tuple[float, Dict[str, float], Dict[str, float]]:
    normalized = normalize_score_weights(score_weights)
    weighted_components = {
        key: round(float(metrics.get(key, 0.0)) * normalized[key], 4)
        for key in SCORE_WEIGHT_KEYS
    }
    final_score = round(sum(weighted_components.values()), 4)
    return final_score, normalized, weighted_components


def evaluate_with_score_weights(
    request_dict: Dict[str, Any],
    response_dict: Dict[str, Any],
    score_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    evaluation = clone_data(evaluate_solution(request_dict, response_dict))
    raw_weights = {
        key: max(0.0, float((score_weights or {}).get(key, default)))
        for key, default in DEFAULT_SCORE_WEIGHTS.items()
    }
    normalized = normalize_score_weights(raw_weights)

    evaluation["score_weights"] = raw_weights
    evaluation["normalized_score_weights"] = {
        key: round(value, 4) for key, value in normalized.items()
    }

    if not evaluation.get("valid", False):
        evaluation["base_final_score"] = float(evaluation.get("final_score", 0.0))
        evaluation["weighted_components"] = {
            key: 0.0 for key in SCORE_WEIGHT_KEYS
        }
        evaluation["final_score"] = 0.0
        return evaluation

    metrics = evaluation.get("metrics", {})
    weighted_score, _, weighted_components = compute_weighted_score(metrics, raw_weights)
    evaluation["base_final_score"] = float(evaluation.get("final_score", 0.0))
    evaluation["weighted_components"] = weighted_components
    evaluation["final_score"] = weighted_score
    return evaluation


def run_solver_for_request(
    request_dict: Dict[str, Any],
    strategy: str = "portfolio_block",
    time_budget_ms: int = 900,
    model_dir: str = "models",
    n_restarts: int = 10,
    beam_width: Optional[int] = None,
    score_weights: Optional[Dict[str, float]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    task_id, pallet, boxes = request_to_models(request_dict)
    started = time.perf_counter()
    solution = solve(
        task_id=task_id,
        pallet=pallet,
        boxes=boxes,
        request_dict=request_dict,
        n_restarts=n_restarts,
        time_budget_ms=time_budget_ms,
        beam_width=beam_width,
        model_dir=model_dir,
        strategy=strategy,
    )
    response_dict = solution_to_dict(solution)
    evaluation = evaluate_with_score_weights(
        request_dict=request_dict,
        response_dict=response_dict,
        score_weights=score_weights,
    )
    evaluation["wall_time_ms"] = int((time.perf_counter() - started) * 1000)
    return response_dict, evaluation


class ExperimentService:
    """Stores experiment state, history, and reproducible solver runs."""

    def __init__(
        self,
        history_path: str | Path = ".history/ui_request_history.json",
        model_dir: str = "models",
    ) -> None:
        self.history_path = Path(history_path)
        self.model_dir = model_dir
        self._records: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        self._load_history()

    def _load_history(self) -> None:
        if not self.history_path.exists():
            return
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, list):
            return

        for item in payload:
            if not isinstance(item, dict):
                continue
            experiment_id = item.get("id")
            request_dict = item.get("request")
            if not experiment_id or not isinstance(request_dict, dict):
                continue
            self._records[experiment_id] = item
            self._order.append(experiment_id)

    def _save_history(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [self._records[experiment_id] for experiment_id in self._order]
        self.history_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _move_to_front(self, experiment_id: str) -> None:
        if experiment_id in self._order:
            self._order.remove(experiment_id)
        self._order.insert(0, experiment_id)

    def _touch(self, record: Dict[str, Any]) -> None:
        record["updated_at"] = utc_now_iso()

    def _summary(self, record: Dict[str, Any]) -> Dict[str, Any]:
        request_dict = record.get("request", {})
        response_dict = record.get("response") or {}
        evaluation = record.get("evaluation") or {}
        total_items = sum(int(box.get("quantity", 0)) for box in request_dict.get("boxes", []))
        placed = len(response_dict.get("placements", []))
        return {
            "id": record["id"],
            "name": record.get("name", "Experiment"),
            "scenario_type": record.get("scenario_type", "custom"),
            "seed": record.get("seed"),
            "status": record.get("status", "draft"),
            "strategy": record.get("strategy", "portfolio_block"),
            "time_budget_ms": record.get("time_budget_ms", 900),
            "score": float(evaluation.get("final_score", 0.0)),
            "base_score": float(evaluation.get("base_final_score", evaluation.get("final_score", 0.0))),
            "placed": placed,
            "total_items": total_items,
            "solve_time_ms": response_dict.get("solve_time_ms", 0),
            "valid": bool(evaluation.get("valid", False)),
            "error": evaluation.get("error"),
            "updated_at": record.get("updated_at"),
            "response_source": record.get("response_source", "solver"),
            "notes": record.get("notes", ""),
        }

    def list_summaries(self) -> List[Dict[str, Any]]:
        return [self._summary(self._records[experiment_id]) for experiment_id in self._order]

    def list_records(self) -> List[Dict[str, Any]]:
        return [clone_data(self._records[experiment_id]) for experiment_id in self._order]

    def latest_id(self) -> Optional[str]:
        return self._order[0] if self._order else None

    def get_record(self, experiment_id: str) -> Dict[str, Any]:
        if experiment_id not in self._records:
            raise KeyError(f"Unknown experiment id: {experiment_id}")
        return clone_data(self._records[experiment_id])

    def _new_record(
        self,
        *,
        name: str,
        scenario_type: str,
        seed: Optional[int],
        request_dict: Dict[str, Any],
        strategy: str,
        time_budget_ms: int,
        n_restarts: int,
        beam_width: Optional[int],
        score_weights: Optional[Dict[str, float]],
        notes: str,
    ) -> Dict[str, Any]:
        record_id = f"exp_{uuid.uuid4().hex[:10]}"
        now = utc_now_iso()
        return {
            "id": record_id,
            "name": name,
            "scenario_type": scenario_type,
            "seed": seed,
            "request": clone_data(request_dict),
            "response": None,
            "evaluation": None,
            "strategy": strategy,
            "time_budget_ms": int(time_budget_ms),
            "n_restarts": int(n_restarts),
            "beam_width": beam_width,
            "score_weights": {
                key: float((score_weights or DEFAULT_SCORE_WEIGHTS).get(key, DEFAULT_SCORE_WEIGHTS[key]))
                for key in SCORE_WEIGHT_KEYS
            },
            "notes": notes,
            "status": "draft",
            "response_source": "solver",
            "created_at": now,
            "updated_at": now,
        }

    def ensure_default_experiment(self) -> Dict[str, Any]:
        if self._order:
            return self.get_record(self._order[0])
        scenario_type, seed = ("random_mixed", 45)
        return self.create_experiment_from_scenario(
            scenario_type=scenario_type,
            seed=seed,
            name=f"{scenario_type} · seed {seed}",
        )

    def create_experiment_from_scenario(
        self,
        *,
        scenario_type: str,
        seed: int,
        name: Optional[str] = None,
        strategy: str = "portfolio_block",
        time_budget_ms: int = 900,
        n_restarts: int = 10,
        beam_width: Optional[int] = None,
        score_weights: Optional[Dict[str, float]] = None,
        notes: str = "",
        run_now: bool = True,
    ) -> Dict[str, Any]:
        request_dict = generate_request_from_scenario(
            scenario_type=scenario_type,
            seed=seed,
            name=name or scenario_type,
        )
        experiment = self.create_experiment(
            name=name or f"{scenario_type} · seed {seed}",
            scenario_type=scenario_type,
            seed=seed,
            request_dict=request_dict,
            strategy=strategy,
            time_budget_ms=time_budget_ms,
            n_restarts=n_restarts,
            beam_width=beam_width,
            score_weights=score_weights,
            notes=notes,
            run_now=run_now,
        )
        return experiment

    def create_experiment(
        self,
        *,
        name: str,
        scenario_type: str,
        seed: Optional[int],
        request_dict: Dict[str, Any],
        strategy: str = "portfolio_block",
        time_budget_ms: int = 900,
        n_restarts: int = 10,
        beam_width: Optional[int] = None,
        score_weights: Optional[Dict[str, float]] = None,
        notes: str = "",
        run_now: bool = True,
    ) -> Dict[str, Any]:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        record = self._new_record(
            name=name,
            scenario_type=scenario_type,
            seed=seed,
            request_dict=request_dict,
            strategy=strategy,
            time_budget_ms=time_budget_ms,
            n_restarts=n_restarts,
            beam_width=beam_width,
            score_weights=score_weights,
            notes=notes,
        )
        self._records[record["id"]] = record
        self._move_to_front(record["id"])
        if run_now:
            self.run_experiment(record["id"])
        else:
            self._save_history()
        return self.get_record(record["id"])

    def clone_experiment(self, experiment_id: str) -> Dict[str, Any]:
        source = self._records[experiment_id]
        record = self._new_record(
            name=f"{source['name']} copy",
            scenario_type=source.get("scenario_type", "custom"),
            seed=source.get("seed"),
            request_dict=source["request"],
            strategy=source.get("strategy", "portfolio_block"),
            time_budget_ms=source.get("time_budget_ms", 900),
            n_restarts=source.get("n_restarts", 10),
            beam_width=source.get("beam_width"),
            score_weights=source.get("score_weights"),
            notes=source.get("notes", ""),
        )
        record["response"] = clone_data(source.get("response"))
        record["evaluation"] = clone_data(source.get("evaluation"))
        record["status"] = source.get("status", "ready")
        record["response_source"] = source.get("response_source", "solver")
        self._records[record["id"]] = record
        self._move_to_front(record["id"])
        self._save_history()
        return self.get_record(record["id"])

    def delete_experiment(self, experiment_id: str) -> None:
        if experiment_id not in self._records:
            return
        self._records.pop(experiment_id, None)
        if experiment_id in self._order:
            self._order.remove(experiment_id)
        self._save_history()

    def update_request(
        self,
        experiment_id: str,
        request_dict: Dict[str, Any],
        *,
        name: Optional[str] = None,
        scenario_type: Optional[str] = None,
        seed: Optional[int] = None,
        strategy: Optional[str] = None,
        time_budget_ms: Optional[int] = None,
        n_restarts: Optional[int] = None,
        beam_width: Optional[int] = None,
        notes: Optional[str] = None,
        run_now: bool = True,
    ) -> Dict[str, Any]:
        record = self._records[experiment_id]
        record["request"] = clone_data(request_dict)
        if name is not None:
            record["name"] = name
        if scenario_type is not None:
            record["scenario_type"] = scenario_type
        if seed is not None:
            record["seed"] = seed
        if strategy is not None:
            if strategy not in STRATEGIES:
                raise ValueError(f"Unknown strategy: {strategy}")
            record["strategy"] = strategy
        if time_budget_ms is not None:
            record["time_budget_ms"] = int(time_budget_ms)
        if n_restarts is not None:
            record["n_restarts"] = int(n_restarts)
        if beam_width is not None:
            record["beam_width"] = beam_width
        if notes is not None:
            record["notes"] = notes
        record["status"] = "draft"
        record["response"] = None
        record["evaluation"] = None
        record["response_source"] = "solver"
        self._touch(record)
        self._move_to_front(experiment_id)
        if run_now:
            self.run_experiment(experiment_id)
        else:
            self._save_history()
        return self.get_record(experiment_id)

    def update_score_weights(
        self,
        experiment_id: str,
        score_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        record = self._records[experiment_id]
        record["score_weights"] = {
            key: max(0.0, float(score_weights.get(key, DEFAULT_SCORE_WEIGHTS[key])))
            for key in SCORE_WEIGHT_KEYS
        }
        if record.get("response") is not None:
            record["evaluation"] = evaluate_with_score_weights(
                request_dict=record["request"],
                response_dict=record["response"],
                score_weights=record["score_weights"],
            )
        self._touch(record)
        self._move_to_front(experiment_id)
        self._save_history()
        return self.get_record(experiment_id)

    def update_response(
        self,
        experiment_id: str,
        response_dict: Dict[str, Any],
        *,
        response_source: str = "manual",
    ) -> Dict[str, Any]:
        record = self._records[experiment_id]
        record["response"] = clone_data(response_dict)
        record["response_source"] = response_source
        record["status"] = "ready"
        record["evaluation"] = evaluate_with_score_weights(
            request_dict=record["request"],
            response_dict=record["response"],
            score_weights=record["score_weights"],
        )
        self._touch(record)
        self._move_to_front(experiment_id)
        self._save_history()
        return self.get_record(experiment_id)

    def update_metadata(
        self,
        experiment_id: str,
        *,
        strategy: Optional[str] = None,
        time_budget_ms: Optional[int] = None,
        n_restarts: Optional[int] = None,
        beam_width: Optional[int] = None,
        notes: Optional[str] = None,
        run_now: bool = False,
    ) -> Dict[str, Any]:
        record = self._records[experiment_id]
        if strategy is not None:
            if strategy not in STRATEGIES:
                raise ValueError(f"Unknown strategy: {strategy}")
            record["strategy"] = strategy
        if time_budget_ms is not None:
            record["time_budget_ms"] = int(time_budget_ms)
        if n_restarts is not None:
            record["n_restarts"] = int(n_restarts)
        if beam_width is not None:
            record["beam_width"] = beam_width
        if notes is not None:
            record["notes"] = notes
        self._touch(record)
        self._move_to_front(experiment_id)
        if run_now and record.get("request"):
            self.run_experiment(experiment_id)
        else:
            self._save_history()
        return self.get_record(experiment_id)

    def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        record = self._records[experiment_id]
        record["status"] = "running"
        self._touch(record)
        response_dict, evaluation = run_solver_for_request(
            request_dict=record["request"],
            strategy=record["strategy"],
            time_budget_ms=record["time_budget_ms"],
            model_dir=self.model_dir,
            n_restarts=record.get("n_restarts", 10),
            beam_width=record.get("beam_width"),
            score_weights=record["score_weights"],
        )
        record["response"] = response_dict
        record["evaluation"] = evaluation
        record["status"] = "ready"
        record["response_source"] = "solver"
        self._touch(record)
        self._move_to_front(experiment_id)
        self._save_history()
        return self.get_record(experiment_id)

    def get_visualization_scenario(self, experiment_id: str) -> Dict[str, Any]:
        record = self._records[experiment_id]
        response_dict = record.get("response") or {"placements": [], "unplaced": []}
        evaluation = record.get("evaluation") or {}
        return build_scenario_viz_data(
            request_dict=record["request"],
            response_dict=response_dict,
            scenario_name=record["name"],
            score=float(evaluation.get("final_score", 0.0)),
        )


def make_experiment_draft(
    source: Optional[Dict[str, Any]] = None,
    scenario_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if source is None:
        base_scenario = scenario_type or BENCHMARK_SCENARIO_NAMES[0]
        base_seed = seed if seed is not None else BENCHMARK_SCENARIOS[0][1]
        request_dict = generate_request_from_scenario(
            scenario_type=base_scenario,
            seed=base_seed,
            name=f"{base_scenario} experiment",
        )
        return {
            "name": f"{base_scenario} · seed {base_seed}",
            "scenario_type": base_scenario,
            "seed": base_seed,
            "strategy": "portfolio_block",
            "time_budget_ms": 900,
            "n_restarts": 10,
            "beam_width": None,
            "score_weights": dict(DEFAULT_SCORE_WEIGHTS),
            "request": request_dict,
            "notes": "",
        }

    request_dict = clone_data(source["request"])
    return {
        "name": source.get("name", "Experiment"),
        "scenario_type": source.get("scenario_type", "custom"),
        "seed": source.get("seed"),
        "strategy": source.get("strategy", "portfolio_block"),
        "time_budget_ms": source.get("time_budget_ms", 900),
        "n_restarts": source.get("n_restarts", 10),
        "beam_width": source.get("beam_width"),
        "score_weights": clone_data(source.get("score_weights", DEFAULT_SCORE_WEIGHTS)),
        "request": request_dict,
        "notes": source.get("notes", ""),
    }
