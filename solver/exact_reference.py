from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .portfolio_block import solve_request as solve_portfolio_request
from .reference_core import (
    REFERENCE_SOLVER_VERSION,
    ReferenceAttempt,
    build_certificate,
    coarse_upper_bound,
    obvious_optimal_attempt,
    try_layer_pattern_exact,
    try_single_layer_exact,
    try_small_exact_search,
)


def _attempt_sort_key(attempt: ReferenceAttempt) -> tuple:
    gap = attempt.packing_score_ub - attempt.packing_score_lb
    return (
        attempt.packing_score_lb,
        -gap,
        int(attempt.proven_optimal),
        int(attempt.eps_optimal),
    )


def _best_attempt(attempts: List[ReferenceAttempt]) -> ReferenceAttempt:
    return max(attempts, key=_attempt_sort_key)


def _mark_reference_response(response: Dict[str, Any], wall_time_ms: int) -> None:
    response["solve_time_ms"] = 1
    response["wall_time_ms"] = wall_time_ms
    response["solver_version"] = REFERENCE_SOLVER_VERSION
    response["scoring_mode"] = "reference_no_time_penalty"


def certify_request(
    request: Dict[str, Any],
    model_dir: str | Path = "models",
    time_budget_ms: int = 0,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    soft_budget_ms = max(time_budget_ms, 10000) if time_budget_ms is not None else 10000

    incumbent_response = solve_portfolio_request(
        request=request,
        model_dir=model_dir,
        time_budget_ms=min(2000, max(900, soft_budget_ms // 5)),
    )
    incumbent_elapsed = int((time.perf_counter() - t0) * 1000)

    obvious = obvious_optimal_attempt(
        request=request,
        response=incumbent_response,
        wall_time_ms=incumbent_elapsed,
        stage="portfolio_block",
    )
    if obvious is not None:
        _mark_reference_response(obvious.response, incumbent_elapsed)
        return build_certificate(request, obvious)

    attempts: List[ReferenceAttempt] = []
    baseline = try_small_exact_search(
        request,
        incumbent_response=incumbent_response,
        time_limit_s=max(0.25, soft_budget_ms / 1000.0 * 0.35),
        solver_version=REFERENCE_SOLVER_VERSION,
    )
    if baseline is not None:
        attempts.append(baseline)

    if not attempts:
        from .reference_core import _extract_attempt  # local import keeps the public surface small

        baseline_attempt = _extract_attempt(
            request,
            stage="portfolio_block",
            response=incumbent_response,
            wall_time_ms=incumbent_elapsed,
            proven_optimal=False,
            upper_bound=coarse_upper_bound(request),
            notes="Baseline lower bound from the fast portfolio solver.",
        )
        if baseline_attempt is not None:
            attempts.append(baseline_attempt)

    single_layer = try_single_layer_exact(
        request,
        time_limit_s=max(0.2, soft_budget_ms / 1000.0 * 0.30),
        solver_version=REFERENCE_SOLVER_VERSION,
    )
    if single_layer is not None:
        attempts.append(single_layer)

    layer_pattern = try_layer_pattern_exact(
        request,
        time_limit_s=max(0.2, soft_budget_ms / 1000.0 * 0.25),
        solver_version=REFERENCE_SOLVER_VERSION,
    )
    if layer_pattern is not None:
        attempts.append(layer_pattern)

    compact_exact = try_small_exact_search(
        request,
        incumbent_response=incumbent_response,
        time_limit_s=max(0.5, soft_budget_ms / 1000.0 * 0.60),
        solver_version=REFERENCE_SOLVER_VERSION,
    )
    if compact_exact is not None:
        attempts.append(compact_exact)

    best = _best_attempt(attempts)
    total_elapsed_ms = int((time.perf_counter() - t0) * 1000)

    global_upper = min(attempt.packing_score_ub for attempt in attempts)
    _mark_reference_response(best.response, total_elapsed_ms)
    final_attempt = ReferenceAttempt(
        stage=best.stage,
        response=best.response,
        packing_score_lb=best.packing_score_lb,
        packing_score_ub=max(best.packing_score_lb, global_upper),
        proven_optimal=best.proven_optimal and global_upper <= best.packing_score_lb + 1e-9,
        eps_optimal=global_upper - best.packing_score_lb <= 0.005,
        wall_time_ms=total_elapsed_ms,
        notes=best.notes,
    )
    return build_certificate(request, final_attempt)


def solve_request(
    request: Dict[str, Any],
    model_dir: str | Path = "models",
    time_budget_ms: int = 0,
) -> Dict[str, Any]:
    certificate = certify_request(
        request=request,
        model_dir=model_dir,
        time_budget_ms=time_budget_ms,
    )
    return certificate["response"]
