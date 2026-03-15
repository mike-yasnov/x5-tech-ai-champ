"""FastAPI + NiceGUI app for interactive pallet packing experiments."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from nicegui import ui

from experiment_service import (
    DEFAULT_SCORE_WEIGHTS,
    SCORE_WEIGHT_KEYS,
    ExperimentService,
    clone_data,
    generate_request_from_scenario,
    make_experiment_draft,
    make_task_id,
)
from scenario_catalog import BENCHMARK_SCENARIO_NAMES
from solver.solver import STRATEGIES
from visualize import generate_scenario_html

fastapi_app = FastAPI(
    title="X5 Packing Lab",
    description="Interactive pallet packing playground powered by FastAPI and NiceGUI.",
)
service = ExperimentService()


@fastapi_app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@fastapi_app.get("/api/experiments")
def api_experiments() -> list[Dict[str, Any]]:
    return service.list_summaries()


@fastapi_app.get("/api/experiments/{experiment_id}")
def api_experiment(experiment_id: str) -> Dict[str, Any]:
    try:
        return service.get_record(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@fastapi_app.post("/api/experiments/{experiment_id}/rerun")
def api_rerun(experiment_id: str) -> Dict[str, Any]:
    try:
        return service.run_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@fastapi_app.get("/visualization/{experiment_id}", response_class=HTMLResponse)
def visualization(experiment_id: str) -> HTMLResponse:
    try:
        scenario = service.get_visualization_scenario(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return HTMLResponse(generate_scenario_html(scenario))


ui.run_with(
    fastapi_app,
    title="X5 Packing Lab",
    language="ru",
    dark=True,
    storage_secret="x5-packing-lab",
    show_welcome_message=False,
)

APP_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
"""

APP_CSS = """
/* ═══════════════════════════════════════════════════════
   X5 Tech Design System
   ═══════════════════════════════════════════════════════ */

:root {
  /* Dark base (60%) */
  --bg-deepest:    #111111;
  --bg-deep:       #171717;
  --bg-card:       #1F1F1F;
  --bg-elevated:   #2A2A2A;
  --bg-hover:      #333333;

  /* Brand green (10%) */
  --green-primary: #67DB3A;
  --green-bright:  #6FEA3A;
  --green-stable:  #59C436;
  --green-light:   #A8F36A;
  --green-soft:    rgba(103, 219, 58, 0.12);
  --green-glow:    rgba(103, 219, 58, 0.25);

  /* Text hierarchy */
  --text-primary:   #FFFFFF;
  --text-secondary: #EDEDED;
  --text-tertiary:  #BDBDBD;
  --text-muted:     #8E8E8E;

  /* Soft accents (5%) */
  --lavender:  #CBB7F6;
  --sand:      #EEDCA8;
  --mint:      #BFE8C6;
  --lime:      #D7F4A8;

  /* Semantic */
  --error:   #EF4444;
  --warning: #F59E0B;
  --success: #67DB3A;

  /* Structure */
  --border:       rgba(255, 255, 255, 0.08);
  --border-hover: rgba(255, 255, 255, 0.15);
  --border-green: rgba(103, 219, 58, 0.3);
  --shadow-lg:    0 24px 48px rgba(0, 0, 0, 0.4);
  --shadow-md:    0 12px 24px rgba(0, 0, 0, 0.3);
  --shadow-sm:    0 4px 12px rgba(0, 0, 0, 0.2);
  --radius-xl:  24px;
  --radius-lg:  18px;
  --radius-md:  12px;
  --radius-sm:  8px;
  --radius-pill: 999px;
}

/* ── Global ─────────────────────────────────────────── */

html, body, #q-app {
  min-height: 100vh;
  margin: 0;
  padding: 0;
  background: var(--bg-deepest) !important;
  color: var(--text-secondary);
  font-family: "Space Grotesk", "Segoe UI", sans-serif;
}
body { overflow-x: hidden; }

/* NiceGUI page container reset */
.nicegui-content {
  padding: 0 !important;
  margin: 0 !important;
}

.mono, .mono textarea, .mono pre, code, pre {
  font-family: "IBM Plex Mono", Consolas, monospace !important;
}

/* ── App Layout Grid ────────────────────────────────── */

.app-layout {
  display: grid;
  grid-template-columns: auto 1fr auto;
  grid-template-rows: auto 1fr auto;
  height: 100vh;
  width: 100%;
  max-width: 100vw;
  overflow: hidden;
}

/* ── Header ─────────────────────────────────────────── */

.app-header {
  grid-column: 1 / -1;
  grid-row: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 56px;
  background: var(--bg-deep);
  border-bottom: 1px solid var(--border);
  z-index: 50;
  gap: 12px;
}

.logo-text {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  white-space: nowrap;
}
.logo-accent { color: var(--green-primary); }

.header-experiment-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-tertiary);
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ── Sidebars ───────────────────────────────────────── */

.sidebar-left {
  grid-column: 1;
  grid-row: 2;
  width: 310px;
  overflow-y: auto;
  background: var(--bg-deep);
  border-right: 1px solid var(--border);
  padding: 16px;
  transition: width 0.25s ease, padding 0.25s ease, opacity 0.25s ease;
}
.sidebar-left.collapsed {
  width: 0;
  padding: 0;
  overflow: hidden;
  opacity: 0;
}

.sidebar-right {
  grid-column: 3;
  grid-row: 2;
  width: 330px;
  overflow-y: auto;
  background: var(--bg-deep);
  border-left: 1px solid var(--border);
  padding: 16px;
  transition: width 0.25s ease, padding 0.25s ease, opacity 0.25s ease;
}
.sidebar-right.collapsed {
  width: 0;
  padding: 0;
  overflow: hidden;
  opacity: 0;
}

/* ── Main Center ────────────────────────────────────── */

.main-center {
  grid-column: 2;
  grid-row: 2;
  overflow: hidden;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  min-height: 0;
}

/* ── Bottom Panel ───────────────────────────────────── */

.bottom-panel {
  grid-column: 1 / -1;
  grid-row: 3;
  border-top: 1px solid var(--border);
  background: var(--bg-deep);
  max-height: 38vh;
  overflow-y: auto;
  padding: 16px 20px;
  transition: max-height 0.3s ease, padding 0.3s ease, opacity 0.3s ease;
}
.bottom-panel.collapsed {
  max-height: 0;
  padding: 0 20px;
  overflow: hidden;
  opacity: 0;
}

/* ── Cards ──────────────────────────────────────────── */

.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-sm);
}

.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
}
.metric-card::before {
  content: '';
  position: absolute;
  top: -30px;
  right: -30px;
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background: var(--green-soft);
  pointer-events: none;
}

/* ── Typography helpers ─────────────────────────────── */

.section-kicker {
  color: var(--text-muted);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  font-size: 0.7rem;
  font-weight: 500;
}

.metric-value {
  font-size: 2.2rem;
  font-weight: 700;
  color: var(--green-primary);
  line-height: 1.1;
  letter-spacing: -0.02em;
}

.metric-label {
  font-size: 0.7rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  margin-bottom: 2px;
}

.metric-note {
  font-size: 0.8rem;
  color: var(--text-tertiary);
  margin-top: 4px;
}

.detail-note {
  color: var(--text-muted);
  font-size: 0.84rem;
  line-height: 1.55;
}

.score-value {
  color: var(--green-primary);
  font-weight: 700;
  font-size: 1.1rem;
}

.sidebar-title {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text-primary);
}

.card-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

/* ── Pills / Badges ─────────────────────────────────── */

.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: var(--radius-pill);
  font-size: 0.75rem;
  font-weight: 500;
  background: var(--bg-elevated);
  color: var(--text-tertiary);
  border: 1px solid var(--border);
  white-space: nowrap;
}
.pill.good {
  background: var(--green-soft);
  color: var(--green-light);
  border-color: rgba(103, 219, 58, 0.2);
}
.pill.warn {
  background: rgba(245, 158, 11, 0.12);
  color: var(--sand);
  border-color: rgba(245, 158, 11, 0.2);
}
.pill.cool {
  background: rgba(203, 183, 246, 0.1);
  color: var(--lavender);
  border-color: rgba(203, 183, 246, 0.2);
}

/* ── Buttons ────────────────────────────────────────── */

.btn-primary {
  background: var(--green-primary) !important;
  color: var(--bg-deepest) !important;
  font-weight: 600 !important;
  border-radius: var(--radius-pill) !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  padding: 6px 20px !important;
  transition: background 0.2s ease, box-shadow 0.2s ease !important;
}
.btn-primary:hover {
  background: var(--green-bright) !important;
  box-shadow: 0 0 24px var(--green-glow) !important;
}

.btn-secondary {
  background: var(--bg-elevated) !important;
  color: var(--text-secondary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-pill) !important;
  text-transform: none !important;
  font-weight: 500 !important;
  padding: 6px 16px !important;
}
.btn-secondary:hover {
  border-color: var(--border-hover) !important;
  background: var(--bg-hover) !important;
}

.btn-icon {
  color: var(--text-muted) !important;
}
.btn-icon:hover {
  color: var(--text-primary) !important;
}

/* ── History Cards ──────────────────────────────────── */

.history-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 14px 16px;
  cursor: pointer;
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.history-card:hover {
  transform: translateY(-2px);
  border-color: var(--border-hover);
  box-shadow: var(--shadow-md);
}
.history-card.active {
  border-color: var(--border-green);
  box-shadow: 0 0 0 1px var(--green-soft), var(--shadow-md);
}

/* ── Visual Frame ───────────────────────────────────── */

.visual-frame {
  width: 100%;
  height: 100%;
  border: 0;
  border-radius: var(--radius-xl);
  background: var(--bg-deepest);
}

.viz-wrapper {
  position: relative;
  flex: 1;
  min-height: 0;
  width: 100%;
  display: flex;
}

/* Ensure NiceGUI columns/rows inside main-center stretch */
.main-center > .nicegui-column,
.main-center > .nicegui-row,
.main-center .nicegui-column {
  min-height: 0;
}
.main-center > .nicegui-column > .nicegui-column {
  min-height: 0;
}

/* ── Code / JSON ────────────────────────────────────── */

.request-code {
  border: 1px solid var(--border);
  background: var(--bg-deepest);
  border-radius: var(--radius-lg);
  overflow: auto;
}
.request-code pre {
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}

.json-editor textarea {
  min-height: 30rem !important;
  font-family: "IBM Plex Mono", Consolas, monospace !important;
  font-size: 0.84rem !important;
  line-height: 1.6 !important;
  background: var(--bg-deepest) !important;
  color: var(--text-secondary) !important;
}

/* ── Dialog ─────────────────────────────────────────── */

.dialog-card {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-xl) !important;
}

.soft-line {
  border-top: 1px solid var(--border);
}

/* ── Quasar Overrides ───────────────────────────────── */

.q-field__control {
  background: var(--bg-elevated) !important;
  color: var(--text-secondary) !important;
}
.q-field__label { color: var(--text-muted) !important; }
.q-field--focused .q-field__label { color: var(--green-primary) !important; }
.q-field--focused .q-field__control::after { border-color: var(--green-primary) !important; }
.q-field__native, .q-field__input { color: var(--text-secondary) !important; }

.q-toggle__inner--truthy .q-toggle__track { background: var(--green-soft) !important; }
.q-toggle__inner--truthy .q-toggle__thumb:after { background: var(--green-primary) !important; }

.q-slider { width: 100% !important; min-width: 0 !important; }
.q-slider__track-container--h { width: 100% !important; }
.q-slider__track-container--h .q-slider__selection { background: var(--green-primary) !important; }
.q-slider__thumb .q-slider__thumb-container .q-slider__focus-ring { background: var(--green-glow) !important; }

/* Ensure sidebar slider containers stretch full width */
.sidebar-right .nicegui-slider,
.sidebar-right .q-slider {
  width: 100% !important;
  box-sizing: border-box !important;
}

.q-tab { color: var(--text-muted) !important; text-transform: none !important; }
.q-tab--active { color: var(--green-primary) !important; }
.q-tab-indicator { background: var(--green-primary) !important; }
.q-tab-panels { background: transparent !important; }
.q-tab-panel { padding: 16px 0 !important; }

.q-menu {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
.q-item { color: var(--text-secondary) !important; }
.q-item--active { color: var(--green-primary) !important; }

.q-dialog__backdrop { background: rgba(0, 0, 0, 0.7) !important; }

.q-notification { border-radius: var(--radius-md) !important; }

.q-card { background: var(--bg-card) !important; }

.q-separator { background: var(--border) !important; }

/* ── Scrollbar ──────────────────────────────────────── */

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--bg-elevated);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--bg-hover); }

/* ── Skeleton / Loading ─────────────────────────────── */

@keyframes shimmer {
  0% { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}

.skeleton {
  background: linear-gradient(90deg,
    var(--bg-elevated) 25%,
    var(--bg-hover) 50%,
    var(--bg-elevated) 75%
  );
  background-size: 800px 100%;
  animation: shimmer 1.8s ease-in-out infinite;
  border-radius: var(--radius-sm);
}

.skeleton-text {
  height: 1em;
  width: 60%;
  margin: 4px 0;
}

.skeleton-value {
  height: 2.2rem;
  width: 80%;
  margin: 6px 0;
}

.skeleton-block {
  height: 100%;
  width: 100%;
  min-height: 200px;
}

.loading-overlay {
  position: absolute;
  inset: 0;
  background: rgba(17, 17, 17, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  border-radius: inherit;
  backdrop-filter: blur(4px);
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.spinner {
  width: 36px;
  height: 36px;
  border: 3px solid var(--bg-elevated);
  border-top-color: var(--green-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.spinner-sm {
  width: 20px;
  height: 20px;
  border-width: 2px;
}

/* ── Sidebar close (mobile only) ───────────────────── */

.sidebar-close { display: none; }

/* ── Sidebar backdrop (mobile only) ────────────────── */

.sidebar-backdrop {
  display: none;
  position: fixed;
  grid-column: 1 / -1;
  grid-row: 1 / -1;
}

/* ── Responsive ─────────────────────────────────────── */

@media (max-width: 1200px) {
  .sidebar-left { width: 260px; }
  .sidebar-right { width: 280px; }
}

@media (max-width: 768px) {
  /* Grid: zero-width sidebar columns, center takes all space */
  .app-layout {
    grid-template-columns: 0px 1fr 0px;
    grid-template-rows: auto 1fr auto;
  }

  /* Header: shrink padding, hide experiment pills */
  .app-header {
    padding: 0 12px;
    height: 48px;
    gap: 8px;
  }
  .header-pills { display: none; }
  .logo-text { font-size: 16px; }

  /* Sidebars: fixed drawers, fully removed from grid flow */
  .sidebar-left, .sidebar-right {
    position: fixed;
    top: 48px;
    bottom: 0;
    z-index: 200;
    width: 85vw !important;
    max-width: 340px;
    box-shadow: var(--shadow-lg);
    background: var(--bg-deep);
    transition: transform 0.25s ease, opacity 0.25s ease;
    padding: 16px;
    overflow-y: auto;
    /* Override desktop grid-column width */
    min-width: 0;
  }
  .sidebar-left { left: 0; transform: translateX(0); }
  .sidebar-right { right: 0; transform: translateX(0); }
  .sidebar-left.collapsed {
    transform: translateX(-100%);
    opacity: 0;
    pointer-events: none;
  }
  .sidebar-right.collapsed {
    transform: translateX(100%);
    opacity: 0;
    pointer-events: none;
  }

  /* Main center fills entire row on mobile */
  .main-center {
    grid-column: 1 / -1;
    grid-row: 2;
  }

  /* Show sidebar close button on mobile */
  .sidebar-close { display: flex !important; }

  /* Sidebar backdrop on mobile */
  .sidebar-backdrop {
    display: block;
    position: fixed;
    inset: 0;
    top: 48px;
    background: rgba(0, 0, 0, 0.5);
    z-index: 199;
    backdrop-filter: blur(2px);
  }
  .sidebar-backdrop.hidden { display: none; }

  /* Main center: less padding */
  .main-center { padding: 8px; gap: 8px; }

  /* Metrics: horizontal scroll strip */
  .metrics-row {
    flex-wrap: nowrap !important;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    scroll-snap-type: x mandatory;
    gap: 8px !important;
    padding-bottom: 4px;
  }
  .metric-card {
    min-width: 130px !important;
    flex: 0 0 auto !important;
    scroll-snap-align: start;
    padding: 12px 14px;
  }
  .metric-value { font-size: 1.6rem; }
  .metric-card::before { width: 60px; height: 60px; top: -20px; right: -20px; }

  /* Bottom panel */
  .bottom-panel { max-height: 50vh; padding: 12px; }

  /* Visualization */
  .viz-wrapper { min-height: 300px; }

  /* Cards */
  .glass-card { border-radius: var(--radius-lg); }
  .history-card { padding: 12px; }
  .history-card:hover { transform: none; box-shadow: none; }

  /* Dialogs: full screen */
  .dialog-card {
    width: 100vw !important;
    max-width: 100vw !important;
    height: 100vh !important;
    max-height: 100vh !important;
    border-radius: 0 !important;
  }
  .json-editor textarea { min-height: 16rem !important; }

  /* Action buttons: compact for mobile header */
  .app-header .q-btn {
    height: 34px !important;
    min-height: 0 !important;
    max-height: 34px !important;
    padding: 0 12px !important;
    font-size: 0.75rem !important;
  }
  .app-header .q-btn--round,
  .app-header .q-btn--fab-mini {
    height: 30px !important;
    width: 30px !important;
    min-height: 0 !important;
    min-width: 0 !important;
    max-height: 30px !important;
    padding: 0 !important;
  }
  .app-header .q-btn__content {
    padding: 0 !important;
    min-height: 0 !important;
  }
  .app-header .q-icon {
    font-size: 18px !important;
  }
  /* Hide button labels on mobile, keep icons only */
  .app-header .q-btn__content {
    font-size: 0 !important;
    gap: 0 !important;
  }
  .app-header .q-btn__content .q-icon {
    font-size: 18px !important;
    margin: 0 !important;
  }

  /* Sidebar content height */
  .sidebar-left .overflow-auto { max-height: calc(100vh - 140px) !important; }

  /* Prevent text selection on interactive elements */
  .history-card, .metric-card, .btn-primary, .btn-secondary {
    -webkit-tap-highlight-color: transparent;
    user-select: none;
  }

  /* Hide scrollbar on mobile */
  ::-webkit-scrollbar { display: none; }
}

@media (max-width: 480px) {
  .app-header { height: 44px; }
  .logo-text { font-size: 14px; }
  .sidebar-left, .sidebar-right { top: 44px; }
  .sidebar-backdrop { top: 44px; }

  /* Header actions: only show key buttons */
  .header-actions-secondary { display: none; }

  .metric-card { min-width: 110px !important; padding: 10px 12px; }
  .metric-value { font-size: 1.3rem; }
  .metric-label { font-size: 0.6rem; }

  .main-center { padding: 6px; gap: 6px; }

  .sidebar-left, .sidebar-right { padding: 12px; }

  .section-kicker { font-size: 0.6rem; }
  .card-title { font-size: 0.9rem; }

  .bottom-panel { max-height: 60vh; }
}
"""

SCORE_LABELS = {
    "volume_utilization": ("Volume", "Плотность использования объема паллеты"),
    "item_coverage": ("Coverage", "Доля размещенных коробок"),
    "fragility_score": ("Fragility", "Штраф за тяжелое на хрупком"),
    "time_score": ("Time", "Очки за скорость работы солвера"),
}


def pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def short_timestamp(value: Optional[str]) -> str:
    if not value:
        return "без времени"
    return value.replace("T", " ").replace("+00:00", " UTC")


def total_requested_items(record: Dict[str, Any]) -> int:
    return sum(int(box.get("quantity", 0)) for box in record.get("request", {}).get("boxes", []))


@ui.page("/")
def index() -> None:
    ui.add_head_html(APP_HEAD)
    ui.add_css(APP_CSS)

    initial = service.ensure_default_experiment()
    state: Dict[str, Any] = {
        "selected_id": initial["id"],
        "autorun": True,
        "history_query": "",
        "left_open": False,
        "right_open": False,
        "bottom_open": False,
        "loading": False,
    }

    # ── Refs for sidebar/bottom elements ──
    left_ref = None
    right_ref = None
    bottom_ref = None
    backdrop_ref = None

    def current_record() -> Dict[str, Any]:
        selected_id = state.get("selected_id")
        if selected_id:
            try:
                return service.get_record(selected_id)
            except KeyError:
                pass
        fallback = service.ensure_default_experiment()
        state["selected_id"] = fallback["id"]
        return fallback

    def select_experiment(experiment_id: str) -> None:
        state["selected_id"] = experiment_id
        refresh_all()

    def refresh_all() -> None:
        header_info.refresh()
        header_actions.refresh()
        control_panel.refresh()
        metrics_panel.refresh()
        visualization_panel.refresh()
        details_panel.refresh()
        history_panel.refresh()

    def _set_loading(on: bool) -> None:
        state["loading"] = on
        header_actions.refresh()
        metrics_panel.refresh()
        visualization_panel.refresh()

    async def rerun_selected() -> None:
        _set_loading(True)
        await ui.run_javascript("void(0)")  # yield to render
        try:
            record = current_record()
            updated = service.run_experiment(record["id"])
            state["selected_id"] = updated["id"]
        finally:
            state["loading"] = False
            refresh_all()
        ui.notify(f"Пересчитан «{updated['name']}»", color="positive")

    async def update_solver_setting(key: str, value: Any) -> None:
        record = current_record()
        kwargs = {key: value, "run_now": bool(state["autorun"])}
        if state["autorun"]:
            _set_loading(True)
            await ui.run_javascript("void(0)")
        try:
            updated = service.update_metadata(record["id"], **kwargs)
            state["selected_id"] = updated["id"]
        finally:
            state["loading"] = False
            refresh_all()

    async def update_weight(key: str, value: float) -> None:
        record = current_record()
        weights = clone_data(record.get("score_weights", DEFAULT_SCORE_WEIGHTS))
        weights[key] = float(value)
        updated = service.update_score_weights(record["id"], weights)
        if state["autorun"]:
            _set_loading(True)
            await ui.run_javascript("void(0)")
            try:
                updated = service.run_experiment(updated["id"])
            finally:
                state["loading"] = False
        state["selected_id"] = updated["id"]
        refresh_all()

    async def reset_weights() -> None:
        record = current_record()
        updated = service.update_score_weights(record["id"], dict(DEFAULT_SCORE_WEIGHTS))
        if state["autorun"]:
            _set_loading(True)
            await ui.run_javascript("void(0)")
            try:
                updated = service.run_experiment(updated["id"])
            finally:
                state["loading"] = False
        state["selected_id"] = updated["id"]
        refresh_all()
        ui.notify("Веса score сброшены к дефолтным 50/30/10/10", color="primary")

    def clone_selected() -> None:
        record = current_record()
        cloned = service.clone_experiment(record["id"])
        state["selected_id"] = cloned["id"]
        refresh_all()
        ui.notify(f"Создана копия «{cloned['name']}»", color="primary")

    def delete_experiment(experiment_id: str) -> None:
        was_selected = state.get("selected_id") == experiment_id
        service.delete_experiment(experiment_id)
        if was_selected:
            latest_id = service.latest_id()
            if latest_id is None:
                latest_id = service.ensure_default_experiment()["id"]
            state["selected_id"] = latest_id
        refresh_all()
        ui.notify("Эксперимент удален из истории", color="warning")

    def _update_backdrop() -> None:
        nonlocal backdrop_ref
        if backdrop_ref:
            if state["left_open"] or state["right_open"]:
                backdrop_ref.classes(remove="hidden")
            else:
                backdrop_ref.classes(add="hidden")

    def toggle_left() -> None:
        nonlocal left_ref
        state["left_open"] = not state["left_open"]
        if left_ref:
            if state["left_open"]:
                left_ref.classes(remove="collapsed")
            else:
                left_ref.classes(add="collapsed")
        _update_backdrop()

    def toggle_right() -> None:
        nonlocal right_ref
        state["right_open"] = not state["right_open"]
        if right_ref:
            if state["right_open"]:
                right_ref.classes(remove="collapsed")
            else:
                right_ref.classes(add="collapsed")
        _update_backdrop()

    def toggle_bottom() -> None:
        nonlocal bottom_ref
        state["bottom_open"] = not state["bottom_open"]
        if bottom_ref:
            if state["bottom_open"]:
                bottom_ref.classes(remove="collapsed")
            else:
                bottom_ref.classes(add="collapsed")

    # ── Dialogs ──

    def _make_experiment_name(scenario_type: str, seed: int, request_dict: dict) -> str:
        """Auto-generate experiment name: preset uses 'type · seed N', custom uses box summary."""
        if scenario_type in BENCHMARK_SCENARIO_NAMES:
            return f"{scenario_type} · seed {seed}"
        n_skus = len(request_dict.get("boxes", []))
        total_qty = sum(int(b.get("quantity", 0)) for b in request_dict.get("boxes", []))
        return f"custom · {n_skus} SKU · {total_qty} boxes · seed {seed}"

    def open_request_editor(mode: str) -> None:
        source = current_record() if mode == "edit" else None
        draft = make_experiment_draft(source)

        dialog = ui.dialog()
        with dialog, ui.card().classes("dialog-card w-[96vw] max-w-6xl p-0"):
            with ui.column().classes("w-full gap-0"):
                with ui.row().classes("w-full items-center justify-between px-7 py-5"):
                    with ui.column().classes("gap-1"):
                        ui.label("EXPERIMENT BUILDER").classes("section-kicker")
                        ui.label(
                            "Создайте новый сценарий или подправьте текущий request."
                        ).classes("card-title")
                    ui.button(icon="close", on_click=dialog.close).props("flat round dense").classes("btn-icon")

                ui.separator().classes("soft-line")

                with ui.row().classes("w-full gap-6 px-7 py-6 items-start flex-wrap xl:flex-nowrap"):
                    with ui.column().classes("w-full xl:w-[24rem] gap-4"):
                        ui.label("Общие настройки").classes("section-kicker")
                        scenario_options = list(BENCHMARK_SCENARIO_NAMES)
                        if draft["scenario_type"] not in scenario_options:
                            scenario_options.append(draft["scenario_type"])
                        scenario_select = ui.select(
                            scenario_options,
                            label="Сценарий",
                            value=draft["scenario_type"],
                        ).classes("w-full")
                        seed_input = ui.number(
                            "Seed",
                            value=draft["seed"] if draft["seed"] is not None else 42,
                            min=0,
                            step=1,
                        ).classes("w-full")
                        strategy_select = ui.select(
                            list(STRATEGIES),
                            label="Стратегия солвера",
                            value=draft["strategy"],
                        ).classes("w-full")
                        time_budget_input = ui.number(
                            "Time budget, ms",
                            value=draft["time_budget_ms"],
                            min=50,
                            step=50,
                        ).classes("w-full")
                        effort_input = ui.number(
                            "Legacy effort",
                            value=draft["n_restarts"],
                            min=1,
                            step=1,
                        ).classes("w-full")
                        notes_input = ui.textarea(
                            "Заметки",
                            value=draft.get("notes", ""),
                            placeholder="Что именно хотите проверить в этом эксперименте?",
                        ).classes("w-full")

                        def load_preset() -> None:
                            scenario_type = str(scenario_select.value)
                            seed = int(seed_input.value or 0)
                            request_dict = generate_request_from_scenario(
                                scenario_type=scenario_type,
                                seed=seed,
                                name=f"{scenario_type} · seed {seed}",
                            )
                            request_editor.value = pretty_json(request_dict)
                            ui.notify(
                                f"Пресет «{scenario_type}» с seed={seed} загружен",
                                color="primary",
                            )

                        with ui.row().classes("w-full gap-3"):
                            ui.button(
                                "Загрузить пресет",
                                on_click=load_preset,
                                icon="auto_awesome",
                            ).classes("btn-primary")

                    with ui.column().classes("min-w-0 flex-1 gap-3"):
                        ui.label("Request JSON").classes("section-kicker")
                        request_editor = ui.textarea(
                            placeholder="Здесь можно править request целиком, включая boxes и pallet.",
                            value=pretty_json(draft["request"]),
                        ).classes("w-full mono json-editor")

                ui.separator().classes("soft-line")

                with ui.row().classes("w-full items-center justify-between px-7 py-5"):
                    ui.label(
                        "После сохранения эксперимент сразу перезапустится."
                    ).classes("detail-note")

                    async def save_request() -> None:
                        try:
                            request_dict = json.loads(request_editor.value or "{}")
                        except json.JSONDecodeError as exc:
                            ui.notify(f"Request JSON не парсится: {exc}", color="negative")
                            return
                        if not isinstance(request_dict, dict):
                            ui.notify("Request JSON должен быть объектом", color="negative")
                            return
                        exp_name = _make_experiment_name(
                            str(scenario_select.value or "custom"),
                            int(seed_input.value or 0),
                            request_dict,
                        )
                        request_dict["task_id"] = make_task_id(
                            exp_name,
                            str(scenario_select.value or "custom"),
                            int(seed_input.value or 0),
                        )
                        if "pallet" not in request_dict or "boxes" not in request_dict:
                            ui.notify("В request должны быть ключи pallet и boxes", color="negative")
                            return
                        dialog.close()
                        _set_loading(True)
                        await ui.run_javascript("void(0)")
                        try:
                            if mode == "edit":
                                updated = service.update_request(
                                    current_record()["id"],
                                    request_dict=request_dict,
                                    name=exp_name,
                                    scenario_type=str(scenario_select.value or "custom"),
                                    seed=int(seed_input.value or 0),
                                    strategy=str(strategy_select.value or "portfolio_block"),
                                    time_budget_ms=int(time_budget_input.value or 900),
                                    n_restarts=int(effort_input.value or 10),
                                    notes=str(notes_input.value or ""),
                                    run_now=True,
                                )
                                state["selected_id"] = updated["id"]
                            else:
                                created = service.create_experiment(
                                    name=exp_name,
                                    scenario_type=str(scenario_select.value or "custom"),
                                    seed=int(seed_input.value or 0),
                                    request_dict=request_dict,
                                    strategy=str(strategy_select.value or "portfolio_block"),
                                    time_budget_ms=int(time_budget_input.value or 900),
                                    n_restarts=int(effort_input.value or 10),
                                    score_weights=current_record().get("score_weights", DEFAULT_SCORE_WEIGHTS),
                                    notes=str(notes_input.value or ""),
                                    run_now=True,
                                )
                                state["selected_id"] = created["id"]
                        finally:
                            state["loading"] = False
                            refresh_all()
                        ui.notify("Эксперимент сохранен и запущен", color="positive")

                    with ui.row().classes("gap-3"):
                        ui.button("Отмена", on_click=dialog.close).classes("btn-secondary")
                        ui.button(
                            "Сохранить и запустить",
                            on_click=save_request,
                            icon="play_arrow",
                        ).classes("btn-primary")
        dialog.open()

    def open_result_editor() -> None:
        record = current_record()
        default_response = record.get("response") or {
            "task_id": record["request"]["task_id"],
            "solver_version": "manual-edit",
            "solve_time_ms": 0,
            "placements": [],
            "unplaced": [],
        }
        dialog = ui.dialog()
        with dialog, ui.card().classes("dialog-card w-[96vw] max-w-6xl p-0"):
            with ui.column().classes("w-full gap-0"):
                with ui.row().classes("w-full items-center justify-between px-7 py-5"):
                    with ui.column().classes("gap-1"):
                        ui.label("RESULT EDITOR").classes("section-kicker")
                        ui.label("Подправьте response и сразу увидите новый score.").classes("card-title")
                    ui.button(icon="close", on_click=dialog.close).props("flat round dense").classes("btn-icon")

                ui.separator().classes("soft-line")
                with ui.column().classes("w-full px-7 py-6 gap-3"):
                    ui.label(
                        "Если структура станет невалидной, валидатор это покажет."
                    ).classes("detail-note")
                    response_editor = ui.textarea(
                        value=pretty_json(default_response),
                        placeholder="Измените response JSON",
                    ).classes("w-full mono json-editor")

                ui.separator().classes("soft-line")
                with ui.row().classes("w-full items-center justify-end gap-3 px-7 py-5"):
                    ui.button("Отмена", on_click=dialog.close).classes("btn-secondary")

                    async def save_response() -> None:
                        try:
                            response_dict = json.loads(response_editor.value or "{}")
                        except json.JSONDecodeError as exc:
                            ui.notify(f"Response JSON не парсится: {exc}", color="negative")
                            return
                        if not isinstance(response_dict, dict):
                            ui.notify("Response JSON должен быть объектом", color="negative")
                            return
                        response_dict["task_id"] = record["request"]["task_id"]
                        dialog.close()
                        _set_loading(True)
                        await ui.run_javascript("void(0)")
                        try:
                            updated = service.update_response(record["id"], response_dict)
                            state["selected_id"] = updated["id"]
                        finally:
                            state["loading"] = False
                            refresh_all()
                        ui.notify("Результат обновлен и пересчитан", color="positive")

                    ui.button(
                        "Сохранить результат",
                        on_click=save_response,
                        icon="save",
                    ).classes("btn-primary")
        dialog.open()

    def history_matches(record: Dict[str, Any]) -> bool:
        query = str(state.get("history_query", "")).strip().lower()
        if not query:
            return True
        haystack = " ".join(
            [
                str(record.get("name", "")),
                str(record.get("scenario_type", "")),
                str(record.get("strategy", "")),
                str(record.get("notes", "")),
            ]
        ).lower()
        return query in haystack

    # ══════════════════════════════════════════════════════
    #  LAYOUT
    # ══════════════════════════════════════════════════════

    with ui.element("div").classes("app-layout"):

        # ── HEADER ──────────────────────────────────────
        with ui.element("div").classes("app-header"):
            with ui.row().classes("items-center gap-3 no-wrap"):
                ui.button(icon="menu", on_click=toggle_left).props("flat round dense").classes("btn-icon")
                ui.html('<span class="logo-text">X5 <span class="logo-accent">Packing Lab</span></span>')

            @ui.refreshable
            def header_info() -> None:
                record = current_record()
                evaluation = record.get("evaluation") or {}
                valid = bool(evaluation.get("valid", False))
                exp_name = record.get("name", "Experiment")
                with ui.row().classes("items-center gap-2 no-wrap header-pills"):
                    ui.label(exp_name).style(
                        "color: var(--text-primary); font-size: 14px; font-weight: 500;"
                        "max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                    )
                    ui.html(
                        f'<span class="pill {"good" if valid else "warn"}">'
                        f'{"VALID" if valid else "INVALID"}'
                        f"</span>"
                    )
                    ui.html(f'<span class="pill cool">{record.get("strategy", "portfolio_block")}</span>')
                    ui.html(f'<span class="pill">seed {record.get("seed", "?")}</span>')
                    ui.html(
                        f'<span class="pill">{len(record.get("request", {}).get("boxes", []))} SKU</span>'
                    )

            header_info()

            @ui.refreshable
            def header_actions() -> None:
                loading = state.get("loading", False)
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.button("Новый", on_click=lambda: open_request_editor("create"), icon="add").classes(
                        "btn-primary"
                    ).props("disable" if loading else "")
                    with ui.row().classes("items-center gap-1 no-wrap header-actions-secondary"):
                        ui.button(icon="content_copy", on_click=clone_selected).props(
                            "flat round dense" + (" disable" if loading else "")
                        ).classes("btn-icon").tooltip("Дублировать")
                        ui.button(icon="tune", on_click=lambda: open_request_editor("edit")).props(
                            "flat round dense" + (" disable" if loading else "")
                        ).classes("btn-icon").tooltip("Редактировать request")
                        ui.button(icon="edit_note", on_click=open_result_editor).props(
                            "flat round dense" + (" disable" if loading else "")
                        ).classes("btn-icon").tooltip("Редактировать result")
                    if loading:
                        with ui.row().classes("items-center gap-2 no-wrap"):
                            ui.html('<div class="spinner spinner-sm" style="display:inline-block"></div>')
                            ui.label("Solving...").style("color: var(--green-primary); font-size: 0.8rem; white-space: nowrap")
                    else:
                        ui.button("Запуск", on_click=rerun_selected, icon="play_arrow").classes("btn-primary")
                    ui.button(icon="tune", on_click=toggle_right).props("flat round dense").classes("btn-icon").tooltip("Панель управления")

            header_actions()

        # ── Mobile Backdrop ────────────────────────────
        backdrop_ref = ui.element("div").classes("sidebar-backdrop hidden")
        backdrop_ref.on("click", lambda: (
            toggle_left() if state["left_open"] else None,
            toggle_right() if state["right_open"] else None,
        ))

        # ── LEFT SIDEBAR: History ───────────────────────
        left_ref = ui.element("div").classes("sidebar-left collapsed")
        with left_ref:
            @ui.refreshable
            def history_panel() -> None:
                summaries = service.list_summaries()
                with ui.column().classes("w-full gap-4"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("История").classes("sidebar-title")
                        ui.button(icon="close", on_click=toggle_left).props("flat round dense").classes("btn-icon sidebar-close")
                    ui.input(
                        "Поиск",
                        value=state.get("history_query", ""),
                        placeholder="scenario, strategy, notes...",
                        on_change=lambda e: (state.__setitem__("history_query", e.value), history_panel.refresh()),
                    ).classes("w-full").props('dense outlined')
                    with ui.column().classes("w-full gap-3 overflow-auto pr-1").style("max-height: calc(100vh - 180px)"):
                        filtered = [s for s in summaries if history_matches(s)]
                        if not filtered:
                            ui.label("Ничего не найдено").classes("detail-note")
                        for summary in filtered:
                            active = summary["id"] == state.get("selected_id")
                            card = ui.element("div").classes(
                                "history-card " + ("active" if active else "")
                            )
                            card.on("click", lambda _=None, eid=summary["id"]: select_experiment(eid))
                            with card:
                                with ui.row().classes("w-full items-start justify-between gap-2"):
                                    with ui.column().classes("gap-0 min-w-0 flex-1"):
                                        ui.label(summary["name"]).classes("text-sm font-semibold").style("color: var(--text-primary)")
                                        ui.label(
                                            f"{summary['scenario_type']} · {summary['strategy']}"
                                        ).classes("detail-note").style("font-size: 0.75rem")
                                    ui.label(f"{summary['score']:.4f}").classes("score-value")
                                with ui.row().classes("w-full gap-2 flex-wrap mt-2"):
                                    ui.html(
                                        f'<span class="pill {"good" if summary["valid"] else "warn"}" style="font-size:0.68rem;padding:2px 8px">'
                                        f'{"valid" if summary["valid"] else "invalid"}'
                                        f"</span>"
                                    )
                                    ui.html(f'<span class="pill" style="font-size:0.68rem;padding:2px 8px">{summary["placed"]}/{summary["total_items"]}</span>')
                                    ui.html(f'<span class="pill" style="font-size:0.68rem;padding:2px 8px">{summary["solve_time_ms"]} ms</span>')
                                with ui.row().classes("w-full justify-between items-center mt-1"):
                                    ui.label(short_timestamp(summary["updated_at"])).classes("detail-note").style("font-size: 0.7rem")
                                    with ui.row().classes("gap-1"):
                                        ui.button(
                                            icon="content_copy",
                                            on_click=lambda _=None, eid=summary["id"]: (
                                                state.__setitem__("selected_id", service.clone_experiment(eid)["id"]),
                                                refresh_all(),
                                            ),
                                        ).props("flat round dense").classes("btn-icon").style("font-size: 0.7rem")
                                        if len(summaries) > 1:
                                            ui.button(
                                                icon="delete",
                                                on_click=lambda _=None, eid=summary["id"]: delete_experiment(eid),
                                            ).props("flat round dense").classes("btn-icon").style("font-size: 0.7rem")

            history_panel()

        # ── MAIN CENTER ─────────────────────────────────
        with ui.element("div").classes("main-center"):

            # Metrics row
            @ui.refreshable
            def metrics_panel() -> None:
                loading = state.get("loading", False)
                if loading:
                    with ui.row().classes("w-full gap-3 flex-wrap metrics-row"):
                        for _ in range(5):
                            with ui.element("div").classes("metric-card flex-1").style("min-width: 150px"):
                                ui.element("div").classes("skeleton skeleton-text").style("width: 40%")
                                ui.element("div").classes("skeleton skeleton-value")
                                ui.element("div").classes("skeleton skeleton-text").style("width: 70%")
                    return
                record = current_record()
                evaluation = record.get("evaluation") or {}
                metrics = evaluation.get("metrics") or {}
                total_items = total_requested_items(record)
                placed = len((record.get("response") or {}).get("placements", []))
                solve_time = (record.get("response") or {}).get("solve_time_ms", 0)
                cards_data = [
                    (
                        "Score",
                        f"{evaluation.get('final_score', 0.0):.4f}",
                        f"Base: {evaluation.get('base_final_score', 0.0):.4f}",
                    ),
                    (
                        "Coverage",
                        f"{metrics.get('item_coverage', 0.0):.2%}",
                        f"{placed}/{total_items} items",
                    ),
                    (
                        "Volume",
                        f"{metrics.get('volume_utilization', 0.0):.2%}",
                        "Заполнение объема",
                    ),
                    (
                        "Fragility",
                        f"{metrics.get('fragility_score', 0.0):.4f}",
                        "Хрупкость",
                    ),
                    (
                        "Time",
                        f"{metrics.get('time_score', 0.0):.4f}",
                        f"{solve_time} ms",
                    ),
                ]
                with ui.row().classes("w-full gap-3 flex-wrap metrics-row"):
                    for title, value, note in cards_data:
                        with ui.element("div").classes("metric-card flex-1").style("min-width: 150px"):
                            ui.label(title).classes("metric-label")
                            ui.label(value).classes("metric-value")
                            ui.label(note).classes("metric-note")

            metrics_panel()

            # 3D Visualization (dominant)
            @ui.refreshable
            def visualization_panel() -> None:
                loading = state.get("loading", False)
                record = current_record()
                evaluation = record.get("evaluation") or {}
                valid = bool(evaluation.get("valid", False))
                src = f"/visualization/{record['id']}?v={record.get('updated_at', '')}"
                with ui.column().classes("w-full flex-1 gap-2 min-h-0"):
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label("3D VISUALIZATION").classes("section-kicker")
                            if loading:
                                ui.html('<div class="spinner spinner-sm" style="display:inline-block"></div>')
                                ui.label("Solving...").style("color: var(--green-primary); font-size: 0.75rem")
                            else:
                                ui.html(
                                    f'<span class="pill {"good" if valid else "warn"}" style="font-size:0.68rem;padding:2px 8px">'
                                    f'{"OK" if valid else "Error"}'
                                    f"</span>"
                                )
                        ui.button(
                            "Diagnostics" if not state["bottom_open"] else "Hide diagnostics",
                            on_click=toggle_bottom,
                            icon="terminal",
                        ).classes("btn-secondary").style("font-size: 0.75rem; padding: 4px 12px !important")
                    if not valid and not loading and evaluation.get("error"):
                        ui.label(f"Валидатор: {evaluation['error']}").style("color: var(--error); font-size: 0.85rem")
                    with ui.element("div").classes("viz-wrapper"):
                        frame = ui.element("iframe")
                        frame.props(f'src="{src}"')
                        frame.classes("visual-frame")
                        if loading:
                            with ui.element("div").classes("loading-overlay"):
                                ui.element("div").classes("spinner")

            visualization_panel()

        # ── RIGHT SIDEBAR: Controls ─────────────────────
        right_ref = ui.element("div").classes("sidebar-right collapsed")
        with right_ref:
            @ui.refreshable
            def control_panel() -> None:
                record = current_record()
                evaluation = record.get("evaluation") or {}
                normalized = evaluation.get("normalized_score_weights") or {
                    key: DEFAULT_SCORE_WEIGHTS[key] / 100.0 for key in SCORE_WEIGHT_KEYS
                }
                with ui.column().classes("w-full gap-5"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("Настройки").classes("sidebar-title")
                        ui.button(icon="close", on_click=toggle_right).props("flat round dense").classes("btn-icon sidebar-close")
                    # Solver controls
                    with ui.column().classes("glass-card rounded-[20px] p-5 gap-4"):
                        ui.label("Solver").classes("section-kicker")
                        ui.label("Параметры запуска").classes("card-title")
                        ui.select(
                            list(STRATEGIES),
                            label="Стратегия",
                            value=record.get("strategy", "portfolio_block"),
                            on_change=lambda e: update_solver_setting("strategy", e.value),
                        ).classes("w-full").props("dense outlined")
                        ui.number(
                            "Time budget, ms",
                            value=record.get("time_budget_ms", 900),
                            min=50,
                            step=50,
                            on_change=lambda e: update_solver_setting("time_budget_ms", e.value),
                        ).classes("w-full").props("dense outlined")
                        ui.number(
                            "Legacy effort",
                            value=record.get("n_restarts", 10),
                            min=1,
                            step=1,
                            on_change=lambda e: update_solver_setting("n_restarts", e.value),
                        ).classes("w-full").props("dense outlined")
                        ui.switch(
                            "Автозапуск при изменении",
                            value=bool(state["autorun"]),
                            on_change=lambda e: state.__setitem__("autorun", bool(e.value)),
                        )

                    # Score weights
                    with ui.column().classes("glass-card rounded-[20px] p-5 gap-4 w-full"):
                        ui.label("Score Weights").classes("section-kicker")
                        ui.label("Live-пересчет score").classes("card-title")
                        for key in SCORE_WEIGHT_KEYS:
                            title, subtitle = SCORE_LABELS[key]
                            current_value = float(record.get("score_weights", DEFAULT_SCORE_WEIGHTS).get(key, 0.0))
                            with ui.column().classes("gap-1 w-full"):
                                with ui.row().classes("w-full items-center justify-between"):
                                    ui.label(title).classes("text-sm font-semibold").style("color: var(--text-primary)")
                                    ui.label(f"{current_value:.0f}").classes("text-sm").style("color: var(--text-tertiary)")
                                ui.slider(
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=current_value,
                                    on_change=lambda e, wk=key: update_weight(wk, float(e.value)),
                                ).classes("w-full").style("width: 100%")
                                ui.label(
                                    f"{subtitle} · {normalized.get(key, 0.0) * 100:.1f}%"
                                ).classes("detail-note").style("font-size: 0.72rem")
                        with ui.row().classes("w-full justify-end pt-2"):
                            ui.button("Сбросить", on_click=reset_weights, icon="restart_alt").classes("btn-secondary").style("font-size: 0.75rem")

            control_panel()

        # ── BOTTOM: Diagnostics ─────────────────────────
        bottom_ref = ui.element("div").classes("bottom-panel collapsed")
        with bottom_ref:
            @ui.refreshable
            def details_panel() -> None:
                record = current_record()
                evaluation = record.get("evaluation") or {}
                with ui.column().classes("w-full gap-3"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("DIAGNOSTICS").classes("section-kicker")
                        ui.button(icon="close", on_click=toggle_bottom).props("flat round dense").classes("btn-icon")
                    with ui.tabs().classes("w-full") as tabs:
                        request_tab = ui.tab("request", label="Request")
                        response_tab = ui.tab("response", label="Response")
                        metrics_tab = ui.tab("metrics", label="Metrics")
                    with ui.tab_panels(tabs, value=request_tab).classes("w-full"):
                        with ui.tab_panel(request_tab).classes("gap-3"):
                            ui.label("Текущий request, который уходит в solver").classes("detail-note")
                            with ui.element("div").classes("request-code mono max-h-[24rem] w-full p-4"):
                                ui.code(pretty_json(record["request"]), language="json")
                        with ui.tab_panel(response_tab).classes("gap-3"):
                            ui.label("Последний response от солвера").classes("detail-note")
                            with ui.element("div").classes("request-code mono max-h-[24rem] w-full p-4"):
                                ui.code(pretty_json(record.get("response") or {}), language="json")
                        with ui.tab_panel(metrics_tab).classes("gap-4"):
                            metrics = evaluation.get("metrics") or {}
                            with ui.row().classes("w-full gap-3 flex-wrap"):
                                for key in SCORE_WEIGHT_KEYS:
                                    title, subtitle = SCORE_LABELS[key]
                                    with ui.element("div").classes("metric-card flex-1").style("min-width: 160px"):
                                        ui.label(title).classes("metric-label")
                                        ui.label(f"{metrics.get(key, 0.0):.4f}").classes("metric-value").style("font-size: 1.8rem")
                                        ui.label(subtitle).classes("metric-note")
                                        ui.label(
                                            f"Норм. вес: {(evaluation.get('normalized_score_weights') or {}).get(key, 0.0) * 100:.1f}%"
                                        ).classes("detail-note").style("font-size: 0.72rem")
                            if evaluation.get("error"):
                                ui.label(f"Ошибка: {evaluation['error']}").style("color: var(--error); font-size: 0.85rem")

            details_panel()

    # Auto-expand sidebars on desktop at page load (they start collapsed for mobile-first)
    async def _auto_expand_sidebars() -> None:
        is_mobile = await ui.run_javascript("window.innerWidth <= 768")
        if not is_mobile:
            toggle_left()   # expand left
            toggle_right()  # expand right

    ui.timer(0.3, _auto_expand_sidebars, once=True)


if __name__ in {"__main__", "__mp_main__"}:
    uvicorn.run(
        fastapi_app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "3030")),
        reload=False,
    )
