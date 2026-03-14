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
    dark=False,
    storage_secret="x5-packing-lab",
    show_welcome_message=False,
)

APP_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
"""

APP_CSS = """
:root {
  --bg: #07111f;
  --bg-soft: rgba(10, 22, 38, 0.84);
  --panel: rgba(10, 18, 31, 0.78);
  --panel-strong: rgba(7, 13, 24, 0.92);
  --line: rgba(148, 163, 184, 0.16);
  --text: #e5f0ff;
  --muted: #8ca0bf;
  --accent: #22c55e;
  --accent-soft: rgba(34, 197, 94, 0.16);
  --warm: #f97316;
  --warm-soft: rgba(249, 115, 22, 0.18);
  --cool: #38bdf8;
  --cool-soft: rgba(56, 189, 248, 0.16);
}

html, body, #q-app {
  min-height: 100%;
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 24%),
    radial-gradient(circle at top right, rgba(249, 115, 22, 0.15), transparent 30%),
    radial-gradient(circle at bottom center, rgba(34, 197, 94, 0.12), transparent 28%),
    linear-gradient(180deg, #07111f 0%, #091423 42%, #050b14 100%);
  color: var(--text);
  font-family: "Space Grotesk", "Segoe UI", sans-serif;
}

body {
  overflow-x: hidden;
}

.mono, .mono textarea, .mono pre, code, pre {
  font-family: "IBM Plex Mono", Consolas, monospace !important;
}

.page-shell {
  position: relative;
}

.hero-panel {
  background:
    linear-gradient(135deg, rgba(10, 18, 31, 0.95), rgba(10, 22, 38, 0.88)),
    radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.12), transparent 32%);
  border: 1px solid var(--line);
  box-shadow: 0 30px 60px rgba(3, 8, 18, 0.45);
  backdrop-filter: blur(22px);
}

.glass-card {
  background: linear-gradient(180deg, rgba(10, 18, 31, 0.9), rgba(7, 13, 24, 0.88));
  border: 1px solid var(--line);
  box-shadow: 0 24px 48px rgba(3, 8, 18, 0.34);
  backdrop-filter: blur(18px);
}

.metric-card {
  background:
    linear-gradient(180deg, rgba(13, 23, 40, 0.96), rgba(8, 15, 28, 0.92)),
    radial-gradient(circle at top right, rgba(56, 189, 248, 0.1), transparent 40%);
  border: 1px solid rgba(148, 163, 184, 0.12);
  box-shadow: 0 16px 40px rgba(3, 8, 18, 0.3);
}

.request-code {
  border: 1px solid rgba(148, 163, 184, 0.12);
  background: rgba(5, 10, 19, 0.88);
  border-radius: 20px;
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
}

.soft-line {
  border-top: 1px solid rgba(148, 163, 184, 0.1);
}

.history-card {
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.history-card:hover {
  transform: translateY(-2px);
  border-color: rgba(56, 189, 248, 0.3);
  box-shadow: 0 18px 40px rgba(3, 8, 18, 0.28);
}

.history-card.active {
  border-color: rgba(56, 189, 248, 0.4);
  box-shadow: 0 20px 44px rgba(56, 189, 248, 0.1);
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  background: rgba(8, 15, 28, 0.65);
  color: var(--muted);
  font-size: 0.78rem;
}

.pill.good {
  background: var(--accent-soft);
  color: #9ae6b4;
}

.pill.warn {
  background: var(--warm-soft);
  color: #fdba74;
}

.pill.cool {
  background: var(--cool-soft);
  color: #bae6fd;
}

.visual-frame {
  width: 100%;
  min-height: 72vh;
  border: 0;
  border-radius: 26px;
  background: #050915;
}

.section-kicker {
  color: var(--muted);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-size: 0.72rem;
}

.detail-note {
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.55;
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
    }

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
        hero_panel.refresh()
        control_panel.refresh()
        metrics_panel.refresh()
        visualization_panel.refresh()
        details_panel.refresh()
        history_panel.refresh()

    def rerun_selected() -> None:
        record = current_record()
        updated = service.run_experiment(record["id"])
        state["selected_id"] = updated["id"]
        refresh_all()
        ui.notify(f"Пересчитан эксперимент «{updated['name']}»", color="positive")

    def update_solver_setting(key: str, value: Any) -> None:
        record = current_record()
        kwargs = {key: value, "run_now": bool(state["autorun"])}
        updated = service.update_metadata(record["id"], **kwargs)
        state["selected_id"] = updated["id"]
        refresh_all()

    def update_weight(key: str, value: float) -> None:
        record = current_record()
        weights = clone_data(record.get("score_weights", DEFAULT_SCORE_WEIGHTS))
        weights[key] = float(value)
        updated = service.update_score_weights(record["id"], weights)
        state["selected_id"] = updated["id"]
        refresh_all()

    def reset_weights() -> None:
        record = current_record()
        updated = service.update_score_weights(record["id"], dict(DEFAULT_SCORE_WEIGHTS))
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

    def open_request_editor(mode: str) -> None:
        source = current_record() if mode == "edit" else None
        draft = make_experiment_draft(source)
        if mode == "create":
            draft["name"] = f"{draft['scenario_type']} · seed {draft['seed']}"

        dialog = ui.dialog()
        with dialog, ui.card().classes("glass-card rounded-[28px] w-[96vw] max-w-6xl p-0"):
            with ui.column().classes("w-full gap-0"):
                with ui.row().classes("w-full items-center justify-between px-7 py-6"):
                    with ui.column().classes("gap-1"):
                        ui.label("Experiment Builder").classes("section-kicker")
                        ui.label(
                            "Создайте новый сценарий или подправьте текущий request перед новым запуском."
                        ).classes("text-2xl font-semibold text-slate-100")
                    ui.button(icon="close", on_click=dialog.close, color=None).props("flat round dense")

                ui.separator().classes("soft-line")

                with ui.row().classes("w-full gap-6 px-7 py-6 items-start flex-wrap xl:flex-nowrap"):
                    with ui.column().classes("w-full xl:w-[24rem] gap-4"):
                        ui.label("Общие настройки").classes("section-kicker")
                        name_input = ui.input("Название", value=draft["name"]).classes("w-full")
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
                                name=str(name_input.value or scenario_type),
                            )
                            request_editor.value = pretty_json(request_dict)
                            ui.notify(
                                f"Пресет «{scenario_type}» с seed={seed} загружен в редактор request",
                                color="primary",
                            )

                        with ui.row().classes("w-full gap-3"):
                            ui.button(
                                "Загрузить пресет",
                                on_click=load_preset,
                                icon="auto_awesome",
                            ).classes("bg-sky-500 text-slate-950 font-semibold")
                            hint = ui.label(
                                "Score-веса задаются на боковой панели после запуска."
                            ).classes("detail-note")
                            ui.button(
                                "Подсказка",
                                on_click=lambda: hint.set_text(
                                    "Можно править request целиком, а потом сразу увидеть новый score и 3D-сцену."
                                ),
                                icon="tips_and_updates",
                                color=None,
                            ).props("flat")

                    with ui.column().classes("min-w-0 flex-1 gap-3"):
                        ui.label("Request JSON").classes("section-kicker")
                        request_editor = ui.textarea(
                            placeholder="Здесь можно править request целиком, включая boxes и pallet.",
                            value=pretty_json(draft["request"]),
                        ).classes("w-full mono json-editor")

                ui.separator().classes("soft-line")

                with ui.row().classes("w-full items-center justify-between px-7 py-5"):
                    ui.label(
                        "После сохранения эксперимент сразу перезапустится, а визуализация и история обновятся."
                    ).classes("detail-note")

                    def save_request() -> None:
                        try:
                            request_dict = json.loads(request_editor.value or "{}")
                        except json.JSONDecodeError as exc:
                            ui.notify(f"Request JSON не парсится: {exc}", color="negative")
                            return
                        if not isinstance(request_dict, dict):
                            ui.notify("Request JSON должен быть объектом", color="negative")
                            return
                        request_dict["task_id"] = make_task_id(
                            str(name_input.value or "experiment"),
                            str(scenario_select.value or "custom"),
                            int(seed_input.value or 0),
                        )
                        if "pallet" not in request_dict or "boxes" not in request_dict:
                            ui.notify("В request должны быть ключи pallet и boxes", color="negative")
                            return
                        if mode == "edit":
                            updated = service.update_request(
                                current_record()["id"],
                                request_dict=request_dict,
                                name=str(name_input.value or "Experiment"),
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
                                name=str(name_input.value or "Experiment"),
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
                        dialog.close()
                        refresh_all()
                        ui.notify("Эксперимент сохранен и запущен", color="positive")

                    with ui.row().classes("gap-3"):
                        ui.button("Отмена", on_click=dialog.close, color=None).props("flat")
                        ui.button(
                            "Сохранить и запустить",
                            on_click=save_request,
                            icon="play_arrow",
                        ).classes("bg-emerald-400 text-slate-950 font-semibold")
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
        with dialog, ui.card().classes("glass-card rounded-[28px] w-[96vw] max-w-6xl p-0"):
            with ui.column().classes("w-full gap-0"):
                with ui.row().classes("w-full items-center justify-between px-7 py-6"):
                    with ui.column().classes("gap-1"):
                        ui.label("Result Editor").classes("section-kicker")
                        ui.label("Подправьте response и сразу увидите новый score.").classes(
                            "text-2xl font-semibold text-slate-100"
                        )
                    ui.button(icon="close", on_click=dialog.close, color=None).props("flat round dense")

                ui.separator().classes("soft-line")
                with ui.column().classes("w-full px-7 py-6 gap-3"):
                    ui.label(
                        "Если структура станет невалидной, валидатор это покажет, а финальный score опустится до 0."
                    ).classes("detail-note")
                    response_editor = ui.textarea(
                        value=pretty_json(default_response),
                        placeholder="Измените response JSON",
                    ).classes("w-full mono json-editor")

                ui.separator().classes("soft-line")
                with ui.row().classes("w-full items-center justify-end gap-3 px-7 py-5"):
                    ui.button("Отмена", on_click=dialog.close, color=None).props("flat")

                    def save_response() -> None:
                        try:
                            response_dict = json.loads(response_editor.value or "{}")
                        except json.JSONDecodeError as exc:
                            ui.notify(f"Response JSON не парсится: {exc}", color="negative")
                            return
                        if not isinstance(response_dict, dict):
                            ui.notify("Response JSON должен быть объектом", color="negative")
                            return
                        response_dict["task_id"] = record["request"]["task_id"]
                        updated = service.update_response(record["id"], response_dict)
                        state["selected_id"] = updated["id"]
                        dialog.close()
                        refresh_all()
                        ui.notify("Результат обновлен и пересчитан", color="positive")

                    ui.button(
                        "Сохранить результат",
                        on_click=save_response,
                        icon="save",
                    ).classes("bg-orange-400 text-slate-950 font-semibold")
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

    with ui.column().classes("page-shell w-full max-w-[1680px] mx-auto gap-6 px-4 py-5 md:px-6 md:py-6 xl:px-8"):
        @ui.refreshable
        def hero_panel() -> None:
            record = current_record()
            evaluation = record.get("evaluation") or {}
            with ui.row().classes("hero-panel rounded-[30px] w-full items-center justify-between gap-5 px-6 py-6 lg:px-8 lg:py-7 flex-wrap"):
                with ui.column().classes("gap-2"):
                    ui.label("X5 Packing Lab").classes("section-kicker")
                    ui.label(record["name"]).classes("text-3xl md:text-4xl font-bold text-slate-50")
                    ui.label(
                        "FastAPI + NiceGUI лаборатория для запуска солвера, live-пересчета score и 3D-разбора укладки."
                    ).classes("detail-note max-w-3xl")
                with ui.row().classes("gap-2 flex-wrap items-center"):
                    valid = bool(evaluation.get("valid", False))
                    ui.html(
                        f'<span class="pill {"good" if valid else "warn"}">'
                        f'{"VALID" if valid else "INVALID"}'
                        f"</span>"
                    )
                    ui.html(f'<span class="pill cool">{record.get("strategy", "portfolio_block")}</span>')
                    ui.html(f'<span class="pill">seed {record.get("seed", "custom")}</span>')
                    ui.html(
                        f'<span class="pill">{len(record.get("request", {}).get("boxes", []))} SKU</span>'
                    )
                    ui.button("Новый эксперимент", on_click=lambda: open_request_editor("create"), icon="add").classes(
                        "bg-sky-400 text-slate-950 font-semibold"
                    )
                    ui.button("Дублировать", on_click=clone_selected, icon="content_copy", color=None).props("outline")
                    ui.button("Редактировать request", on_click=lambda: open_request_editor("edit"), icon="tune", color=None).props("outline")
                    ui.button("Редактировать result", on_click=open_result_editor, icon="edit_note", color=None).props("outline")
                    ui.button("Запустить", on_click=rerun_selected, icon="play_arrow").classes(
                        "bg-emerald-400 text-slate-950 font-semibold"
                    )

        hero_panel()

        with ui.row().classes("w-full gap-6 items-start flex-wrap 2xl:flex-nowrap"):
            @ui.refreshable
            def control_panel() -> None:
                record = current_record()
                evaluation = record.get("evaluation") or {}
                normalized = evaluation.get("normalized_score_weights") or {
                    key: DEFAULT_SCORE_WEIGHTS[key] / 100.0 for key in SCORE_WEIGHT_KEYS
                }
                with ui.column().classes("w-full 2xl:w-[22rem] shrink-0 gap-5"):
                    with ui.card().classes("glass-card rounded-[26px] p-5 gap-4"):
                        ui.label("Solver controls").classes("section-kicker")
                        ui.label("Параметры текущего запуска").classes("text-xl font-semibold text-slate-50")
                        ui.select(
                            list(STRATEGIES),
                            label="Стратегия",
                            value=record.get("strategy", "portfolio_block"),
                            on_change=lambda e: update_solver_setting("strategy", e.value),
                        ).classes("w-full")
                        ui.number(
                            "Time budget, ms",
                            value=record.get("time_budget_ms", 900),
                            min=50,
                            step=50,
                            on_change=lambda e: update_solver_setting("time_budget_ms", e.value),
                        ).classes("w-full")
                        ui.number(
                            "Legacy effort",
                            value=record.get("n_restarts", 10),
                            min=1,
                            step=1,
                            on_change=lambda e: update_solver_setting("n_restarts", e.value),
                        ).classes("w-full")
                        ui.switch(
                            "Автозапуск при изменении solver-настроек",
                            value=bool(state["autorun"]),
                            on_change=lambda e: state.__setitem__("autorun", bool(e.value)),
                        )
                        ui.label(
                            "Изменения score-весов ниже мгновенно пересчитывают итог без обязательного rerun."
                        ).classes("detail-note")

                    with ui.card().classes("glass-card rounded-[26px] p-5 gap-5"):
                        ui.label("Score weights").classes("section-kicker")
                        ui.label("Live-пересчет score").classes("text-xl font-semibold text-slate-50")
                        for key in SCORE_WEIGHT_KEYS:
                            title, subtitle = SCORE_LABELS[key]
                            current_value = float(record.get("score_weights", DEFAULT_SCORE_WEIGHTS).get(key, 0.0))
                            with ui.column().classes("gap-2"):
                                with ui.row().classes("w-full items-center justify-between"):
                                    ui.label(title).classes("text-sm font-semibold text-slate-100")
                                    ui.label(f"{current_value:.0f}").classes("text-sm text-slate-300")
                                ui.slider(
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=current_value,
                                    on_change=lambda e, weight_key=key: update_weight(weight_key, float(e.value)),
                                ).classes("w-full")
                                ui.label(
                                    f"{subtitle}. Текущий вклад: {normalized.get(key, 0.0) * 100:.1f}%"
                                ).classes("detail-note")
                        with ui.row().classes("w-full justify-between items-center pt-2"):
                            ui.label("Весы нормализуются автоматически").classes("detail-note")
                            ui.button("Сбросить", on_click=reset_weights, icon="restart_alt", color=None).props("outline")

            control_panel()

            with ui.column().classes("min-w-0 flex-1 gap-5"):
                @ui.refreshable
                def metrics_panel() -> None:
                    record = current_record()
                    evaluation = record.get("evaluation") or {}
                    metrics = evaluation.get("metrics") or {}
                    total_items = total_requested_items(record)
                    placed = len((record.get("response") or {}).get("placements", []))
                    cards = [
                        (
                            "Score",
                            f"{evaluation.get('final_score', 0.0):.4f}",
                            f"Base validator: {evaluation.get('base_final_score', 0.0):.4f}",
                        ),
                        (
                            "Coverage",
                            f"{metrics.get('item_coverage', 0.0):.4f}",
                            f"{placed}/{total_items} items placed",
                        ),
                        (
                            "Volume",
                            f"{metrics.get('volume_utilization', 0.0):.4f}",
                            "Заполнение объема паллеты",
                        ),
                        (
                            "Fragility / Time",
                            f"{metrics.get('fragility_score', 0.0):.4f} / {metrics.get('time_score', 0.0):.4f}",
                            f"Solve: {(record.get('response') or {}).get('solve_time_ms', 0)} ms",
                        ),
                    ]
                    with ui.row().classes("w-full gap-4 flex-wrap"):
                        for title, value, note in cards:
                            with ui.card().classes("metric-card rounded-[24px] px-5 py-4 gap-2 min-w-[15rem] flex-1"):
                                ui.label(title).classes("section-kicker")
                                ui.label(value).classes("text-2xl font-semibold text-slate-50")
                                ui.label(note).classes("detail-note")

                metrics_panel()

                @ui.refreshable
                def visualization_panel() -> None:
                    record = current_record()
                    evaluation = record.get("evaluation") or {}
                    valid = bool(evaluation.get("valid", False))
                    src = f"/visualization/{record['id']}?v={record.get('updated_at', '')}"
                    with ui.card().classes("glass-card rounded-[28px] p-5 gap-4"):
                        with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap"):
                            with ui.column().classes("gap-1"):
                                ui.label("3D Visualization").classes("section-kicker")
                                ui.label("Та же Three.js-сцена, теперь внутри интерактивной лаборатории").classes(
                                    "text-xl font-semibold text-slate-50"
                                )
                            with ui.row().classes("gap-2 flex-wrap items-center"):
                                ui.html(
                                    f'<span class="pill {"good" if valid else "warn"}">'
                                    f'{"Score OK" if valid else "Есть ошибка"}'
                                    f"</span>"
                                )
                                ui.html(
                                    f'<span class="pill">{(record.get("response") or {}).get("solve_time_ms", 0)} ms</span>'
                                )
                        if not valid and evaluation.get("error"):
                            ui.label(f"Валидатор: {evaluation['error']}").classes("text-sm text-orange-300")
                        frame = ui.element("iframe")
                        frame.props(f"src={src}")
                        frame.classes("visual-frame")

                visualization_panel()

                @ui.refreshable
                def details_panel() -> None:
                    record = current_record()
                    evaluation = record.get("evaluation") or {}
                    with ui.card().classes("glass-card rounded-[28px] p-5 gap-4"):
                        ui.label("Diagnostics").classes("section-kicker")
                        with ui.tabs().classes("w-full") as tabs:
                            request_tab = ui.tab("request", label="Request")
                            response_tab = ui.tab("response", label="Response")
                            metrics_tab = ui.tab("metrics", label="Metrics")
                        with ui.tab_panels(tabs, value=request_tab).classes("w-full"):
                            with ui.tab_panel(request_tab).classes("gap-3"):
                                ui.label("Текущий request, который уходит в solver").classes("detail-note")
                                with ui.element("div").classes("request-code mono max-h-[28rem] w-full p-4"):
                                    ui.code(pretty_json(record["request"]), language="json")
                            with ui.tab_panel(response_tab).classes("gap-3"):
                                ui.label(
                                    "Последний response. Его можно вручную редактировать через кнопку сверху."
                                ).classes("detail-note")
                                with ui.element("div").classes("request-code mono max-h-[28rem] w-full p-4"):
                                    ui.code(pretty_json(record.get("response") or {}), language="json")
                            with ui.tab_panel(metrics_tab).classes("gap-4"):
                                metrics = evaluation.get("metrics") or {}
                                with ui.row().classes("w-full gap-3 flex-wrap"):
                                    for key in SCORE_WEIGHT_KEYS:
                                        title, subtitle = SCORE_LABELS[key]
                                        with ui.card().classes("metric-card rounded-[22px] px-4 py-4 gap-2 min-w-[13rem] flex-1"):
                                            ui.label(title).classes("section-kicker")
                                            ui.label(f"{metrics.get(key, 0.0):.4f}").classes("text-xl font-semibold text-slate-50")
                                            ui.label(subtitle).classes("detail-note")
                                            ui.label(
                                                f"Норм. вес: {(evaluation.get('normalized_score_weights') or {}).get(key, 0.0) * 100:.1f}%"
                                            ).classes("detail-note")
                                if evaluation.get("error"):
                                    ui.label(f"Ошибка: {evaluation['error']}").classes("text-sm text-orange-300")

                details_panel()

            @ui.refreshable
            def history_panel() -> None:
                summaries = service.list_summaries()
                with ui.column().classes("w-full 2xl:w-[22rem] shrink-0 gap-4"):
                    with ui.card().classes("glass-card rounded-[26px] p-5 gap-4"):
                        ui.label("History").classes("section-kicker")
                        ui.label("История запросов и экспериментов").classes("text-xl font-semibold text-slate-50")
                        ui.input(
                            "Фильтр",
                            value=state.get("history_query", ""),
                            placeholder="scenario, strategy, notes",
                            on_change=lambda e: (state.__setitem__("history_query", e.value), history_panel.refresh()),
                        ).classes("w-full")
                        with ui.column().classes("w-full gap-3 max-h-[72vh] overflow-auto pr-1"):
                            filtered = [summary for summary in summaries if history_matches(summary)]
                            if not filtered:
                                ui.label("По фильтру ничего не найдено").classes("detail-note")
                            for summary in filtered:
                                active = summary["id"] == state.get("selected_id")
                                card = ui.card().classes(
                                    "history-card glass-card rounded-[22px] p-4 gap-3 cursor-pointer "
                                    + ("active" if active else "")
                                )
                                card.on("click", lambda _=None, experiment_id=summary["id"]: select_experiment(experiment_id))
                                with card:
                                    with ui.row().classes("w-full items-start justify-between gap-3"):
                                        with ui.column().classes("gap-1 min-w-0"):
                                            ui.label(summary["name"]).classes("text-base font-semibold text-slate-50")
                                            ui.label(
                                                f"{summary['scenario_type']} · {summary['strategy']}"
                                            ).classes("detail-note")
                                        ui.label(f"{summary['score']:.4f}").classes("text-lg font-semibold text-sky-200")
                                    with ui.row().classes("w-full gap-2 flex-wrap"):
                                        ui.html(
                                            f'<span class="pill {"good" if summary["valid"] else "warn"}">'
                                            f'{"valid" if summary["valid"] else "invalid"}'
                                            f"</span>"
                                        )
                                        ui.html(f'<span class="pill">{summary["placed"]}/{summary["total_items"]}</span>')
                                        ui.html(f'<span class="pill">{summary["solve_time_ms"]} ms</span>')
                                    ui.label(short_timestamp(summary["updated_at"])).classes("detail-note")
                                    with ui.row().classes("w-full justify-end gap-2"):
                                        ui.button(
                                            icon="content_copy",
                                            on_click=lambda _=None, experiment_id=summary["id"]: (
                                                state.__setitem__("selected_id", service.clone_experiment(experiment_id)["id"]),
                                                refresh_all(),
                                            ),
                                            color=None,
                                        ).props("flat round dense")
                                        if len(summaries) > 1:
                                            ui.button(
                                                icon="delete",
                                                on_click=lambda _=None, experiment_id=summary["id"]: delete_experiment(experiment_id),
                                                color=None,
                                            ).props("flat round dense")

            history_panel()


if __name__ in {"__main__", "__mp_main__"}:
    uvicorn.run(
        fastapi_app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
