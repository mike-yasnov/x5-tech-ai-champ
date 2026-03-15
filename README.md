# AI CHEMP — X5 Tech Smart 3D Packing

3D Pallet Packing Optimizer для задачи укладки коробов на паллеты в фуд-ритейле.

Текущий runtime по умолчанию: `portfolio_block`.

Версия пакета: `2.1.0`.

## Что есть в репозитории

- Default solver `portfolio_block`: портфель конструктивных стратегий, локальный поиск порядка, repair и postprocess.
- Legacy solver'ы для сравнения: `legacy_hybrid` и `legacy_greedy`.
- Встроенный валидатор `validator.py` с hard constraints и официальной формулой score.
- Генератор сценариев, benchmark, constraint-benchmark, сравнение baseline-сортировок.
- Веб-лаборатория на FastAPI + NiceGUI с историей экспериментов и встроенной 3D-визуализацией.
- Шипованный XGBoost selector в [`models/selector_xgb.json`](models/selector_xgb.json); block ranker можно дообучить отдельно.

Подробнее о постановке: [docs/task.md](docs/task.md)

## Структура проекта

```text
├── solver/
│   ├── __init__.py                # Версия пакета
│   ├── __main__.py                # python -m solver
│   ├── cli.py                     # CLI
│   ├── solver.py                  # Публичный entrypoint и dispatch стратегий
│   ├── portfolio_block.py         # Default runtime solver
│   ├── scenario_selector.py       # Request fingerprint + XGBoost selector
│   ├── block_ranker.py            # Optional XGBoost ranker for block candidates
│   ├── block_features.py          # Feature extraction для ranker
│   ├── packer.py                  # Greedy / layered / column packing primitives
│   ├── scoring.py                 # Baseline placement scoring
│   ├── pallet_state.py            # Baseline pallet state
│   ├── models.py                  # Box, Pallet, Placement, Solution
│   ├── orientations.py            # Rotation logic
│   └── hybrid/                    # Legacy hybrid beam-search solver
├── tests/                         # 12 pytest-файлов
├── models/
│   ├── selector_xgb.json          # Шипованный selector artifact
│   └── selector_meta.json         # Метаданные selector'а
├── docs/
│   ├── task.md                    # Описание кейса
│   ├── TEST_SCENARIOS.md          # Каталог benchmark-сценариев
│   └── selector_report.md         # Генерируемый отчёт selector evaluation
├── generator.py                   # Генератор synthetic request'ов
├── validator.py                   # Валидация + score
├── benchmark.py                   # Основной benchmark по 29 сценариям
├── benchmark_constraints.py       # 76 constraint stress-tests
├── benchmark_strategies.py        # Сравнение baseline greedy sort heuristics
├── visualize.py                   # Генерация standalone HTML-визуализаций
├── webapp.py                      # FastAPI + NiceGUI web UI
├── experiment_service.py          # Experiment storage / rerun / weighted evaluation
├── collect_selector_data.py       # Сбор датасета selector'а
├── train_selector.py              # Обучение selector'а
├── evaluate_selector.py           # Held-out + benchmark evaluation selector'а
├── collect_block_ranker_data.py   # Сбор датасета ranker'а
├── train_block_ranker.py          # Обучение block ranker'а
├── scenario_catalog.py            # Organizer / extended / diagnostic / private scenarios
└── .github/workflows/ci.yml       # CI: tests + benchmarks + PR artifacts
```

## Быстрый старт

```bash
# Клонировать репо
git clone git@github.com:mike-yasnov/x5-tech-ai-champ.git
cd x5-tech-ai-champ

# Установить зависимости
pip install -r requirements.txt

# Сгенерировать локальные примеры request_*.json
python generator.py

# Решить один request
python -m solver request_heavy_water.json -o response.json

# Прогнать тесты
pytest tests/ -v

# Основной benchmark
python benchmark.py

# Веб-лаборатория
python webapp.py
```

После запуска web UI откройте `http://127.0.0.1:3030/`.

Порт и хост можно переопределить через переменные окружения `PORT` и `HOST`.

## CLI

```bash
python -m solver <input.json> [options]
```

Поддерживается один или несколько входных JSON-файлов.

### Основные опции

| Опция           | Значение по умолчанию | Что делает                                                                    |
| --------------- | --------------------: | ----------------------------------------------------------------------------- |
| `inputs`        |                     — | Один или несколько request JSON                                               |
| `-o, --output`  |                  auto | Явный output only для single-input запуска                                    |
| `--strategy`    |     `portfolio_block` | `portfolio_block`, `legacy_hybrid`, `legacy_greedy`                           |
| `--time-budget` |                 `900` | Тайм-бюджет на задачу в миллисекундах                                         |
| `--model-dir`   |              `models` | Папка с optional ML artifact'ами                                              |
| `--beam-width`  |               derived | Явный beam width для `legacy_hybrid`                                          |
| `--restarts`    |                  `10` | Legacy effort knob; если `--beam-width` не задан, конвертируется в beam width |
| `--log-level`   |                `INFO` | `DEBUG`, `INFO`, `WARN`, `WARNING`, `ERROR`                                   |

### Примеры

```bash
# Один файл -> один output
python -m solver request_heavy_water.json -o result.json

# Batch: output-файлы будут названы как response_<name>.json
python -m solver request_*.json

# Принудительно включить legacy_hybrid и указать beam width
python -m solver request_random_mixed.json --strategy legacy_hybrid --beam-width 4

# Увеличить time budget
python -m solver request_random_mixed.json --time-budget 5000
```

## Python API

```python
from solver.models import load_request
from solver.solver import solve

task_id, pallet, boxes = load_request("request_heavy_water.json")

solution = solve(
    task_id=task_id,
    pallet=pallet,
    boxes=boxes,
    request_dict=None,          # optional; если есть, полезно для validator/debug
    strategy="portfolio_block",
    time_budget_ms=900,
    model_dir="models",
    score_weights=None,         # optional local search weights
)
```

Важно: `score_weights` влияют на внутренний proxy-objective в `portfolio_block` и на локальную лабораторию экспериментов, но официальный benchmark-валидатор по умолчанию использует фиксированные веса `50/30/10/10`.

## Текущая архитектура

### Dispatch стратегий

```text
solver.solve(..., strategy=X)
  ├── portfolio_block   # default
  ├── legacy_hybrid
  └── legacy_greedy
```

### Default solver: `portfolio_block`

`portfolio_block` сейчас делает не один жадный проход, а короткий runtime-портфель кандидатов:

1. Считает `ScenarioFingerprint` по request.
2. Ранжирует `seed_family` через fallback-эвристику и optional XGBoost selector.
3. Запускает только верхние 2-3 seed family вместо полного перебора.
4. Для greedy seed'ов при необходимости пробует специальные варианты:
   - `pack_upright_layered`
   - `pack_small_column_volume_first`
   - staged / prefill-последовательности
   - строгий fragile-aware прогон
   - маленький beam-кандидат на компактных diverse-case входах
5. Для лучшего greedy-кандидата делает local order search по первым SKU.
6. Применяет repair-стадию:
   - remove-and-refill
   - fragility micro-repack
   - block repair для block-based runs
7. Финализирует placements и unplaced.

### Seed families

В рантайме используются 6 семейств:

- `heavy_base`
- `liquid_fill`
- `mixed_volume`
- `fragile_density`
- `block_structured`
- `coverage_tie`

`block_structured` внутри дополнительно прогоняет 3 constructive policy:

- `foundation`
- `fragile_last`
- `coverage_fill`

### Legacy стратегии

- `legacy_hybrid` — beam search / greedy fallback из старого hybrid pipeline.
- `legacy_greedy` — чистый greedy baseline без portfolio-логики.

## Валидация и scoring

Источник истины для benchmark score: [`validator.py`](validator.py).

### Hard constraints

- Коробы должны лежать внутри паллеты.
- Не допускаются 3D-пересечения.
- Для коробов выше пола требуется опора не менее `60%` площади основания.
- Для `strict_upright: true` запрещены вращения, меняющие исходную высоту по оси `Z`.
- Суммарный вес размещенных коробов не может превышать `max_weight_kg`.
- На `stackable: false` нельзя ставить другие коробы сверху.

### Официальная формула score

```text
final_score = 0.50 * volume_utilization
            + 0.30 * item_coverage
            + 0.10 * fragility_score
            + 0.10 * time_score
```

### Time score

- `<= 1000 ms` -> `1.0`
- `<= 5000 ms` -> `0.7`
- `<= 30000 ms` -> `0.3`
- `> 30000 ms` -> `0.0`

### Fragility score

Валидатор считает cumulative non-fragile load по support chain. За каждую fragile-коробку, на которую сверху через цепочку опор давит больше `2 кг` non-fragile груза, применяется штраф `0.05`.

## Ротации

Поддерживаются стандартные rotation code:

| Код   | Mapping                              |
| ----- | ------------------------------------ |
| `LWH` | Length -> X, Width -> Y, Height -> Z |
| `LHW` | Length -> X, Height -> Y, Width -> Z |
| `WLH` | Width -> X, Length -> Y, Height -> Z |
| `WHL` | Width -> X, Height -> Y, Length -> Z |
| `HLW` | Height -> X, Length -> Y, Width -> Z |
| `HWL` | Height -> X, Width -> Y, Length -> Z |

Для `strict_upright=True` остаются только ориентации, в которых исходная высота товара остаётся по оси `Z`.

## Сценарии и benchmark'и

Основной каталог задаётся в [`scenario_catalog.py`](scenario_catalog.py):

- `4` organizer scenarios
- `5` extended realistic scenarios
- `5` diagnostic/sanity scenarios
- `15` private-style scenarios

Итого: `29` benchmark-сценариев.

Подробное описание: [docs/TEST_SCENARIOS.md](docs/TEST_SCENARIOS.md)

### Основной benchmark

```bash
python benchmark.py
python benchmark.py --strategy legacy_hybrid --output benchmark_results.json
python benchmark.py --model-dir models --viz viz
```

`benchmark.py --viz viz`:

- сохраняет `viz/benchmark_viz.json`
- сразу генерирует HTML-визуализации для каждого сценария

### Constraint benchmark

```bash
python benchmark_constraints.py
python benchmark_constraints.py --strategy portfolio_block --output constraint_results.json
```

Сейчас в нём `76` сценариев на hard/soft constraints и граничные случаи.

### Сравнение baseline greedy sort heuristics

```bash
python benchmark_strategies.py
python benchmark_strategies.py --limit 8 --markdown-output strategy_output.md
```

Это сравнение только baseline `pack_greedy` сортировок из `solver/packer.py`, а не full `portfolio_block`.

## Визуализация

### Standalone HTML

```bash
# если benchmark_viz.json уже существует
python visualize.py viz/benchmark_viz.json -o viz_html
```

`visualize.py` генерирует standalone HTML-файлы с Three.js-сценой:

- вращение и zoom
- детерминированные цвета по SKU
- отдельное отображение unplaced boxes
- подсветка fragile-нарушений
- пошаговый просмотр укладки

### Web UI

```bash
python webapp.py
```

URL по умолчанию: `http://127.0.0.1:3030/`

Что умеет UI:

- создавать эксперименты из catalog scenario
- редактировать request целиком
- редактировать response вручную и мгновенно перевалидировать его
- переключать runtime strategy
- менять `time_budget_ms`
- менять локальные `score_weights`
- пересчитывать и хранить историю запусков
- смотреть встроенную 3D-визуализацию

История экспериментов хранится в `.history/ui_request_history.json`.

### Доступные HTTP endpoints

| Метод  | Endpoint                      | Что возвращает                         |
| ------ | ----------------------------- | -------------------------------------- |
| `GET`  | `/health`                     | `{ "status": "ok" }`                   |
| `GET`  | `/api/experiments`            | список summary по экспериментам        |
| `GET`  | `/api/experiments/{id}`       | полный record эксперимента             |
| `POST` | `/api/experiments/{id}/rerun` | rerun текущего эксперимента            |
| `GET`  | `/visualization/{id}`         | standalone HTML сцена для эксперимента |

## ML workflows

### Selector

Репозиторий уже содержит selector artifact:

- [`models/selector_xgb.json`](models/selector_xgb.json)
- [`models/selector_meta.json`](models/selector_meta.json)

`portfolio_block` автоматически пытается использовать selector, если:

- artifact существует в `model_dir`
- request не слишком большой (`total_items <= 120`)

Сбор и обучение:

```bash
# train split
python collect_selector_data.py --output selector_train.npz --seed-start 1000 --seed-end 1010

# validation split
python collect_selector_data.py --output selector_val.npz --seed-start 1500 --seed-end 1505

# train
python train_selector.py --train selector_train.npz --val selector_val.npz --model-dir models

# evaluate
python evaluate_selector.py --dataset selector_val.npz --model-dir models --markdown-output docs/selector_report.md
```

### Block ranker

Block ranker в runtime тоже optional, но artifact по умолчанию в репозиторий не шипуется.

Если в `model_dir` появится `xgb_ranker.json`, `portfolio_block` сможет использовать его на сравнительно небольших запросах.

Сбор и обучение:

```bash
# train split
python collect_block_ranker_data.py --output ranker_train.npz --seed-start 1000 --seed-end 1005

# validation split
python collect_block_ranker_data.py --output ranker_val.npz --seed-start 1500 --seed-end 1503

# train
python train_block_ranker.py --train ranker_train.npz --val ranker_val.npz --model-dir models_ranker_trial
```

## Зависимости

Основные зависимости из [`requirements.txt`](requirements.txt):

| Пакет          | Назначение                          |
| -------------- | ----------------------------------- |
| `numpy`        | численные операции                  |
| `scikit-learn` | legacy hybrid ML wrapper (`RF/SVR`) |
| `xgboost`      | selector и optional block ranker    |
| `pytest`       | тесты                               |
| `fastapi`      | backend web UI                      |
| `nicegui`      | frontend web UI                     |
| `uvicorn`      | запуск ASGI-приложения              |

## CI

CI описан в [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

Что делает pipeline:

- на `push` и `pull_request` в `main` запускает `pytest` на Python `3.11` и `3.12`
- на `pull_request` дополнительно запускает:
  - `benchmark.py`
  - `benchmark_constraints.py`
  - `benchmark_strategies.py`
- прикладывает HTML-визуализации как artifact
- публикует benchmark summary в PR-комментарий, если PR открыт из этого же репозитория

Runner: self-hosted Linux.

## Разработка

### Куда смотреть в первую очередь

| Файл                          | Когда полезен                               |
| ----------------------------- | ------------------------------------------- |
| `solver/portfolio_block.py`   | runtime-логика default solver               |
| `solver/packer.py`            | greedy, layered и column packing primitives |
| `solver/scenario_selector.py` | fingerprint и XGBoost selector              |
| `validator.py`                | hard constraints и score                    |
| `benchmark.py`                | основной regression harness                 |
| `experiment_service.py`       | local weighted evaluation и история webapp  |
| `webapp.py`                   | UI и REST endpoints                         |

### Локальная проверка перед изменениями

```bash
pytest tests/ -v
python benchmark.py
python benchmark_constraints.py
```
