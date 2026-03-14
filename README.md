# AI CHEMP — X5 Tech Smart 3D Packing

3D Pallet Packing Optimizer для задачи укладки коробов на паллеты в фуд-ритейле.

## Описание задачи

Разработать алгоритм, который по списку коробов (разный размер, вес, хрупкость) и параметрам паллеты строит оптимальный план укладки. Решение должно:

- Максимизировать плотность укладки (volume utilization)
- Разместить как можно больше товаров (item coverage)
- Учитывать хрупкость (fragile items сверху)
- Работать быстро (< 1 сек = максимальный балл за скорость)

Подробнее: [docs/task.md](docs/task.md)

## Структура проекта

```
├── solver/                # Пакет солвера (основная работа здесь)
│   ├── __init__.py
│   ├── models.py          # Модели данных: Box, Pallet, Placement, Solution
│   ├── orientations.py    # Генерация допустимых ориентаций (strict_upright)
│   ├── pallet_state.py    # Baseline-состояние паллеты для unit-тестов и сравнения
│   ├── scoring.py         # Baseline scoring-функция
│   ├── packer.py          # Baseline greedy packer
│   ├── solver.py          # Публичный entrypoint: адаптер к hybrid solver
│   ├── hybrid/            # Legacy hybrid beam search + postprocess
│   ├── portfolio_block.py # Default solver: block portfolio + optional ranker
│   ├── cli.py             # CLI-интерфейс
│   └── __main__.py        # Запуск через python -m solver
├── tests/                 # Тесты
│   ├── test_models.py
│   ├── test_orientations.py
│   ├── test_pallet_state.py
│   ├── test_scoring.py
│   ├── test_packer.py
│   └── test_solver.py     # Интеграционные тесты (все сценарии)
├── generator.py           # Генератор тестовых сценариев
├── validator.py           # Валидатор решений (hard constraints + scoring)
├── benchmark.py           # Бенчмарк: прогон всех сценариев с отчётом
├── docs/                  # Документация и исследования
│   └── task.md            # Описание задачи
└── requirements.txt       # Зависимости
```

## Быстрый старт

```bash
# Клонировать репо
git clone git@github.com:mike-yasnov/x5-tech-ai-champ.git
cd x5-tech-ai-champ

# Установить зависимости
pip install -r requirements.txt

# Сгенерировать тестовые данные
python generator.py

# Запустить солвер
python -m solver request_heavy_water.json -o response.json

# Запустить все тесты
pytest tests/ -v

# Запустить бенчмарк (все сценарии + скоры)
python benchmark.py
```

## Как внести изменения (workflow для команды)

### 1. Создать ветку

```bash
git checkout main
git pull
git checkout -b my-improvement
```

### 2. Внести изменения в солвер

Основные файлы для улучшений:

| Файл | Что менять |
|------|-----------|
| `solver/portfolio_block.py` | Default solver: block portfolio, repair, optional ranking |
| `solver/hybrid/pipeline.py` | Legacy hybrid solver и параметры поиска |
| `solver/hybrid/search.py` | Beam search / greedy fallback и ranking |
| `solver/hybrid/postprocess.py` | Уплотнение, fragile reorder, дозаполнение |
| `solver/solver.py` | Публичный API и адаптер dataclass <-> request/response |

### 3. Проверить локально

```bash
# Тесты должны проходить
pytest tests/ -v

# Бенчмарк покажет скоры
python benchmark.py
```

### 4. Запушить и создать PR

```bash
git add -A
git commit -m "improve: описание улучшения"
git push -u origin my-improvement
```

Создайте Pull Request в `main`. CI автоматически:
- Прогонит тесты
- Запустит бенчмарк на всех сценариях
- Покажет скоры в комментарии к PR

### 5. Ревью и мёрдж

После проверки скоров и code review — мёрдж в `main`.

## Hard Constraints (нарушение = 0 баллов)

| # | Ограничение | Проверка |
|---|------------|---------|
| 1 | Коробы внутри паллеты | AABB bounds check |
| 2 | Нет пересечений | AABB collision |
| 3 | Опора ≥ 60% площади | Support area calculation |
| 4 | strict_upright → только Z-ось вращения | Orientation filter |
| 5 | Вес ≤ max_weight_kg | Weight accumulator |
| 6 | Нельзя ставить сверху на `stackable: false` | Stackable-below check |

## Scoring

```
final_score = 0.50 × volume_utilization
            + 0.30 × item_coverage
            + 0.10 × fragility_score
            + 0.10 × time_score
```

| Метрика | Вес | Описание |
|---------|-----|----------|
| Volume Utilization | 50% | Плотность укладки |
| Item Coverage | 30% | Доля размещённых товаров |
| Fragility Score | 10% | Штраф за тяжёлое на хрупком |
| Time Score | 10% | ≤1s→1.0, ≤5s→0.7, ≤30s→0.3, >30s→0.0 |

## Формат данных

<details>
<summary>Request JSON</summary>

```json
{
  "task_id": "test_case_042",
  "pallet": {
    "length_mm": 1200,
    "width_mm": 800,
    "max_height_mm": 1800,
    "max_weight_kg": 1500.0
  },
  "boxes": [
    {
      "sku_id": "SKU-SHO-1234",
      "description": "Shoe Box",
      "length_mm": 330,
      "width_mm": 190,
      "height_mm": 115,
      "weight_kg": 1.0,
      "quantity": 10,
      "strict_upright": false,
      "fragile": false,
      "stackable": true
    }
  ]
}
```

</details>

<details>
<summary>Response JSON</summary>

```json
{
  "task_id": "test_case_042",
  "solver_version": "1.0.0",
  "solve_time_ms": 248,
  "placements": [
    {
      "sku_id": "SKU-SHO-1234",
      "instance_index": 0,
      "position": { "x_mm": 0, "y_mm": 0, "z_mm": 0 },
      "dimensions_placed": { "length_mm": 330, "width_mm": 190, "height_mm": 115 },
      "rotation_code": "LWH"
    }
  ],
  "unplaced": []
}
```

</details>

## Legacy Baseline Benchmark

| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |
|----------|-------|--------|----------|-----------|------------|--------|-----------|
| heavy_water | **0.7434** | 0.7281 | 0.5978 | 1.0000 | 1.0000 | 107/179 | 389 |
| fragile_tower | **0.6214** | 0.6382 | 0.3409 | 1.0000 | 1.0000 | 15/44 | 9 |
| liquid_tetris | **0.6846** | 0.3992 | 1.0000 | 0.8500 | 1.0000 | 84/84 | 249 |
| random_mixed | **0.7023** | 0.8198 | 0.4579 | 0.5500 | 1.0000 | 49/107 | 112 |
| **Average** | **0.6879** | | | | | | |

> Дата: 2026-03-14. Это baseline-замер для greedy + multi-restart v1.0.0. Текущий default solver в репозитории использует `portfolio_block`, а legacy hybrid сохранён как отдельная стратегия.

## Архитектура текущего default-солвера

**Auto Portfolio V2: deterministic seed search + local order search + optional selector**

1. **Scenario fingerprint** — считаем request-level признаки: число SKU, доля хрупких, weight ratio, volume ratio, max SKU share и т.д.
2. **Seed portfolio** — запускаем только top-3 быстрых seed-family:
   - `heavy_base`
   - `liquid_fill`
   - `mixed_volume`
   - `fragile_density`
   - `block_structured`
   - `coverage_tie`
3. **Local order search** — для лучшего greedy seed переставляем первые SKU в очереди и переоцениваем только несколько front-priority вариантов.
4. **Repair + Postprocess** — remove-and-refill repair, compact-downward, fragile reorder, повторная вставка unplaced.
5. **Optional selector** — маленький XGBoost classifier в `models/selector_xgb.json` может переупорядочить seed-family по request fingerprint, но fallback-эвристика остаётся основным safety net.

Baseline greedy-модули (`solver/packer.py`, `solver/scoring.py`, `solver/pallet_state.py`) и legacy hybrid сохранены для unit-тестов и сравнительных бенчмарков. Optional block ranker сохранён только для offline-экспериментов и не включён в runtime по умолчанию.

## Selector Workflow

```bash
# Собрать request-level датасет
python collect_selector_data.py --output selector_train.npz --seed-start 1000 --seed-end 1010
python collect_selector_data.py --output selector_val.npz --seed-start 1500 --seed-end 1505

# Обучить selector
python train_selector.py --train selector_train.npz --val selector_val.npz --model-dir models

# Проверить held-out accuracy и benchmark with/without selector
python evaluate_selector.py --dataset selector_val.npz --model-dir models --markdown-output docs/selector_report.md
```

## Направления улучшений

- [x] Beam search вместо greedy
- [x] Учёт `stackable: false` в валидаторе
- [x] Scenario-level selector для seed ranking
- [ ] Layer-based packing (укладка слоями)
- [ ] LNS (Large Neighborhood Search) — локальный поиск с перестроением
- [ ] Тюнинг эвристики / HYB ranking
- [ ] Больше стратегий для postprocess и локального улучшения
- [ ] 3D-визуализация (plotly/matplotlib)
