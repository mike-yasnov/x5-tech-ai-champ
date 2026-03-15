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
│   ├── pallet_state.py    # Состояние паллеты: Extreme Points, проверки
│   ├── scoring.py         # Scoring-функция для ранжирования кандидатов
│   ├── packer.py          # Greedy packer: основной цикл размещения
│   ├── solver.py          # Multi-restart обёртка
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
| `solver/scoring.py` | Веса и компоненты scoring-функции |
| `solver/packer.py` | Алгоритм упаковки, стратегии сортировки |
| `solver/solver.py` | Multi-restart логика, параметры |
| `solver/pallet_state.py` | Extreme Points, проверки, новые предикаты |

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

## Benchmark Results

### Текущие результаты (2026-03-15)

#### Сценарии организаторов

| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |
|----------|-------|--------|----------|-----------|------------|--------|-----------|
| heavy_water | **0.7434** | 0.7281 | 0.5978 | 1.0000 | 1.0000 | 107/179 | 727 |
| fragile_tower | **0.7529** | 0.8294 | 0.4773 | 0.9500 | 1.0000 | 21/44 | 419 |
| liquid_tetris | **0.6996** | 0.3992 | 1.0000 | 1.0000 | 1.0000 | 84/84 | 455 |
| random_mixed | **0.7483** | 0.8570 | 0.5327 | 0.6000 | 1.0000 | 57/107 | 801 |
| **Average** | **0.7360** | | | | | | |

#### Расширенные реалистичные сценарии

| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |
|----------|-------|--------|----------|-----------|------------|--------|-----------|
| weight_limited_repeat | **0.7904** | 0.7617 | 0.6985 | 1.0000 | 1.0000 | 95/136 | 679 |
| fragile_cap_mix | **0.7762** | 0.7002 | 0.7536 | 1.0000 | 1.0000 | 52/69 | 553 |
| mixed_column_repeat | **0.9010** | 0.8019 | 1.0000 | 1.0000 | 1.0000 | 77/77 | 679 |
| small_gap_fill | **0.7917** | 0.5833 | 1.0000 | 1.0000 | 1.0000 | 22/22 | 249 |
| non_stackable_caps | **0.8750** | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 18/18 | 186 |
| **Average** | **0.8269** | | | | | | |

#### Sanity и диагностические сценарии

| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |
|----------|-------|--------|----------|-----------|------------|--------|-----------|
| exact_fit | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4/4 | 7 |
| fragile_mix | **0.9864** | 1.0000 | 0.9545 | 1.0000 | 1.0000 | 21/22 | 191 |
| support_tetris | **0.9571** | 1.0000 | 0.8571 | 1.0000 | 1.0000 | 12/14 | 77 |
| cavity_fill | **0.7111** | 0.5556 | 0.7778 | 1.0000 | 1.0000 | 14/18 | 50 |
| count_preference | **0.9000** | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 2/3 | 4 |
| **Average** | **0.9109** | | | | | | |

**Overall average: 0.8309** | Constraint tests: **74/76 (97.4%)**

> Solver v1.x, greedy + multi-restart (30+ стратегий) + LNS + postprocess, budget 900ms.

### Прогресс от baseline

| Scenario | Baseline | Текущий | Δ |
|----------|----------|---------|---|
| heavy_water | 0.7434 | 0.7434 | — |
| fragile_tower | 0.6214 | **0.7529** | **+0.1315** |
| liquid_tetris | 0.6846 | **0.6996** | **+0.0150** |
| random_mixed | 0.7023 | **0.7483** | **+0.0460** |
| **Organizer avg** | **0.6879** | **0.7360** | **+0.0481** |

### Baseline (2026-03-14, отправная точка)

| Scenario | Score | Volume | Coverage | Fragility | Time Score | Placed | Time (ms) |
|----------|-------|--------|----------|-----------|------------|--------|-----------|
| heavy_water | **0.7434** | 0.7281 | 0.5978 | 1.0000 | 1.0000 | 107/179 | 389 |
| fragile_tower | **0.6214** | 0.6382 | 0.3409 | 1.0000 | 1.0000 | 15/44 | 9 |
| liquid_tetris | **0.6846** | 0.3992 | 1.0000 | 0.8500 | 1.0000 | 84/84 | 249 |
| random_mixed | **0.7023** | 0.8198 | 0.4579 | 0.5500 | 1.0000 | 49/107 | 112 |
| **Average** | **0.6879** | | | | | | |

> Solver v1.0.0, greedy + multi-restart (5 стратегий), budget 900ms.

## Архитектура солвера

**Extreme Points + Greedy + Multi-restart + LNS + Postprocess**

1. **Extreme Points** — генерация кандидатных позиций (cap=200, 5 EP per box + проекции)
2. **Greedy packer** — для каждого короба выбирает лучшую позицию × ориентацию по scoring-функции
3. **Multi-restart** — 30+ стратегий (sort key × weight profile × packer type), adaptive stop
4. **LNS** — destroy-repair post-processing с safe rebuild (валидация support)
5. **Postprocess** — compact_downward → reorder_fragile → try_insert → second reorder_fragile → remove_and_refill → compact

### Стратегии сортировки

`constrained_first`, `base_area_desc`, `fragile_last`, `volume_desc`, `volume_asc`, `density_desc`, `non_stackable_last`, `height_desc`, `weight_desc`, `max_dim_desc`, `heavy_base_fragile_top`, `stackable_base`, `score_per_kg`, `light_fillers_first`, `coverage_optimal`, `random_0..9`

### Weight profiles

`default`, `contact_heavy`, `fill_heavy`, `layer_heavy`, `fragile_avoid`, `compact`, `wall_hugger`, `fragile_strict` (hard-block heavy-on-fragile)

### Ключевые оптимизации (feature/rl-solver)

- **Phase-aware insert ordering** — non-fragile stackable first для стабильной базы
- **Safe LNS rebuild** — validate support с `can_place()`, orphans в repair pool
- **Second reorder_fragile** после insert — fix violations от вставки
- **Fragile-safe candidate tracking** — сравнение 0-violation альтернативы после postprocess
- **Fill_heavy fragile strategies** в top-6 для достижения в 10-restart бюджете
- **Hard-block в fragile_strict** — return -1.0 score для heavy-on-fragile
- **Adaptive strategy head** — prepend weight-aware стратегий при weight_pressure > 1.3
- **Weight-aware sort keys** — `score_per_kg`, `light_fillers_first`, `coverage_optimal`
- **Layer packer** — EP greedy с layer z-level bonus

## Ограничения по физике

| Scenario | Лимитирующий фактор | Примечание |
|----------|---------------------|------------|
| heavy_water | Вес (170% лимита) | 107/179 items = 99.4% веса использовано |
| fragile_tower | Объём (113% паллеты) + 50% fragile | 21/44 items |
| liquid_tetris | Все items помещаются | vol_util=0.40 (мелкие предметы) |
| random_mixed | Вес (115%) + Объём (120%) + 59% fragile | 57/107 items |
