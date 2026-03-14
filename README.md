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

## Архитектура baseline-солвера

**Extreme Points + Greedy + Multi-restart**

1. **Extreme Points** — генерация кандидатных позиций (не полный перебор x,y,z)
2. **Greedy packer** — для каждого короба выбирает лучшую позицию × ориентацию по scoring-функции
3. **Multi-restart** — прогоняет 5 стратегий сортировки, выбирает лучший скор

Стратегии сортировки: `volume_desc`, `weight_desc`, `base_area_desc`, `density_desc`, `constrained_first`

## Направления улучшений

- [ ] Beam search вместо greedy (ширина луча = качество vs скорость)
- [ ] Layer-based packing (укладка слоями)
- [ ] LNS (Large Neighborhood Search) — локальный поиск с перестроением
- [ ] Тюнинг весов scoring-функции
- [ ] Больше стратегий сортировки для multi-restart
- [ ] Учёт `stackable: false` в валидаторе
- [ ] 3D-визуализация (plotly/matplotlib)
