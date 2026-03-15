# Решение: 3D-оптимизатор укладки паллет (Х5 Tech)

---

## Информация о команде

| Поле | Данные |
| :-- | :-- |
| **Название команды** | R2BD |
| **Город** | Санкт-Петербург |
| **Площадка** | ИТМО |

### Состав команды

|  | ФИО | Вуз | Курс / год выпуска |
| :-- | :-- | :-- | :-- |
| 1 | Яснов Михаил | ИТМО | 3 курс/2027 |
| 2 | Козак Борис | ИТМО | 4 курс/2026| |
| 3 | Хачатрян Геворк| ИТМО | 3 курс/2027 |
| 4 | Пепеляев Мирон | СПбГУ| 1 курс магистратуры/2027 |

---

## Описание решения

Проект содержит два независимых солвера для задачи 3D bin packing на паллетах, объединённых в единый бенчмарк и web-интерфейс.

### Структура проекта

```
├── base_solver/          # Солвер 1: Portfolio-Block Hybrid
├── alternative_solver/   # Солвер 2: Multi-restart EP + LNS
├── benchmark.py          # Unified benchmark (сравнение обоих солверов)
├── webapp.py             # Web UI (NiceGUI + FastAPI)
├── experiment_service.py # Сервис экспериментов для web UI
├── generator.py          # Генератор тестовых сценариев
├── validator.py          # Валидатор решений (исправленный)
├── validator_org.py      # Оригинальный валидатор организаторов
├── visualize.py          # 3D визуализация (HTML)
├── scenario_catalog.py   # Каталог benchmark-сценариев
└── tests/                # Тесты
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

# Решить один request (base_solver)
python -m base_solver request_heavy_water.json -o response.json

# Решить один request (alternative_solver)
python -m alternative_solver request_heavy_water.json -o response.json

# Прогнать тесты
pytest tests/ -v

# Основной benchmark (оба солвера)
python benchmark.py

# Benchmark только одного солвера
python benchmark.py --solver base
python benchmark.py --solver alternative

# Веб-лаборатория
python webapp.py
```

После запуска web UI откройте `http://127.0.0.1:3030/`.

Порт и хост можно переопределить через переменные окружения `PORT` и `HOST`.

## CLI

```bash
# base_solver
python -m base_solver <input.json> [options]

# alternative_solver
python -m alternative_solver <input.json> [options]
```

### Примеры

```bash
# Один файл -> один output
python -m base_solver request_heavy_water.json -o result.json

# Batch: output-файлы будут названы как response_<name>.json
python -m base_solver request_*.json

# Принудительно включить legacy_hybrid и указать beam width
python -m base_solver request_random_mixed.json --strategy legacy_hybrid --beam-width 4

# Увеличить time budget
python -m base_solver request_random_mixed.json --time-budget 5000

# Alternative solver
python -m alternative_solver request_heavy_water.json -o result.json --time-budget 5000
```

## Солверы

### base_solver — Portfolio-Block Hybrid

Стратегии: `portfolio_block`, `legacy_hybrid`, `legacy_greedy`.

`portfolio_block` делает не один жадный проход, а короткий runtime-портфель кандидатов:

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

### alternative_solver — Multi-restart EP + LNS

Многопроходный солвер с адаптивным выбором стратегий:

1. Строит портфель из комбинаций (sort_key, weight_profile, packer_type).
2. Адаптивно подбирает стратегии на основе свойств задачи (fragile_ratio, weight_pressure, volume_pressure).
3. Запускает стратегии параллельно (ProcessPoolExecutor) или последовательно с adaptive time stopping.
4. Лучшее решение улучшается через LNS (Large Neighborhood Search) — destroy + repair.
5. Применяет постобработку: compaction, swap, fragility repair.

## Benchmark

```bash
# Оба солвера
python benchmark.py

# Один солвер
python benchmark.py --solver base
python benchmark.py --solver alternative

# С визуализацией
python benchmark.py --viz viz_output/

# Больше restarts
python benchmark.py --restarts 30
```

Benchmark прогоняет 9 сценариев (4 organizer + 5 diagnostic) и выводит сравнительную таблицу с метриками: volume utilization, item coverage, fragility score, time score.

## Архитектура решения

### Алгоритмы
- Extreme Points (EP) — генерация точек размещения
- Greedy packing с различными sort keys и weight profiles
- Layered / staged constructive packing
- Block-based packing (группировка одинаковых SKU)
- Small beam search
- Local order search
- LNS (Large Neighborhood Search) — destroy + repair
- Repair / remove-and-refill
- Portfolio selection — выбор лучшей стратегии
- ML ranking (optional):
  - XGBoost selector для выбора стратегии
  - XGBoost ranker для block-кандидатов

### Ключевые особенности
- Два независимых солвера с единым интерфейсом
- Runtime portfolio — классифицирует заказ, потом решает, как паковать
- Сочетает несколько типов конструктивных алгоритмов
- Общая геометрия и feasibility-логика
- ML не заменяет solver, а ранжирует варианты
- Учитывает fragility и stackability внутри поиска
- Специальные режимы для upright/fragile/overload кейсов
- Локальное улучшение и repair после построения решения

### Скоринг

```
final_score = 0.50 * volume_utilization
            + 0.30 * item_coverage
            + 0.10 * fragility_score
            + 0.10 * time_score
```

## Web UI

Интерактивная лаборатория для экспериментов с укладкой:

- Выбор сценария и параметров солвера
- 3D визуализация укладки
- Live-пересчёт score при изменении весов
- История экспериментов с возможностью клонирования и сравнения

```bash
python webapp.py
# → http://127.0.0.1:3030/
```
