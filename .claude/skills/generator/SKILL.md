---
name: generator
description: Generate test scenarios for 3D pallet packing. Creates JSON test cases with different box archetypes, pallet types, and complexity levels. Use when you need new test data or want to extend the scenario generator.
argument-hint: [scenario-type or "new-archetype <name>" or "batch <count>"]
allowed-tools: Read Edit Write Bash(python3 generator.py*) Bash(python3 -c *) Glob Grep
metadata:
  author: ai-chemp
  version: "1.0"
  category: testing
---

# Generator Skill

Управление генератором тестовых сценариев для задачи 3D Pallet Packing.

## Контекст проекта

Генератор находится в `generator.py` в корне проекта. Он создаёт JSON-файлы с тестовыми сценариями для укладки коробов на паллеты.

## Архитектура генератора

### Паллеты (PALLETS)
- `EUR_1200x800` — европаллета, макс. высота 1800мм, макс. вес 1000кг
- `EUR_1200x1000` — увеличенная европаллета, макс. высота 2000мм
- `US_48x40` — американский стандарт (1219x1016мм)

### Архетипы коробов (FOOD_RETAIL_ARCHETYPES)
| Ключ     | Описание       | Д×Ш×В (мм)      | Вес (кг) | Upright | Fragile |
|----------|---------------|------------------|----------|---------|---------|
| banana   | Bananas Box    | 502×394×239      | 19.0     | да      | нет     |
| sugar    | Sugar 10kg     | 400×300×150      | 10.0     | нет     | нет     |
| water    | Water Pack     | 280×190×330      | 9.2      | да      | нет     |
| wine     | Wine Case      | 250×170×320      | 8.0      | да      | да      |
| chips    | Chips Carton   | 600×400×400      | 1.8      | нет     | да      |
| eggs     | Eggs 360pcs    | 630×320×350      | 22.0     | да      | да      |
| canned   | Canned Peas    | 300×200×120      | 6.0      | да      | нет     |

### Типы сценариев
- `heavy_water` — тяжёлые грузы (вода + сахар), тест на вес
- `fragile_tower` — хрупкие товары (бананы + чипсы + яйца), тест на fragility
- `liquid_tetris` — жидкости + консервы (вода + вино + консервы), тест на upright
- `random_mixed` — случайный микс 4-7 архетипов, общий тест

### Шум размеров
Функция `_noise_int()` добавляет ±2% шум к размерам для реалистичности.

## Команды

### Генерация стандартных сценариев
```bash
python3 generator.py
```
Создаёт 4 файла: `request_heavy_water.json`, `request_fragile_tower.json`, `request_liquid_tetris.json`, `request_random_mixed.json`.

### Режимы работы

**При вызове без аргументов** — покажи текущие сценарии и предложи сгенерировать.

**При `scenario-type`** — сгенерируй конкретный сценарий:
1. Вызови `python3 -c "from generator import *; import json; task = generate_scenario('task_X', '<type>', seed=<seed>); print(json.dumps(task, indent=2))"`
2. Покажи статистику: кол-во коробов, общий вес, соответствие ограничениям паллеты

**При `new-archetype <name>`:**
1. Запроси у пользователя параметры (размеры, вес, upright, fragile)
2. Добавь новый архетип в `FOOD_RETAIL_ARCHETYPES` в `generator.py`
3. Предложи добавить его в существующий или новый сценарий

**При `batch <count>`:**
1. Сгенерируй `<count>` сценариев с разными seed-ами
2. Используй все типы сценариев по кругу
3. Сохрани в директорию `test_cases/`

## Правила

- Всегда используй фиксированный seed для воспроизводимости
- При добавлении новых архетипов проверяй реалистичность размеров и веса
- Не изменяй формат выходного JSON — он должен соответствовать спецификации в `docs/task.md`
- При генерации batch выводи сводную статистику по всем сценариям
