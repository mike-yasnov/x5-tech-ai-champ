---
name: validator
description: Validate 3D packing solutions against hard constraints and calculate quality scores. Use to check solver output, debug placement issues, or understand scoring. Run after solver produces a response JSON.
argument-hint: [request.json response.json | "explain" | "debug <sku_id>"]
allowed-tools: Read Edit Write Bash(python3 validator.py*) Bash(python3 -c *) Glob Grep
metadata:
  author: ai-chemp
  version: "1.0"
  category: testing
---

# Validator Skill

Валидация решений задачи 3D Pallet Packing и расчёт скоринга.

## Контекст проекта

Валидатор находится в `validator.py` в корне проекта. Проверяет корректность укладки и считает итоговый балл.

## Hard Constraints (нарушение = 0 баллов)

1. **Перевес** — суммарный вес размещённых коробов ≤ `max_weight_kg` паллеты
2. **Границы паллеты** — все коробы внутри `[0, length] × [0, width] × [0, max_height]`
3. **Upright constraint** — если `strict_upright: true`, высота после размещения = оригинальная высота (вращение только вокруг Z)
4. **Коллизии** — AABB-пересечение между любой парой коробов = 0
5. **Гравитация** — опора ≥ 60% площади основания (либо z=0, либо на верхней грани других коробов)

### Anti-cheat
- Размеры после размещения должны быть перестановкой оригинальных размеров
- Количество размещённых экземпляров SKU ≤ `quantity` из запроса

## Soft Metrics (скоринг)

| Метрика             | Вес  | Формула                                              |
|---------------------|------|------------------------------------------------------|
| Volume Utilization  | 50%  | sum(box_volumes) / pallet_volume                     |
| Item Coverage       | 30%  | placed_items / total_requested_items                 |
| Fragility Score     | 10%  | max(0, 1 - 0.05 × fragility_violations)             |
| Time Score          | 10%  | ≤1s→1.0, ≤5s→0.7, ≤30s→0.3, >30s→0.0               |

**Fragility violation:** тяжёлый короб (>2кг) стоит на хрупком коробе (непосредственный контакт z_top = z_bottom + overlap > 0).

**Final score** = 0.50×vol + 0.30×coverage + 0.10×fragility + 0.10×time

## Команды

### Валидация решения
```bash
python3 -c "
from validator import evaluate_solution
import json
req = json.load(open('<request.json>'))
resp = json.load(open('<response.json>'))
result = evaluate_solution(req, resp)
print(json.dumps(result, indent=2))
"
```

### Режимы работы

**При `<request.json> <response.json>`:**
1. Запусти валидацию
2. Если `valid: false` — объясни причину ошибки и предложи исправление
3. Если `valid: true` — покажи метрики и подсказки по улучшению скора

**При `explain`:**
- Объясни все проверки и метрики валидатора простым языком

**При `debug <sku_id>`:**
1. Найди все placements данного SKU в response
2. Проверь каждый на все hard constraints
3. Покажи визуализацию позиции и конфликтов

## Формат входных/выходных данных

### Request JSON
```json
{
  "task_id": "string",
  "pallet": { "length_mm", "width_mm", "max_height_mm", "max_weight_kg" },
  "boxes": [{ "sku_id", "length_mm", "width_mm", "height_mm", "weight_kg", "quantity", "strict_upright", "fragile", "stackable" }]
}
```

### Response JSON
```json
{
  "task_id": "string",
  "solve_time_ms": number,
  "placements": [{
    "sku_id", "instance_index",
    "position": { "x_mm", "y_mm", "z_mm" },
    "dimensions_placed": { "length_mm", "width_mm", "height_mm" },
    "rotation_code": "LWH|LHW|WLH|WHL|HLW|HWL"
  }],
  "unplaced": [{ "sku_id", "quantity_unplaced", "reason" }]
}
```

## Правила

- Не изменяй пороги hard constraints без согласования (60% опоры, 2кг fragility)
- При изменении метрик обновляй веса в соответствии с `docs/task.md`
- `stackable: false` пока НЕ проверяется в валидаторе — учитывай при доработке
