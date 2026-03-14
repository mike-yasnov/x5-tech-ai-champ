---
name: docs
description: Access and navigate project documentation for AI CHEMP 3D Pallet Packing challenge. Use to find task requirements, scoring rules, data formats, or research reports.
argument-hint: [topic or "search <query>"]
allowed-tools: Read Glob Grep WebFetch WebSearch
metadata:
  author: ai-chemp
  version: "1.0"
  category: documentation
---

# Documentation Skill

Навигация по документации проекта AI CHEMP (X5 Tech Smart 3D Packing).

## Структура документации

### Основные файлы

| Файл | Содержание |
|------|-----------|
| `docs/task.md` | Полное описание задачи, форматы данных, метрики оценки |
| `docs/deep-research-report.md` | Исследовательский отчёт #1 |
| `docs/deep-research-report_2.md` | Исследовательский отчёт #2 |
| `docs/deep-research-report_3.md` | Исследовательский отчёт #3 |
| `docs/s10732-026-09586-5.pdf` | Научная статья по bin packing |

### Ключевые разделы task.md

- **Контекст** — описание проблемы в ритейл-логистике
- **Задача** — что нужно реализовать (солвер + визуализация + генератор)
- **Формат входных данных** — JSON-спецификация request
- **Формат выходных данных** — JSON-спецификация response
- **Hard Constraints** — жёсткие ограничения (5 правил)
- **Soft Metrics** — метрики качества (4 показателя с весами)
- **Дополнительная информация** — рекомендации по алгоритмам

## Режимы работы

**При вызове без аргументов:**
- Покажи краткую сводку по задаче и ключевые метрики

**При `<topic>`:**
- Найди релевантную информацию по теме в документации
- Темы: `constraints`, `metrics`, `format`, `algorithms`, `research`

**При `search <query>`:**
- Поиск по всем файлам документации
- Выведи релевантные фрагменты с указанием источника

## Краткая сводка задачи

**Цель:** Оптимальная 3D укладка коробов на паллету (NP-hard задача)

**Вход:** Параметры паллеты + список SKU с размерами, весом, ограничениями
**Выход:** Координаты и ориентация каждого короба

**Hard Constraints (нарушение = 0 баллов):**
1. Коробы внутри границ паллеты
2. Нет пересечений (коллизий)
3. Опора ≥ 60% площади основания
4. `strict_upright` — только вращение вокруг Z
5. Суммарный вес ≤ `max_weight_kg`

**Scoring:**
- Volume Utilization: 50%
- Item Coverage: 30%
- Fragility Compliance: 10%
- Execution Time: 10%

**Рекомендуемые подходы:** Эвристики, генетические алгоритмы, simulated annealing, RL

## Правила

- Всегда ссылайся на конкретный файл и раздел при цитировании
- При противоречиях между документами приоритет у `docs/task.md`
- Исследовательские отчёты содержат анализ алгоритмов — используй их для рекомендаций по реализации солвера
