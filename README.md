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

### Подход

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

### Архитектура решения
Алгоритмы, используемые в решении
- Extreme Points
- Greedy packing
- Layered / staged constructive packing
- Block-based packing
- Small beam search
- Local order search
- Repair / remove-and-refill
- Portfolio selection
- частично ML ranking:
  - selector для выбора стратегии
  - ranker для block-кандидатов
### Ключевые особенности
- не один solver, а runtime portfolio
- сначала классифицирует заказ, потом решает, как его паковать
- сочетает несколько типов конструктивных алгоритмов
- общая геометрия и общая feasibility-логика для всех путей
- ML не заменяет solver, а только ранжирует варианты
- учитывает fragility и stackability не только в validator, но и внутри поиска
- умеет включать специальные режимы для upright/fragile/overload кейсов
- после построения решения делает локальное улучшение и repair
- ориентирован на быстрый выбор сильных кандидатов, а не на полный перебор
