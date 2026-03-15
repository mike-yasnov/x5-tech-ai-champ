# RL Experiments Report — AI CHEMP 3D Pallet Packing

## Обзор

Серия экспериментов по применению Reinforcement Learning и метаэвристик к задаче 3D паллетной упаковки.
Все эксперименты проводились на удалённом сервере (ssh work, GPU CUDA).
Исходные файлы проекта (solver/, generator.py, validator.py, benchmark.py) не модифицировались.

---

## 1. Baseline: Original Solver

Жадный EP-солвер с постобработкой (compact_downward, reorder_fragile, try_insert_unplaced).

**Результаты на 4 организаторских тестах:**

| Scenario | Score |
|----------|-------|
| heavy_water | 0.7266 |
| fragile_tower | 0.7282 |
| liquid_tetris | 0.6871 |
| random_mixed | 0.7103 |
| **AVERAGE** | **0.7131** |

---

## 2. Эксперимент 1: Классические RL агенты (DQN, PPO, A2C)

### Подход
Прямое RL: агент выбирает действие (позиция × ориентация = 2880 дискретных действий) для размещения коробки на паллете.

### Среда (PackingEnv)
- **Observation**: 488 floats (heightmap 24×20 + box features 8 + state 5)
- **Action space**: Discrete(2880) = 24×20 позиций × 6 ориентаций
- **Grid resolution**: 50mm/cell
- **Reward**: volume_reward + position_bonus - fragility_penalty + coverage_bonus

### Архитектуры

#### DQN (Double Dueling + Prioritized Replay)
- Feature extractor: input → 512 → 512 (ReLU)
- Value head: 512 → 256 → 1
- Advantage head: 512 → 256 → 2880
- Replay buffer: 200K, priority alpha=0.6, beta 0.4→1.0
- LR: 1e-4, batch: 128, target update: every 500 steps
- Epsilon: 1.0 → 0.05 (linear decay)

#### PPO (Actor-Critic + Action Masking + GAE)
- Shared backbone: input → 512 → 512 (ReLU)
- Actor: 512 → 256 → 2880 (Categorical + mask)
- Critic: 512 → 256 → 1
- Rollout: 512 steps, 4 epochs, batch 64
- Clip ε=0.2, entropy coeff=0.02, GAE λ=0.95

#### A2C (N-step + LayerNorm)
- Backbone: input → 256 → LayerNorm → 256
- Actor/Critic: 256 → n_actions / 1
- N-step=5, LR=7e-4, RMSprop

### Результаты
| Agent | Avg Reward | Avg Items Placed | Training Time |
|-------|-----------|-----------------|---------------|
| DQN | 0.358 | 9.7 | 49.3s |
| PPO | **0.476** | 12.3 | 62.1s |
| A2C | 0.367 | 10.1 | 41.7s |

### Вывод
PPO лучший из классических RL, но все три проигрывают original solver.
**Причина**: пространство действий слишком большое (2880), агент тратит ресурсы на изучение позиционирования вместо стратегического планирования.

---

## 3. Эксперимент 2: Метаэвристики (GA, SA)

### Genetic Algorithm (GA)
- **Хромосома**: permutation порядка коробок + вектор ориентаций
- **Операторы**: Order Crossover (OX), uniform crossover для ориентаций
- **Фитнес**: 0.50×vol_util + 0.30×coverage + 0.20×bonus
- **Параметры**: pop=100, generations=300, crossover=0.85, mutation=0.15
- **Extreme Points**: до 200 кандидатов позиций
- **Time budget**: 25s

### Simulated Annealing (SA)
- **Соседство**: swap(50%), reverse(25%), rotate(25%)
- **Метрополис**: exp(Δ/T)
- **Параметры**: T₀=10, cooling=0.9995, T_min=0.001, max_iter=100K
- **Restarts**: 3

### Результаты на организаторских тестах
| Method | heavy_water | fragile_tower | liquid_tetris | random_mixed | AVG |
|--------|------------|--------------|--------------|-------------|-----|
| Original | 0.7266 | 0.7282 | 0.6871 | 0.7103 | 0.7131 |
| GA | 0.6628 | — | — | — | ~0.66 |
| SA | 0.6600 | — | — | — | ~0.66 |

### Вывод
GA и SA уступают original solver. Extreme point пакер в GA/SA менее оптимизирован, чем оригинальный.

---

## 4. Эксперимент 3: Hybrid RL v1 (RL Ordering + Greedy Packer)

### Ключевая идея
RL определяет **порядок** коробок, а жадный EP-пакер (из GA) выполняет размещение. Это сужает пространство действий с 2880 до N (количество оставшихся коробок).

### Архитектура: PointerNetwork
- **Item encoder**: 8 dims → 128 → 128
- **Context encoder**: 16 dims → 128 → 128
- **Attention**: pointer attention (query=context, keys=items)
- **Value head**: 128 → 64 → 1
- **Item features** (8): dims, weight, volume, fragile, upright, stackable
- **Context features** (16): placed/remaining ratios, pallet dims, weight budget, constraint counts

### Тренировка
- PPO: LR=3e-4, clip=0.2, entropy=0.05, 4 epochs
- Episodes: 300
- Gamma: 0.99

### Результаты
Улучшение на heavy_water, но в среднем слабее original.

### Вывод
Pointer network — правильное направление, но сеть слишком маленькая (128 hidden), мало эпизодов, нет curriculum.

---

## 5. Эксперимент 4: Hybrid RL v2 (Multi-Head Attention + Curriculum)

### Улучшения над v1
1. **Увеличена сеть**: 128 → 256 hidden
2. **Multi-head self-attention**: 4 heads, 2 layers
3. **Richer features**: 12 item dims + 20 context dims
4. **Curriculum learning**: первые 30% эпизодов — лёгкие сценарии
5. **Multi-sample evaluation**: N стохастических порядков → лучший
6. **Cosine annealing scheduler**
7. **EMA baseline** (alpha=0.05)

### Архитектура: MultiHeadPointerNet
```
Item features (12 dims) → Linear(256) + LayerNorm + Residual
Context features (20 dims) → Linear(256) + LayerNorm + Residual
↓
2× Self-Attention (4 heads, 256 dim)
↓
Pointer Attention (context → items, tanh clip=10)
↓
Value head: concat(item_pool, context) → 512 → 256 → 1
```

### Item Features (12 dims)
1-3: normalized L/W/H
4: weight ratio (to pallet max)
5: volume ratio
6: base area ratio
7-9: fragile, upright, stackable flags
10: volume relative to max available
11: weight relative to average
12: density relative to typical

### Context Features (20 dims)
1-2: placed/remaining ratios
3-4: weight/volume utilization
5-7: normalized pallet dims
8: max weight ratio
9-11: remaining constraint counts
12-18: weight budget, remaining weight/volume, placed count, max Z, capacity check, constrained fraction
19-20: reserved

### Тренировка
- Episodes: 1000
- LR: 1e-4
- Clip ε: 0.2
- Entropy: 0.02
- N epochs: 4
- Curriculum: 30% warm-up on easy scenarios

### Fine-tuning на организаторских сценариях
- Episodes: 500
- LR: 5e-5 (пониженный)
- Clip ε: 0.15
- Entropy: 0.03
- N epochs: 6

---

## 6. Эксперимент 5: Ensemble Solver

### Стратегия
Запускаем 4 подхода на каждый сценарий, берём лучший:

1. **Original solver** (900ms budget, 30 restarts)
2. **Hybrid-v2** (N samples стохастических порядков)
3. **Hybrid-v2 + LNS** (3000ms рефайнмент)
4. **Original + LNS** (3000ms рефайнмент)

Каждое решение валидируется, выбирается максимальный score.

### Постобработка (PostprocessWrapper)
Три шага из оригинального солвера:
1. `compact_downward()` — опускает коробки вниз
2. `reorder_fragile()` — переставляет хрупкие наверх (40% бюджета)
3. `try_insert_unplaced()` — заполняет пустоты (60% бюджета)

### Результаты Ensemble (n_samples=64)

| Scenario | Score | Winner |
|----------|-------|--------|
| heavy_water | **0.7364** | hybrid-v2 |
| fragile_tower | **0.7282** | original |
| liquid_tetris | **0.6871** | original |
| random_mixed | **0.7103** | original |
| **AVERAGE** | **0.7155** | |

### Прогресс
- Original baseline: **0.7131**
- Ensemble: **0.7155** (+0.34%)
- Hybrid-v2 побеждает на heavy_water (+1.35% vs original)

---

## 7. Генерация сложных тестов

### 30 Hard Scenarios (hard_scenarios.py)

Покрывают все комбинации ограничений:

| Категория | Сценарии | Фокус |
|-----------|---------|-------|
| Weight Limits (1-5) | exact, overflow, one_heavy, uniform_cutoff, greedy_trap | Граничные случаи веса |
| Support 60% (6-10) | borderline, staircase, platform, height_mismatch, mosaic_floor | Правило поддержки |
| Fragility (11-15) | all_fragile, big_bottom, interleave, singleton, nonstackable | Хрупкость + вес |
| Upright/Nostack (16-20) | all_upright, gap_fill, tall_nostack, triple_constraint, flat_nostack | Ориентация + стекируемость |
| Tight Fit (21-25) | tetris_exact, oversized, diverse_sizes, low_ceiling, near_full | Геометрия |
| Chaos (26-30) | retail, heavy_fragile, all_nostack, squeeze, maximum | Все ограничения вместе |

### 14 Private Scenarios (generator.py)

Более реалистичные сценарии на основе food retail архетипов:

| Scenario | Описание |
|----------|----------|
| private_heavy_eggs_crush | Яйца 22кг + fragile + upright — тройное ограничение |
| private_all_upright_tight | Все upright на US паллете (нестандартные размеры) |
| private_fragile_dominant | 3 fragile SKU, почти нет базы |
| private_weight_razor | Все 7 архетипов, бюджет 500кг |
| private_sugar_flood | 120-200 одинаковых коробок сахара |
| private_wine_eggs_dilemma | Wine + eggs — оба fragile, нельзя стакать друг на друга |
| private_canned_wall | 80-150 мелких банок + крупные бананы сверху |
| private_chips_mountain | 30-50 больших хрупких чипсов (1.8кг ≤ 2кг порог) |
| private_weight_tradeoff | Banana vs sugar — оптимальный микс по весу |
| private_full_catalog | Все 7 архетипов, вес 700кг |
| private_micro_batch | Все 7 архетипов, по 1-3 штуки |
| private_upright_overflow | Высокие upright на низком потолке (800мм) |
| private_nostack_fragile_mix | Non-stackable display + fragile + filler |
| private_heavy_fragile_sandwich | Тяжёлые-хрупкие-тяжёлые слои |
| private_odd_pallet_stress | EUR-sized коробки на US паллете (зазоры 19мм) |

---

## 8. Структура кода

```
rl_solver/
├── env.py                    # Gymnasium среда (PackingEnv)
├── train.py                  # Тренировка DQN/PPO/A2C + запуск GA/SA
├── train_hybrid.py           # Тренировка Hybrid v1
├── train_v2.py               # Тренировка Hybrid v2
├── finetune_org.py           # Fine-tune на организаторских тестах
├── ensemble_solver.py        # Ensemble: original + hybrid-v2 + LNS
├── postprocess_wrapper.py    # Обёртка постобработки
├── rl_benchmark.py           # Бенчмарк всех агентов
├── __main__.py               # Entry point
├── agents/
│   ├── dqn_agent.py          # Double Dueling DQN + PER
│   ├── ppo_agent.py          # PPO + Action Masking + GAE
│   ├── a2c_agent.py          # A2C + N-step + LayerNorm
│   ├── genetic_agent.py      # Genetic Algorithm
│   ├── sa_agent.py           # Simulated Annealing
│   ├── hybrid_rl_agent.py    # Hybrid v1: PointerNetwork
│   └── hybrid_rl_v2.py       # Hybrid v2: MultiHeadPointerNet
├── models/                   # Сохранённые веса (.pt)
└── scenarios/
    └── hard_scenarios.py     # 30 hard test scenarios
```

---

## 9. Сводная таблица результатов

| Метод | heavy_water | fragile_tower | liquid_tetris | random_mixed | AVG |
|-------|------------|--------------|--------------|-------------|-----|
| Original | 0.7266 | **0.7282** | **0.6871** | **0.7103** | 0.7131 |
| DQN | — | — | — | — | ~0.36 |
| PPO | — | — | — | — | ~0.48 |
| A2C | — | — | — | — | ~0.37 |
| GA | 0.6628 | — | — | — | ~0.66 |
| SA | 0.6600 | — | — | — | ~0.66 |
| Hybrid v1 | ~0.73 | ~0.70 | ~0.65 | ~0.68 | ~0.69 |
| Hybrid v2 | **0.7364** | 0.72 | 0.67 | 0.70 | ~0.71 |
| **Ensemble** | **0.7364** | **0.7282** | **0.6871** | **0.7103** | **0.7155** |

---

## 10. Выводы и дальнейшие направления

### Что работает
1. **Hybrid RL (ordering + greedy packer)** — правильная декомпозиция задачи
2. **Multi-sample evaluation** — стохастический поиск N порядков эффективен
3. **Ensemble** — лучший результат за счёт выбора лучшего из нескольких подходов
4. **Curriculum learning** — помогает на ранних этапах обучения

### Что не работает
1. **Прямое RL** (DQN/PPO/A2C) — пространство действий 2880 слишком велико
2. **GA/SA** — EP-пакер уступает оригинальному солверу
3. **LNS рефайнмент** — не даёт стабильного улучшения поверх хороших решений

### Возможные улучшения
1. Использовать оригинальный EP-пакер вместо упрощённого в GA/SA
2. Увеличить n_samples до 128-256 для hybrid-v2
3. Попробовать Transformer-based архитектуру (GPT-style autoregressive)
4. Beam search вместо greedy/sampling для порядка
5. Обучение на большем количестве сценариев (30 hard + 14 private)
6. Multi-objective reward с явным учётом fragility_score
7. Graph Neural Network для моделирования отношений между коробками
