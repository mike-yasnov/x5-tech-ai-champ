# Результаты Private Test: средний скор 0.7368

## Разбор по сценариям (от худшего к лучшему)

| Сценарий                       | Score  | Слабое место           | Почему сложно                                                                      |
| ------------------------------ | ------ | ---------------------- | ---------------------------------------------------------------------------------- |
| private_chips_mountain         | 0.6263 | coverage=0.28 (12/43)  | Огромные коробки 600x400x400 — мало помещается на слой, height теряется            |
| private_micro_batch            | 0.6384 | volume=0.28            | 11 разных мелких коробок — низкая плотность при максимальном разнообразии          |
| private_upright_overflow       | 0.6702 | coverage=0.35 (32/92)  | Все upright h=320-350mm, потолок 800mm — всего 2 слоя, 60+ items не влезают        |
| private_odd_pallet_stress      | 0.6989 | volume=0.52            | US-паллета (1219x1016) + EUR-размерные коробки — остатки 19mm/216mm не заполняются |
| private_weight_razor           | 0.7299 | coverage=0.40 (36/90)  | 7 SKU но только 500kg — banana(19kg)+eggs(22kg) моментально съедают бюджет         |
| private_wine_eggs_dilemma      | 0.7376 | fragility=0.0          | Wine и eggs оба fragile+upright — стэкинг друг на друга = взаимные violations      |
| private_full_catalog           | 0.7365 | coverage=0.23 (40/175) | 7 SKU x ~25 = 175 items, 700kg → жёсткий weight cutoff, мало влезает               |
| private_heavy_fragile_sandwich | 0.7520 | coverage=0.48          | banana+water(heavy) vs chips+eggs(fragile) — бутерброд constraint                  |
| private_fragile_dominant       | 0.7566 | fragility=0.0          | 3 fragile SKU (wine+chips+eggs), мало non-fragile основы → massive violations      |
| private_weight_tradeoff        | 0.7604 | coverage=0.41          | banana(19kg) vs sugar(10kg) при 600kg — жадный выбор критичен                      |
| private_canned_wall            | 0.7565 | volume=0.51            | 105 мелких canned (300x200x120) — много щелей, low density                         |
| private_all_upright_tight      | 0.7817 | fragility=0.4          | 5 SKU all upright, в т.ч. wine+eggs fragile — violations неизбежны                 |
| private_sugar_flood            | 0.7837 | coverage=0.66          | 154 одинаковых коробок sugar — тест на масштаб single-SKU                          |
| private_heavy_eggs_crush       | 0.7933 | fragility=0.55         | Eggs (22kg+fragile+upright) — самый проблемный архетип                             |
| private_nostack_fragile_mix    | 0.8299 | coverage=0.54          | Non-stackable display caps + fragile wine+chips                                    |

## Ключевые слабости солвера (по private tests)

### 1. Fragility = 0.0 в 2 сценариях (private_fragile_dominant, private_wine_eggs_dilemma)

- Когда большинство SKU fragile — солвер не может избежать стэкинга fragile-на-fragile
- Wine+eggs дилемма: оба fragile, оба >2kg, стэкинг в любом порядке = violation

### 2. Low coverage при weight constraints (private_weight_razor: 36/90, private_full_catalog: 40/175)

- При жёстком лимите веса солвер не оптимизирует выбор "какие items пропустить"

### 3. Odd pallet gaps (private_odd_pallet_stress: volume=0.52)

- US-паллета 1219x1016 плохо сочетается с EUR-sized коробками — 19mm/216mm остатки

### 4. Upright overflow (private_upright_overflow: 32/92)

- При низком потолке + все upright — вертикальное пространство исчерпывается быстро
