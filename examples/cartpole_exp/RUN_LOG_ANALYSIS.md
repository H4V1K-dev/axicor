# Полный анализ run_log.json

**Прогон:** 10 000 эпизодов, seed 12345, ~69 мин (4165 сек)

---

## 1. Сводка

| Метрика               | Значение                          |
| --------------------- | --------------------------------- |
| Episodes              | 10 000                            |
| Total steps           | 2 598 978                         |
| Mean score            | **259.9**                         |
| Max score             | **1167**                          |
| episodes_to_threshold | **null** (порог 700 не достигнут) |
| Ticks/sec             | ~12 478                           |

---

## 2. Фазы и переходы

### 2.1. Хронология фаз

| Фаза         | Эпизоды   | Длительность |
| ------------ | --------- | ------------ |
| EXPLORATION  | 0–2417    | 2418 эп.     |
| DISTILLATION | 2418–2982 | **565 эп.**  |
| EXPLORATION  | 2983–9409 | 6427 эп.     |
| DISTILLATION | 9410–9443 | **34 эп.**   |
| EXPLORATION  | 9444–9999 | 556 эп.      |

### 2.2. Переходы

| Ep   | Переход                               | Score | SMA        |
| ---- | ------------------------------------- | ----- | ---------- |
| 2418 | EXPLORATION → DISTILLATION            | 474   | 366 (≥350) |
| 2983 | DISTILLATION → EXPLORATION (rollback) | 144   | 207 (<210) |
| 9410 | EXPLORATION → DISTILLATION            | 569   | 363        |
| 9444 | DISTILLATION → EXPLORATION (rollback) | 165   | 209        |

**Вывод:** Дважды достигнут порог входа в DISTILLATION (SMA ≥ 350), оба раза произошёл откат из‑за падения SMA ниже 210.

---

## 3. Распределение score

| Диапазон | Кол-во | %         |
| -------- | ------ | --------- |
| 100–200  | 2 531  | 25.3%     |
| 200–300  | 5 171  | **51.7%** |
| 300–400  | 1 569  | 15.7%     |
| 400–500  | 461    | 4.6%      |
| 500–700  | 229    | 2.3%      |
| 700–1000 | 35     | 0.3%      |
| 1000+    | 4      | 0.04%     |

**Мода:** 200–300 (половина эпизодов).  
**Высокие score (≥500):** 268 эпизодов, max 1167.

---

## 4. Score по фазам

| Фаза         | Эпизодов | Mean score | Max score |
| ------------ | -------- | ---------- | --------- |
| EXPLORATION  | 9 401    | 259.6      | 1167      |
| DISTILLATION | 599      | 263.8      | 1069      |

В DISTILLATION средний score чуть выше, но откаты происходят из‑за падения SMA.

---

## 5. Punishment softening

- **Episodes с punishment_modifier < 1:** 582 (5.8%)
- **По фазам:** 580 в DISTILLATION, 2 в EXPLORATION
- **Score при смягчении:** 127–1069, mean 261.4

Смягчение punishment срабатывает в основном в DISTILLATION при SMA < 350.

---

## 6. Структура сети (SHM)

| Метрика                   | Значение        | Комментарий                |
| ------------------------- | --------------- | -------------------------- |
| structure_active_synapses | **115** (const) | Не меняется за весь прогон |
| structure_avg_weight      | ~1207.6         | Почти константа            |
| structure_max_weight      | 3242            | Const                      |

**Примечание:** `baked_structure_stats` даёт 471 808 синапсов и avg_weight 8000 — это, вероятно, полный baked brain. 115 синапсов в логе могут относиться к другой зоне/матрице (например, только к выходному слою или к одному шарду).

---

## 7. Output и spike

| Метрика             | Min | Max   | Mean   |
| ------------------- | --- | ----- | ------ |
| output_spiking_mean | 0   | 4     | 0.19   |
| spike_rate          | 0   | 0.003 | 0.0012 |

`output_spiking_mean` > 0 в части эпизодов — выходной слой иногда спайкует (в отличие от out.json, где было 0).

---

## 8. Конфигурация

- **Adaptive leak:** mode=1, dopamine_gain=1000, burst_gain=24, leak_min=50, leak_max=800 (variant 2)
- **Tuner:** prune 30/100, night 10000/7500, rollback 0.3, distillation_enter 0.5

---

## 9. Выводы

### Что работает

1. **Tuner:** Переходы EXPLORATION ↔ DISTILLATION и rollback работают по SMA.
2. **Punishment softening:** 582 эпизода с modifier < 1, в основном в DISTILLATION.
3. **Score:** Mean 260, max 1167 — сеть даёт осмысленные решения.
4. **output_spiking_mean > 0** — выходной слой активен в части эпизодов.

### Проблемы

1. **Порог 700 не достигнут** — episodes_to_threshold = null.
2. **Откаты из DISTILLATION** — оба захода в DISTILLATION заканчиваются rollback при SMA < 210.
3. **Структура статична** — 115 синапсов, веса ~1207; R-STDP, по логам SHM, не меняет структуру.
4. **Короткие фазы DISTILLATION** — 565 и 34 эпизода; дистилляция не успевает стабилизироваться.

### Рекомендации

1. **Снизить порог входа в DISTILLATION** — например, 0.4 × target (280 вместо 350), чтобы чаще входить в фазу.
2. **Смягчить rollback** — порог 0.25 × target (175) вместо 210, чтобы реже откатываться.
3. **Проверить R-STDP и зону** — почему structure_active_synapses и веса не меняются.
4. **Увеличить прогон** — 10k эпизодов может быть мало для устойчивого выхода на 700+.

---

## 10. Верификация SHM и R-STDP

### Что проверено

| Компонент     | Статус | Детали                                                                                                                         |
| ------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **zone_hash** | ✓      | `fnv1a_32(b"SensoryCortex")` = 0x273FD103 — совпадает в agent, node, baker                                                     |
| **Layout**    | ✓      | GPU: `col_idx = slot*padded_n + tid`; Python: `(128, padded_n)` C-order; SHM: weights → targets → handovers → flags            |
| **Offsets**   | ✓      | memory.py читает `weights_offset`, `targets_offset`, `flags_offset` из заголовка SHM                                           |
| **Fallback**  | ✓      | `sample_runtime_metrics` использует baked_state только при `active_synapses==0 && avg_weight==0` — 115 это реальные данные SHM |

### Источники данных

- **structure_active_synapses (115)** — `GenesisMemory.get_network_stats()` → `np.sum(targets != 0)` в SHM
- **baked_structure_stats (471 808)** — `load_baked_state_stats(manifest_path)` → `shard.state` на диске

### Причина расхождения 115 vs 471 808 (исправлено)

**Баг:** Baker открывает SHM с `truncate(false)` — старые данные сохраняются. Node пишет в SHM только в Night Phase. При перезапуске agent читал stale data от предыдущего прогона.

**Фикс:** `shard_thread.rs` — Initial DMA после boot: VRAM → SHM до входа в main loop. Agent видит актуальное состояние сразу.

### Скрипт верификации

```bash
# При работающем genesis-node:
cd examples/cartpole_exp
python verify_shm.py
```

Скрипт выводит: padded_n, active_synapses из SHM и baked, layout, проверку изменения весов (R-STDP).

### Действие при расхождении

Если `verify_shm.py` показывает 115 в SHM и 471 808 в baked:

```bash
# Удалить checkpoint — node загрузит полный shard.state
del Genesis-Models\CartPole-example\baked\SensoryCortex\checkpoint.state
del Genesis-Models\CartPole-example\baked\SensoryCortex\checkpoint.axons
# Перезапустить node
```
