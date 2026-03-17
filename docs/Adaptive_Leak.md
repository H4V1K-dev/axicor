# Adaptive Leak — документация

Адаптивная модуляция мембранной утечки (leak) в Genesis. Позволяет нейрону менять временной масштаб в зависимости от дофамина и активности (burst).

> Подробный roadmap и обоснование: [LTC_Adaptive_Leak_Roadmap.md](LTC_Adaptive_Leak_Roadmap.md)

---

## Кратко

- **Без adaptive leak:** у каждого типа нейрона фиксированный `leak_rate`.
- **С adaptive leak:** эффективный leak ограничен окном `[leak_min, leak_max]` и модулируется:
  - **дофамином** — reward/punishment сигнал
  - **burst_count** — число спайков в текущем батче (0–7)

---

## Параметры

| Параметр             | Тип | Описание                                |
| -------------------- | --- | --------------------------------------- |
| `adaptive_leak_mode` | u8  | Режим: 0=выкл, 1=continuous, 2=discrete |
| `dopamine_leak_gain` | i16 | Коэффициент модуляции от дофамина       |
| `burst_leak_gain`    | i16 | Коэффициент модуляции от burst_count    |
| `leak_min`           | i16 | Нижняя граница эффективного leak        |
| `leak_max`           | i16 | Верхняя граница эффективного leak       |

**Условие включения:** `adaptive_leak_mode != 0` и `leak_min < leak_max`. Иначе используется `base_leak_rate`.

---

## Режимы

### 0 — выключен

Используется `base_leak_rate` из `VariantParameters`. Поведение как в baseline.

### 1 — continuous

Непрерывная модуляция:

```
leak_mod = (dopamine * dopamine_leak_gain) >> 7 + burst_count * burst_leak_gain
effective_leak = clamp(base_leak_rate + leak_mod, leak_min, leak_max)
```

- `dopamine` — глобальный reward (обычно -210..+10 в CartPole).
- `burst_count` — накопление спайков в батче (0–7), сбрасывается в начале батча.
- `>> 7` — фиксированный shift для масштабирования.

### 2 — discrete

Режимы мембраны: `stable`, `responsive`, `excited`, `recovery`.

| Режим        | Условие                         | effective_leak              |
| ------------ | ------------------------------- | --------------------------- |
| `stable`     | модуляция в dead band           | `base` (clamped)            |
| `responsive` | combined_mod ≤ -band            | `(base + leak_min) / 2`     |
| `excited`    | combined_mod ≥ band             | `(base + leak_max + 1) / 2` |
| `recovery`   | burst_count ≥ 4 и burst_mod > 0 | `leak_max`                  |

`band = max((leak_max - leak_min) >> 2, 1)` — dead band для stable.

---

## Конфигурация

### Blueprints (baked)

В `blueprints.toml` для каждого `[[neuron_type]]`:

```toml
adaptive_leak_mode = 0
dopamine_leak_gain = 0
burst_leak_gain = 0
leak_min = 0
leak_max = 0
```

### Runtime (Python SDK)

Через `GenesisControl` после загрузки manifest:

```python
from genesis.control import GenesisControl

control = GenesisControl(manifest_path)
control.set_adaptive_leak(
    variant_id=2,              # Motor_Pyramidal
    adaptive_leak_mode=1,
    dopamine_leak_gain=1000,
    burst_leak_gain=24,
    leak_min=50,
    leak_max=800,
)
```

### Benchmark scenarios (CartPole)

| Сценарий            | Описание                                 |
| ------------------- | ---------------------------------------- |
| `baseline`          | adaptive_leak_mode=0                     |
| `dopamine_only`     | dopamine gain=1000, burst=0, leak 50–800 |
| `burst_only`        | burst gain=24, dopamine=0                |
| `combined`          | dopamine + burst, 50–800                 |
| `discrete_combined` | discrete mode, dopamine + burst          |

---

## Пример: combined для Motor_Pyramidal

Motor_Pyramidal (variant 2) — выходной слой CartPole. `base_leak_rate=223`.

```python
AdaptiveLeakConfig(
    adaptive_leak_mode=1,
    dopamine_leak_gain=1000,
    burst_leak_gain=24,
    leak_min=50,
    leak_max=800,
    variant_ids=[2],
)
```

- **leak_min=50, leak_max=800** — окно модуляции (base 223 внутри).
- **dopamine_leak_gain=1000** — при reward=10: mod ≈ 78; при punishment=-210: mod ≈ -1640 (clamped).
- **burst_leak_gain=24** — при burst_count=7: mod = 168.

---

## Телеметрия

- `effective_leak_mean` — средний эффективный leak в зоне.
- `mode_counts` — число нейронов в каждом режиме (discrete).
- `mean_burst_count` — средний burst_count в батче.

---

## Инварианты

- Integer physics.
- Детерминизм.
- При `adaptive_leak_mode=0` поведение бит-в-бит совпадает с baseline.
- `effective_leak` всегда ≥ 1.

---

## Ссылки

- [LTC_Adaptive_Leak_Roadmap.md](LTC_Adaptive_Leak_Roadmap.md) — roadmap и обоснование
- [03_neuron_model.md](specs/03_neuron_model.md) — модель нейрона
- [05_signal_physics.md](specs/05_signal_physics.md) — физика сигналов
