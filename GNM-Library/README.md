# GNM-Lib - Genesis Neuron Model Library

Библиотека цифровых блюпринтов нейронов для движка **Genesis**.
Каждый `.toml` файл описывает один биологически аппроксимированный тип нейрона - мембранные свойства, морфологию роста, параметры пластичности - в целочисленной физике Genesis.

---

## Содержание

1. [Структура директории](#структура-директории)
2. [Формат TOML файла](#формат-toml-файла)
3. [Справочник параметров](#справочник-параметров)
4. [Формулы конвертации](#формулы-конвертации)
5. [Источники данных](#источники-данных)
6. [Pipeline генерации](#pipeline-генерации)
7. [Credits & Attribution](#credits--attribution)
8. [Инварианты](#инварианты)

---

## Структура директории

```
GNM-Library/
├── Cortex/                     # ~1780 типов от Allen Cell Types API
│   ├── L1/                     # Cortical layer 1
│   │   ├── aspiny/             # Тормозные интернейроны
│   │   ├── sparsely spiny/     # Переходный тип
│   │   └── spiny/              # Возбуждающие пирамиды
│   │       ├── VISp5/          # Зона → вложенные варианты
│   │       │   ├── 1.toml
│   │       │   └── 2.toml
│   │       └── SSp-un5.toml    # Единственный вариант → плоский файл
│   ├── L2/ ... L6b/            # Слои L2, L23, L3, L4, L5, L6, L6a, L6b
│
├── Cerebellum/                 # Мозжечок (hardcoded из литературы)
│   ├── Mouse/
│   ├── Rat/
│   └── Zebrafish/
│
├── Hippocampus/                # Гиппокамп
│   ├── Mouse/
│   └── Rat/
│
├── Striatum/                   # Стриатум
│   └── Mouse/ Rat/
│
└── Thalamus/                   # Таламус
    └── Mouse/ Rat/
```

### Соглашение об именовании

**Cortex:** `L{layer}_{dendrite_type}_{brain_area}[_{index}]`
- Пример: `L5_spiny_VISp5_42` - 42-й вариант spiny-нейрона зоны VISp5 в слое L5

**Подкорковые:** `{Region}_{CellType}`
- Пример: `Thalamus_TC`, `Hippocampus_PV_Basket`

---

## Формат TOML файла

```toml
name = "L5_spiny_SSp-un5"
is_inhibitory = false

# Мембрана
threshold = 35718                           # i32, µV (GNM)
rest_potential = 11607                      # i32, µV (GNM)
leak_rate = 204                             # i32, dV/tick
refractory_period = 9                       # u8, ticks
spontaneous_firing_period_ticks = 854       # u32, ticks (0 = off)

# Адаптация и Тайминги
homeostasis_penalty = 1000                  # i32, µV penalty per spike
homeostasis_decay = 2                       # u16, decay rate
synapse_refractory_period = 15              # u8, ticks
signal_propagation_length = 20              # u8, segments

# Рост и Морфология
steering_fov_deg = 34.4                     # f32, degrees
steering_radius_um = 66.7                   # f32, µm
growth_vertical_bias = 0.95                 # f32, 0.0–1.0
dendrite_radius_um = 420.0                  # f32, µm
type_affinity = 0.8                         # f32, 0.0–1.0
sprouting_weight_distance = 0.5             # f32, weight
sprouting_weight_power = 0.4                # f32, weight
sprouting_weight_explore = 0.1              # f32, weight
sprouting_weight_type = 0.2                 # f32, weight
steering_weight_inertia = 0.5               # f32, weight
steering_weight_sensor = 0.4                # f32, weight
steering_weight_jitter = 0.1                # f32, weight

# Пластичность GSOP
initial_synapse_weight = 168                # u16, absolute
gsop_potentiation = 108                     # u16
gsop_depression = 69                        # u16
inertia_curve = [128, 121, 115, ...]        # [u8; 16], fixed-point (128 = 1.0×)
prune_threshold = 5                         # i16, min weight to survive
```

---

## Справочник параметров

### Мембрана

| Параметр | Тип | Единицы | Диапазон | Описание |
|----------|-----|---------|----------|----------|
| `threshold` | i32 | µV (GNM) | 0–65 000 | Порог возбуждения. Когда потенциал ≥ threshold - нейрон генерирует спайк. |
| `rest_potential` | i32 | µV (GNM) | 0–30 000 | Потенциал покоя. Мембрана стремится к этому значению между спайками. |
| `leak_rate` | i32 | dV/tick | 10–12 000 | Скорость утечки мембраны. Выше = быстрее возврат к rest_potential. |
| `refractory_period` | u8 | ticks | 5–25 | Абсолютный рефрактерный период. Нейрон не может спайкнуть в течение этого времени. |
| `spontaneous_firing_period_ticks` | u32 | ticks | 0–100 000 | Период спонтанной активности (heartbeat). 0 = отключён. |

### Адаптация и Тайминги

| Параметр | Тип | Единицы | Диапазон | Описание |
|----------|-----|---------|----------|----------|
| `homeostasis_penalty` | i32 | µV | 1 000–20 000 | Повышение порога после каждого спайка (spike-frequency adaptation). |
| `homeostasis_decay` | u16 | rate | 2–80 | Скорость сброса адаптивного порога обратно к baseline. |
| `synapse_refractory_period` | u8 | ticks | 15 | Рефрактерный период синапса между повторными активациями. |
| `signal_propagation_length` | u8 | segments | 3–100 | Длина Active Tail сигнала по аксону (1 сегмент ≈ 20 µm). |

### Рост и Морфология (Baker Pipeline)

| Параметр | Тип | Единицы | Диапазон | Описание |
|----------|-----|---------|----------|----------|
| `steering_fov_deg` | f32 | degrees | 20–180 | Угол обзора конуса при Cone Tracing (поиск целей аксоном). |
| `steering_radius_um` | f32 | µm | 20–500 | Радиус шага роста аксона. |
| `growth_vertical_bias` | f32 | - | 0.0–0.95 | Вертикальный bias роста (1.0 = строго вверх/вниз по слоям). |
| `dendrite_radius_um` | f32 | µm | 15–500 | Радиус дендритного поля вокруг сомы. |
| `type_affinity` | f32 | - | 0.0–1.0 | Притяжение к сомам своего vs чужого типа. spiny=0.8, aspiny=0.2. |
| `sprouting_weight_distance` | f32 | - | 0.0–1.0 | Вес расстояния в Sprouting Score. |
| `sprouting_weight_power` | f32 | - | 0.0–1.0 | Вес мощности сигнала в Sprouting Score. |
| `sprouting_weight_explore` | f32 | - | 0.0–0.5 | Вес исследования (exploration) в Sprouting Score. |
| `sprouting_weight_type` | f32 | - | 0.0–1.0 | Вес типового соответствия в Sprouting Score. |
| `steering_weight_inertia` | f32 | - | 0.0–1.0 | Вес инерции направления при росте аксона. |
| `steering_weight_sensor` | f32 | - | 0.0–1.0 | Вес сенсорного сигнала (притяжение к целям). |
| `steering_weight_jitter` | f32 | - | 0.0–0.5 | Вес случайного шума при росте. |

### Пластичность (GSOP - Genesis Synaptic Ordering Protocol)

| Параметр | Тип | Единицы | Диапазон | Описание |
|----------|-----|---------|----------|----------|
| `initial_synapse_weight` | u16 | - | 50–32 767 | Начальный вес нового синапса. ~1/50 от (threshold − rest). |
| `gsop_potentiation` | u16 | - | 2–32 767 | Скорость потенциации (LTP). Больше = быстрее усиление. |
| `gsop_depression` | u16 | - | 2–32 767 | Скорость депрессии (LTD). Больше = быстрее ослабление. |
| `inertia_curve` | [u8; 16] | fixed-point | 2–255 | Сопротивление изменению веса по рангу. 128 = 1.0× (без модификации). |
| `prune_threshold` | i16 | - | 5–8 000 | Минимальный вес для выживания синапса. Ниже = прунинг. |
| `is_inhibitory` | bool | - | - | true = тормозной (ГАМК), false = возбуждающий (глутамат). |

---

## Формулы конвертации

### Система координат GNM

```
Биологический ноль:  -82 mV = 0 µV (GNM)
Масштаб:             1 mV  = 1000 µV (GNM)
Временной тик:       0.1 ms = 1 tick
Тиков в секунду:     10 000
```

### Мембранные параметры

| Параметр | Формула | Источник Allen |
|----------|---------|----------------|
| `threshold` | `(ef__fast_trough_v_long_square − (−82)) × 1000` | `ef__fast_trough_v_long_square` |
| `rest_potential` | `(ef__vrest − (−82)) × 1000` | `ef__vrest` |
| `leak_rate` | `ef__tau × 10` | `ef__tau` (мс) |
| `refractory_period` | `29 − 4 × ef__upstroke_downstroke_ratio` , clamp [5, 25] | `ef__upstroke_downstroke_ratio_long_square` |
| `spontaneous_firing_period_ticks` | **Приоритет 1:** `10000 / ef__avg_firing_rate` | `ef__avg_firing_rate` (Hz) |
| | **Приоритет 2:** `10000 / ((0.5 × Rin) / ΔV)`, ΔV = threshold − rest | `ef__ri` (MΩ) |

### Адаптация

| Параметр | Формула | Источник Allen |
|----------|---------|----------------|
| `homeostasis_penalty` | `max(1000, ef__adaptation × 20000)` | `ef__adaptation` |
| `homeostasis_decay` | `250 / ef__avg_isi`, clamp [2, 80] | `ef__avg_isi` (мс) |
| `signal_propagation_length` | `nr__max_euclidean_distance / 20`, clamp [3, 100] | `nr__max_euclidean_distance` (µm) |

### Морфология роста

| Параметр | Формула | Примечание |
|----------|---------|------------|
| `steering_fov_deg` | aspiny: 150°, sparsely spiny: 75°, spiny: 40° - коррекция: `−min(20, axon_length/100)` | Интернейроны = широкий поиск, пирамиды = узкий луч |
| `steering_radius_um` | `axon_total_length_um × 0.12`, clamp [20, 500] | 12% от длины аксона из SWC |
| `growth_vertical_bias` | **Exc:** `|soma_z − 0.05| × 2000 / axon_length`, clamp [0.3, 0.95] | Пирамиды L5 тянутся к L1 |
| | **Inh:** `0.1 + soma_z × 0.2`, max 0.4 | Интернейроны ветвятся локально |
| `type_affinity` | spiny=0.8, aspiny=0.2, sparsely spiny=0.5 | Фиксированное правило |
| `sprouting_weight_explore` | `0.05 + nr__number_bifurcations / 300`, clamp [0.05, 0.5] | Больше бифуркаций = больше exploration |

### Steering weights (по длине аксона)

| Длина аксона | Тип | distance | power | type | inertia | sensor | jitter |
|------------|------|----------|-------|------|---------|--------|--------|
| < 300 µm | Local interneuron | 0.7 | 0.3 | 0.3 | 0.2 | 0.6 | 0.2 |
| 300–800 µm | Medium range | 0.5 | 0.4 | 0.2 | 0.5 | 0.4 | 0.1 |
| > 800 µm | Long-range projector | 0.3 | 0.5 | 0.1 | 0.8 | 0.1 | 0.1 |

### Пластичность GSOP

| Параметр | Формула | Примечание |
|----------|---------|------------|
| `initial_synapse_weight` | `(threshold − rest) / 50` | PV/aspiny inh: ×3 (быстрый отклик) |
| `gsop_potentiation` | **Exc:** `100 + adaptation × 400` | Высокая адаптация = высокая пластичность |
| | **PV inh:** `200 + adaptation × 100` (ratio 100:1) | PV держат связи стабильно |
| `gsop_depression` | **Exc:** `potentiation / (1.5 + adaptation × 2)` | |
| | **PV inh:** `potentiation / 100` | |
| `inertia_curve` | `128 × exp(−steepness × rank × 3.5 / 15)` | steepness = penalty_norm × 0.6 + adaptation × 0.4 |
| `prune_threshold` | `initial_weight − depression × 30`, clamp [5, initial_weight / 4] | Синапс должен пережить 30 тиков депрессии |

### Dead Zone Guard (Patch)

Условие живого обучения на каждом ранге:
```
(gsop_potentiation × inertia[rank]) >> 7 ≥ 1
```
Если нарушено - `inertia[rank]` поднимается до `⌈128 / gsop_potentiation⌉`.

---

## Источники данных

### Cortex (~1780 нейронов)

| Источник | Что даёт | Endpoint / Файл |
|----------|----------|------------------|
| **Allen Cell Types API** | Электрофизиология: Vrest, τ, Rin, firing rate, adaptation, upstroke/downstroke ratio, dendrite type, cortical layer, brain area | `https://api.brain-map.org/api/v2/data/query.json` model: `ApiCellTypesSpecimenDetail` |
| **Allen SWC morphometry** | Морфология аксона и дендритов: `axon_total_length_um`, `dendrite_max_radius_um` | SWC файлы через `well_known_file_download/{nrwkf_id}` |

### Подкорковые структуры (hardcoded)

Параметры заданы вручную по литературным данным:

| Регион | Типы | Источники |
|--------|------|-----------|
| **Thalamus** | TC, TRN | Sherman & Guillery 2006 |
| **Hippocampus** | CA1 Pyramidal, CA3 Pyramidal, Dentate Granule, PV Basket | Spruston 2008, Klausberger & Somogyi 2008 |
| **Striatum** | MSN | Kreitzer & Malenka 2008 |
| **Cerebellum** | Purkinje, Granule | Häusser & Clark 1997 |

---

## Pipeline генерации

Библиотека была сгенерирована собственным алгоритмом:

```
1. download_neuro_data.py      Скачивает raw_data/ из Allen API + SWC
         ↓
2. generate_gnm_library.py     Allen JSON → ~1780 индивидуальных .toml (Cortex)
   generate_subcortical_library.py  Hardcoded → ~12 .toml (подкорка)
         ↓
3. patch_gnm_library.py        Post-hoc исправления:
                                 - signal_length ≥ refractory + 1
                                 - Dead Zone guard для inertia × potentiation
                                 - Масштабирование initial_weight от delta
                                 - Синхронизация prune_threshold
```

---

## Credits & Attribution

### Allen Institute for Brain Science

Cortical neuron electrophysiology and morphometry data derived from the
**Allen Cell Types Database**.

> © 2015 Allen Institute for Brain Science. Allen Cell Types Database.
> Available from: [celltypes.brain-map.org](https://celltypes.brain-map.org)
>
> Citation: *Allen Institute for Brain Science (2015). Allen Cell Types Database.
> Available from celltypes.brain-map.org*

The Allen Institute Terms of Use permit use of their data in research and derivative works
with attribution. GNM-Library `.toml` files are derivative works - biological parameters
(mV, ms, Hz, µm) were transformed through custom conversion formulas into Genesis
integer physics units. No raw Allen data is redistributed.

### Literature Sources (Subcortical)

Subcortical neuron parameters are based on published electrophysiological measurements:

- Sherman, S. M. & Guillery, R. W. (2006). *Exploring the Thalamus and Its Role in Cortical Function.* MIT Press.
- Spruston, N. (2008). Pyramidal neurons: dendritic structure and synaptic integration. *Nature Reviews Neuroscience*, 9(3), 206–221.
- Klausberger, T. & Somogyi, P. (2008). Neuronal diversity and temporal dynamics. *Science*, 321(5885), 53–57.
- Kreitzer, A. C. & Malenka, R. C. (2008). Striatal plasticity and basal ganglia circuit function. *Neuron*, 60(4), 543–554.
- Häusser, M. & Clark, B. A. (1997). Tonic synaptic inhibition modulates neuronal output pattern. *Neuron*, 19(3), 665–678.

### License

GNM-Library is part of the Genesis project and is distributed under **GPLv3**
(see root LICENSE). The conversion formulas and resulting neuron blueprints are
original work; source data is used under the respective providers' terms of use.

---

## Инварианты

- `inertia_curve` - ровно 16 элементов `[u8; 16]`, 128 = 1.0×
- `is_inhibitory = true` → вес синапса интерпретируется как отрицательный
- `signal_propagation_length ≥ refractory_period + 1` (иначе сигнал не успевает пройти)
- `Σ sprouting_weight_* ≈ 1.0`
- Dead Zone: `(gsop_potentiation × inertia[any_rank]) >> 7 ≥ 1`
