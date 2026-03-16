# Brain Builder: HFT-Коннектом и Offline-Компиляция

В Axicor вы не создаете нейроны в оперативной памяти Питона через `new Neuron()`. Это убьет сборщик мусора (GC) и сделает невозможным Zero-Copy загрузку на GPU. 

Вместо этого мы используем паттерн **"ДНК Мозга"**. С помощью Python-класса `BrainBuilder` вы генерируете строгую топологию в виде TOML-конфигов. Затем утилита `genesis-baker` "запекает" эту ДНК в плоские бинарные C-ABI массивы (`.state` и `.axons`). Это основа движка Axicor.

## 1. Жизненный цикл (Pipeline Clarity)

1. **Python Script (`builder.build()`):** Генерирует иерархию папок и TOML-файлов (`anatomy.toml`, `blueprints.toml`, `io.toml`, `shard.toml`).
2. **Genesis Baker (CPU Compiler):** `cargo run -p genesis-baker -- --brain config/brain.toml`. Читает TOML, расставляет нейроны, "проращивает" аксоны через 3D-пространство (Cone Tracing) и сохраняет выровненные по варпам VRAM-дампы.
3. **Genesis Node (GPU Runtime):** `cargo run -p genesis-node`. Делает `mmap` запеченных дампов в память ОС напрямую, минуя аллокации.

---

## 2. Создание HFT-Коннектома (Эталон)

При работе в высокочастотном цикле (100+ Гц) со средами вроде Gymnasium, базовая пластичность из биологической библиотеки убьет веса. Мы обязаны глушить фоновый рост и переводить сеть в режим **Reward-Gated Plasticity** (рост только при дофамине).

```python
from genesis.builder import BrainBuilder

# 1. Инициализация архитектора
builder = BrainBuilder(project_name="HftAgent", output_dir="Genesis-Models/")

# Тонкая настройка физики под HFT-цикл (1 ms барьер)
builder.sim_params["sync_batch_ticks"] = 10
builder.sim_params["tick_duration_us"] = 100

# 2. Создание зоны (Ширина, Глубина, Высота в вокселях)
# Размер вокселя по умолчанию = 25 мкм
cortex = builder.add_zone("SensoryCortex", width_vox=64, depth_vox=64, height_vox=63)

# 3. Загрузка типов и HFT-тюнинг пластичности
# В HFT-режиме рост весов происходит ТОЛЬКО при вливании дофамина (pot=0)
# Фоновая депрессия (dep=2) медленно выжигает случайный шум. Это важно для Axicor.
exc_type = builder.gnm_lib("VISp4/141").set_plasticity(pot=0, dep=2)
inh_type = builder.gnm_lib("VISp4/114").set_plasticity(pot=0, dep=2)

# 4. Наполнение слоями (Bottom-Up дизайн)
# ВНИМАНИЕ: height_pct слоев в сумме должны давать строго 1.0!
cortex.add_layer("L4_Input", height_pct=0.1, density=0.3)\
      .add_population(exc_type, fraction=0.8)\
      .add_population(inh_type, fraction=0.2)

cortex.add_layer("L23_Hidden", height_pct=0.6, density=0.15)\
      .add_population(exc_type, fraction=0.8)\
      .add_population(inh_type, fraction=0.2)

cortex.add_layer("L5_Motor", height_pct=0.3, density=0.1)\
      .add_population(exc_type, fraction=1.0)

# 5. Определение Входов и Выходов (I/O Matrix)
cortex.add_input("sensors", width=8, height=8, entry_z="top")
cortex.add_output("motors", width=16, height=8, target_type="All")

# 6. Компиляция ДНК
builder.build()
```

## 3. Как это работает под капотом (DOD)

1. **Никакого RNG для популяций (Hard Quotas):** Метод `add_population(fraction=0.8)` устанавливает жесткую квоту. Если в слое помещается 1000 нейронов по объему и density, движок гарантированно создаст ровно 800 нейронов одного типа и 200 другого. Типы перемешиваются детерминированным шаффлом (master_seed).
2. **Матричный маппинг:** Входы и выходы (`add_input`, `add_output`) растягиваются на 3D-сетку зоны (UV-проекция). Один пиксель матрицы охватывает популяцию физических нейронов. Выбор конкретной сомы для I/O детерминирован алгоритмом Spatial Hashing.
3. **C-ABI выравнивание:** Baker автоматически вычисляет `padded_n` и добивает количество нейронов нулями до числа, кратного 32 (Warp Alignment). GPU будет читать эту память без Divergence и Cache Misses.
