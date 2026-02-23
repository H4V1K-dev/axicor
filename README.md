# Genesis

Движок для воплощённого интеллекта. Пластичная нейросеть с растущими аксонами и двухфазным циклом жизни. Собственный закон (GSOP), целочисленная физика, Rust + CUDA.

---

## Что это

Genesis — реализация биологически-вдохновленной нейросети с собственным законом пластичности, целочисленной физикой сигналов и двухфазным циклом жизни (Day/Night). Полность детерминирована: воспроизводимый результат на любом железе.

Цель: воплощённый AGI в руках каждого — от одного GPU до кластера. Масштаб зависит только от ваших ресурсов.

---

## Ключевые решения

| Решение | Почему |
|---|---|
| **Integer Physics** | Детерминизм, скорость, воспроизводимость на любом железе |
| **GSOP вместо STDP** | Пространственное перекрытие Active Tail, без хранения истории спайков |
| **Day/Night Cycle** | GPU — только физика. CPU — структурная пластичность. Без конфликтов |
| **Columnar SoA Layout** | 100% Coalesced Access на варп. Нет AoS, нет кэш-промахов |
| **Population Coding** | Сила = количество активных нейронов, не частота. Мгновенный отклик |
| **Pub/Sub Connectivity** | Аксон вещает. Дендрит слушает. Нет списков подписчиков на аксоне |
| **Strict BSP** | Детерминированная синхронизация шардов через барьеры |
| **Planar Sharding** | Шардирование по XY-плоскостям — топология сохраняет физическую геометрию мозга |
| **Cone Tracing** | Аксоны растут по вектору с FOV. Рождение связей через пространственный поиск |
| **Ghost Axons** | Виртуальные копии на границах шардов — межшардовая пластичность без разрывов |

---

## Архитектура

Спецификация разбита на 7 документов в [`docs/specs/`](./docs/specs/):

| Файл | Содержание |
|---|---|
| [`01_foundations.md`](./docs/specs/01_foundations.md) | Физические законы, детерминизм, тороидальная топология |
| [`02_configuration.md`](./docs/specs/02_configuration.md) | TOML-конфиги, Baking pipeline, SoA layout |
| [`03_neuron_model.md`](./docs/specs/03_neuron_model.md) | GLIF-модель, гомеостаз, 4-битная маска типа |
| [`04_connectivity.md`](./docs/specs/04_connectivity.md) | Cone Tracing, GSOP, Inertia Curves, Ghost Axons |
| [`05_signal_physics.md`](./docs/specs/05_signal_physics.md) | CUDA-ядра: PropagateAxons, UpdateNeurons, ApplyGSOP |
| [`06_distributed.md`](./docs/specs/06_distributed.md) | Planar Sharding, BSP, Ring Buffer, Ping-Pong |
| [`07_gpu_runtime.md`](./docs/specs/07_gpu_runtime.md) | VRAM layout, Day/Night Cycle, Lifecycle Invariants |

---

## Статус

**Pre-alpha. Активная разработка.**

| Компонент | Статус | Описание |
|---|---|---|
| Спецификация | ✅ Готово | 7 документов, ~3000 строк. Вся архитектура |
| `genesis-core` | 🔨 В работе | Общие типы, константы, SoA layout |
| `genesis-baker` | 🔨 В работе | TOML → `.state` / `.axons` / `.positions` |
| `genesis-runtime` | 🔨 В работе | Orchestrator, Day/Night Phase, BSP, UDP Router, WebSocket telemetry |
| `genesis-ide` | 🔨 В работе | Bevy-based 3D визуализатор с live telemetry |

### genesis-baker

Компилятор конфигов. Принимает 4 TOML-файла, выдаёт бинарные блобы готовые к загрузке на GPU:

```
baker compile \
  --simulation simulation.toml \
  --blueprints blueprints.toml \
  --anatomy     anatomy.toml \
  --io          io.toml \
  --output      baked/
```

Пайплайн: парсинг → валидация инвариантов → размещение нейронов → Cone Tracing (рост аксонов) → Atlas Routing (белое вещество) → подключение дендритов → запись `.state` / `.axons` / `.positions`.

Запись атомарная: `.tmp` → `rename`.

### genesis-runtime

Daemon одного шарда распределённой сети. Запускается как:

```
genesis-node --config shard_04.toml --port 8000
```

Реализовано:
- Загрузка `.state` / `.axons` напрямую в VRAM (zero-copy)
- UDP Fast Path (BSP-барьер для межшардовой синхронизации)
- TCP Geometry Server (slow path, структурная пластичность)
- WebSocket Telemetry Server (порт + 2) — live поток спайков
- Эфемерный цикл: Day Phase (GPU batch) → Night Phase (CPU plasticity)

### genesis-ide

3D-визуализатор на [Bevy](https://bevyengine.org/). Читает `.positions` из baker, подключается к runtime по WebSocket и показывает активность сети в реальном времени.

Плагины:
- `LoaderPlugin` — загрузка `.positions`, переход в состояние `Running`
- `WorldPlugin` — 3D точки-нейроны, цветовая кодировка по типу
- `CameraPlugin` — орбитальная камера + Fly-режим, управление мышью и клавиатурой
- `HudPlugin` — оверлей: FPS, кол-во нейронов, аксонов, выбранный нейрон
- `TelemetryPlugin` — WebSocket-клиент, подсветка активных нейронов по спайкам

---

## Быстрый старт

```bash
# 1. Скомпилировать конфиги в бинарные блобы
cargo run -p genesis-baker -- compile --output baked/

# 2. Запустить runtime-ноду
cargo run -p genesis-runtime -- --config config/shard_00.toml --port 8000

# 3. Открыть визуализатор
cargo run -p genesis-ide
```

---

## Стек

- **Rust** — весь движок (baker, runtime, IDE)
- **CUDA** — GPU-ядра (Day Phase, в разработке via FFI)
- **Bevy** — 3D visualization (genesis-ide)
- **Tokio** — async runtime, WebSocket, UDP

---

## Лицензия

GPLv3 + коммерческое лицензирование. Подробности в [LICENSE](./LICENSE).

Copyright (C) 2026  Oleksandr Arzamazov