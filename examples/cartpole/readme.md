# CartPole — Genesis Example (3-Layer Cortical Hierarchy)

> **Статус:** 🔬 В исследовании — активная настройка параметров нейрогенеза
> ИНСТРУКЦИЯ НЕ АКТУАЛЬНА
> **Текущий рекорд:** 🏆 100+ баллов (@H4V1K-dev / Antigravity)
>
> **Архитектура:** `SensoryCortex (L4) → HiddenCortex (L2/3) → MotorCortex (L5)` — 450K+ нейронов

Демонстрирует трёхслойную кортикальную иерархию с GSOP-пластичностью на задаче балансировки CartPole.
Текущая открытая проблема: тормозные (Inh) нейроны в ряде конфигураций побеждают в конкуренции с возбуждающими — баланс Exc/Inh требует донастройки. Именно это сейчас исследуется.

---

## 🚀 Быстрый старт

### 1. Подготовка окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib gymnasium pygame
```

### 2. Запекание сети (Baking)

Останавливает старые процессы, очищает SHM и пересобирает топологию:

```bash
pkill -f genesis-node 2>/dev/null; pkill -f genesis-baker-daemon 2>/dev/null
rm -rf baked/ /dev/shm/genesis_shard_*
cargo run --release -p genesis-baker --bin baker -- \
  --brain examples/cartpole/config/brain.toml
```

### 3. Запуск Симуляции (3 шарда — IntraNode)
## ⚠️ Если у вас слабый ПК или давно не меняли термопасту, термопрокладки, то лучше перенастроить симуляцию на меньшее количество нейронов и выбрать `--cpu-profile balanced` (средний) или `eco` (экономный), т.к. в режиме `aggressive` видеокарта и процессор будут греться как в майнинге буквально.
```bash
cargo run --release -p genesis-node -- \
  --manifest baked/SensoryCortex/manifest.toml \
  --manifest baked/HiddenCortex/manifest.toml \
  --manifest baked/MotorCortex/manifest.toml \
  --batch-size 100 \
  --cpu-profile aggressive
```

### 4. Запуск клиента и мониторинга

В отдельных терминалах:

```bash
# Обучение
python3 examples/cartpole/cartpole_client.py

# Live-мониторинг: SMA-25/100/300, TPS, скроллинг
python3 scripts/live_dashboard.py
```

---

## ⚠️ Первый запуск — ожидаемое поведение

После старта TPS будет очень низким — это нормально. Сеть обучается.
> [АКТУАЛЬНО] Сейчас поломана мультипоточность, в следующем коммите будет исправлено.
```
⚠️ [Genesis] RX Timeout — waiting for node...
```

Эти сообщения в клиенте — норма на старте. Сеть ещё не оптимизирована.

| Этап | ~Такты | TPS |
|---|---|---|
| Старт | 0 – 100K | 1× (базовый) |
| Начало обучения | 100K – 500K | 5–15× |
| Зрелая сеть | 3–4M | до 45× |

> 100 000 тактов на зрелой сети пролетают за 20–25 секунд.

---

## 🧠 Главная задача: баланс Exc/Inh

**Цель:** соотношение 4:1 (возбуждающие : тормозные) во время обучения.

Главный ориентир — **SMA-300**: должен монотонно расти до 100+. Если падает ниже 100, сеть деградирует (чаще всего — доминирование Inh).

**Ключевые параметры в `blueprints.toml`:**

| Параметр | Что регулирует |
|---|---|
| `initial_synapse_weight` | Стартовая сила нового синапса (Sprouting). Увеличьте для Exc-типов. |
| `gsop_potentiation` | Скорость обучения (LTP). Увеличьте для Exc при закреплении дофаминовых паттернов. |
| `save_checkpoints_interval_ticks` | Частота автосохранения. В начале — меньше, на зрелой сети — больше (щадим SSD). |

**Диагностика весов:**

```bash
python3 scripts/weight_checker.py
```

Читает `checkpoint.state` и показывает статистику отклонения от целевого Exc/Inh баланса.

---

## 🧠 Визуализация морфологии нейронов

Чтобы увидеть 3D-структуру отдельного нейрона (ствол аксона + дендритные отростки):

```bash
# Визуализация нейрона 3D (ID нейрона) в SensoryCortex
python3 scripts/visualize_neuron.py --MotorCortex --id 1488

# Или с сохранением в файл PNG
python3 scripts/visualize_neuron.py --MotorCortex --id 1488 --save
```

---

## 🧠 Визуализация межнейронных связей (Ghost Network)

Чтобы увидеть 3D-структуру межнейронных связей (ghost connections) между кортикальными слоями:
- Работает так себе. Непонятно, как правильно настроить, чтобы было красиво.
- Слишком много обьектов.
```bash
# Показать интерактивное окно с 3D-визуализацией
python3 scripts/visualize_ghosts.py --show

# Или сохранить в файл PNG
python3 scripts/visualize_ghosts.py --save
```

---

## ♻️ Hot-Reload (без перезапуска)

После Baking в `baked/` появляются `manifest.toml` для каждого шарда. Их можно редактировать **на ходу** — подхватывается за ~0.5 с (иногда дольше, разбираюсь с этим сейчас).

**Безопасно менять:**
- `[settings] night_interval_ticks` — частота нейрогенеза
- `[settings] save_checkpoints_interval_ticks` — частота чекпоинтов
- `[settings.plasticity] prune_threshold` — порог удаления слабых связей
- `[[variants]]` — любые параметры физики нейронов

> [!WARNING]
> `ghost_capacity`, `padded_n`, порты и пути требуют **перезапуска** — они определяют структуру VRAM.

---

## 📊 Аналитика и визуализация

```bash
# Внутренние веса (Green/Red Map)
python3 scripts/visualize_internal_weights.py baked/HiddenCortex/checkpoint.state

# Межзональные связи (3D Ghosts)
python3 scripts/visualize_ghosts.py

# Детальный дебаг нейронов
python3 scripts/brain_debugger.py baked/MotorCortex/checkpoint.state
```

---

## 🏆 Hall of Fame

| Участник | Рекорд | Дата | Изменения |
| :--- | :---: | :---: | :--- |
| **@H4V1K-dev** | **100+** | 09.03.2026 | 3-слойная иерархия, GNM L2/3, Persistence Logic |
| **@shuanat** | **71** | 07.03.2026 | Добавил `random` при равенстве (старая 2-слойка) |
