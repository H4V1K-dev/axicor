# 🛒 CartPole Genesis Environment (3-Layer Hierarchy)

> **Статус:** ✅ Стабильный (GNM v2)
> **Текущий рекорд:** 🏆 100+ баллов (@H4V1K-dev / Antigravity)
> **Архитектура:** `Sensory (L4) → Hidden (L2/3) → Motor (L5)`

Система использует полноценную трехслойную кортикальную иерархию (450,000+ нейронов). Благодаря промежуточному слою `HiddenCortex` и GNM-пластичности, сеть обучается балансировать маятник, достигая **100 баллов** за ~50-100 эпизодов при TPS **3000+**.

---

## 🚀 Быстрый старт

### 1. Подготовка окружения
При первом запуске создайте виртуальное окружение:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib gymnasium pygame
```

### 2. Запекание сети (Baking)
Очистка SHM, остановка старых процессов и сборка топологии:
```bash
rm -rf baked/ && \
pkill -f genesis-node; \
pkill -f genesis-baker-daemon; \
rm -f /dev/shm/genesis_shard_* && \
cargo run --release -p genesis-baker --bin baker -- --brain examples/cartpole/config/brain.toml
```

### 3. Запуск Симуляции (3 Шарда)
Запуск всех трех слоев коры в одном процессе (IntraNode):
```bash
cargo run --release -p genesis-node -- \
  --manifest baked/SensoryCortex/manifest.toml \
  --manifest baked/HiddenCortex/manifest.toml \
  --manifest baked/MotorCortex/manifest.toml \
  --batch-size 100 \
  --cpu-profile aggressive
```

### 4. Запуск Клиента и Дашборда
В разных терминалах:
```bash
# Обучение
python3 examples/cartpole/cartpole_client.py

# Мониторинг (SMA-25/100/300, TPS, Скроллинг)
python3 scripts/live_dashboard.py
```

---

## 📊 Аналитика и Визуализация

- **Внутренние веса (Green/Red Map):**
```bash
python3 scripts/visualize_internal_weights.py baked/HiddenCortex/checkpoint.state
```
- **Межзональные связи (3D Ghosts):**
```bash
python3 scripts/visualize_ghosts.py
```
- **Дебаг нейронов:**
```bash
python3 scripts/brain_debugger.py baked/MotorCortex/checkpoint.state
```

---

## 🏆 Hall of Fame

| Участник | Рекорд | Дата | Изменения |
| :--- | :---: | :---: | :--- |
| **@H4V1K-dev** | **100+** | 09.03.2026 | **3-слойная иерархия**, GNM L2/3, Persistence Logic |
| **@shuanat** | **71** | 07.03.2026 | Добавил `random` при равенстве (старая 2-слойка) |

## Удачи в обучении! 🚀
