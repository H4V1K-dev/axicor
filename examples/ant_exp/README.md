# 🤖 Genesis HFT: Ant-v4 Example

Высокопроизводительный Embodied AI агент для среды Ant-v4, построенный на базе Spiking Neural Networks (SNN) с использованием 3-зонной архитектуры (DOD/WTA) и обучения через инъекции дофамина (R-STDP).

## 🚀 Как запустить (Zero-Magic Pipeline)

**Шаг 0. Активируйте виртуальное окружение**
```bash
source .venv/bin/activate
```

**Шаг 1. Сгенерируйте и запеките мозг (WTA Architecture)**
Скрипт создаст 3-зонную топологию (Sensory, Thoracic, Motor) с 60% плотностью тормозных нейронов в моторной коре для реализации Winner-Takes-All динамики.
```bash
python3 examples/ant/build_brain.py
```

**Шаг 2. Запустите HFT-реактор на GPU (Dual-Backend)**
Оркестратор загрузит бинарные дампы VRAM и перейдет в режим ожидания тиков от агента.

# Для NVIDIA (CUDA)
```bash
cargo run --release -p genesis-node -- --brain AntConnectome --log
```

# Для AMD (ROCm / HIP)
```bash
cargo run --release -p genesis-node --features amd -- --brain AntConnectome --log
```

**Шаг 3. Подключите среду (DOD Hot Loop)**
Запустите Python-шлюз. Он работает в бесконечном цикле без аллокаций, передавая состояния среды и управляя обучением через `TARGET_TIME` и `TARGET_SCORE`.

```bash
python3 examples/ant_exp/ant_agent.py
```

Муравей никогда не падает с нулевой эпохи.
