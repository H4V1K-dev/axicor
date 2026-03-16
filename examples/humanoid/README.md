# 🤖 Genesis HFT: Humanoid-v4 Example

<h1 style="color: red;"> ВРЕМЕННО НЕ РАБОТАЕТ </h1>
<h3 style="color: red;"> Но можно запускать и експериментировать </h3>



Высокопроизводительный Embodied AI агент для сложной бипедальной локомоции (Humanoid-v4). Архитектура использует 5-зонный нейро-стек для разделения функций баланса, ритма и сенсорной интеграции.

## 🚀 Как запустить (Zero-Magic Pipeline)

**Шаг 0. Активируйте виртуальное окружение**
```bash
source .venv/bin/activate
```

**Шаг 1. Сгенерируйте и запеките мозг (WTA Architecture)**
Скрипт создаст 5-зонную топологию HumanoidAgent и вызовет Rust-компилятор (`genesis-baker`) для нарезки VRAM-дампов.
```bash
python3 examples/humanoid/build_brain.py
```

**Шаг 2. Запустите HFT-реактор на GPU (Dual-Backend)**
Оркестратор мгновенно загрузит VRAM-дампы (Zero-Copy) и перейдет в режим ожидания тиков от агента.

# Для NVIDIA (CUDA)
```bash
cargo run --release -p genesis-node -- --brain HumanoidAgent --log
```

# Для AMD (ROCm / HIP)
```bash
cargo run --release -p genesis-node --features amd -- --brain HumanoidAgent --log
```

**Шаг 3. Подключите bipedal-агента (DOD Hot Loop)**
Запустите Python-шлюз. Он работает в бесконечном цикле без аллокаций, управляя обучением через дофаминовую модуляцию и Auto-Tuner.

```bash
python3 examples/humanoid/humanoid_agent.py
```

Это самый сложный пример в Genesis. Если агент стабилизирует походку, это будет абсолютный разьеб — 17 степеней свободы на Spiking Neural Networks без классического обучения с учителем!
