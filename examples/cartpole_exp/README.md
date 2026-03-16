# 🤖 Genesis HFT: CartPole Example

Полностью рабочий Embodied AI агент, который учится балансировать палку в реальном времени с использованием SNN (Spiking Neural Networks) и инъекций дофамина (R-STDP). Никакого Backpropagation.

## 🚀 Как запустить (Zero-Magic Pipeline)

**Шаг 0. Активируйте виртуальное окружение**
```bash
source .venv/bin/activate
```

**Шаг 1. Сгенерируйте и запеките мозг (One-Click Build)**
Скрипт сгенерирует TOML-топологию через Python SDK и автоматически вызовет Rust-компилятор (`genesis-baker`), чтобы нарезать бинарные VRAM-дампы.
```bash
python3 examples/cartpole/build_brain.py
```

**Шаг 2. Запустите HFT-реактор на GPU (Dual-Backend)**
Оркестратор мгновенно загрузит VRAM-дампы (Zero-Copy) и заблокируется на барьере ожидания данных от среды.

# Для NVIDIA (CUDA)
```bash
cargo run --release -p genesis-node -- --brain CartPole-example --log
```

# Для AMD (ROCm / HIP)
```bash
cargo run --release -p genesis-node --features amd -- --brain CartPole-example --log
```

**Шаг 3. Подключите среду (RL Agent)**
В новом терминале запустите Python-шлюз. Он начнет слать состояния маятника в виде битовых масок и обучать сеть инъекциями дофамина (-255) при падении и вознаграждать не линейно при успехе.

```bash
python3 examples/cartpole_exp/agent.py
```

Смотрите на логи Питона. Вы увидите, как сеть сама выжигает слабые связи и стабилизирует рефлекс.
