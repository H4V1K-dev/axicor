# 10. Hardware Backends (Roadmap & Integration)

> Расширение экосистемы [Genesis](../../README.md) за пределы NVIDIA CUDA. Стратегия портирования вычислительного ядра на альтернативный кремний: от серверных GPU AMD до встраиваемых систем (ESP32) и нейроморфных процессоров.

---

## 1. Концепция: Compute Backend Abstraction

**Инвариант:** Ядро симуляции (физика GLIF, пластичность GSOP) детерминировано и не зависит от конкретного API (CUDA/HIP/OpenCL). Рантайм разделяется на `Orchestrator` (Rust) и `Compute Backend` (Native code).

### 1.1. Backend Trait (Rust Side)
Для поддержки разных типов железа вводится абстракция вычислителя:

```rust
pub trait GenesisBackend {
    fn load_shard(&mut self, state: &ShardState) -> Result<()>;
    fn step(&mut self, inputs: &InputBatch) -> Result<OutputBatch>;
    fn sync_night_phase(&mut self) -> Result<ShardState>; // Download & maintenance
}
```

---

## 2. Tier 1: High-Performance Compute (AMD ROCm/HIP)

**Цель:** Прямая альтернатива NVIDIA для серверных кластеров.

### 2.1. Стратегия Портирования
// TODO: Внедрить трансляцию `kernel.cu` -> `kernel.hip` через `hipify-perl`.
// - **Интеграция:** Поддержка `hipRuntime` в `genesis-compute`.
// - **Оптимизация:** Адаптация под архитектуру CDNA (AMD Instinct). Использование Matrix Cores для векторизованного вычисления дендритов.
// - **Zero-Copy:** Реализация `hipHostMalloc` для обеспечения того же уровня производительности, что и `cudaHostAlloc`.

---

## 3. Tier 2: Edge Bare Metal (ESP32 & Embedded)

**Цель:** Автономные воплощенные агенты (робототехника) на сверхдешевом железе.

### 3.1. Bare Metal Runtime
// TODO: Разработать `genesis-lite` — runtime на чистом Си/C++ для встраиваемых систем.
// - **ESP32-S3 (AI Instruction Set):** Использование векторных инструкций Xtensa для ускорения целочисленной физики GLIF.
// - **Ограничения:** Уменьшенный размер `dendrite_slots` (32 вместо 128) для вписывания в SRAM.
// - **Flash-Mapped DNA:** Использование `mmap` (или аналога) для чтения аксонов напрямую из Flash-памяти без копирования в RAM.
// - **I/O:** Прямое мапирование GPIO на `InputMatrix` (сенсоры) и `OutputMatrix` (сервоприводы).

---

## 4. Tier 3: Future Silicon (Neuromorphic & ASICs)

**Цель:** Энергоэффективность уровня биологического мозга (на порядки выше GPU).

### 4.1. Neuromorphic Integration (Loihi, SpiNNaker)
// TODO: Исследовать мапинг GNM на асинхронные нейроморфные архитектуры.
// - **Event-Driven Execution:** Отказ от глобального тика (BSP) в пользу асинхронных спайков.
// - **On-Chip Learning:** Адаптация GSOP под аппаратные реализации STDP.

### 4.2. ASIC / FPGA (Verilog & VHDL)
// TODO: Проектирование RTL-описания ядра GLIF.
// - **Pipeline Logic:** Нейрон как конечный автомат (FSM) на FPGA.
// - **NoC (Network on Chip):** Аппаратная реализация Ring Buffer для пересылки Ghost Axons между ядрами чипа.
// - **HBM Integration:** Использование памяти с высокой пропускной способностью для хранения весов синапсов.

---

## Connected Documents

| Document | Connection |
|---|---|
| [07_gpu_runtime.md](./07_gpu_runtime.md) | Текущая эталонная реализация на CUDA |
| [01_foundations.md](./01_foundations.md) | Детерминизм и физика, общие для всех бэкендов |
| [06_distributed.md](./06_distributed.md) | Сетевой протокол для гетерогенных кластеров (GPU + ESP32) |
