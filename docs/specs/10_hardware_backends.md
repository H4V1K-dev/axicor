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

## 2. Tier 1: High-Performance Compute (NVIDIA CUDA)

**Статус: [MVP - Реализовано]**

Для компиляции CUDA-бекенда (`genesis-compute/src/cuda/`) применяются строгие аппаратные инварианты:

*   **Архитектура GPU:** Pascal (GTX 10-серия) и выше. Поддерживаются мобильные версии (например, GTX 1060 Mobile).
*   **Версия CUDA Toolkit:** Строго `>= 12.4`. Старые компиляторы `nvcc` имеют баги эвристического анализатора при развертке циклов (`#pragma unroll`) поверх 32-байтных `alignas` структур и не могут корректно распределить регистры для `__shfl_sync` и `__popc` в наших HFT-ядрах.
*   **Хост ОС:** Ubuntu 22.04 LTS / 24.04 LTS. 
*   **Виртуализация:** Компиляция на хостах гипервизоров (например, Proxmox 9.1 на Debian 13) не поддерживается из-за конфликта системных линкеров GCC с проприетарными хидерами. Для виртуальных машин используйте исключительно прямой проброс устройства (PCIe Passthrough) в гостевую Ubuntu.

---

## 3. Tier 1: High-Performance Compute (AMD ROCm/HIP)

**Статус: [MVP - Реализовано]**

### 3.1. Архитектура: Dual-Backend C-ABI

Вместо хрупкого инструментария вроде `hipify-perl` (автоматическая трансляция CUDA→HIP) был реализован более чистый подход - **Dual-Backend C-ABI абстракция**.

`genesis-compute` содержит два зеркальных каталога с нативными реализациями одного и того же интерфейса:

```
genesis-compute/src/
├── cuda/
│   ├── bindings.cu   ← NVIDIA CUDA (nvcc, Warp 32)
│   └── physics.cu
└── amd/
    ├── bindings.hip  ← AMD ROCm/HIP (hipcc, Wavefront 64)
    └── physics.hip
```

**Выбор бэкенда - на этапе компиляции** через feature-флаги Cargo. `build.rs`動динамически вызывает `nvcc` или `hipcc`:

```sh
# NVIDIA (по умолчанию)
cargo build -p genesis-node

# AMD
cargo build -p genesis-node --features amd
```

### 3.2. Почему Портирование Обошлось Дёшево

Благодаря изначальной **Data-Oriented архитектуре без графов вычислений** (никакого PyTorch/cuDNN/cuBLAS) портирование математики свелось к механическим заменам API:

| NVIDIA CUDA | AMD HIP |
|---|---|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `__syncthreads()` | `__syncthreads()` (идентично) |
| `blockDim.x = 32` (Warp) | `blockDim.x = 64` (Wavefront) |

Единственная содержательная адаптация - размер локальной группы потоков: AMD-архитектуре (GCN/RDNA) нативен **64-поточный Wavefront** вместо NVIDIA-шного 32-поточного Warp. Все алгоритмы (Integer GLIF, GSOP, Columnar Dendrite Access) масштабируются линейно - волновой фронт большего размера только повышает утилизацию видеопамяти.

---

## 4. Tier 2: Edge Bare Metal (ESP32 & Embedded)

**Цель:** Автономные воплощенные агенты (робототехника) на сверхдешевом железе.

### 4.1. Bare Metal Runtime
Реализацию архитектуры памяти, двухъядерного распараллеливания и сетевого стека см. в [11_edge_bare_metal.md](11_edge_bare_metal.md).
// - **ESP32-S3 (AI Instruction Set):** Использование векторных инструкций Xtensa для ускорения целочисленной физики GLIF.
// - **Ограничения:** Уменьшенный размер `dendrite_slots` (32 вместо 128) для вписывания в SRAM.
// - **Flash-Mapped DNA:** Использование `mmap` (или аналога) для чтения аксонов напрямую из Flash-памяти без копирования в RAM.
// - **I/O:** Прямое мапирование GPIO на `InputMatrix` (сенсоры) и `OutputMatrix` (сервоприводы).

---

## 5. Tier 3: Future Silicon (Neuromorphic & ASICs)

**Цель:** Энергоэффективность уровня биологического мозга (на порядки выше GPU).

### 5.1. Neuromorphic Integration (Loihi, SpiNNaker)
// TODO: Исследовать мапинг GNM на асинхронные нейроморфные архитектуры.
// - **Event-Driven Execution:** Отказ от глобального тика (BSP) в пользу асинхронных спайков.
// - **On-Chip Learning:** Адаптация GSOP под аппаратные реализации STDP.

### 5.2. ASIC / FPGA (Verilog & VHDL)
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
