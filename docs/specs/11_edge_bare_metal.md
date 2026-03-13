# 11. Edge Bare Metal (Genesis-Lite)

> Часть архитектуры Genesis. Спецификация HFT-движка для микроконтроллеров (Tier 2: ESP32-S3 / RISC-V).
> Цель: Автономные воплощенные агенты (Embodied AI) с задержкой < 2 мс и бюджетом памяти 520 КБ SRAM.

---

## 1. Фундаментальные Ограничения и Урезания

В отличие от серверных GPU с гигабайтами VRAM, на ESP32 каждый байт критичен. Чтобы HFT-цикл не выпадал из кэшей, мы вводим жесткие архитектурные лимиты:

1. **Dendrite Limit:** `MAX_DENDRITE_SLOTS` урезан со 128 до **32**. Это радикально снижает потребление памяти (Columnar Layout занимает в 4 раза меньше места).
2. **Integer Physics:** Строго 100% Branchless целочисленная математика. Никаких FPU-инструкций в горячем цикле.
3. **Alignment:** Все структуры выравниваются по границе **32 байта** (`alignas(32)`) для идеального попадания в D-Cache (кэш-линии процессоров Xtensa/RISC-V).

---

## 2. Архитектура Памяти (Hybrid SRAM / Flash)

520 КБ SRAM физически не вместят коннектом на несколько тысяч нейронов. Мы используем гибридную архитектуру памяти, опираясь на Memory-Mapped Flash (XIP - eXecute In Place).

### 2.1. Flash-Mapped DNA (Read-Only)
Топология, которая не меняется в Day Phase, мапится напрямую из Flash-памяти (по интерфейсу QSPI) через `mmap` / `PROGMEM`:
- `dendrite_targets` (PackedTarget, X|Y|Z|Type)
- `soma_to_axon` (маршрутизация)
*Эти данные кэшируются аппаратно D-Cache микроконтроллера при чтении.*

### 2.2. Hot State SRAM (Read/Write)
В быстрой 520 КБ SRAM хранятся только мутирующие в реальном времени массивы:
- `voltage`, `flags`, `refractory_timer`, `threshold_offset` (Hot Soma)
- `axon_heads` (Сдвиговый регистр `BurstHeads8` - 32 байта)
- `dendrite_weights` (Веса синапсов для работы GSOP-пластичности)

---

## 3. Асимметричный Двухъядерный Цикл (FreeRTOS)

ESP32-S3 имеет два ядра. ОС FreeRTOS позволяет нам жестко изолировать HFT-физику от медленных сетевых прерываний.

### 3.1. Core 1 (App Core): Day Phase
Ядро 1 жестко зарезервировано под горячий цикл (Hot Loop). 
- Выполняет только: `Propagate`, `UpdateNeurons` (GLIF), `ApplyGSOP`.
- **Никаких мьютексов.** 
- **Watchdog Protection:** Раз в N тиков (например, каждые 10 мс симуляции) цикл обязан вызывать `vTaskDelay(1 / portTICK_PERIOD_MS)`, уступая планировщику 1 тик для сброса аппаратного Task WDT. Иначе контроллер уйдет в ребут.

### 3.2. Core 0 (Pro Core): I/O & Night Phase
Ядро 0 выполняет всю "грязную" работу:
- Поддержание Wi-Fi / ESP-NOW.
- Чтение датчиков по I2C/SPI через DMA и конвертация float -> spikes (Sensory Encoding).
- **Night Phase:** Сортировка синапсов, Pruning и Sprouting. Передача данных между ядрами идет через Lock-Free очереди, а не мьютексы.

### 3.3. Визуальная схема потоков данных

```text
[ Физический Мир ]                               [ Чип ESP32-S3 ]
      │                                                │
      ▼                                                ▼
  Датчики (I2C)     ──(float)──>  [ Core 0 ] Population Encoder (float -> 8-bit SpikeEvent)
(Гироскоп, Лидар)                     │
                                      ▼
                             [ SRAM: Lock-Free Ring Buffer ] (alignas 32, std::atomic)
                                      │
                                      ▼
                              [ Core 1 ] Hot Loop (Day Phase)
                              1. Inject: Сброс голов аксонов (h0 = 0)
                              2. Propagate: Векторный сдвиг (hX += v_seg)
                              3. GLIF: Интеграция токов, пороги, спайки сомы
                              4. GSOP: Мутация весов дендритов
                              5. Readout: ++к счетчикам моторных нейронов
                                      │
                                      ▼
                           [ SRAM: MotorOut Struct ] (std::atomic)
                                      │
                                      ▼
   Сервоприводы     <──(PWM)───   [ Core 0 ] Population Decoder (spikes -> Duty Cycle)
  (Моторы, LED)                       │
      │                               │
      └───────────────────────────────┘
          (Изменение среды / Угол)
```

---

## 4. Micro-Networking и Сенсоры

### 4.1. ESP-NOW (Swarm Protocol)
UDP-пакеты Genesis-Node (65 КБ) не подходят для ESP32 (ограничение RAM и Wi-Fi стека).
- Используется протокол **ESP-NOW**.
- Жесткий MTU ~250 байт.
- Вместо полного дампа памяти передаются только компактные `SpikeEvent` (8 байт: `ghost_id` + `tick_offset`). 
- Применяется популяционное кодирование для сжатия сенсорных данных.

### 4.2. Прямой Hardware I/O
Вместо виртуальных матриц `Input_Bitmask` по сети, Genesis-Lite умеет биндить сенсорные аксоны напрямую к прерываниям или DMA-буферам датчиков (I2C гироскопы, SPI камеры, PWM сервоприводы).
