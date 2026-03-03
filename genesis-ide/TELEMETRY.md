# Genesis IDE - Telemetry Protocol Documentation

## System Overview

IDE подключается к Runtime на **двух каналах:**

1. **TCP 8001 - GeometryServer** (инициализация)
   - Загрузка геометрии один раз при старте
   - IDE строит O(1) таблицу маршрутизации спайков
   - → [GEOMETRY_PROTOCOL.md](GEOMETRY_PROTOCOL.md)

2. **WebSocket 8002 - TelemetryServer** (real-time)
   - Стрим спайков ~60fps
   - Каждый спайк — это Dense Index (u32)
   - IDE использует routing table для маршрутизации

## WebSocket Спайки (08_ide.md §2.3)

### Протокол

Genesis IDE слушает на **`ws://127.0.0.1:8002/ws`** бинарные фреймы со спайками от Runtime.

#### Формат Spike Frame

```
Offset  | Size | Name         | Value
--------|------|------|-----
[0..4]  | 4    | Magic        | b"SPIK" (0x5350494B in BE)
[4..12] | 8    | Tick         | u64, Little-Endian (кадр движка Runtime)
[12..16]| 4    | Count        | u32, Little-Endian (кол-во спайков)
[16..N] | 4*C  | Spike IDs    | Array of u32, LE (Dense Index или Local ID)
```

**Пример:**
```
Raw bytes: [0x53, 0x50, 0x49, 0x4B,  // Magic "SPIK"
            0xE8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // Tick = 1000
            0x03, 0x00, 0x00, 0x00,  // Count = 3
            0x0A, 0x00, 0x00, 0x00,  // Spike ID = 10
            0x15, 0x00, 0x00, 0x00,  // Spike ID = 21
            0x2C, 0x00, 0x00, 0x00]  // Spike ID = 44
```

### Декодирование

```rust
// В telemetry.rs
fn decode_telemetry_frame(data: &[u8]) -> Option<SpikeFrame> {
    // 1. Проверка magic
    if &data[0..4] != b"SPIK" { return None; }
    
    // 2. Распаковка заголовка
    let tick = u64::from_le_bytes(&data[4..12]);
    let count = u32::from_le_bytes(&data[12..16]) as usize;
    
    // 3. Выборка spike_ids
    let spike_ids = data[16..]
        .chunks_exact(4)
        .take(count)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<u32>>();
    
    Some(SpikeFrame { tick, spike_ids })
}
```

### Интеграция в IDE

```
Genesis Runtime TCP 8002
        ↓
IoTaskPool::spawn(async {
    ws connect() → Message::Binary
        ↓
    decode_telemetry_frame()
        ↓
    crossbeam_channel::tx.try_send(frame)
        ↓ (bounded, capacity 60)
})

[Main Thread]
poll_telemetry_channel():
    for frame in bridge.rx.try_iter() {
        event_writer.send(frame)
    }
        ↓
EventReader<SpikeFrame>
        ↓
apply_telemetry_spikes():
    batch_spikes.extend(&ev.spike_ids)
        ↓
    layer.instances[id].emissive = 1.0
        ↓
    GPU Instancing (Phase 2)
        ↓
    Frame glow effect
```

## Тестирование

### Mock Server (Python)

Для локального тестирования без Runtime:

```python
#!/usr/bin/env python3
import asyncio
import websockets
import struct
import random

async def serve_telemetry(websocket, path):
    tick = 0
    while True:
        # Генерируем фейковые спайки
        count = random.randint(5, 20)
        spike_ids = [random.randint(0, 159999) for _ in range(count)]
        
        # Строим фрейм
        data = bytearray()
        data.extend(b"SPIK")  # Magic
        data.extend(struct.pack("<Q", tick))  # Tick (u64)
        data.extend(struct.pack("<I", count))  # Count (u32)
        for spike_id in spike_ids:
            data.extend(struct.pack("<I", spike_id))  # Spike ID
        
        await websocket.send(bytes(data))
        tick += 1
        await asyncio.sleep(0.016)  # ~60 fps (16ms)

async def main():
    async with websockets.serve(serve_telemetry, "127.0.0.1", 8002):
        print("Mock telemetry server listening on ws://127.0.0.1:8002/ws")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
```

### Запуск

```bash
# Терминал 1: Mock server
python3 telemetry_mock.py

# Терминал 2: Genesis IDE
cargo run -p genesis-ide --release
```

Вы должны увидеть логи:
```
Connecting to Genesis Telemetry at ws://127.0.0.1:8002/ws...
Telemetry connected. Awaiting frames...
IDE Render Tick: Received 12 spikes from GPU Batch #1234
IDE Render Tick: Received 8 spikes from GPU Batch #1235
...
```

И в окне IDE будут мерцать случайные нейроны!

## Архитектурные решения

### Почему НЕ передавать type_id со спайками?
- **Спецификация 08_ide.md §2.3**: стрим только u32[] spike_ids
- **Лишняя bandwidth**: 4 байта → 8 байт на спайк = 100% увеличение трафика
- **Избыточность**: type_id — это КОНСТАНТА, может вычисляться один раз
- **Оптимальное решение**: TCP 8001 GeometryServer загружает type_id один раз при старте
  - IDE строит routing table: spike_id → (batch_id, local_idx)
  - Hot Loop: O(1) lookup вместо поиска/парсинга

### Почему IoTaskPool?
- **Не блокирует Main Thread** - сетевые I/O в фоновом потоке
- **Интегрирован с Bevy** - используется для async операций
- **Масштабируемость** - теоретически можно несколько подключений

### Почему crossbeam_channel?
- **Lock-free** - очень быстро
- **Bounded capacity** - дропаем старые спайки если IDE отстает (real-time)
- **try_send/try_iter** - неблокирующие операции в Hot Loop

### Почему магия "SPIK"?
- Fast Fail: быстрая проверка корруптированных данных
- 4 байта = 32-битное выравнивание (быстро сравнить)
- ASCII понятно при дебъге

### Почему u32 spike_ids?
- Dense Index: 32 бита = 4 млрд нейронов теоретически
- MVP может быть меньше, но протокол масштабируется
- O(1) lookup через routing table

## Known Limitations & Roadmap

✅ **SOLVED (Phase 1.5)**: Spike routing через GeometryServer + O(1) lookup table
  - Реализовано в TCP 8001 GEOM протоколе
  - IDE загружает geometry один раз, строит permanent routing table
  - Hot Loop: spike_id → batch_id, local_idx за 2 индексации

⚠️ **TODO (Phase 2)**: Bandwidth backpressure
   - Bounded channel (60) дропает спайки если IDE медленный
   - TODO: Backpressure сигнал Runtime'у через TCP 8001

⚠️ **TODO (Phase 2)**: WebSocket reconnect logic
   - Если соединение упадет, IDE не переподключится
   - TODO: Reconnect с exponential backoff

⚠️ **TODO (Phase 3)**: GPU Instancing
   - Текущая архитектура использует Mock data + Standard rendering
   - TODO: DynamicStorageBuffer + custom render pass
   - TODO: Масштабирование до 500k нейронов

## Производительность

### Latency (спайк → экран)
```
Runtime спайк
    ↓ 1ms (сетевая задержка)
    ↓
Telemetry poll()
    ↓ 0.016ms (try_iter)
    ↓
apply_telemetry_spikes()
    ↓ 0.001ms (O(1) routing[spike_id] lookup)
    ↓
instances[local_idx].emissive = 1.0
    ↓
GPU upload (Phase 2)
    ↓ 5-10ms (батч)
    ↓
Frame render
    ↓ 3-17ms (display refresh)
---
Total: ~10-30ms (typical 60fps display)
```

### Throughput
- 60 спайков/кадр × 60 кадров/сек = 3600 спайков/сек (Mock)
- Реально: Runtime может отправлять 1000+ спайков/кадр (100k нейр/сек потенциально)
- Канал capacity 60 = ~1 сек буфер при 60fps

## Дебъг

### Включить логи
```rust
// В main.rs добавить
.init_resource::<bevy::log::LogSettings>()

// Или ENV переменная
RUST_LOG=debug cargo run -p genesis-ide
```

### Проверить соединение
```bash
# Терминал 1: Mock server
python3 telemetry_mock.py

# Терминал 2: Проверить соединение через nc
nc -L 127.0.0.1 8002  # или socat/websocat

# Терминал 3: telemetry.rs должен логировать
cargo run -p genesis-ide 2>&1 | grep -i telemetry
```

### Дебъг фреймов
Распечатать полученные спайки в `apply_telemetry_spikes`:
```rust
for &id in &batch_spikes {
    println!("Spike ID: {}", id);
}
```

## Next: GPU Instancing (Phase 2)

Спайки готовы к GPU! Следующий этап:
1. DynamicStorageBuffer для emissive значений
2. Кастомный render pass
3. PackedPosition → мировые координаты в WGSL
4. Масштабирование с 160k до 500k нейронов
