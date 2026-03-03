# Genesis IDE - Архитектура визуализации

## Обзор
Полная изолированная система визуализации нейронной сети в Bevy 0.13, оптимизированная для масштабирования до 500k нейронов. Архитектура базируется на:
- **16 типов нейронов** = 16 Draw Call максимум
- **Lock-free WebSocket** - спайки через IoTaskPool + crossbeam_channel
- **Асинхронная загрузка** - AsyncComputeTaskPool без блокировки Main Thread
- **FPS-навигация** - Alt-захват, WASD + Space/Shift, полная свобода

## Принципы (Visual Sympathy)
1. **Никаких иерархий Transform** - сокращаем overhead Main Thread propagate_transforms
2. **16 типов максимум** - 16 Entity = 16 Draw Calls максимум в сцене
3. **Zero-Cost спайки** - обновляем только emissive значения во время Hot Loop
4. **PackedPosition декодирование на GPU** - 32-битное кодирование координат (X:11, Y:11, Z:10)
5. **Асинхронная загрузка** - AsyncComputeTaskPool без блокировки Main Thread
6. **FPS-навигация** - Alt-захват курсора, WASD + Space/Shift, полная свобода

## Структура файлов

### `src/camera.rs` - FPS-камера и входные данные

**Компоненты:**
- `IdeCamera` - состояние камеры (скорость, pitch, yaw)

**Ресурсы:**
- `CameraMode` - флаг захвата курсора (Free | Captured)

**Системы:**
```rust
fn setup_camera()
```
- Инициализирует Camera3d в начальной позиции

```rust
fn toggle_camera_mode()
```
- Alt или Esc переключают режим захвата
- Locked = невидимый курсор, свободное вращение
- Free = видимый курсор, UI взаимодействие

```rust
fn camera_movement_system()
```
- **WASD** - движение вперед/назад/влево/вправо (relative to view direction)
- **Space/Shift** - движение вверх/вниз (глобальное Y)
- **Mouse** - вращение (yaw/pitch)
- Sensitivity: 0.002 rad/px, max pitch ±88°

### `src/loader.rs` - Асинхронная загрузка геометрии

**События:**
- `GeometryChunkReceived` - геометрия для type_id загружена в RAM

**Компоненты:**
- `GeometryFetchTask` - обертка Task для отслеживания фоновой загрузки

**Системы:**
```rust
fn spawn_geometry_request()
```
- Спавнит Task в AsyncComputeTaskPool на Startup
- **Пока Mock Data**: 16 типов × 10k инстансов = 160k нейронов
- **TODO на Phase 2**: TCP 8001 запрос к GeometryServer

```rust
fn poll_geometry_tasks()
```
- Опрашивает Future каждый кадр (O(1) poll без блокировки)
- Когда готово: отправляет Event и despawn'ит Task

```rust
fn apply_loaded_geometry()
```
- Zero-Cost обновление: копирует instances в NeuronLayerData
- Устанавливает needs_buffer_update = true для GPU инжекции

### `src/world.rs` - Батчинг нейронов и спайки

#### Структуры данных
- **`NeuronInstance`** - POD структура для GPU (8 байт)
  - `packed_pos: u32` - Сжатые координаты
  - `emissive: f32` - Интенсивность свечения (0.0-1.0)

- **`NeuronLayerData`** - ECS компонент для каждого типа
  - `type_id: u8` - Идентификатор типа (0..15)
  - `instances: Vec<NeuronInstance>` - Буфер на CPU
  - `needs_buffer_update: bool` - Флаг PCIe синхронизации

```rust
fn apply_telemetry_spikes()
```
- Hot Loop каждый кадр, слушает EventReader<SpikeFrame>
- **Фаза 1: Затухание** - emissive -= 0.05 (fade-out спайков)
- **Фаза 2: Инжекция** - новые spike_ids → emissive = 1.0 (но ID mappig - TODO по типам)
- **Фаза 3: Грязь** - только если были изменения, sets needs_buffer_update

### `src/telemetry.rs` - WebSocket спайки от Runtime

**События:**
- `SpikeFrame` - батч спайков от genesis-runtime (tick + Vec<spike_ids>)

**Ресурсы:**
- `TelemetryBridge` - lock-free канал (bounded, capacity 60)

**Системы:**
```rust
fn spawn_telemetry_client()
```
- Спавнит асинхронную задачу в IoTaskPool на Startup
- Подключается к `ws://127.0.0.1:8002/ws` (неблокирующий)
- Слушает Message::Binary с магией "SPIK"
- Декодирует: Tick(u64) + Count(u32) + spike_ids(u32[])
- Отправляет в bounded канал (если переполнен, дропает старые - real-time)

```rust
fn decode_telemetry_frame()
```
- Парсит сырый бинарный фрейм
- Магия: b"SPIK" (Fast Fail)
- Выборка u32 спайков из payload
- O(1) allocate Vec, O(n) decode payload

```rust
fn poll_telemetry_channel()
```
- Hot Loop каждый кадр
- try_iter() без блокировки
- Отправляет в EventWriter<SpikeFrame>
- Zero overhead если канал пуст

### `assets/shaders/neuron_instanced.wgsl` - WGSL шейдер (заготовка)
- Декодирует PackedPosition в вершинном шейдере
- Применяет emissive glow в фрагментном шейдере

## Road Map

### Phase 1: MVP завершено ✓
- [x] FPS-камера с Blender-style Alt-захватом
- [x] Асинхронная загрузка геометрии (Mock Data)
- [x] WebSocket спайки от Runtime (IoTaskPool, lock-free)
- [x] 16 батчей для 500k нейронов (архитектура)
- [x] Компиляция и работа

### Phase 2: GPU Instancing (следующий этап)
```
Текущий: StandardMaterial + Transform per neuron layer
Целевой: DynamicStorageBuffer + кастомный render pass
```
- [ ] Кастомный Material с AsBindGroup
- [ ] DynamicStorageBuffer для PackedPosition/emissive
- [ ] Render pass через RenderSet::Render
- [ ] WGSL декодирование PackedPosition → world coords
- [ ] **ГЛАВНОЕ**: Масштабировать с 160k (Mock) до 500k real нейронов

### Phase 3: GPU оптимизации + Network
- [ ] TCP 8001: запрос GeometryServer (вместо Mock)
- [ ] Spike ID mapping по типам (сейчас global ID modulo)
- [ ] LOD по дистанции от камеры
- [ ] Frustum culling на GPU
- [ ] Bloom/Glow эффекты для спайков

## Интеграция модулей

```
Genesis Runtime (TCP 8002, binary protocol)
       ↓ WebSocket
IoTaskPool (tokio-tungstenite, async)
       ↓ decode_telemetry_frame()
crossbeam_channel (bounded, try_send)
       ↓ poll_telemetry_channel()
SpikeFrame Event
       ↓ apply_telemetry_spikes()
NeuronLayerData::instances[].emissive = 1.0
       ↓ needs_buffer_update flag
Bevy render pass [каждый кадр]
       ↓
StandardMaterial emissive glow
       ↓
GPU: 500k сфер с мерцанием спайков
```

**Навигация:**
```
FPS Camera [RenderSets::Camera]
       ↓ camera_movement_system()
Transform обновления
       ↓
Render pipeline
       ↓
Screen
```

## Сборка и запуск

```bash
cargo build -p genesis-ide
cargo run -p genesis-ide
```

Ожидаемый логи:
```
[world] Initialized neuron layer 0
[world] Initialized neuron layer 1
...
[world] 16 neuron layers ready for GPU Instancing
[loader] Geometry fetch task completed.
[loader] Loaded 10000 instances for neuron type 0
...

IDE Controls:
  Alt           - Toggle camera capture
  WASD          - Move camera
  Space/Shift   - Up/Down
  Mouse         - Look around (when Alt captured)
  Esc           - Release camera
```

## Стабильность кода
- ✓ Модульность: каждый файл отвечает за одно
- ✓ AsyncComputeTaskPool без блокировок (геометрия)
- ✓ IoTaskPool без блокировок (WebSocket спайки)
- ✓ Lock-free crossbeam_channel (спайки → ECS)
- ✓ Pod/Zeroable для GPU-safety
- ✓ ECS Events для разъединения систем
- ✓ Zero overhead в Hot Loop (try_iter контроль)
- ⚠️ Пока Mock Data (160k нейронов)
- ⚠️ Без GPU Instancing (но архитектура подготовлена)
- ⚠️ spike_id mapping global (TODO: per-type mapping)
