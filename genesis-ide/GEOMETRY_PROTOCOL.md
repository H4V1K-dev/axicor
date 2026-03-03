# Genesis IDE - GeometryServer Protocol (TCP 8001)

## Overview

IDE запрашивает геометрию один раз при старте через TCP 8001. Runtime отправляет бинарный фрейм GEOM, содержащий позиции и типы всех нейронов. IDE парсит этот фрейм и строит **O(1) таблицу маршрутизации спайков** (spike_id → batch_id, local_idx).

**Почему не передавать type_id с каждым спайком?**
- Нарушает спецификацию (08_ide.md §2.3 требует только u32[])
- Увеличивает сетевую нагрузку в 8x (вместо 4 байт/спайк → 8 байт/спайк)
- **Тип — это константа**, вычисляется один раз, используется вечно

## Protocol Format

### Connection
```
IDE → Runtime (TCP)
127.0.0.1:8001 (Runtime слушает)
```

### GEOM Frame Structure

```
Offset | Size | Name         | Type     | Value
-------|------|------|----------|------
[0..4] | 4    | Magic        | [u8; 4] | b"GEOM"
[4..8] | 4    | TotalNeurons | u32 LE  | Всего Dense Indices
[8..N] | 8*N  | GeomData     | [u8]    | Array of NeuronGeomData
```

### NeuronGeomData (8 bytes each)

```rust
#[repr(C)]
pub struct NeuronGeomData {
    pub packed_pos: u32,  // 4 bytes: X(11) | Y(11) | Z(10)
    pub flags: u32,       // 4 bytes: type_id = (flags >> 4) & 0xF
}
```

**Декодирование PackedPos:**
```rust
let x = packed_pos & 0x7FF;           // Bits 0..10 (2048 max)
let y = (packed_pos >> 11) & 0x7FF;   // Bits 11..21
let z = (packed_pos >> 22) & 0x3FF;   // Bits 22..31 (1024 max)

// Нормализация в [-1..1]
let world_x = (x as f32 / 1024.0) - 1.0;
let world_y = (y as f32 / 1024.0) - 1.0;
let world_z = (z as f32 / 512.0) - 1.0;
```

**Декодирование Type:**
```rust
let type_id = ((flags >> 4) & 0xF) as u8;  // Bits 4..7
```

## Spike Routing (Hot Loop O(1))

После загрузки GEOM фрейма IDE строит таблицу маршрутизации:

```rust
pub struct SpikeRouter {
    pub routing: Vec<(u8, u32)>,  // routing[spike_id] = (batch_id, local_idx)
}
```

**Построение (один раз):** O(N)
```rust
for global_idx in 0..total_neurons {
    let geom = parse_neuron_geom(global_idx);
    let batch_id = geom.type_id();
    routing[global_idx] = (batch_id, counts[batch_id]);
    counts[batch_id] += 1;
}
```

**Использование (каждый спайк фрейм):** O(1)
```rust
for &spike_id in &spikes {
    let (batch_id, local_idx) = routing[spike_id];  // O(1)
    layer[batch_id].instances[local_idx].emissive = 1.0;
}
```

## Пример: 160k нейронов

```
TotalNeurons = 160,000
Types = 16 (batch_id 0..15)
Neurons per type = 10,000

GEOM Frame Size:
  Header: 8 bytes
  Data: 160,000 × 8 = 1,280,000 bytes
  Total: 1,280,008 bytes (~1.28 MB)

Routing Table:
  160,000 × (u8, u32) = 160,000 × 5 bytes = 800,000 bytes
  (или Vec<(u8, u32)> в Rust немного больше за счет alignment)
```

Memory Footprint:
- GeomData buffer: 1.28 MB (можно discard после парсинга)
- SpikeRouter: ~1 MB (постоянно в памяти)
- NeuronInstance батчи: 160k × 8 bytes = 1.28 MB

## Implementation (genesis-ide/src/loader.rs)

### Structures
```rust
#[repr(C)]
#[derive(Pod, Zeroable)]
pub struct GeometryHeader {
    pub magic: [u8; 4],
    pub total_neurons: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable)]
pub struct NeuronGeomData {
    pub packed_pos: u32,
    pub flags: u32,
}

#[derive(Resource)]
pub struct SpikeRouter {
    pub routing: Vec<(u8, u32)>,
}
```

### Fetch
```rust
async fn fetch_geometry_from_runtime() -> Result<SpikeRouter, String> {
    let mut stream = TcpStream::connect("127.0.0.1:8001")?;
    parse_geometry_stream(&mut stream)
}

fn parse_geometry_stream(stream: &mut TcpStream) -> Result<SpikeRouter> {
    // 1. Read header
    let mut header_bytes = [0u8; 8];
    stream.read_exact(&mut header_bytes)?;
    let header: GeometryHeader = bytemuck::cast(header_bytes);
    
    // 2. Validate magic
    assert_eq!(&header.magic, b"GEOM");
    
    // 3. Read all NeuronGeomData
    let geom_size = header.total_neurons as usize * 8;
    let mut geom_bytes = vec![0u8; geom_size];
    stream.read_exact(&mut geom_bytes)?;
    
    // 4. Build routing table
    let mut routing = Vec::with_capacity(header.total_neurons as usize);
    let mut counts = [0u32; 16];
    
    for chunk in geom_bytes.chunks_exact(8) {
        let geom: NeuronGeomData = bytemuck::cast_slice(chunk)[0];
        let batch_id = geom.type_id();
        routing.push((batch_id, counts[batch_id as usize]));
        counts[batch_id as usize] += 1;
    }
    
    Ok(SpikeRouter { routing })
}
```

### Spike Routing (in world.rs)
```rust
fn apply_telemetry_spikes(
    mut query: Query<&mut NeuronLayerData>,
    mut spike_events: EventReader<SpikeFrame>,
    router: Option<Res<SpikeRouter>>,
) {
    let Some(router) = router else { return; };
    
    // Pre-compute assignments
    let mut layer_spikes: Vec<Vec<u32>> = vec![Vec::new(); 16];
    for &spike_id in batch_spikes {
        let (batch_id, local_idx) = router.routing[spike_id as usize];
        layer_spikes[batch_id as usize].push(local_idx);
    }
    
    // Apply
    for mut layer in query.iter_mut() {
        for &local_idx in &layer_spikes[layer.type_id as usize] {
            layer.instances[local_idx as usize].emissive = 1.0;
        }
    }
}
```

## Fallback: Mock Router

Если Runtime недоступен (TCP 8001 соединение не установлено):

```rust
fn generate_mock_router() -> SpikeRouter {
    // 160k нейронов, 16 типов
    let mut routing = Vec::with_capacity(160_000);
    for global_idx in 0..160_000 {
        let batch_id = (global_idx / 10_000) as u8;
        let local_idx = (global_idx % 10_000) as u32;
        routing.push((batch_id, local_idx));
    }
    SpikeRouter { routing }
}
```

## Advantages

1. **O(1) spike routing** - никаких поисков в Hot Loop
2. **Масштабируемо** - работает так же для 500k нейронов
3. **Низкая latency** - спайк → батч → GPU за 1 индексацию
4. **Spec-compliant** - не нарушает 08_ide.md §2.3
5. **Storage efficient** - routing table меньше одного VRAM буфера

## Future: GPU Memory Upload

Cuando Phase 2 (GPU Instancing):
1. Загруженная geom_data может быть закеширована для инициализации DynamicStorageBuffer
2. PackedPos → мировые координаты в WGSL шейдере
3. Spikes пишутся в separate emissive_buffer[spike_id] = 1.0
4. Один render pass для 16 батчей вместо 160 тысяч draw calls
