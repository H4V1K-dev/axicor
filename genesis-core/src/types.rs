// ---------------------------------------------------------------------------
// Spec 01 §1.1 — Три системы координат
// ---------------------------------------------------------------------------

/// Абсолютная пространственная единица: 1.0 = 1 мкм.
/// Используется для: длин аксонов, радиусов поиска дендритов, скоростей, физики диффузии.
/// Позволяет использовать реальные нейробиологические константы без магических коэффициентов.
pub type Microns = f32;

/// Нормализованная координата [0.0, 1.0].
/// Используется для: границ слоёв (height_pct, population_pct) и топологии зон.
/// При инициализации умножается на world_dim (в мкм) → абсолютные координаты.
pub type Fraction = f32;

/// Дискретная координата в вокселях.
/// = floor(Microns / voxel_size_um). Квант пространства — размер вокселя из конфига.
/// Используется для: Spatial Hashing, поиска соседей, GPU-индексации.
pub type VoxelCoord = u32;

/// Packed voxel coordinate: [Type(4b) | Z(8b) | Y(10b) | X(10b)]
/// Bit layout: t << 28 | z << 20 | y << 10 | x
pub type PackedPosition = u32;

/// Dendrite target: [31..10] axon_id (22 bits) | [9..0] segment_index (10 bits).
/// Layout: `axon_id << TARGET_AXON_SHIFT | seg_idx & TARGET_SEG_MASK`
/// Ёмкость: до 4 194 303 аксонов, до 1023 сегментов на аксон.
/// 0 = пустой слот (нет соединения). Используйте `coords::pack_target` / `unpack_target`.
pub type PackedTarget = u32;

/// Индекс сегмента внутри аксона (§1.2). 10 бит → 0..=1023.
/// Сегмент — атомарная единица пути: дендрит соединяется с `(Axon_ID, SegmentIndex)`,
/// а не с координатой `(X, Y, Z)`.
pub type SegmentIndex = u32;

/// Счётчик тиков (§1.4). Квант = `TICK_DURATION_US` мкс = 0.1 мс.
/// Все таймеры (рефрактерность, decay, Night Phase интервал) задаются в тиках.
/// Пример: 5 мс рефрактерность = 50 тиков. Используйте `time::ms_to_ticks` для конвертации.
pub type Tick = u64;

/// Synaptic weight. Sign encodes excitatory (+) or inhibitory (-).
/// Range: -32768..+32767. Baked in during Night Phase, frozen during Day Phase.
pub type Weight = i16;

/// Neuron membrane voltage accumulator.
pub type Voltage = i32;

/// Axon head position (segment index). AXON_SENTINEL when inactive.
pub type AxonHead = u32;

/// Variant ID (2 bits in flags byte: bits 6-7).
/// 0..3 → index into VariantParameters[4] in Constant Memory.
pub type VariantId = u8;

/// Neuron flags byte:
/// [7:6] variant_id | [5] is_spiking | [4] reserved | [3:0] type_mask
pub type NeuronFlags = u8;
