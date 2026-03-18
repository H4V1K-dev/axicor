use serde::Deserialize;

/// Полный `blueprints.toml` — список типов нейронов.
#[derive(Debug, Deserialize, Clone)]
pub struct BlueprintsConfig {
    #[serde(rename = "neuron_type")]
    pub neuron_types: Vec<NeuronType>,
}

/// Один [[neuron_type]] блок из blueprints.toml.
/// Объединяет параметры как для baker (рост), так и для runtime (GSOP, Night Phase).
#[derive(Debug, Deserialize, Clone, Default)]
pub struct NeuronType {
    /// Уникальное имя типа. Используется как ключ в anatomy.toml.
    pub name: String,

    // --- Параметры Мембраны (i32, microVolts) ---
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_rate: i32,
    
    #[serde(default)]
    pub spontaneous_firing_period_ticks: u32,

    // --- Тайминги (u8, Ticks) ---
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,

    // --- Физика Сигнала (u16) ---
    
    #[serde(default = "default_propagation_length")]
    pub signal_propagation_length: u8,

    // --- Рост аксона (Steering) ---
    
    #[serde(default = "default_steering_fov")]
    pub steering_fov_deg: f32,
    
    #[serde(default = "default_steering_radius")]
    pub steering_radius_um: f32,
    
    #[serde(default = "default_steering_inertia")]
    pub steering_weight_inertia: f32,
    
    #[serde(default = "default_steering_sensor")]
    pub steering_weight_sensor: f32,
    
    #[serde(default = "default_steering_jitter")]
    pub steering_weight_jitter: f32,

    // [DOD FIX] Индивидуальный морфологический радиус дендритного дерева
    #[serde(default = "default_dendrite_radius")]
    pub dendrite_radius_um: f32,

    // --- Направленность роста ---
    #[serde(default = "default_growth_vertical_bias")]
    pub growth_vertical_bias: f32,

    #[serde(default = "default_type_affinity")]
    pub type_affinity: f32,

    // --- Знак синапса ---
    /// true = тормозной (inhibitory, отрицательный вес синапса)
    /// false = возбуждающий (excitatory, положительный вес синапса)
    #[serde(default)]
    pub is_inhibitory: bool,

    /// Начальный вес синапса (абсолютное значение). Знак определяется `is_inhibitory`.
    #[serde(default = "default_initial_weight")]
    pub initial_synapse_weight: u16,

    /// Список имён типов, от которых этот тип принимает дендритные связи.
    /// Пустой список = принимает от всех типов (без фильтрации).
    #[serde(default)]
    pub dendrite_whitelist: Vec<String>, //TODO рефакторинг логики

    // --- Гомеостаз (Adaptive Threshold) ---
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: u16, // u16 (для runtime)

    // --- Slot Decay (Fixed-point: 128 = 1.0×) ---
    pub slot_decay_ltm: u8,
    pub slot_decay_wm: u8,

    pub ltm_slot_count: u8,

    // --- Plasticity Inertia (16 рангов) ---
    #[serde(default = "default_inertia_curve")]
    pub inertia_curve: [u8; 16],

    // --- Adaptive Leak ---
    #[serde(default)]
    pub adaptive_leak_max: i32,
    #[serde(default)]
    pub adaptive_leak_gain: u16,
    #[serde(default)]
    pub adaptive_mode: u8,

    // --- GSOP ---
    #[serde(default = "default_gsop_pot")]
    pub gsop_potentiation: u16,
    
    #[serde(default = "default_gsop_dep")]
    pub gsop_depression: u16,

    #[serde(default = "default_affinity")]
    pub d1_affinity: u8,

    #[serde(default = "default_affinity")]
    pub d2_affinity: u8,

    // --- Sprouting Score Weights (f32, sum ≈ 1.0) ---
    #[serde(default = "default_sprouting_dist")]
    pub sprouting_weight_distance: f32,
    
    #[serde(default = "default_sprouting_power")]
    pub sprouting_weight_power: f32,
    
    #[serde(default = "default_sprouting_exp")]
    pub sprouting_weight_explore: f32,

    #[serde(default = "default_sprouting_type")]
    pub sprouting_weight_type: f32, //TODO не работает до рефакторинга Night Phase

    // --- Night Phase ---
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: i16,
}

// Дефолтные значения для опциональных полей
fn default_propagation_length() -> u8 { 10 }

fn default_steering_fov() -> f32 { 60.0 }
fn default_steering_radius() -> f32 { 100.0 }
fn default_steering_inertia() -> f32 { 0.6 }
fn default_steering_sensor() -> f32 { 0.3 }
fn default_steering_jitter() -> f32 { 0.1 }
fn default_dendrite_radius() -> f32 { 150.0 }
fn default_growth_vertical_bias() -> f32 { 0.7 }
fn default_type_affinity() -> f32 { 0.5 }
fn default_initial_weight() -> u16 { 74 }
fn default_gsop_pot() -> u16 { 60 }
fn default_gsop_dep() -> u16 { 30 }
fn default_affinity() -> u8 { 128 }
fn default_sprouting_dist() -> f32 { 0.4 }
fn default_sprouting_power() -> f32 { 0.4 }
fn default_sprouting_exp() -> f32 { 0.1 }
fn default_sprouting_type() -> f32 { 0.1 }
fn default_prune_threshold() -> i16 { 15 }
fn default_inertia_curve() -> [u8; 16] {
    [128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8]
}

impl NeuronType {
    /// Суммарный вес sprouting score (должна быть ≈ 1.0).
    pub fn sprouting_weight_sum(&self) -> f32 {
        self.sprouting_weight_distance + self.sprouting_weight_power + self.sprouting_weight_explore + self.sprouting_weight_type
    }
}

impl BlueprintsConfig {
    /// Парсит конфиг из TOML строки.
    pub fn parse(src: &str) -> Result<Self, String> {
        toml::from_str(src).map_err(|e| format!("TOML parse error: {}", e))
    }

    /// Загружает конфиг с диска.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {:?}: {}", path, e))?;
        Self::parse(&content)
    }
}

/// Параметры физики и пластичности для одного типа нейрона (Variant).
/// Выровнено ровно до 64 байт для идеального кэширования в L1 Constant Memory GPU.
// use serde::Serialize;

// VariantParameters is now defined in crate::layout.rs
/*
pub struct VariantParameters {
    ...
}
*/

// Заменяет твой старый код, если он там был.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GenesisConstantMemory {
    pub variants: [crate::layout::VariantParameters; 16],
}

// Гаранты детерминизма. Компилятор упадёт здесь, если кто-то сломает выравнивание.
/*
const _: () = assert!(
    std::mem::size_of::<VariantParameters>() == 64,
    "VariantParameters MUST be exactly 64 bytes for optimal Constant Memory caching"
);
const _: () = assert!(
    std::mem::size_of::<GenesisConstantMemory>() == 1024,
    "GenesisConstantMemory MUST be exactly 1024 bytes"
);
*/

// #[cfg(test)]
// #[path = "test_blueprints.rs"]
// mod test_blueprints;
