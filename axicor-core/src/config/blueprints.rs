use serde::{Deserialize, Serialize};

/// Full `blueprints.toml`  list of neuron types.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BlueprintsConfig {
    #[serde(rename = "neuron_type")]
    pub neuron_types: Vec<NeuronType>,
}

/// A single [[neuron_type]] block from blueprints.toml.
/// Combines parameters for both baker (growth) and runtime (GSOP, Night Phase).
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct NeuronType {
    /// Unique type name. Used as a key in anatomy.toml.
    pub name: String,

    // --- Membrane Parameters (i32, microVolts) ---
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_shift: u32,

    #[serde(default)]
    pub spontaneous_firing_period_ticks: u32,

    // --- Timings (u8, Ticks) ---
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,

    // --- Signal Physics (u16) ---
    #[serde(default = "default_propagation_length")]
    pub signal_propagation_length: u8,

    // --- Axon growth (Steering) ---
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

    // [DOD FIX] Individual morphological radius of the dendritic tree
    #[serde(default = "default_dendrite_radius")]
    pub dendrite_radius_um: f32,

    // --- Growth directionality ---
    #[serde(default = "default_growth_vertical_bias")]
    pub growth_vertical_bias: f32,

    #[serde(default = "default_type_affinity")]
    pub type_affinity: f32,

    // --- Synapse sign ---
    /// true = inhibitory (negative synapse weight)
    /// false = excitatory (positive synapse weight)
    #[serde(default)]
    pub is_inhibitory: bool,

    /// Initial synapse weight (absolute value). Sign is determined by `is_inhibitory`.
    #[serde(default = "default_initial_weight")]
    pub initial_synapse_weight: u16,

    /// List of type names from which this type accepts dendritic connections.
    /// Empty list = accepts from all types (no filtering).
    #[serde(default)]
    pub dendrite_whitelist: Vec<String>, // TODO: logic refactoring

    // --- Homeostasis (Adaptive Threshold) ---
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: u16, // u16 (for runtime)

    // --- Slot Decay (Fixed-point: 128 = 1.0) ---
    #[serde(default = "default_slot_decay_ltm")]
    pub slot_decay_ltm: u8,
    #[serde(default = "default_slot_decay_wm")]
    pub slot_decay_wm: u8,

    #[serde(default = "default_ltm_slot_count")]
    pub ltm_slot_count: u8,

    // --- Plasticity Inertia (8 ranks) ---
    #[serde(default = "default_inertia_curve")]
    pub inertia_curve: [u8; 8],

    // --- Adaptive Leak ---
    #[serde(default)]
    pub adaptive_leak_min_shift: i32,
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

    // --- Sprouting Score Weights (f32, sum  1.0) ---
    #[serde(default = "default_sprouting_dist")]
    pub sprouting_weight_distance: f32,

    #[serde(default = "default_sprouting_power")]
    pub sprouting_weight_power: f32,

    #[serde(default = "default_sprouting_exp")]
    pub sprouting_weight_explore: f32,

    #[serde(default = "default_sprouting_type")]
    pub sprouting_weight_type: f32, // TODO: does not work until Night Phase refactoring

    // --- Night Phase ---
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: i16,

    // --- AHP Amplitude ---
    #[serde(default)]
    pub ahp_amplitude: u16,
}

// Default values for optional fields
fn default_propagation_length() -> u8 {
    10
}

fn default_steering_fov() -> f32 {
    60.0
}
fn default_steering_radius() -> f32 {
    100.0
}
fn default_steering_inertia() -> f32 {
    0.6
}
fn default_steering_sensor() -> f32 {
    0.3
}
fn default_steering_jitter() -> f32 {
    0.1
}
fn default_dendrite_radius() -> f32 {
    150.0
}
fn default_growth_vertical_bias() -> f32 {
    0.7
}
fn default_type_affinity() -> f32 {
    0.5
}
fn default_initial_weight() -> u16 {
    74
}
fn default_gsop_pot() -> u16 {
    60
}
fn default_gsop_dep() -> u16 {
    30
}
fn default_affinity() -> u8 {
    128
}
fn default_sprouting_dist() -> f32 {
    0.4
}
fn default_sprouting_power() -> f32 {
    0.4
}
fn default_sprouting_exp() -> f32 {
    0.1
}
fn default_sprouting_type() -> f32 {
    0.1
}
fn default_prune_threshold() -> i16 {
    15
}
fn default_inertia_curve() -> [u8; 8] {
    [
        128, 112, 96, 80, 64, 48, 32, 16,
    ]
}
fn default_slot_decay_ltm() -> u8 { 128 }
fn default_slot_decay_wm() -> u8 { 128 }
fn default_ltm_slot_count() -> u8 { 8 }

impl NeuronType {
    /// Total sprouting score weight (should be  1.0).
    pub fn sprouting_weight_sum(&self) -> f32 {
        self.sprouting_weight_distance
            + self.sprouting_weight_power
            + self.sprouting_weight_explore
            + self.sprouting_weight_type
    }
}

impl BlueprintsConfig {
    /// Parses config from TOML string.
    pub fn parse(src: &str) -> Result<Self, String> {
        toml::from_str(src).map_err(|e| format!("TOML parse error: {}", e))
    }

    /// Loads config from disk.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {:?}: {}", path, e))?;
        Self::parse(&content)
    }
}

// Aligned to exactly 64 bytes for optimal caching in GPU L1 Constant Memory.
// VariantParameters is now defined in crate::layout.rs
/*
pub struct VariantParameters {
    ...
}
*/
// Replaces your old code, if it was there.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AxicorConstantMemory {
    pub variants: [crate::layout::VariantParameters; 16],
}

// Determinism guarantees. Compiler will fail here if alignment is broken.
const _: () = assert!(
    std::mem::size_of::<crate::layout::VariantParameters>() == 64,
    "VariantParameters MUST be exactly 64 bytes for optimal Constant Memory caching"
);
const _: () = assert!(
    std::mem::size_of::<AxicorConstantMemory>() == 1024,
    "AxicorConstantMemory MUST be exactly 1024 bytes for GPU Constant Memory limits"
);

// #[cfg(test)]
// #[path = "test_blueprints.rs"]
// mod test_blueprints;
