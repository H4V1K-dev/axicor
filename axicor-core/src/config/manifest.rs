use serde::{Deserialize, Serialize};
// use crate::layout::VariantParameters;
// use std::fmt;

/// 1. DTO: Read from manifest.toml. Contains String, lives on Heap.
///    Must NOT be passed via raw pointer to C++!
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestVariant {
    pub id: u8,
    pub name: String,
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_shift: u32, // [DOD FIX] Shift-based exponential leak
    pub homeostasis_penalty: i32,
    pub spontaneous_firing_period_ticks: u32,
    pub initial_synapse_weight: u16,
    pub gsop_potentiation: u16,
    pub gsop_depression: u16,
    pub homeostasis_decay: u16,
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,
    pub signal_propagation_length: u8,
    pub is_inhibitory: bool,
    pub inertia_curve: [u8; 8], // [DOD FIX] Compressed curve
    pub ahp_amplitude: u16, // [DOD FIX] 
    pub adaptive_leak_min_shift: i32, // [DOD FIX]
    pub adaptive_leak_gain: u16,
    pub adaptive_mode: u8,
    pub d1_affinity: u8,
    pub d2_affinity: u8,
    pub heartbeat_m: u32,
}

impl ManifestVariant {
    /// Zero-cost conversion to strict C-ABI
    pub fn into_gpu(self) -> crate::layout::VariantParameters {
        // [DOD FIX] Pre-calculate DDS multiplier if not provided or to ensure sync with Python
        let m = if self.heartbeat_m > 0 {
            self.heartbeat_m
        } else {
            65536u32.checked_div(self.spontaneous_firing_period_ticks).unwrap_or(0)
        };

        crate::layout::VariantParameters {
            threshold: self.threshold,
            rest_potential: self.rest_potential,
            leak_shift: self.leak_shift, // [DOD FIX]
            homeostasis_penalty: self.homeostasis_penalty,
            spontaneous_firing_period_ticks: self.spontaneous_firing_period_ticks,
            initial_synapse_weight: self.initial_synapse_weight,
            gsop_potentiation: self.gsop_potentiation,
            gsop_depression: self.gsop_depression,
            homeostasis_decay: self.homeostasis_decay,
            refractory_period: self.refractory_period,
            synapse_refractory_period: self.synapse_refractory_period,
            signal_propagation_length: self.signal_propagation_length,
            is_inhibitory: self.is_inhibitory as u8,
            inertia_curve: self.inertia_curve, // [DOD FIX]
            ahp_amplitude: self.ahp_amplitude, // [DOD FIX]
            _pad: [0; 6], // [DOD FIX] Maintain C-ABI size
            adaptive_leak_min_shift: self.adaptive_leak_min_shift, // [DOD FIX]
            adaptive_leak_gain: self.adaptive_leak_gain,
            adaptive_mode: self.adaptive_mode,
            _leak_pad: [0; 3],
            d1_affinity: self.d1_affinity,
            d2_affinity: self.d2_affinity,
            heartbeat_m: m,
        }
    }
}

use crate::config::brain::SimulationConfigRef;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ZoneManifest {
    pub magic: u32,
    #[serde(default)]
    pub depart_id_v1: Option<crate::config::sys::SystemMeta>,
    pub zone_hash: u32,
    pub blueprints_path: String,
    pub simulation: Option<SimulationConfigRef>,
    pub memory: ManifestMemory,
    pub network: ManifestNetwork,
    #[serde(default)]
    pub settings: ManifestSettings,
    pub variants: Vec<ManifestVariant>,
    #[serde(default)]
    pub connections: Vec<ManifestConnection>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ManifestSettings {
    pub night_interval_ticks: u64,
    pub save_checkpoints_interval_ticks: u64,
    #[serde(default)]
    pub plasticity: ManifestPlasticity,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestPlasticity {
    pub prune_threshold: i16,
    #[serde(default = "default_max_sprouts")]
    pub max_sprouts: u16,
}

fn default_max_sprouts() -> u16 {
    4
}

impl Default for ManifestPlasticity {
    fn default() -> Self {
        Self {
            prune_threshold: 15,
            max_sprouts: default_max_sprouts(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestConnection {
    #[serde(default)]
    pub conn_id_v1: Option<crate::config::sys::SystemMeta>,
    pub from: String,
    pub to: String,
    pub width: Option<u16>,
    pub height: Option<u16>,
}

/// [DOD] Global manifest of the entire model (Grandfather).
/// Contains links to department manifests (Fathers).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelManifest {
    pub magic: u32, // 0x4D4F444C "MODL"
    #[serde(default)]
    pub model_id_v1: Option<crate::config::sys::SystemMeta>,
    pub departments: std::collections::HashMap<String, String>, // Name -> Path to Department Folder
    pub connections: Vec<ManifestConnection>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestMemory {
    pub padded_n: usize,
    pub virtual_axons: usize,
    #[serde(default)]
    pub injected_ghosts: u32,
    pub ghost_capacity: usize,
    pub v_seg: u16,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestNetwork {
    pub slow_path_tcp: u16,
    pub external_udp_in: u16,
    pub external_udp_out: u16,
    #[serde(default)]
    pub external_udp_out_target: Option<String>,
    pub fast_path_udp_local: u16,
    pub fast_path_peers: std::collections::HashMap<String, String>, // [DOD FIX] N-Zone Routing
}
