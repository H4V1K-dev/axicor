use serde::{Deserialize, Serialize};
// use crate::layout::VariantParameters;
// use std::fmt;

/// 1. DTO: Читается из manifest.toml. Содержит String, живет в куче (Heap).
/// Ни в коем случае не передается по сырому указателю в C++!
fn default_affinity() -> u8 { 128 }

/// 1. DTO: Читается из manifest.toml.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestVariant {
    pub id: u8,
    pub name: String,
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_rate: i32,
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: i32,
    pub gsop_potentiation: i32,
    pub gsop_depression: i32,
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,
    pub slot_decay_ltm: u8,
    pub slot_decay_wm: u8,
    pub signal_propagation_length: u8,
    #[serde(default = "default_ltm_slot_count")]
    pub ltm_slot_count: u8,
    #[serde(default = "default_inertia_curve")]
    pub inertia_curve: [i16; 15],
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: i16,
    #[serde(default)]
    pub heartbeat_m: u16,
    
    // [DOD FIX] Новые поля рецепторов
    #[serde(default = "default_affinity")]
    pub d1_affinity: u8,
    #[serde(default = "default_affinity")]
    pub d2_affinity: u8,
}

fn default_prune_threshold() -> i16 { 15 }
fn default_ltm_slot_count() -> u8 { 80 }
fn default_inertia_curve() -> [i16; 15] {
    [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]
}

impl ManifestVariant {
    /// Zero-cost конвертация в строгий C-ABI
    pub fn into_gpu(self) -> crate::layout::VariantParameters {
        crate::layout::VariantParameters {
            threshold: self.threshold,
            rest_potential: self.rest_potential,
            leak_rate: self.leak_rate,
            homeostasis_penalty: self.homeostasis_penalty,
            homeostasis_decay: self.homeostasis_decay as u16,
            gsop_potentiation: self.gsop_potentiation as i16,
            gsop_depression: self.gsop_depression as i16,
            refractory_period: self.refractory_period,
            synapse_refractory_period: self.synapse_refractory_period,
            slot_decay_ltm: self.slot_decay_ltm,
            slot_decay_wm: self.slot_decay_wm,
            signal_propagation_length: self.signal_propagation_length,
            d1_affinity: self.d1_affinity,
            heartbeat_m: self.heartbeat_m,
            d2_affinity: self.d2_affinity,
            ltm_slot_count: self.ltm_slot_count,
            inertia_curve: {
                let mut curve = [0i16; 15];
                for i in 0..15.min(self.inertia_curve.len()) {
                    curve[i] = self.inertia_curve[i];
                }
                curve
            },
            prune_threshold: self.prune_threshold,
        }
    }
}

use crate::config::brain::SimulationConfigRef;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ZoneManifest {
    pub magic: u32,
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
}

impl Default for ManifestPlasticity {
    fn default() -> Self {
        Self { prune_threshold: 15 }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestConnection {
    pub from: String,
    pub to: String,
    pub width: Option<u16>,
    pub height: Option<u16>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestMemory {
    pub padded_n: usize,
    pub virtual_axons: usize,
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
