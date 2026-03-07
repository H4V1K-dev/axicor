use serde::{Deserialize, Serialize};
// use crate::layout::VariantParameters;
// use std::fmt;

/// 1. DTO: Читается из manifest.toml. Содержит String, живет в куче (Heap).
/// Ни в коем случае не передается по сырому указателю в C++!
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
    pub inertia_curve: [i16; 16],
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: i16,
    #[serde(default)]
    pub heartbeat_m: u16,
}

fn default_prune_threshold() -> i16 { 15 }

fn default_ltm_slot_count() -> u8 { 80 }
fn default_inertia_curve() -> [i16; 16] {
    [128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8]
}

/// 2. GPU Layout: Строгий C-формат. Отсутствуют ссылки и объекты кучи.
/// Общий размер 28 байт. Выравнивание (align) = 4. 
#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GpuVariantParameters {
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
    pub ltm_slot_count: u8,
    pub heartbeat_m: u16, // [DOD FIX] DDS Multiplier instead of _pad1
    pub inertia_curve: [i16; 16],
    pub prune_threshold: i16,
    pub _pad2a: [u8; 32],
    pub _pad2b: [u8; 26],
}

impl ManifestVariant {
    /// Zero-cost конвертация
    pub fn into_gpu(self) -> GpuVariantParameters {
        GpuVariantParameters {
            threshold: self.threshold,
            rest_potential: self.rest_potential,
            leak_rate: self.leak_rate,
            homeostasis_penalty: self.homeostasis_penalty,
            homeostasis_decay: self.homeostasis_decay,
            gsop_potentiation: self.gsop_potentiation,
            gsop_depression: self.gsop_depression,
            refractory_period: self.refractory_period,
            synapse_refractory_period: self.synapse_refractory_period,
            slot_decay_ltm: self.slot_decay_ltm,
            slot_decay_wm: self.slot_decay_wm,
            signal_propagation_length: self.signal_propagation_length,
            ltm_slot_count: self.ltm_slot_count,
            heartbeat_m: self.heartbeat_m, // Replaces _pad1
            inertia_curve: self.inertia_curve,
            prune_threshold: self.prune_threshold,
            _pad2a: [0; 32],
            _pad2b: [0; 26],
        }
    }
}

use crate::config::brain::SimulationConfigRef;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ZoneManifest {
    pub magic: u32,
    pub zone_hash: u32,
    pub simulation: Option<SimulationConfigRef>,
    pub memory: ManifestMemory,
    pub network: ManifestNetwork,
    pub variants: Vec<ManifestVariant>,
    #[serde(default)]
    pub connections: Vec<ManifestConnection>,
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
    pub fast_path_peers: Vec<String>,
}
