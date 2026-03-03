use serde::{Deserialize, Serialize};
// use crate::layout::VariantParameters;
// use std::fmt;

/// 1. DTO: Читается из manifest.toml. Содержит String, живет в куче (Heap).
/// Ни в коем случае не передается по сырому указателю в C++!
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ManifestVariant {
    pub id: u8,
    pub name: String, // Теперь ты можешь писать name = "Pyramidal" в TOML
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_rate: i32,
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: u16,
    pub gsop_potentiation: i16,
    pub gsop_depression: i16,
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,
    pub slot_decay_ltm: u8,
    pub slot_decay_wm: u8,
    pub signal_propagation_length: u16,
}

/// 2. GPU Layout: Строгий C-формат. Отсутствуют ссылки и объекты кучи.
/// Общий размер 28 байт. Выравнивание (align) = 4. 
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct GpuVariantParameters {
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_rate: i32,
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: u16,
    pub gsop_potentiation: i16,
    pub gsop_depression: i16,
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,
    pub slot_decay_ltm: u8,
    pub slot_decay_wm: u8,
    pub signal_propagation_length: u16,
    pub _padding: u16, // Выравнивание до 28 байт
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
            _padding: 0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ZoneManifest {
    pub magic: u32,
    pub zone_hash: u32,
    pub memory: ManifestMemory,
    pub network: ManifestNetwork,
    pub variants: Vec<ManifestVariant>, // Вектор легален для TOML парсера
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
