use serde::Deserialize;
use genesis_core::layout::VariantParameters;

#[derive(Debug, Deserialize)]
pub struct TomlNeuronType {
    pub is_inhibitory: bool,
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_rate: i32,
    pub refractory_period: u8,
    pub spontaneous_firing_period_ticks: u32,
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: u16,
    pub synapse_refractory_period: u8,
    pub signal_propagation_length: u8,
    pub initial_synapse_weight: u16,
    pub gsop_potentiation: u16,
    pub gsop_depression: u16,
    pub inertia_curve: [u8; 16],
    pub adaptive_leak_max: i32,
    pub adaptive_leak_gain: u16,
    pub adaptive_mode: u8,
}

impl From<TomlNeuronType> for VariantParameters {
    fn from(dto: TomlNeuronType) -> Self {
        Self {
            threshold: dto.threshold,
            rest_potential: dto.rest_potential,
            leak_rate: dto.leak_rate,
            homeostasis_penalty: dto.homeostasis_penalty,
            spontaneous_firing_period_ticks: dto.spontaneous_firing_period_ticks,
            initial_synapse_weight: dto.initial_synapse_weight,
            gsop_potentiation: dto.gsop_potentiation,
            gsop_depression: dto.gsop_depression,
            homeostasis_decay: dto.homeostasis_decay,
            refractory_period: dto.refractory_period,
            synapse_refractory_period: dto.synapse_refractory_period,
            signal_propagation_length: dto.signal_propagation_length,
            is_inhibitory: dto.is_inhibitory as u8,
            inertia_curve: dto.inertia_curve,
            adaptive_leak_max: dto.adaptive_leak_max,
            adaptive_leak_gain: dto.adaptive_leak_gain,
            adaptive_mode: dto.adaptive_mode,
            _leak_pad: [0; 3],
            _pad: [0; 6],
        }
    }
}
