use axicor_core::config::blueprints::{BlueprintsConfig, AxicorConstantMemory, NeuronType};
use axicor_core::layout::VariantParameters;
use std::collections::HashMap;

/// Full node bootstrap sequence. Standard "Axicor Sequence" pipeline.
pub fn parse_blueprints(
    toml_content: &str,
) -> (AxicorConstantMemory, Vec<NeuronType>, HashMap<String, u8>) {
    let config =
        BlueprintsConfig::parse(toml_content).expect("Fatal: Failed to parse blueprints.toml");

    let num_types = config.neuron_types.len();
    if num_types > 16 {
        panic!(
            "Fatal: Architecture hard limit exceeded. Max 16 neuron types allowed, got {}. \
            (4-bit type_mask index constraint)",
            num_types
        );
    }

    let mut memory = AxicorConstantMemory {
        variants: [VariantParameters::default(); 16],
    };
    let mut name_map: HashMap<String, u8> = HashMap::new();

    for (i, nt) in config.neuron_types.iter().enumerate() {
        name_map.insert(nt.name.clone(), i as u8);

        memory.variants[i] = VariantParameters {
            threshold: nt.threshold,
            rest_potential: nt.rest_potential,
            leak_shift: nt.leak_shift,
            homeostasis_penalty: nt.homeostasis_penalty,
            spontaneous_firing_period_ticks: nt.spontaneous_firing_period_ticks,
            initial_synapse_weight: nt.initial_synapse_weight,
            gsop_potentiation: nt.gsop_potentiation,
            gsop_depression: nt.gsop_depression,
            homeostasis_decay: nt.homeostasis_decay,
            refractory_period: nt.refractory_period,
            synapse_refractory_period: nt.synapse_refractory_period,
            signal_propagation_length: nt.signal_propagation_length,
            is_inhibitory: nt.is_inhibitory as u8,
            inertia_curve: {
                let mut curve = [0u8; 8];
                for (idx, &val) in nt.inertia_curve.iter().enumerate().take(8) {
                    curve[idx] = val;
                }
                curve
            },
            ahp_amplitude: nt.ahp_amplitude,
            _pad: [0; 6],
            adaptive_leak_min_shift: nt.adaptive_leak_min_shift,
            adaptive_leak_gain: nt.adaptive_leak_gain,
            adaptive_mode: nt.adaptive_mode,
            _leak_pad: [0; 3],
            d1_affinity: nt.d1_affinity,
            d2_affinity: nt.d2_affinity,
            // [DOD FIX] Zero-cost: предвычисляем DDS-фазу при компиляции, чтобы избавить GPU от деления
            heartbeat_m: 65536u32.checked_div(nt.spontaneous_firing_period_ticks).unwrap_or(0),
        };
    }

    (memory, config.neuron_types, name_map)
}
