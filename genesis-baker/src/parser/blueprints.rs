use genesis_core::config::blueprints::{GenesisConstantMemory, BlueprintsConfig, NeuronType};
use genesis_core::layout::{VariantParameters};
use std::collections::HashMap;

pub fn parse_blueprints(toml_content: &str) -> (GenesisConstantMemory, Vec<NeuronType>, HashMap<String, u8>) {
    let config = BlueprintsConfig::parse(toml_content)
        .expect("Fatal: Failed to parse blueprints.toml");

    let num_types = config.neuron_types.len();
    if num_types > 16 {
        panic!(
            "Fatal: Architecture hard limit exceeded. Max 16 neuron types allowed, got {}. \
            (4-bit type_mask index constraint)", 
            num_types
        );
    }

    let mut memory = GenesisConstantMemory {
        variants: [VariantParameters::default(); 16],
    };
    let mut name_map: HashMap<String, u8> = HashMap::new();

    for (i, nt) in config.neuron_types.iter().enumerate() {
        // [!IMPORTANT] Валидация GSOP Dead Zone
        if nt.gsop_potentiation > 0 {
            for (rank, &inertia) in nt.inertia_curve.iter().enumerate() {
                let effective_pot = (nt.gsop_potentiation as i32 * inertia as i32) >> 7;
                assert!(
                    effective_pot >= 1,
                    "Validation failed for type '{}': inertia_curve[{}] creates a GSOP dead zone.",
                    nt.name, rank
                );
            }
        }

        name_map.insert(nt.name.clone(), i as u8);

        let mut inertia = [0i16; 16];
        for (idx, &val) in nt.inertia_curve.iter().enumerate() {
            inertia[idx] = val as i16;
        }

        memory.variants[i] = VariantParameters {
            threshold: nt.threshold,
            rest_potential: nt.rest_potential,
            leak_rate: nt.leak_rate,
            homeostasis_penalty: nt.homeostasis_penalty,
            gsop_potentiation: nt.gsop_potentiation as i16,
            gsop_depression: nt.gsop_depression as i16,
            homeostasis_decay: nt.homeostasis_decay as u16,
            signal_propagation_length: nt.signal_propagation_length as u8,
            conduction_velocity: nt.conduction_velocity as u8,
            slot_decay_ltm: nt.slot_decay_ltm as u8,
            slot_decay_wm: nt.slot_decay_wm as u8,
            refractory_period: nt.refractory_period,
            synapse_refractory_period: nt.synapse_refractory_period,
            inertia_curve: inertia,
            ..VariantParameters::default()
        };
    }

    (memory, config.neuron_types, name_map)
}

