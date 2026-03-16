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

        let mut inertia = [0i16; 15];
        for (idx, &val) in nt.inertia_curve.iter().enumerate() {
            if idx < 15 {
                inertia[idx] = val as i16;
            }
        }

        let heartbeat_m = if nt.spontaneous_firing_period_ticks == 0 {
            0
        } else {
            (65536u32 / nt.spontaneous_firing_period_ticks.max(1)).clamp(1, 65535) as u16
        };

        // [DOD FIX] Алгоритмическая деривация D1/D2 рецепторов
        let (d1_affinity, d2_affinity) = if nt.is_inhibitory {
            (0u8, 0u8)
        } else {
            let mut d1_f: f32 = 128.0;
            let mut d2_f: f32 = 128.0;

            if nt.spontaneous_firing_period_ticks > 0 {
                let explorer_mod = (500.0 / nt.spontaneous_firing_period_ticks as f32).clamp(1.0, 2.0);
                d1_f *= explorer_mod;
                d2_f /= explorer_mod.sqrt();
            }

            let thresh_mod = (nt.threshold as f32 / 40000.0).clamp(0.5, 1.5);
            d1_f *= thresh_mod;

            let proj_mod = (nt.signal_propagation_length as f32 / 10.0).clamp(0.5, 2.0);
            d2_f *= proj_mod;

            (
                d1_f.round().clamp(0.0, 255.0) as u8,
                d2_f.round().clamp(0.0, 255.0) as u8,
            )
        };

        memory.variants[i] = VariantParameters {
            threshold: nt.threshold,
            rest_potential: nt.rest_potential,
            leak_rate: nt.leak_rate,
            homeostasis_penalty: nt.homeostasis_penalty,
            homeostasis_decay: nt.homeostasis_decay as u16,
            gsop_potentiation: nt.gsop_potentiation as i16,
            gsop_depression: nt.gsop_depression as i16,
            refractory_period: nt.refractory_period,
            synapse_refractory_period: nt.synapse_refractory_period,
            slot_decay_ltm: nt.slot_decay_ltm as u8,
            slot_decay_wm: nt.slot_decay_wm as u8,
            signal_propagation_length: nt.signal_propagation_length as u8,
            d1_affinity,
            heartbeat_m,
            d2_affinity,
            ltm_slot_count: nt.ltm_slot_count,
            inertia_curve: inertia,
            prune_threshold: nt.prune_threshold,
            adaptive_leak_mode: nt.adaptive_leak_mode,
            _pad_adaptive: 0,
            dopamine_leak_gain: nt.dopamine_leak_gain,
            burst_leak_gain: nt.burst_leak_gain,
            leak_min: nt.leak_min,
            leak_max: nt.leak_max,
            _reserved: [0; 6],
        };
    }

    (memory, config.neuron_types, name_map)
}
