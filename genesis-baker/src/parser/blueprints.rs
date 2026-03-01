use genesis_core::config::blueprints::{GenesisConstantMemory, VariantParameters};
use std::collections::HashMap;
use serde::Deserialize;

/// DTO для парсинга TOML. Содержит String (name), который отбрасывается при упаковке для GPU.
#[derive(Deserialize, Debug)]
pub struct BlueprintsToml {
    #[serde(rename = "neuron_type")]
    pub neuron_types: Vec<NeuronTypeToml>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(default)]
pub struct NeuronTypeToml {
    pub name: String,
    
    // 1. Potentials & Homeostasis
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_rate: i32,
    pub homeostasis_penalty: i32,

    // 2. Plasticity & Geometry
    pub gsop_potentiation: i16,
    pub gsop_depression: i16,
    pub homeostasis_decay: u16,
    pub signal_propagation_length: u16,
    pub conduction_velocity: u16,
    pub slot_decay_ltm: u16,
    pub slot_decay_wm: u16,

    // 3. Timers
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,

    // 4. LUT Curves
    pub inertia_curve: [u8; 16],
}

pub fn parse_blueprints(toml_content: &str) -> (GenesisConstantMemory, HashMap<String, u8>) {
    let parsed: BlueprintsToml = toml::from_str(toml_content)
        .expect("Fatal: Failed to parse blueprints.toml");

    let num_types = parsed.neuron_types.len();
    if num_types > 16 {
        panic!(
            "Fatal: Architecture hard limit exceeded. Max 16 neuron types allowed, got {}. \
            (4-bit type_mask index constraint)", 
            num_types
        );
    }

    // Инициализируем нулями. Неиспользуемые типы (если их < 16) останутся пустыми,
    // что безопасно, так как анатомия не позволит аллоцировать нейроны несуществующего типа.
    let mut memory = GenesisConstantMemory {
        variants: [VariantParameters {
            threshold: 0, rest_potential: 0, leak_rate: 0, homeostasis_penalty: 0,
            gsop_potentiation: 0, gsop_depression: 0, homeostasis_decay: 0,
            signal_propagation_length: 0, conduction_velocity: 0,
            slot_decay_ltm: 0, slot_decay_wm: 0,
            refractory_period: 0, synapse_refractory_period: 0,
            inertia_curve: [0; 16], _reserved: [0; 16],
        }; 16],
    };
    let mut name_map: HashMap<String, u8> = HashMap::new();

    for (i, nt) in parsed.neuron_types.into_iter().enumerate() {
        // [!IMPORTANT] Валидация GSOP Dead Zone
        // Проверяем, что нелинейное сопротивление не убьет обучение намертво.
        if nt.gsop_potentiation > 0 {
            for (rank, &inertia) in nt.inertia_curve.iter().enumerate() {
                let effective_pot = (nt.gsop_potentiation as i32 * inertia as i32) >> 7;
                assert!(
                    effective_pot >= 1,
                    "Validation failed for type '{}': inertia_curve[{}] creates a GSOP dead zone. \
                    (potentiation * inertia) >> 7 must be >= 1. Got 0.",
                    nt.name, rank
                );
            }
        }

        name_map.insert(nt.name.clone(), i as u8);
        memory.variants[i] = VariantParameters {
            threshold: nt.threshold,
            rest_potential: nt.rest_potential,
            leak_rate: nt.leak_rate,
            homeostasis_penalty: nt.homeostasis_penalty,
            gsop_potentiation: nt.gsop_potentiation,
            gsop_depression: nt.gsop_depression,
            homeostasis_decay: nt.homeostasis_decay,
            signal_propagation_length: nt.signal_propagation_length,
            conduction_velocity: nt.conduction_velocity,
            slot_decay_ltm: nt.slot_decay_ltm,
            slot_decay_wm: nt.slot_decay_wm,
            refractory_period: nt.refractory_period,
            synapse_refractory_period: nt.synapse_refractory_period,
            inertia_curve: nt.inertia_curve,
            _reserved: [0; 16], // Гарантия отсутствия мусора
        };
    }

    (memory, name_map)
}

