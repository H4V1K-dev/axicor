use crate::{Runtime, VariantParameters, GenesisConstantMemory};
use genesis_core::config::{BlueprintsConfig, InstanceConfig};

pub struct ZoneRuntime {
    pub name: String,
    pub runtime: Runtime,
    pub const_mem: GenesisConstantMemory,
    pub config: InstanceConfig,
    pub prune_threshold: i16,
}

impl ZoneRuntime {
    pub fn build_constant_memory(blueprints: &BlueprintsConfig) -> GenesisConstantMemory {
        let mut const_mem = GenesisConstantMemory::default();
        for (i, nt) in blueprints.neuron_types.iter().take(16).enumerate() {
            const_mem.variants[i] = VariantParameters {
                threshold:            nt.threshold,
                rest_potential:       nt.rest_potential,
                leak:                 nt.leak_rate,
                homeostasis_penalty:  nt.homeostasis_penalty,
                homeostasis_decay:    nt.homeostasis_decay as i32,
                gsop_potentiation:    nt.gsop_potentiation,
                gsop_depression:      nt.gsop_depression,
                refractory_period:    nt.refractory_period,
                synapse_refractory:   nt.synapse_refractory_period,
                slot_decay_ltm:       nt.slot_decay_ltm,
                slot_decay_wm:        nt.slot_decay_wm,
                propagation_length:   nt.signal_propagation_length,
                ltm_slot_count:       nt.ltm_slot_count,
                inertia_curve:        nt.inertia_curve,
                _padding:             [0; 14],
            };
        }
        const_mem
    }
}
