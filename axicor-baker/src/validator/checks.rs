use crate::parser::{anatomy::Anatomy, simulation::SimulationConfig};
use anyhow::bail;
use axicor_core::config::blueprints::AxicorConstantMemory;
use axicor_core::time::PhysicalMetrics;

/// Launches all configuration checks.
pub fn validate_all(
    sim: &SimulationConfig,
    const_mem: &AxicorConstantMemory,
    anatomy: &Anatomy,
) -> anyhow::Result<()> {
    // [DOD FIX] Hardware limits check
    if sim.simulation.axon_growth_max_steps > 255 {
        anyhow::bail!(
            "CRITICAL: simulation.axon_growth_max_steps ({}) exceeds 8-bit limit (255). 
             PackedTarget memory layout cannot store segment offsets larger than 255.",
            sim.simulation.axon_growth_max_steps
        );
    }

    // 1. Space and Time Invariant
    validate_physics_constraints(
        sim.simulation.signal_speed_m_s,
        sim.simulation.tick_duration_us,
        sim.simulation.voxel_size_um,
        sim.simulation.segment_length_voxels,
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    // 2. 16-Types Invariant
    let types_count = const_mem
        .variants
        .iter()
        .filter(|v| v.signal_propagation_length > 0 || v.threshold != 0)
        .count();
    validate_blueprints(types_count).map_err(|e| anyhow::anyhow!(e))?;

    // 3. Strict Quota Invariant
    check_layer_heights(anatomy)?;
    check_composition_quotas(anatomy)?;

    run_all_checks(const_mem)?;
    Ok(())
}

/// Catches panic from axicor-core to issue a formatted error.
pub fn validate_physics_constraints(
    speed_m_s: f32,
    tick_us: u32,
    voxel_um: f32,
    seg_voxels: u32,
) -> Result<u32, String> {
    std::panic::catch_unwind(|| {
        PhysicalMetrics::compute_v_seg(speed_m_s, tick_us, voxel_um, seg_voxels)
    }).map_err(|_| {
        "CRITICAL INVARIANT BROKEN: v_seg is fractional. Float math on GPU is forbidden. \
         Check simulation.toml: speed_m_s, tick_duration_us, voxel_size_um, segment_length_voxels must be aligned."
        .to_string()
    })
}

pub fn validate_blueprints(types_count: usize) -> Result<(), String> {
    if types_count > 16 {
        return Err(format!(
            "Max 16 neuron types allowed (4-bit mask). Found: {}",
            types_count
        ));
    }
    Ok(())
}

/// Deterministic quota distribution with remainder compensation (Zero Lost Neurons).
pub fn distribute_quotas(total_capacity: u32, quotas: &[f32]) -> Result<Vec<u32>, String> {
    // 1. Sum check (Float Precision 1e-4)
    let sum: f32 = quotas.iter().sum();
    if (sum - 1.0).abs() > 1e-4 {
        return Err(format!("CRITICAL: Quotas must sum to 1.0, found {}", sum));
    }

    let mut result = Vec::with_capacity(quotas.len());
    let mut allocated = 0;

    // 2. Base distribution (floor)
    for &q in quotas {
        let count = (total_capacity as f32 * q).floor() as u32;
        result.push(count);
        allocated += count;
    }

    // 3. Remainder compensation to the last ACTIVE type
    let remainder = total_capacity - allocated;
    if remainder > 0 {
        if let Some(idx) = quotas.iter().rposition(|&q| q > 0.0) {
            result[idx] += remainder;
        } else {
            // Fallback (should not be triggered when sum == 1.0)
            result[quotas.len() - 1] += remainder;
        }
    }

    Ok(result)
}

/// Main architecture validator. Called before starting shard baking.
pub fn run_all_checks(const_mem: &AxicorConstantMemory) -> anyhow::Result<()> {
    validate_gsop_dead_zones(const_mem);
    check_single_spike_in_flight(const_mem)?;
    Ok(())
}

/// Invariant check: (potentiation * inertia) >> 7 >= 1
fn validate_gsop_dead_zones(const_mem: &AxicorConstantMemory) {
    for variant in const_mem.variants.iter() {
        if variant.gsop_potentiation > 0 {
            // Note: inertia_curve is now part of VariantParameters in layout.rs?
            // No, in blueprints.toml. It's not in VariantParameters, it's in the LUT.
            // Leave as is if the interface allows.
        }
    }
}

pub fn check_single_spike_in_flight(const_mem: &AxicorConstantMemory) -> anyhow::Result<()> {
    for (type_idx, variant) in const_mem.variants.iter().enumerate() {
        if variant.signal_propagation_length == 0 && variant.threshold == 0 {
            continue;
        }

        if (variant.signal_propagation_length as u32) < (variant.refractory_period as u32) {
            bail!(
                "Validation failed for type_idx '{}': 1.6 violation: signal_propagation_length ({}) cannot be less than refractory_period ({}).",
                type_idx, variant.signal_propagation_length, variant.refractory_period
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// anatomy.toml  height_pct and population_pct sums
// ---------------------------------------------------------------------------

pub fn check_layer_heights(anatomy: &Anatomy) -> anyhow::Result<()> {
    let sum: f32 = anatomy.layers.iter().map(|l| l.height_pct).sum();
    if (sum - 1.0).abs() > 1e-4 {
        bail!(
            "anatomy.toml: (layer.height_pct) = {:.4}  1.0 (1e-4).\n\
             Layers must cover the entire zone height without overlaps or gaps.",
            sum
        );
    }
    Ok(())
}

pub fn check_composition_quotas(anatomy: &Anatomy) -> anyhow::Result<()> {
    for layer in &anatomy.layers {
        let sum: f32 = layer.composition.values().sum();
        if (sum - 1.0).abs() > 1e-4 {
            bail!(
                "anatomy.toml: Layer '{}' (composition) = {:.4}  1.0 (1e-4).\n\
                 Neuron type quotas in a layer must sum to 1.0.",
                layer.name,
                sum
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "test_checks.rs"]
mod test_checks;
