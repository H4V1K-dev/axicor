use crate::parser::{anatomy::Anatomy, blueprints::Blueprints, simulation::SimulationConfig};
use anyhow::bail;

/// Запускает все проверки конфигурации.
/// Возвращает Ok(()) или первую критическую ошибку.
pub fn validate_all(
    sim: &SimulationConfig,
    blueprints: &Blueprints,
    anatomy: &Anatomy,
) -> anyhow::Result<()> {
    check_v_seg_divisible(sim)?;
    check_layer_heights(anatomy)?;
    check_layer_populations(anatomy)?;
    check_sprouting_weights(blueprints)?;
    check_composition_quotas(anatomy)?;
    check_propagation_covers_v_seg(sim, blueprints)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// §1.6 — v_seg делимость — делегат в genesis_core::physics
// ---------------------------------------------------------------------------

/// Проверяет инвариант §1.6 через `genesis_core::physics::compute_derived_physics`.
/// Физика живёт в core, baker только транслирует конфиг и пробрасывает ошибку.
pub fn check_v_seg_divisible(sim: &SimulationConfig) -> anyhow::Result<()> {
    genesis_core::physics::compute_derived_physics(
        sim.simulation.signal_speed_um_tick as u32,
        sim.simulation.voxel_size_um,
        sim.simulation.segment_length_voxels,
    )
    .map(|_| ())
    .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Проверяет инвариант §1.1 — propagation_length >= v_seg.
pub fn check_propagation_covers_v_seg(sim: &SimulationConfig, blueprints: &Blueprints) -> anyhow::Result<()> {
    let physics = genesis_core::physics::compute_derived_physics(
        sim.simulation.signal_speed_um_tick as u32,
        sim.simulation.voxel_size_um,
        sim.simulation.segment_length_voxels,
    ).map_err(|e| anyhow::anyhow!("{e}"))?;
    
    let v_seg = physics.v_seg;
    for nt in &blueprints.neuron_types {
        if (nt.signal_propagation_length as u32) < v_seg {
            bail!(
                "blueprints.toml: NeuronType '{}' нарушает §1.1 Invariant.\n\
                 signal_propagation_length ({}) < v_seg ({})!\n\
                 Это приведёт к 'пропускам' сегментов при движении аксона (мертвые синапсы).",
                nt.name,
                nt.signal_propagation_length,
                v_seg
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// anatomy.toml — суммы height_pct и population_pct
// ---------------------------------------------------------------------------

/// Сумма `height_pct` всех слоёв должна быть ≈ 1.0.
/// Биологический инвариант: слои покрывают всю высоту зоны.
pub fn check_layer_heights(anatomy: &Anatomy) -> anyhow::Result<()> {
    let sum: f32 = anatomy.layers.iter().map(|l| l.height_pct).sum();
    if (sum - 1.0).abs() > 0.001 {
        bail!(
            "anatomy.toml: Σ(layer.height_pct) = {:.4} ≠ 1.0 (±0.01).\n\
             Слои обязаны покрывать всю высоту зоны без перекрытий и пробелов.",
            sum
        );
    }
    Ok(())
}

/// Сумма `population_pct` всех слоёв должна быть ≈ 1.0.
pub fn check_layer_populations(anatomy: &Anatomy) -> anyhow::Result<()> {
    let sum: f32 = anatomy.layers.iter().map(|l| l.population_pct).sum();
    if (sum - 1.0).abs() > 0.001 {
        bail!(
            "anatomy.toml: Σ(layer.population_pct) = {:.4} ≠ 1.0 (±0.01).\n\
             Бюджет нейронов должен быть распределён полностью.",
            sum
        );
    }
    Ok(())
}

/// Сумма весов composition каждого слоя должна быть ≈ 1.0.
pub fn check_composition_quotas(anatomy: &Anatomy) -> anyhow::Result<()> {
    for layer in &anatomy.layers {
        let sum: f32 = layer.composition.values().sum();
        if (sum - 1.0).abs() > 0.01 {
            bail!(
                "anatomy.toml: Layer '{}' Σ(composition) = {:.4} ≠ 1.0 (±0.01).\n\
                 Квоты типов нейронов в слое обязаны суммироваться в 1.0.",
                layer.name,
                sum
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// blueprints.toml — sprouting weights
// ---------------------------------------------------------------------------

/// Сумма sprouting weights каждого типа должна быть ≈ 1.0.
/// (04_connectivity.md §1.6.1: weight_distance + weight_power + weight_explore)
pub fn check_sprouting_weights(blueprints: &Blueprints) -> anyhow::Result<()> {
    for nt in &blueprints.neuron_types {
        let sum = nt.sprouting_weight_sum();
        let max_range = nt.axon_growth_step as f32 * 1000.0; // dummy heuristic
        if (sum - 1.0).abs() > 0.02 {
            bail!(
                "blueprints.toml: NeuronType '{}' sprouting weights sum = {:.4} ≠ 1.0 (±0.02).\n\
                 (weight_distance + weight_power + weight_explore должны суммироваться в 1.0)",
                nt.name,
                sum
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "test_checks.rs"]
mod test_checks;
