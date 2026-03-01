use crate::parser::{anatomy::Anatomy, simulation::SimulationConfig};
use anyhow::bail;
use genesis_core::config::blueprints::GenesisConstantMemory;

/// Запускает все проверки конфигурации.
/// Возвращает Ok(()) или первую критическую ошибку.
pub fn validate_all(
    sim: &SimulationConfig,
    const_mem: &GenesisConstantMemory,
    anatomy: &Anatomy,
) -> anyhow::Result<()> {
    check_v_seg_divisible(sim)?;
    check_layer_heights(anatomy)?;
    check_layer_populations(anatomy)?;
    check_composition_quotas(anatomy)?;
    run_all_checks(const_mem);
    Ok(())
}

/// Главный валидатор архитектуры. Вызывается перед началом запекания шарда.
pub fn run_all_checks(const_mem: &GenesisConstantMemory) {
    validate_gsop_dead_zones(const_mem);
    validate_refractory_physics(const_mem);
}

/// Проверка инварианта: (potentiation * inertia) >> 7 >= 1
/// Защита от вечной заморозки синапсов (Мёртвая зона пластичности).
fn validate_gsop_dead_zones(const_mem: &GenesisConstantMemory) {
    for (type_idx, variant) in const_mem.variants.iter().enumerate() {
        if variant.gsop_potentiation > 0 {
            for (rank, &inertia) in variant.inertia_curve.iter().enumerate() {
                let effective_pot = (variant.gsop_potentiation as i32 * inertia as i32) >> 7;
                assert!(
                    effective_pot >= 1,
                    "Validation failed for type_idx '{}': inertia_curve[{}] creates a GSOP dead zone. \
                    (potentiation * inertia) >> 7 must be >= 1. Got 0.",
                    type_idx, rank
                );
            }
        }
    }
}

/// Проверка целочисленной физики
fn validate_refractory_physics(const_mem: &GenesisConstantMemory) {
    for (type_idx, variant) in const_mem.variants.iter().enumerate() {
        // Если сигнал идет по аксону быстрее, чем сома выходит из рефрактерности, 
        // мы нарушаем инвариант Single-Tick Pulse (получим 2 спайка в одном аксоне)
        
        // Skip validation for unused types (where parameters are 0)
        if variant.signal_propagation_length == 0 && variant.refractory_period == 0 {
            continue;
        }

        assert!(
            variant.signal_propagation_length >= variant.refractory_period as u16,
            "Validation failed for type_idx '{}': signal_propagation_length ({}) cannot be less than refractory_period ({}).",
            type_idx, variant.signal_propagation_length, variant.refractory_period
        );
    }
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

#[cfg(test)]
#[path = "test_checks.rs"]
mod test_checks;
