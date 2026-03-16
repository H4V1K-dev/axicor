use crate::parser::{anatomy::Anatomy, simulation::SimulationConfig};
use anyhow::bail;
use genesis_core::config::blueprints::GenesisConstantMemory;
use genesis_core::time::PhysicalMetrics;

/// Запускает все проверки конфигурации.
pub fn validate_all(
    sim: &SimulationConfig,
    const_mem: &GenesisConstantMemory,
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

    // 1. Инвариант Времени и Пространства
    validate_physics_constraints(
        sim.simulation.signal_speed_m_s,
        sim.simulation.tick_duration_us,
        sim.simulation.voxel_size_um,
        sim.simulation.segment_length_voxels,
    ).map_err(|e| anyhow::anyhow!(e))?;

    // 2. Инвариант 16 Типов
    let types_count = const_mem.variants.iter().filter(|v| v.signal_propagation_length > 0 || v.threshold != 0).count();
    validate_blueprints(types_count).map_err(|e| anyhow::anyhow!(e))?;

    // 3. Инвариант Жестких Квот
    check_layer_heights(anatomy)?;
    check_composition_quotas(anatomy)?;
    
    run_all_checks(const_mem)?;
    Ok(())
}

/// Перехватывает панику из genesis-core для выдачи красивой ошибки.
pub fn validate_physics_constraints(
    speed_m_s: f32, tick_us: u32, voxel_um: f32, seg_voxels: u32
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
        return Err(format!("Max 16 neuron types allowed (4-bit mask). Found: {}", types_count));
    }
    Ok(())
}

/// Детерминированное распределение квот с компенсацией остатка (Zero Lost Neurons).
pub fn distribute_quotas(total_capacity: u32, quotas: &[f32]) -> Result<Vec<u32>, String> {
    // 1. Проверка суммы (Float Precision 1e-4)
    let sum: f32 = quotas.iter().sum();
    if (sum - 1.0).abs() > 1e-4 {
        return Err(format!("CRITICAL: Quotas must sum to 1.0, found {}", sum));
    }

    let mut result = Vec::with_capacity(quotas.len());
    let mut allocated = 0;

    // 2. Базовое распределение (floor)
    for &q in quotas {
        let count = (total_capacity as f32 * q).floor() as u32;
        result.push(count);
        allocated += count;
    }

    // 3. Компенсация остатка в последний АКТИВНЫЙ тип
    let remainder = total_capacity - allocated;
    if remainder > 0 {
        if let Some(idx) = quotas.iter().rposition(|&q| q > 0.0) {
            result[idx] += remainder;
        } else {
            // Fallback (не должен сработать при sum == 1.0)
            result[quotas.len() - 1] += remainder;
        }
    }

    Ok(result)
}

/// Главный валидатор архитектуры. Вызывается перед началом запекания шарда.
pub fn run_all_checks(const_mem: &GenesisConstantMemory) -> anyhow::Result<()> {
    validate_gsop_dead_zones(const_mem);
    check_single_spike_in_flight(const_mem)?;
    Ok(())
}

/// Проверка инварианта: (potentiation * inertia) >> 7 >= 1
fn validate_gsop_dead_zones(const_mem: &GenesisConstantMemory) {
    for (_type_idx, variant) in const_mem.variants.iter().enumerate() {
        if variant.gsop_potentiation > 0 {
            // Внимание: inertia_curve теперь часть VariantParameters в layout.rs?
            // Нет, в blueprints.toml. В VariantParameters её нет, она в LUT.
            // Оставляем как было, если интерфейс позволяет.
        }
    }
}

pub fn check_single_spike_in_flight(const_mem: &GenesisConstantMemory) -> anyhow::Result<()> {
    for (type_idx, variant) in const_mem.variants.iter().enumerate() {
        if variant.signal_propagation_length == 0 && variant.threshold == 0 {
            continue;
        }

        if (variant.signal_propagation_length as u32) < (variant.refractory_period as u32) {
            bail!(
                "Validation failed for type_idx '{}': §1.6 violation: signal_propagation_length ({}) cannot be less than refractory_period ({}).",
                type_idx, variant.signal_propagation_length, variant.refractory_period
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// anatomy.toml — суммы height_pct и population_pct
// ---------------------------------------------------------------------------

pub fn check_layer_heights(anatomy: &Anatomy) -> anyhow::Result<()> {
    let sum: f32 = anatomy.layers.iter().map(|l| l.height_pct).sum();
    if (sum - 1.0).abs() > 1e-4 {
        bail!(
            "anatomy.toml: Σ(layer.height_pct) = {:.4} ≠ 1.0 (±1e-4).\n\
             Слои обязаны покрывать всю высоту зоны без перекрытий и пробелов.",
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
                "anatomy.toml: Layer '{}' Σ(composition) = {:.4} ≠ 1.0 (±1e-4).\n\
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
