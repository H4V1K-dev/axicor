/// Тесты для единого конфига `simulation.toml`.
use super::*;
use std::path::PathBuf;

#[test]
fn test_default_values() {
    let toml = r#"
        [world]
        width_um = 100
        depth_um = 100
        height_um = 100

        [simulation]
        tick_duration_us = 100
        total_ticks = 1000
        master_seed = "TEST"
        global_density = 0.05
        voxel_size_um = 25
        signal_speed_um_tick = 50
        sync_batch_ticks = 10
    "#;

    let cfg = SimulationConfig::parse(toml).unwrap();
    // Проверяем дефолты
    assert_eq!(cfg.simulation.segment_length_voxels, 5);
    assert_eq!(cfg.simulation.axon_growth_max_steps, 2000);
    assert_eq!(cfg.simulation.night_interval_ticks, 0);
    assert_eq!(cfg.simulation.num_virtual_axons, None);
}

#[test]
fn test_neuron_budget_calc() {
    let toml = r#"
        [world]
        width_um = 100
        depth_um = 100
        height_um = 100

        [simulation]
        voxel_size_um = 25
        global_density = 0.5
        # ... обязательные поля
        tick_duration_us = 100
        total_ticks = 1000
        master_seed = "TEST"
        signal_speed_um_tick = 50
        sync_batch_ticks = 10
    "#;

    let cfg = SimulationConfig::parse(toml).unwrap();
    // width 100/25 = 4
    // total voxels = 4*4*4 = 64
    assert_eq!(cfg.total_voxels(), 64);
    // budget = 64 * 0.5 = 32
    assert_eq!(cfg.neuron_budget(), 32);
}

#[test]
fn test_missing_required_fields() {
    let toml = r#"
        [world]
        width_um = 100
    "#; // Нет секции simulation и других обязательных полей
    assert!(SimulationConfig::parse(toml).is_err());
}
