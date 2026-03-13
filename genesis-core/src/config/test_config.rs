/// Тесты для единого конфига `simulation.toml`.
use super::*;
// use std::path::PathBuf;

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
        voxel_size_um = 25.0
        signal_speed_m_s = 0.5
        sync_batch_ticks = 10
    "#;

    let cfg = SimulationConfig::parse(toml).unwrap();
    // Проверяем дефолты
    assert_eq!(cfg.simulation.segment_length_voxels, 5);
    assert_eq!(cfg.simulation.axon_growth_max_steps, 2000);
}

#[test]
fn test_missing_required_fields() {
    let toml = r#"
        [world]
        width_um = 100
    "#; // Нет секции simulation и других обязательных полей
    assert!(SimulationConfig::parse(toml).is_err());
}
