use serde::Deserialize;

/// Полный `simulation.toml`.
#[derive(Debug, Deserialize)]
pub struct SimulationConfig {
    pub world: World,
    pub simulation: Simulation,
}

/// [world] — Физические размеры пространства (в микрометрах).
#[derive(Debug, Deserialize)]
pub struct World {
    pub width_um: u32,
    pub depth_um: u32,
    pub height_um: u32,
}

/// [simulation] — Глобальные параметры симуляции.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct Simulation {
    /// Шаг времени в микросекундах. 100 = 0.1 мс.
    pub tick_duration_us: u32,
    /// Количество тиков в симуляции (может быть 0 = бесконечно).
    pub total_ticks: u64,
    /// Глобальный сид (строка, хэшируется в u64 при запуске).
    pub master_seed: String,
    /// Процент вокселей с телами нейронов (0.0..1.0).
    pub global_density: f32,
    /// Размер вокселя (квант пространства в микрометрах).
    pub voxel_size_um: u32,
    /// Скорость распространения сигнала (мкм/тик).
    pub signal_speed_um_tick: u16,
    /// Количество тиков автономного расчета между синхронизациями шардов.
    pub sync_batch_ticks: u32,
    /// Количество виртуальных аксонов (сетчатка). Опционально.
    pub num_virtual_axons: Option<u32>,
}

impl SimulationConfig {
    /// Общее число вокселей для заданного размера вокселя (в мкм).
    pub fn total_voxels(&self) -> u64 {
        let v_um = self.simulation.voxel_size_um;
        let w = (self.world.width_um / v_um) as u64;
        let d = (self.world.depth_um / v_um) as u64;
        let h = (self.world.height_um / v_um) as u64;
        w * d * h
    }

    /// Максимальное число нейронов = total_voxels * global_density.
    pub fn neuron_budget(&self) -> u64 {
        (self.total_voxels() as f64 * self.simulation.global_density as f64) as u64
    }
}

/// Парсит `simulation.toml` из строки.
pub fn parse(src: &str) -> anyhow::Result<SimulationConfig> {
    let cfg: SimulationConfig = toml::from_str(src)?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"
[world]
width_um  = 3500
depth_um  = 3500
height_um = 10250

[simulation]
tick_duration_us = 100
total_ticks      = 10000
master_seed      = "GENESIS"
global_density   = 0.04
voxel_size_um    = 25
signal_speed_um_tick = 50
sync_batch_ticks = 1000
"#;

    #[test]
    fn parse_simulation_example() {
        let cfg = parse(EXAMPLE).expect("parse failed");
        assert_eq!(cfg.simulation.tick_duration_us, 100);
        assert_eq!(cfg.simulation.master_seed, "GENESIS");
        assert!((cfg.simulation.global_density - 0.04).abs() < 1e-6);
        assert_eq!(cfg.simulation.voxel_size_um, 25);
        assert_eq!(cfg.simulation.signal_speed_um_tick, 50);
        assert_eq!(cfg.simulation.sync_batch_ticks, 1000);
    }

    #[test]
    fn neuron_budget_sanity() {
        let cfg = parse(EXAMPLE).expect("parse failed");
        let voxels = cfg.total_voxels();
        // 140 × 140 × 410 = 8_036_000
        assert_eq!(voxels, 8_036_000);
        let budget = cfg.neuron_budget();
        // 8_036_000 * 0.04 = 321_440
        // 8_036_000 * 0.04 = 321_440.0, but f32 precision may give 321_439
        let diff = budget as i64 - 321_440;
        assert!(diff.abs() <= 1, "neuron_budget far off: {}", budget);
    }
}
