/// Временна́я метрика (Spec 01 §1.4).
///
/// Квант времени: 1 Тик = `TICK_DURATION_US` мкс = 0.1 мс.
/// Все таймеры (рефрактерность, decay, ночные интервалы) задаются в тиках.
/// Пример: 5 мс рефрактерность = 50 тиков.
use crate::constants::TICK_DURATION_US;
use crate::types::Tick;

/// Миллисекунды → тики.
/// Пример: `ms_to_ticks(5.0)` = 50 (при TICK_DURATION_US=100).
#[inline]
pub fn ms_to_ticks(ms: f32) -> Tick {
    let us = ms * 1000.0;
    (us / TICK_DURATION_US as f32).round() as Tick
}

/// Микросекунды → тики.
/// Пример: `us_to_ticks(500)` = 5.
#[inline]
pub fn us_to_ticks(us: u32) -> Tick {
    (us / TICK_DURATION_US) as Tick
}

/// Тики → миллисекунды (для логов и отладки).
#[inline]
pub fn ticks_to_ms(ticks: Tick) -> f32 {
    ticks as f32 * TICK_DURATION_US as f32 / 1000.0
}

pub struct PhysicalMetrics;

impl PhysicalMetrics {
    /// Вычисляет дискретную скорость распространения сигнала (v_seg).
    /// Вызывает panic (Baking Phase), если параметры не согласованы.
    ///
    /// v_seg = speed_um_tick / segment_length_um
    pub fn compute_v_seg(
        signal_speed_m_s: f32,
        tick_duration_us: u32,
        voxel_size_um: f32,
        segment_length_voxels: u32,
    ) -> u32 {
        // speed_um_tick = speed_m_s * 10^6 * tick_duration_us / 10^6
        let speed_um_tick = signal_speed_m_s * (tick_duration_us as f32);
        let segment_length_um = voxel_size_um * (segment_length_voxels as f32);

        let v_seg_f32 = speed_um_tick / segment_length_um;
        let v_seg = v_seg_f32.round() as u32;

        // Контракт: Дробная часть должна быть строго равна нулю (с учетом погрешности float).
        let diff = (v_seg_f32 - (v_seg as f32)).abs();
        assert!(
            diff < 1e-5,
            "CRITICAL INVARIANT BROKEN: signal_speed_um_tick ({}) is not divisible by segment_length_um ({}). v_seg = {}. Float math in GPU hot-loop is strictly forbidden.",
            speed_um_tick, segment_length_um, v_seg_f32
        );

        v_seg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v_seg_valid_config() {
        // 0.5 m/s, 100 us, 25 um, 2 voxels -> speed_um_tick=50, seg_um=50 -> v_seg=1
        let v = PhysicalMetrics::compute_v_seg(0.5, 100, 25.0, 2);
        assert_eq!(v, 1);
    }

    #[test]
    #[should_panic(expected = "CRITICAL INVARIANT BROKEN")]
    fn test_v_seg_invalid_config_panics() {
        // 0.6 m/s, 100 us, 25 um, 2 voxels -> speed_um_tick=60, seg_um=50 -> v_seg=1.2 (Error)
        PhysicalMetrics::compute_v_seg(0.6, 100, 25.0, 2);
    }
}

#[cfg(test)]
#[path = "test_time.rs"]
mod test_time;
