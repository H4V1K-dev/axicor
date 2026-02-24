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

#[cfg(test)]
#[path = "test_time.rs"]
mod test_time;
