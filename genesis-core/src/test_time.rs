/// Тесты временно́й метрики (§1.4).
use super::*;
use crate::constants::TICK_DURATION_US;

#[test]
fn ms_to_ticks_basic() {
    // 5 мс / 0.1 мс = 50 тиков
    assert_eq!(ms_to_ticks(5.0), 50);
    // 1 мс = 10 тиков
    assert_eq!(ms_to_ticks(1.0), 10);
    // 0.1 мс = 1 тик
    assert_eq!(ms_to_ticks(0.1), 1);
}

#[test]
fn us_to_ticks_basic() {
    // 500 мкс / 100 мкс = 5 тиков
    assert_eq!(us_to_ticks(500), 5);
    // 100 мкс = 1 тик
    assert_eq!(us_to_ticks(TICK_DURATION_US), 1);
    // 0 мкс = 0 тиков
    assert_eq!(us_to_ticks(0), 0);
}

#[test]
fn ticks_to_ms_roundtrip() {
    let ms = 7.5f32;
    let ticks = ms_to_ticks(ms);
    let back = ticks_to_ms(ticks);
    assert!((back - ms).abs() < 0.01, "roundtrip: {ms} ms → {ticks} ticks → {back} ms");
}

#[test]
fn biological_refractory_example() {
    // Спека §1.4: 5 мс рефрактерность = 50 тиков
    assert_eq!(ms_to_ticks(5.0), 50);
}
