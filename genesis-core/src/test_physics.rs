/// Тесты конвейера пересчёта физики (§1.5 + §1.6).
use super::*;
use crate::constants::{SEGMENT_LENGTH_UM, SIGNAL_SPEED_UM_TICK, V_SEG, VOXEL_SIZE_UM, SEGMENT_LENGTH_VOXELS};

/// Штатная конфигурация из спека → v_seg = 1.
#[test]
fn valid_config_from_spec() {
    let p = compute_derived_physics(50, 25, 2).unwrap();
    assert_eq!(p.signal_speed_um_tick, 50);
    assert_eq!(p.segment_length_um, 50);
    assert_eq!(p.v_seg, 1);
}

/// Runtime-результат совпадает с compile-time константами в constants.rs.
#[test]
fn derived_matches_compile_time_constants() {
    let p = compute_derived_physics(
        SIGNAL_SPEED_UM_TICK,
        VOXEL_SIZE_UM,
        SEGMENT_LENGTH_VOXELS,
    ).unwrap();

    assert_eq!(p.v_seg, V_SEG,
        "runtime v_seg должен совпадать с compile-time V_SEG");
    assert_eq!(p.segment_length_um, SEGMENT_LENGTH_UM,
        "runtime segment_length_um должен совпадать с SEGMENT_LENGTH_UM");
}

/// Нарушение §1.6 — v_seg дробное → Err.
#[test]
fn non_divisible_speed_returns_err() {
    // speed=50, segment=30 → 50 % 30 ≠ 0
    let result = compute_derived_physics(50, 30, 1);
    assert!(result.is_err(), "должна быть ошибка §1.6");
    let msg = result.unwrap_err();
    assert!(msg.contains("§1.6"), "сообщение должно ссылаться на §1.6: {msg}");
}

/// Конфигурация с v_seg=2 (быстрый аксон) валидна.
#[test]
fn v_seg_two_is_valid() {
    // speed=100, segment_length_um=50 → v_seg=2
    let p = compute_derived_physics(100, 25, 2).unwrap();
    assert_eq!(p.v_seg, 2);
}

/// Нулевая длина сегмента → Err (деление на 0).
#[test]
fn zero_segment_length_is_err() {
    let result = compute_derived_physics(50, 0, 2);
    assert!(result.is_err());
}
