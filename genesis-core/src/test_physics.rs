/// Тесты конвейера пересчёта физики (§1.5 + §1.6).
use super::*;
use crate::constants::{SEGMENT_LENGTH_UM, V_SEG, VOXEL_SIZE_UM, SEGMENT_LENGTH_VOXELS, TICK_DURATION_US};

/// Штатная конфигурация из спека → v_seg = 1.
#[test]
fn valid_config_from_spec() {
    // 0.5 m/s at 100us = 50um/tick. voxel=25, segment=2vox -> 50um/segment -> v_seg=1
    let p = compute_derived_physics(0.5, 100, 25.0, 2).unwrap();
    assert_eq!(p.signal_speed_um_tick, 50);
    assert_eq!(p.segment_length_um, 50);
    assert_eq!(p.v_seg, 1);
}

/// Runtime-результат совпадает с compile-time константами в constants.rs.
#[test]
fn derived_matches_compile_time_constants() {
    // 50um/tick / 100us = 0.5 m/s
    let p = compute_derived_physics(
        0.5,
        TICK_DURATION_US,
        VOXEL_SIZE_UM as f32,
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
    // 0.1 m/s at 100us = 10um/tick. voxel=30, segment=1 -> 30um/segment -> 10/30 = 0.33
    let result = compute_derived_physics(0.1, 100, 30.0, 1);
    assert!(result.is_err(), "должна быть ошибка §1.6");
    let msg = result.unwrap_err();
    assert!(msg.contains("§1.6"), "сообщение должно ссылаться на §1.6: {msg}");
}

/// Конфигурация с v_seg=2 (быстрый аксон) валидна.
#[test]
fn v_seg_two_is_valid() {
    // 1.0 m/s at 100us = 100um/tick. segment_length_um=50 -> v_seg=2
    let p = compute_derived_physics(1.0, 100, 25.0, 2).unwrap();
    assert_eq!(p.v_seg, 2);
}

/// Нулевая длина сегмента → Err (деление на 0).
#[test]
fn zero_segment_length_is_err() {
    let result = compute_derived_physics(0.5, 100, 0.0, 2);
    assert!(result.is_err());
}
