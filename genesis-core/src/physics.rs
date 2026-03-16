/// Конвейер пересчёта физических констант при старте (Spec 01 §1.5).
///
/// При старте движок берёт человеко-читаемые значения из конфига и вычисляет
/// «сырые» GPU-константы. Горячий цикл не делает умножений — он оперирует
/// уже готовыми числами из Constant Memory.
///
/// Инвариант §1.6: `signal_speed_um_tick % segment_length_um == 0`.
/// Нарушение → возврат Err до любого GPU-upload.

/// Производные физические константы готовые к загрузке в GPU Constant Memory.
///
/// Вычисляются один раз при старте через `compute_derived_physics`.
/// Поля намеренно плоские — прямой маппинг в C-структуру для `cudaMemcpyToSymbol`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedPhysics {
    /// Скорость сигнала в мкм/тик (§1.5 п.1).
    /// GPU прибавляет это к позиции головы каждый тик.
    pub signal_speed_um_tick: u32,
    /// Длина одного сегмента в мкм = voxel_size_um × segment_length_voxels.
    pub segment_length_um: u32,
    /// Дискретная скорость: сегментов/тик (§1.5 п.2).
    /// `v_seg = signal_speed_um_tick / segment_length_um` — гарантированно целое.
    /// GPU: `axon_head += v_seg` за тик. Никаких float, никакой интерполяции.
    pub v_seg: u32,
}

/// Вычисляет `DerivedPhysics` из значений конфига — конвейер пересчёта §1.5.
///
/// Возвращает `Err` если нарушен инвариант §1.6 (`v_seg` дробное).
///
/// # Аргументы
/// - `signal_speed_um_tick` — из `simulation.signal_speed_um_tick`
/// - `voxel_size_um`        — из `simulation.voxel_size_um`
/// - `segment_length_vox`   — из `simulation.segment_length_voxels`
///
/// # Пример (конфигурация из спека)
/// ```
/// # use genesis_core::physics::compute_derived_physics;
/// let p = compute_derived_physics(0.5, 100, 25.0, 2).unwrap();
/// assert_eq!(p.v_seg, 1);
/// ```
pub fn compute_derived_physics(
    signal_speed_m_s: f32,
    tick_duration_us: u32,
    voxel_size_um: f32,
    segment_length_vox: u32,
) -> Result<DerivedPhysics, String> {
    let tick_ms = tick_duration_us as f32 / 1000.0;
    // 1 m/s = 1000 um/ms
    let signal_speed_um_tick = signal_speed_m_s * 1000.0 * tick_ms;
    let segment_length_um = voxel_size_um * segment_length_vox as f32;

    if segment_length_um < 1e-6 {
        return Err("segment_length_um must be > 0".to_string());
    }

    let v_seg_f32 = signal_speed_um_tick / segment_length_um;
    let v_seg = v_seg_f32.round() as u32;

    let diff = (v_seg_f32 - v_seg as f32).abs();
    if diff > 1e-5 {
        return Err(format!(
            "§1.6 violation: v_seg ({v_seg_f32:.4}) is not an integer. \
             Physically impossible for Integer Physics engine. \
             Check simulation.signal_speed_m_s ({} m/s) and simulation.voxel_size_um ({} um).",
             signal_speed_m_s, voxel_size_um
        ));
    }

    Ok(DerivedPhysics { 
        signal_speed_um_tick: signal_speed_um_tick.round() as u32, 
        segment_length_um: segment_length_um.round() as u32, 
        v_seg 
    })
}

// ---------------------------------------------------------------------------
// GLIF Dynamics (Spec 03 §2, §3)
// ---------------------------------------------------------------------------

/// Вычисляет новое напряжение мембраны (Integer GLIF).
///
/// `dV = -(V - V_rest) / leak_rate + I_input`
///
/// Деление целочисленное (truncation toward zero) — детерминизм на CPU и GPU.
#[inline(always)]
pub const fn compute_glif(
    voltage: i32,
    rest_potential: i32,
    leak_rate: i32,
    input_current: i32,
) -> i32 {
    debug_assert!(leak_rate > 0, "Leak rate must be > 0");
    let leak = (voltage - rest_potential) / leak_rate;
    voltage - leak + input_current
}

// ---------------------------------------------------------------------------
// Adaptive Leak (LTC Roadmap Milestone 2)
// ---------------------------------------------------------------------------

/// Dopamine-driven effective leak (LTC Roadmap §Formulas).
///
/// `leak_mod = (dopamine * dopamine_leak_gain) >> 7`
/// `effective_leak = clamp(base_leak_rate + leak_mod, leak_min, leak_max)`
///
/// When `adaptive_leak_mode == 0` or clamp window invalid (`leak_min >= leak_max`),
/// returns `base_leak_rate`. Branchless, integer-only.
#[inline(always)]
pub const fn compute_effective_leak(
    adaptive_leak_mode: u8,
    base_leak_rate: i32,
    dopamine: i16,
    dopamine_leak_gain: i16,
    leak_min: i16,
    leak_max: i16,
) -> i32 {
    if adaptive_leak_mode == 0 || leak_min >= leak_max {
        return base_leak_rate;
    }
    let leak_mod = (dopamine as i32 * dopamine_leak_gain as i32) >> 7;
    let raw = base_leak_rate + leak_mod;
    let lo = leak_min as i32;
    let hi = leak_max as i32;
    // Branchless clamp: min(max(raw, lo), hi)
    let clamped_lo = raw - ((raw - lo) & ((raw - lo) >> 31));
    let clamped_hi = clamped_lo + ((hi - clamped_lo) & ((hi - clamped_lo) >> 31));
    // Ensure positive for GLIF (avoid div-by-zero): max(result, 1)
    let result = clamped_hi;
    result + ((1 - result) & ((result - 1) >> 31))
}

// ---------------------------------------------------------------------------
// Homeostasis (Spec 03 §3.2 — Adaptive Threshold)
// ---------------------------------------------------------------------------

/// Branchless обновление адаптивного порога.
///
/// Логика: `max(0, offset - decay) + (is_spiking ? penalty : 0)`
///
/// Использует `wrapping_sub` + arithmetic shift для branchless clamp к 0.
/// Никаких паник в debug builds при underflow.
#[inline(always)]
pub const fn update_homeostasis(
    threshold_offset: i32,
    decay: u16,
    is_spiking: bool,
    penalty: i32,
) -> i32 {
    // wrapping_sub: безопасен даже если offset < decay (даёт отрицательное число в i32)
    let decayed = threshold_offset.wrapping_sub(decay as i32);
    // Arithmetic shift: 0xFFFFFFFF если < 0, иначе 0x00000000
    let mask = decayed >> 31;
    // Branchless max(0, decayed): обнулить если отрицательное
    let clamped = decayed & !mask;

    // Penalty добавляется только при спайке
    let spike_penalty = if is_spiking { penalty } else { 0 };
    clamped + spike_penalty
}

// ---------------------------------------------------------------------------
// GSOP Plasticity (Spec 04 §2.4, §2.5)
// ---------------------------------------------------------------------------

/// Inertia Rank: разделяем диапазон |weight| (0..32767) на 15 рангов по 2048.
/// `rank = abs_w >> 11`. Это 1 такт ALU, без ветвлений.
#[inline(always)]
pub const fn inertia_rank(abs_weight: i32) -> usize {
    let rank = (abs_weight >> 11) as usize;
    if rank > 14 { 14 } else { rank }
}

/// Вычисляет новый вес синапса с учетом GSOP, инерции и слот-декея.
///
/// # Аргументы
/// - `current_weight`: текущий вес (i16). Знак = тип синапса (Dale's Law).
/// - `is_active_tail`: true = каузальное совпадение (potentiation).
/// - `potentiation`: GSOP delta при совпадении (из Constant Memory).
/// - `depression`: GSOP delta при промахе (из Constant Memory).
/// - `inertia_multiplier`: Значение из `inertia_curve[rank]` (0..255, fixed-point base=128).
/// - `slot_decay`: Множитель слота (LTM или WM, fixed-point base=128).
///
/// # Контракт
/// - Знак веса НИКОГДА не меняется (Закон Дейла).
/// - `i16::MIN` (-32768) обрабатывается корректно (каст в i32 перед abs).
/// - Результат clamped к `[-32767, 32767]` (abs_w clamped к `[0, 32767]`).
#[inline(always)]
pub const fn compute_gsop_weight(
    current_weight: i16,
    is_active_tail: bool,
    potentiation: i16,
    depression: i16,
    inertia_multiplier: u8,
    slot_decay: u8,
) -> i16 {
    // 1. Знак — фиксируем раз и навсегда (Dale's Law)
    let sign: i32 = if current_weight < 0 { -1 } else { 1 };

    // 2. Безопасный abs: i16::MIN (-32768) as i32 = -32768, abs = 32768.
    //    Далее clamp до 32767.
    let abs_w = (current_weight as i32).abs();

    // 3. Inertia: масштабируем delta на кривую инерции (fixed-point >>7)
    let raw_delta = if is_active_tail {
        (potentiation as i32 * inertia_multiplier as i32) >> 7
    } else {
        -((depression as i32 * inertia_multiplier as i32) >> 7)
    };

    // 4. Slot Decay: масштабируем delta на коэффициент слота (fixed-point >>7)
    let delta = (raw_delta * slot_decay as i32) >> 7;

    // 5. Вычисляем новый абсолютный вес
    let mut new_abs = abs_w + delta;

    // 6. Клэмп [0, 32767] — вес не переходит через 0 и не превышает i16::MAX
    if new_abs < 0 {
        new_abs = 0;
    } else if new_abs > 32767 {
        new_abs = 32767;
    }

    // 7. Восстанавливаем знак
    (sign * new_abs) as i16
}

#[cfg(test)]
#[path = "test_physics.rs"]
mod test_physics;

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // GLIF
    // -----------------------------------------------------------------------

    #[test]
    fn test_glif_leak_positive_offset() {
        // voltage=100, rest=-70, leak=2, input=0
        // leak = (100 - (-70)) / 2 = 85
        // new_v = 100 - 85 + 0 = 15
        assert_eq!(compute_glif(100, -70, 2, 0), 15);
    }

    #[test]
    fn test_glif_leak_negative_offset() {
        // voltage=-100, rest=-70, leak=2, input=0
        // leak = (-100 - (-70)) / 2 = -15
        // new_v = -100 - (-15) + 0 = -85
        assert_eq!(compute_glif(-100, -70, 2, 0), -85);
    }

    #[test]
    fn test_glif_at_rest_no_change() {
        // voltage == rest -> leak=0 -> stays at rest (no input)
        assert_eq!(compute_glif(-70, -70, 2, 0), -70);
    }

    #[test]
    fn test_glif_with_input_current() {
        // voltage=-70, rest=-70, leak=2, input=50
        // leak=0, new_v = -70 + 50 = -20
        assert_eq!(compute_glif(-70, -70, 2, 50), -20);
    }

    // -----------------------------------------------------------------------
    // Adaptive Leak (Milestone 2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_effective_leak_mode_off_returns_base() {
        assert_eq!(compute_effective_leak(0, 10, 100, 64, 1, 20), 10);
    }

    #[test]
    fn test_effective_leak_invalid_window_returns_base() {
        assert_eq!(compute_effective_leak(1, 10, 100, 64, 5, 5), 10);
        assert_eq!(compute_effective_leak(1, 10, 100, 64, 10, 5), 10);
    }

    #[test]
    fn test_effective_leak_dopamine_modulation() {
        // leak_mod = (127 * 64) >> 7 = 64, effective = 10 + 64 = 74, clamp(1,20)=20
        assert_eq!(compute_effective_leak(1, 10, 127, 64, 1, 20), 20);
        // leak_mod = (0 * 64) >> 7 = 0, effective = 10
        assert_eq!(compute_effective_leak(1, 10, 0, 64, 1, 20), 10);
        // leak_mod = (-128 * 64) >> 7 = -64, effective = 10 - 64 = -54, clamp(1,20)=1
        assert_eq!(compute_effective_leak(1, 10, -128, 64, 1, 20), 1);
    }

    #[test]
    fn test_effective_leak_integration_with_glif() {
        let eff = compute_effective_leak(1, 2, 10, 64, 1, 20);
        assert!(eff >= 1 && eff <= 20);
        assert_eq!(compute_glif(100, -70, eff, 0), 100 - (170 / eff) + 0);
    }

    // -----------------------------------------------------------------------
    // Homeostasis
    // -----------------------------------------------------------------------

    #[test]
    fn test_homeostasis_branchless_clamp() {
        // offset=5, decay=10 -> decayed = 5 - 10 = -5 -> clamped to 0
        assert_eq!(update_homeostasis(5, 10, false, 100), 0);
    }

    #[test]
    fn test_homeostasis_normal_decay() {
        // offset=100, decay=3, not spiking -> 100 - 3 = 97
        assert_eq!(update_homeostasis(100, 3, false, 15), 97);
    }

    #[test]
    fn test_homeostasis_spike_adds_penalty() {
        // offset=100, decay=3, spiking, penalty=15 -> 97 + 15 = 112
        assert_eq!(update_homeostasis(100, 3, true, 15), 112);
    }

    #[test]
    fn test_homeostasis_zero_stays_zero() {
        // offset=0, decay=0, not spiking -> 0
        assert_eq!(update_homeostasis(0, 0, false, 15), 0);
    }

    #[test]
    fn test_homeostasis_spike_from_zero() {
        // offset=0, decay=0, spiking -> 0 + penalty
        assert_eq!(update_homeostasis(0, 0, true, 15), 15);
    }

    // -----------------------------------------------------------------------
    // GSOP
    // -----------------------------------------------------------------------

    #[test]
    fn test_gsop_potentiation_basic() {
        // weight=100, causal, pot=80, dep=40, inertia=128 (rank 0 full), slot=128 (1.0x)
        // raw_delta = (80 * 128) >> 7 = 80
        // delta = (80 * 128) >> 7 = 80
        // new_abs = 100 + 80 = 180
        assert_eq!(compute_gsop_weight(100, true, 80, 40, 128, 128), 180);
    }

    #[test]
    fn test_gsop_depression_basic() {
        // weight=100, not causal, pot=80, dep=40, inertia=128, slot=128
        // raw_delta = -((40 * 128) >> 7) = -40
        // delta = (-40 * 128) >> 7 = -40
        // new_abs = 100 - 40 = 60
        assert_eq!(compute_gsop_weight(100, false, 80, 40, 128, 128), 60);
    }

    #[test]
    fn test_gsop_i16_min() {
        // current_weight = -32768 (i16::MIN).
        // Must NOT panic. abs in i32 = 32768 -> clamp to 32767 after delta.
        // potentiation: raw_delta = (80 * 128) >> 7 = 80, delta = (80 * 128) >> 7 = 80
        // new_abs = 32768 + 80 = 32848 -> clamp to 32767
        // sign = -1 -> -32767
        let w = compute_gsop_weight(i16::MIN, true, 80, 40, 128, 128);
        assert_eq!(w, -32767);

        // depression: raw_delta = -40, delta = -40
        // new_abs = 32768 - 40 = 32728
        // sign = -1 -> -32728
        let w2 = compute_gsop_weight(i16::MIN, false, 80, 40, 128, 128);
        assert_eq!(w2, -32728);
    }

    #[test]
    fn test_gsop_sign_preservation_negative() {
        // Negative weight should NEVER become positive
        let w = compute_gsop_weight(-500, false, 80, 40, 128, 128);
        assert!(w <= 0, "Dale's Law violated: negative weight became positive: {w}");
        // -500 -> abs=500, delta=-40, new_abs=460 -> -460
        assert_eq!(w, -460);
    }

    #[test]
    fn test_gsop_sign_preservation_depression_to_zero() {
        // Small negative weight, heavy depression -> should clamp to 0, NOT become positive
        let w = compute_gsop_weight(-5, false, 80, 40, 128, 128);
        // abs=5, delta=-40, new_abs = 5-40 = -35 -> clamp 0 -> sign*0 = 0
        assert_eq!(w, 0);
    }

    #[test]
    fn test_gsop_clamp_max() {
        // Large weight + potentiation -> clamp at 32767
        let w = compute_gsop_weight(32767, true, 80, 40, 128, 128);
        assert_eq!(w, 32767);
    }

    #[test]
    fn test_gsop_inertia_rank_calculation() {
        assert_eq!(inertia_rank(0), 0);
        assert_eq!(inertia_rank(2047), 0);
        assert_eq!(inertia_rank(2048), 1);
        assert_eq!(inertia_rank(4095), 1);
        assert_eq!(inertia_rank(4096), 2);
        assert_eq!(inertia_rank(32767), 14);
		assert_eq!(inertia_rank(40000), 14);
    }

    #[test]
    fn test_gsop_slot_decay_effect() {
        // slot_decay=64 (0.5x) should halve the delta compared to slot_decay=128 (1.0x)
        let w_full = compute_gsop_weight(1000, true, 80, 40, 128, 128);
        let w_half = compute_gsop_weight(1000, true, 80, 40, 128, 64);
        // w_full: delta= (80*128)>>7=80, (80*128)>>7=80 -> 1080
        // w_half: delta= (80*128)>>7=80, (80*64)>>7=40 -> 1040
        assert_eq!(w_full, 1080);
        assert_eq!(w_half, 1040);
    }
}
