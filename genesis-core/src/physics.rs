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
/// let p = compute_derived_physics(50, 25, 2).unwrap();
/// assert_eq!(p.v_seg, 1);
/// ```
pub fn compute_derived_physics(
    signal_speed_um_tick: u32,
    voxel_size_um: u32,
    segment_length_vox: u32,
) -> Result<DerivedPhysics, String> {
    let segment_length_um = voxel_size_um
        .checked_mul(segment_length_vox)
        .ok_or_else(|| "segment_length_um overflow".to_string())?;

    if segment_length_um == 0 {
        return Err("segment_length_um must be > 0".to_string());
    }

    // §1.6 инвариант — v_seg обязан быть целым
    if signal_speed_um_tick % segment_length_um != 0 {
        return Err(format!(
            "§1.6 violation: signal_speed_um_tick ({signal_speed_um_tick}) \
             must be divisible by segment_length_um ({segment_length_um}). \
             v_seg = {}/{} — не целое число. \
             Нарушает Integer Physics детерминизм.",
            signal_speed_um_tick, segment_length_um,
        ));
    }

    let v_seg = signal_speed_um_tick / segment_length_um;

    Ok(DerivedPhysics { signal_speed_um_tick, segment_length_um, v_seg })
}

#[cfg(test)]
#[path = "test_physics.rs"]
mod test_physics;
