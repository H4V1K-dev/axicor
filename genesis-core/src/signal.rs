/// Физика сигнала: «Активный Хвост» (Active Tail, Spec 01 §1.3).
///
/// Сигнал — это «поезд», скользящий по сегментам аксона.
/// `axon_head` стартует с `AXON_SENTINEL - length * V_SEG` и растёт на `V_SEG` каждый тик.
/// Когда `axon_head.wrapping_sub(segment_idx) < propagation_length` — сегмент «горит».
///
/// Целочисленная арифметика с переполнением u32 гарантирует детерминизм на GPU без float.
use crate::constants::{AXON_SENTINEL, V_SEG};
use crate::types::AxonHead;

/// Проверяет, находится ли сегмент `segment_idx` в «активном хвосте» за данный тик.
///
/// # Аргументы
/// - `axon_head` — текущая позиция головы аксона (u32, wrapping)
/// - `segment_idx` — индекс проверяемого сегмента
/// - `propagation_length` — длина хвоста в сегментах (`signal_propagation_length` из blueprints)
///
/// # Возвращает
/// `true` если сегмент входит в активный хвост `[head - propagation_length, head)`.
#[inline]
pub fn is_segment_active(
    axon_head: AxonHead,
    segment_idx: u32,
    propagation_length: u32,
) -> bool {
    if axon_head == AXON_SENTINEL {
        return false;
    }
    axon_head.wrapping_sub(segment_idx) < propagation_length
}

/// Вычисляет начальную позицию головы аксона для N сегментов.
/// `head = AXON_SENTINEL - length * V_SEG`
///
/// Это позволяет `propagate_axons` корректно «доехать» до конца в первый же тик.
#[inline]
pub fn initial_axon_head(length_segments: u32) -> AxonHead {
    AXON_SENTINEL.wrapping_sub(length_segments * V_SEG)
}

#[cfg(test)]
#[path = "test_signal.rs"]
mod test_signal;

#[cfg(test)]
#[path = "test_train_model.rs"]
mod test_train_model;
