/// Физика сигнала: «Активный Хвост» (Active Tail, Spec 01 §1.3).
///
/// Сигнал — это «поезд», скользящий по сегментам аксона.
/// `axon_head` стартует с `AXON_SENTINEL - length * V_SEG` и растёт на `V_SEG` каждый тик.
/// Когда `axon_head.wrapping_sub(segment_idx) < propagation_length` — сегмент «горит».
///
/// Целочисленная арифметика с переполнением u32 гарантирует детерминизм на GPU без float.
use crate::constants::{AXON_SENTINEL, V_SEG};
use crate::types::AxonHead;

/// Branchless Active Tail check для GPU Hot Loop (Spec 03 §1.3).
///
/// Проверяет, находится ли дендритный сегмент внутри активного хвоста сигнала.
/// Нет ветвлений — AXON_SENTINEL (0x80000000) обрабатывается автоматически:
/// `0x80000000.wrapping_sub(any_small_idx)` ≈ 2.1 млрд > любого propagation_length.
///
/// # Гарантии
/// - Zero Warp Divergence на GPU (нет `if`)
/// - Deterministic: одинаковый результат на CPU и GPU
/// - AXON_SENTINEL всегда возвращает `false`
#[inline(always)]
pub const fn is_in_active_tail(head_idx: u32, segment_idx: u32, propagation_length: u8) -> bool {
    let dist = head_idx.wrapping_sub(segment_idx);
    dist < (propagation_length as u32)
}


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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::AXON_SENTINEL;

    #[test]
    fn test_active_tail_normal_overlap() {
        // head=10, segment=8, prop=3 -> dist=2 < 3 -> true
        assert!(is_in_active_tail(10, 8, 3));
        // head=10, segment=10, prop=3 -> dist=0 < 3 -> true
        assert!(is_in_active_tail(10, 10, 3));
        // head=10, segment=9, prop=3 -> dist=1 < 3 -> true
        assert!(is_in_active_tail(10, 9, 3));
    }

    #[test]
    fn test_active_tail_outside() {
        // head=10, segment=7, prop=3 -> dist=3, NOT < 3 -> false
        assert!(!is_in_active_tail(10, 7, 3));
        // head=10, segment=0, prop=3 -> dist=10 -> false
        assert!(!is_in_active_tail(10, 0, 3));
    }

    #[test]
    fn test_sentinel_edge_case() {
        // AXON_SENTINEL (0x80000000) - 5 = 0x7FFFFFFB ≈ 2.1 billion -> always >= prop
        assert!(!is_in_active_tail(AXON_SENTINEL, 5, 3));
        assert!(!is_in_active_tail(AXON_SENTINEL, 0, 255));
        assert!(!is_in_active_tail(AXON_SENTINEL, 1000, 100));
    }

    #[test]
    fn test_active_tail_zero_propagation() {
        // prop=0 means no segment is ever active
        assert!(!is_in_active_tail(10, 10, 0));
        assert!(!is_in_active_tail(10, 9, 0));
    }
}
