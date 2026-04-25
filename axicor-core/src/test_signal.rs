/// All signal.rs tests (1.3 Active Tail).
/// Included via: `#[path = "test_signal.rs"] mod test_signal;` in signal.rs
use super::*;
use crate::constants::{AXON_SENTINEL, V_SEG};

#[test]
fn sentinel_is_never_active() {
    assert!(!is_segment_active(AXON_SENTINEL, 0, 10));
    assert!(!is_segment_active(AXON_SENTINEL, 100, 1));
}

#[test]
fn segment_in_tail_is_active() {
    let head = 10u32;
    assert!(is_segment_active(head, 8, 3));
    assert!(is_segment_active(head, 9, 3));
    assert!(is_segment_active(head, 10, 3)); // dist=0 < 3  active
}

#[test]
fn segment_outside_tail_is_inactive() {
    let head = 10u32;
    assert!(!is_segment_active(head, 7, 3)); // dist=3, not < 3
    assert!(!is_segment_active(head, 0, 3)); // too far
}

#[test]
fn initial_head_is_not_sentinel() {
    let head = initial_axon_head(10);
    assert_ne!(head, AXON_SENTINEL);
    assert_ne!(head, 0);
}

/// initial_axon_head starts such that head reaches AXON_SENTINEL after length ticks.
/// In intermediate ticks (before SENTINEL) a segment in the tail must be active.
#[test]
fn axon_fires_after_length_ticks() {
    let length = 5u32;
    let propagation = 3u32;
    let mut head = initial_axon_head(length);

    // In length - propagation ticks, head hasn't "reached" SENTINEL yet,
    // but has shifted enough for segment to fall into the tail.
    // At tick length - 1, head is V_SEG away from SENTINEL.
    for _ in 0..(length - propagation) {
        head = head.wrapping_add(V_SEG);
    }

    // Now dist to initial segment < propagation  it must be active
    let starting_segment = AXON_SENTINEL.wrapping_sub(length * V_SEG);
    assert!(
        is_segment_active(head, starting_segment, propagation),
        "segment should be in active tail mid-flight. head={head:#010x}, seg={starting_segment:#010x}"
    );
}

/// Active Tail with width P covers exactly P segments.
#[test]
fn tail_width_equals_propagation_length() {
    let head = 100u32;
    let p = 7u32;

    let active_count = (0u32..200)
        .filter(|&s| is_segment_active(head, s, p))
        .count();

    assert_eq!(
        active_count, p as usize,
        "Active tail must cover exactly propagation_length={p} segments, got {active_count}"
    );
}

/// Determinism: two calls with same arguments yield same result.
#[test]
fn is_segment_active_is_pure() {
    let head = 42u32;
    let seg = 40u32;
    let p = 3u32;
    assert_eq!(
        is_segment_active(head, seg, p),
        is_segment_active(head, seg, p)
    );
}

/// Edge case: propagation_length = 0  no segment is ever active.
#[test]
fn zero_propagation_length_never_active() {
    let head = 100u32;
    for seg in 95..=105 {
        assert!(
            !is_segment_active(head, seg, 0),
            "propagation=0 must never be active, seg={seg}"
        );
    }
}

/// sentinel is always inactive regardless of propagation_length.
#[test]
fn sentinel_always_dead() {
    for p in [0u32, 1, 10, u32::MAX] {
        assert!(
            !is_segment_active(AXON_SENTINEL, 0, p),
            "AXON_SENTINEL must be inactive for propagation={p}"
        );
    }
}

#[cfg(test)]
mod burst_heads_tests {
    use crate::constants::AXON_SENTINEL;

    /// Test Vector 1: The Wrap-Around Wrap (Overflow Arithmetic)
    /// Проверяем, что инициализация головы как `0 - v_seg` корректно отрабатывает
    /// переход через нуль и не вызывает ложных активаций до фактического смещения.
    #[test]
    fn test_burst_wrap_around_overflow() {
        let v_seg = 3u32;
        let prop = 10u32;
        let seg_idx = 0u32;

        // Имитируем рождение спайка (как в C-ядре: h0 = 0 - v_seg)
        let h0_born = 0u32.wrapping_sub(v_seg); // 0xFFFFFFFD

        // На этом тике (ДО сдвига) дистанция до нулевого сегмента должна быть огромной
        let dist_born = h0_born.wrapping_sub(seg_idx);
        assert!(dist_born > prop, "Fatal: Born spike falsely triggered segment 0");

        // Тик 1: сдвиг
        let h0_tick1 = h0_born.wrapping_add(v_seg); // Ровно 0x00000000
        let dist_tick1 = h0_tick1.wrapping_sub(seg_idx);
        assert!(dist_tick1 < prop, "Fatal: Spike failed to activate segment 0 after shift");

        // Тик 2: быстрый аксон пролетает дальше
        let h0_tick2 = h0_tick1.wrapping_add(v_seg * 2); // 6
        let dist_tick2 = h0_tick2.wrapping_sub(5); // 6 - 5 = 1 < 10
        assert!(dist_tick2 < prop, "Fatal: Spike lost Active Tail tracking");
    }

    /// Test Vector 2: Sentinel Collision (Death by Old Age)
    /// Убеждаемся, что старый спайк, приближающийся к 0x80000000, 
    /// никогда не даст дистанцию < prop для реальных дендритов.
    #[test]
    fn test_burst_sentinel_collision() {
        let prop = 20u32;
        let seg_idx = 10u32;

        // Спайк умирает от старости (подходит к лимиту Sentinel)
        let dying_head = AXON_SENTINEL.wrapping_sub(5); // 0x7FFFFFFB

        let dist_dying = dying_head.wrapping_sub(seg_idx);
        // 0x7FFFFFFB - 10 = 0x7FFFFFF1. Это гигантское число (больше 2 миллиардов)
        assert!(
            dist_dying > prop,
            "Fatal: Dying spike hallucinated a connection!"
        );

        // Сам Sentinel никогда не должен ничего активировать
        let dist_sentinel = AXON_SENTINEL.wrapping_sub(seg_idx);
        assert!(
            dist_sentinel > prop,
            "Fatal: Sentinel activated a dendrite!"
        );
    }

    /// Test Vector 3: Burst Compression (Bitwise OR Merging)
    /// Тестируем Branchless-логику из GPU-ядра. Если 2 спайка накладываются
    /// в одном сегменте, они должны сжаться в 1 (True), без Branching.
    #[test]
    fn test_burst_compression_bitwise_or() {
        let prop = 5u32;
        let seg_idx = 10u32;

        // Два спайка летят вплотную (Active Tails пересекаются)
        let h0 = 12u32; // Dist = 2 (Hit)
        let h1 = 14u32; // Dist = 4 (Hit)
        let h2 = AXON_SENTINEL; // Miss

        // Эмуляция железа: 8-way bitwise OR
        let hit = ((h0.wrapping_sub(seg_idx) < prop) as u32)
            | ((h1.wrapping_sub(seg_idx) < prop) as u32)
            | ((h2.wrapping_sub(seg_idx) < prop) as u32);

        assert_eq!(
            hit, 1,
            "Fatal: Burst Compression failed to merge overlapping spikes!"
        );

        // Смещаем сегмент так, чтобы оба спайка промахнулись
        let seg_far = 20u32;
        let hit_far = ((h0.wrapping_sub(seg_far) < prop) as u32)
            | ((h1.wrapping_sub(seg_far) < prop) as u32)
            | ((h2.wrapping_sub(seg_far) < prop) as u32);

        assert_eq!(
            hit_far, 0,
            "Fatal: Branchless OR falsely detected a hit on a distant segment!"
        );
    }
}
