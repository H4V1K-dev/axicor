/// Тесты The Train Model (§1.1) с различными `v_seg` и `propagation_length`.
/// Позволяет убедиться, что "дыр" при прыжках головы нет, если `prop_len >= v_seg`.

use super::*;
use crate::constants::{AXON_SENTINEL, V_SEG};

#[test]
fn test_no_gap_v_seg_1() {
    let v_seg = 1;
    let prop_len = 5;
    
    // Эмуляция 50 тиков
    let start_head = 100u32;
    let mut current_head = start_head;
    
    let mut covered_segments = vec![];
    
    for _ in 0..50 {
        current_head = current_head.wrapping_add(v_seg);
        
        // В каждом тике какие-то сегменты активны. Запомним их.
        // Активный хвост: (head - prop_len ..= head]  (если < prop_len)
        for seg in 0..200 {
            if is_segment_active(current_head, seg, prop_len) {
                covered_segments.push(seg);
            }
        }
    }
    
    // Сортируем и убираем дубли
    covered_segments.sort_unstable();
    covered_segments.dedup();
    
    // Проверяем, что нет "дыр" (непройденных сегментов) в диапазоне от start_head до start_head + 50*v_seg
    // Те сегменты, которые лежали внутри этого диапазона, все должны быть в covered_segments.
    let min_expected = start_head + v_seg;
    let max_expected = start_head + 50 * v_seg - prop_len + 1;
    
    for seg in min_expected..=max_expected {
        assert!(covered_segments.contains(&seg), "Segment {} was missed! v_seg={}, prop={}", seg, v_seg, prop_len);
    }
}

#[test]
fn test_no_gap_v_seg_large() {
    let v_seg = 5;
    let prop_len = 5; // prop >= v_seg, дыр быть не должно
    
    let start_head = 100u32;
    let mut current_head = start_head;
    
    let mut covered_segments = vec![];
    
    for _ in 0..20 {
        current_head = current_head.wrapping_add(v_seg);
        for seg in 0..300 {
            if is_segment_active(current_head, seg, prop_len) {
                covered_segments.push(seg);
            }
        }
    }
    
    covered_segments.sort_unstable();
    covered_segments.dedup();
    
    let min_expected = start_head + v_seg;
    let max_expected = start_head + 20 * v_seg - prop_len + 1;
    
    for seg in min_expected..=max_expected {
        assert!(covered_segments.contains(&seg), "Segment {} was missed! v_seg={}, prop={}", seg, v_seg, prop_len);
    }
}

#[test]
fn test_gap_when_prop_lt_v_seg() {
    let v_seg = 5;
    let prop_len = 3; // prop < v_seg, ДОЛЖНЫ быть дыры
    
    let start_head = 100u32;
    let mut current_head = start_head;
    
    let mut covered_segments = vec![];
    
    for _ in 0..20 {
        current_head = current_head.wrapping_add(v_seg);
        for seg in 0..300 {
            if is_segment_active(current_head, seg, prop_len) {
                covered_segments.push(seg);
            }
        }
    }
    
    covered_segments.sort_unstable();
    covered_segments.dedup();
    
    // Прыжки 100 -> 105, 105 -> 110.
    // Если prop=3, то хвост [103..105], [108..110].
    // Сегменты 101, 102, 106, 107 пропущены!
    assert!(!covered_segments.contains(&106), "Segment 106 should be MISSED, but it was covered!");
    assert!(!covered_segments.contains(&107), "Segment 107 should be MISSED, but it was covered!");
}

#[test]
fn test_roundtrip_fire_to_overlap() {
    let length = 10u32;
    // v_seg используем из константы для этого теста
    let mut head = initial_axon_head(length);
    let prop_len = 3;
    
    for _ in 0..length {
        head = head.wrapping_add(V_SEG);
    }
    
    // На тике `length` голова должна оказаться возле 0 для сегмента 0
    let start_segment = AXON_SENTINEL.wrapping_sub(length * V_SEG);
    
    // Если мы сдвинули initial_head на length * V_SEG, мы окажемся в start_segment + length * V_SEG = AXON_SENTINEL
    // Ух ты, начальный сегмент это AXON_SENTINEL - length * V_SEG. Конечный - AXON_SENTINEL.
    // Значит когда голова == AXON_SENTINEL, сигнал "умер".
    // Но давайте проверим на 1 тик РАНЬШЕ:
    
    let mut before_end_head = initial_axon_head(length);
    for _ in 0..(length - 1) {
        before_end_head = before_end_head.wrapping_add(V_SEG);
    }
    
    let last_segment = AXON_SENTINEL.wrapping_sub(V_SEG);
    assert!(is_segment_active(before_end_head, last_segment, prop_len), "Last segment must be active just before sentinel!");
}
