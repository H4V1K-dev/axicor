/// CPU эмуляция и тесты для математики ядра ApplyGSOP (Spec §1.3)
/// Проверяет формулы potentiation, depression, clanp, slot decay и inertia rank.

use crate::config::blueprints::NeuronType;

/// Полная копия branchless логики из `physics.cu -> apply_gsop_kernel`
fn emulate_gsop_math(
    weight: i16,
    is_causal: bool,
    slot_idx: u8,
    p: &NeuronType, // Используем NeuronType как аналог VariantParameters
) -> i16 {
    // 5. Inertia Rank: abs(weight) >> 11 → 0..15
    let abs_w = weight.abs() as u16;
    let rank = (abs_w >> 11) as usize;
    
    // Защита от выхода за пределы (хотя clamp ниже не даст весу стать > 32767)
    let rank_safe = rank.min(15);
    let inertia = p.inertia_curve[rank_safe] as u16;

    // 6. Branchless GSOP Math
    let delta_pot = (p.gsop_potentiation as u16 * inertia) >> 7;
    let delta_dep = (p.gsop_depression as u16 * inertia) >> 7;
    
    let is_causal_int = if is_causal { 1 } else { 0 };
    let not_causal_int = if is_causal { 0 } else { 1 };
    
    let delta = (is_causal_int * delta_pot as i32) - (not_causal_int * delta_dep as i32);

    // 7. Slot Decay
    let decay = if slot_idx < p.ltm_slot_count {
        p.slot_decay_ltm as i32
    } else {
        p.slot_decay_wm as i32
    };
    
    let delta = (delta * decay) >> 7;

    // 8. Signed Clamp ±32767
    let w_sign = if weight < 0 { -1 } else { 1 };
    let mut new_abs = (abs_w as i32) + delta;
    
    // В CUDA: new_abs = (new_abs > 32767) ? 32767 : ((new_abs < 0) ? 0 : new_abs);
    new_abs = if new_abs > 32767 {
        32767
    } else if new_abs < 0 {
        0
    } else {
        new_abs
    };
    
    (w_sign * new_abs) as i16
}

fn test_neuron() -> NeuronType {
    let mut nt = NeuronType::default();
    nt.gsop_potentiation = 80;
    nt.gsop_depression = 40;
    nt.ltm_slot_count = 80;
    nt.slot_decay_ltm = 120; // ~0.93x
    nt.slot_decay_wm = 64;   // ~0.50x
    nt.inertia_curve = [128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8];
    nt
}

#[test]
fn test_gsop_potentiation_basic() {
    let nt = test_neuron();
    let w = emulate_gsop_math(100, true, 0, &nt);
    // weight=100 -> rank=0 -> inertia=128
    // delta_pot = (80 * 128) >> 7 = 80
    // decay (LTM) = 120. delta = (80 * 120) >> 7 = 75
    // new_abs = 100 + 75 = 175
    assert_eq!(w, 175);
}

#[test]
fn test_gsop_depression_basic() {
    let nt = test_neuron();
    let w = emulate_gsop_math(100, false, 0, &nt);
    // weight=100 -> rank=0 -> inertia=128
    // delta_dep = (40 * 128) >> 7 = 40
    // delta = -40
    // decay (LTM) = 120. delta = (-40 * 120) >> 7 = -38
    // new_abs = 100 - 38 = 62
    assert_eq!(w, 62);
}

#[test]
fn test_gsop_clamp_max() {
    let nt = test_neuron();
    // Максимальный вес. potentiation не должен пробить 32767
    let w = emulate_gsop_math(32767, true, 0, &nt);
    assert_eq!(w, 32767);
}

#[test]
fn test_gsop_clamp_zero() {
    let nt = test_neuron();
    // Вес 1, depression. Должен упереться в 0, но не сменить знак!
    let w = emulate_gsop_math(1, false, 0, &nt);
    assert_eq!(w, 0);
}

#[test]
fn test_gsop_sign_preserved() {
    let nt = test_neuron();
    // Inhibitory synapse (-500)
    // Потенциация делает вес "сильнее", то есть модуль растёт.
    // rank=0, delta=75 (см. первый тест)
    // new_abs = 500 + 75 = 575 -> sign * 575 = -575
    let w = emulate_gsop_math(-500, true, 0, &nt);
    assert_eq!(w, -575);

    // Депрессия делает вес слабее (модуль падает)
    // rank=0, delta=-38
    // new_abs = 500 - 38 = 462 -> sign * 462 = -462
    let w2 = emulate_gsop_math(-500, false, 0, &nt);
    assert_eq!(w2, -462);
}

#[test]
fn test_inertia_rank_boundaries() {
    let mut nt = test_neuron();
    // Custom inertia, чтобы точно видеть ранги
    nt.inertia_curve = [0; 16];
	nt.inertia_curve[0] = 64;
	nt.inertia_curve[1] = 128;
	nt.inertia_curve[15] = 255;

    // rank 0: 0..2047
    let w0 = emulate_gsop_math(2000, true, 0, &nt);
    // delta_pot = (80 * 64) >> 7 = 40. decay=120 -> (40*120)>>7 = 37
    assert_eq!(w0, 2037);

    // rank 1: 2048..4095
    let w1 = emulate_gsop_math(2048, true, 0, &nt);
    // delta_pot = (80 * 128) >> 7 = 80. decay=120 -> (80*120)>>7 = 75
    assert_eq!(w1, 2123);

    // rank 15: 30720..32767
    let w15 = emulate_gsop_math(32000, true, 0, &nt);
    // delta_pot = (80 * 255) >> 7 = 159. decay=120 -> (159*120)>>7 = 149
    assert_eq!(w15, 32149);
}

#[test]
fn test_slot_decay_ltm_vs_wm() {
    let mut nt = test_neuron();
    // LTM: 120/128, WM: 64/128
    nt.gsop_potentiation = 128; // для ровного счета. 128 * 128(inertia) = 16384 >> 7 = 128
    
    // Slot 0 < 80. LTM. delta = 128 * 120 >> 7 = 120
    let w_ltm = emulate_gsop_math(100, true, 0, &nt);
    assert_eq!(w_ltm, 220); // 100 + 120

    // Slot 80 >= 80. WM. delta = 128 * 64 >> 7 = 64
    let w_wm = emulate_gsop_math(100, true, 80, &nt);
    assert_eq!(w_wm, 164); // 100 + 64
}
