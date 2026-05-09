/// CPU emulation and tests for ApplyGSOP kernel math (Spec 1.3)
/// Verifies formulas for potentiation, depression, clamp, and inertia rank.
/// [PHASE 8.1] Spatial Cooling removed. 8-way Bitwise OR strategy.
use crate::config::blueprints::NeuronType;

/// Full copy of branchless logic from `physics.cu -> apply_gsop_kernel`
fn emulate_gsop_math(
    weight: i32,
    dopamine: i16,
    is_active: bool,
    burst_count: u8,
    p: &NeuronType,
) -> i32 {
    let sign = if weight >= 0 { 1 } else { -1 };
    let abs_w = weight.abs();

    // 1. Dopamine modulation
    let pot_mod = ((dopamine as i32) * (p.d1_affinity as i32)) >> 7;
    let dep_mod = ((dopamine as i32) * (p.d2_affinity as i32)) >> 7;

    let raw_pot = (p.gsop_potentiation as i32) + pot_mod;
    let raw_dep = (p.gsop_depression as i32) - dep_mod;
    
    // Branchless clamp to 0
    let final_pot = raw_pot & !(raw_pot >> 31);
    let final_dep = raw_dep & !(raw_dep >> 31);

    // 2. Inertia and bursts
    let rank = (abs_w >> 28) as usize;
    let rank_safe = rank.min(7);
    let inertia = p.inertia_curve[rank_safe] as i32;
    let burst_mult = if burst_count > 0 {
        burst_count as i32
    } else {
        1
    };

    let delta_pot = (final_pot * inertia * burst_mult) >> 7;
    let delta_dep = (final_dep * inertia * burst_mult) >> 7;

    // 3. Final delta (Spatial Cooling removed)
    let delta = if is_active {
        delta_pot
    } else {
        -delta_dep
    };

    // 4. Global Decay
    let decay = 128i32;
    let delta = (delta * decay) >> 7;

    // 5. Clamp
    let mut new_abs = abs_w + delta;
    if new_abs < 0 {
        new_abs = 0;
    }
    if new_abs > 2140000000 {
        new_abs = 2140000000;
    }

    sign * new_abs
}

fn test_neuron() -> NeuronType {
    let mut nt = NeuronType::default();
    nt.gsop_potentiation = 80;
    nt.gsop_depression = 40;
    nt.d1_affinity = 128; // 1.0x
    nt.d2_affinity = 128; // 1.0x
    nt.inertia_curve = [
        128, 112, 96, 80, 64, 48, 32, 16,
    ];
    nt
}

#[test]
fn test_gsop_potentiation_basic() {
    let nt = test_neuron();
    // weight=100, dopamine=0, active (is_active=true), no burst
    let w = emulate_gsop_math(100, 0, true, 0, &nt);
    assert_eq!(w, 180);
}

#[test]
fn test_gsop_depression_basic() {
    let nt = test_neuron();
    // weight=100, dopamine=0, inactive (is_active=false), no burst
    let w = emulate_gsop_math(100, 0, false, 0, &nt);
    assert_eq!(w, 60);
}

#[test]
fn test_gsop_clamp_max() {
    let nt = test_neuron();
    let w = emulate_gsop_math(2140000000, 0, true, 0, &nt);
    assert_eq!(w, 2140000000);
}

#[test]
fn test_gsop_inertia_dampening_effect() {
    let nt = test_neuron(); // Использует кривую [128, 112, 96, 80, 64, 48, 32, 16]
    
    // 1. Молодой синапс (Rank 0: 100 >> 28 = 0)
    let w_young_start = 100;
    let w_young_new = emulate_gsop_math(w_young_start, 0, true, 0, &nt);
    let delta_young = w_young_new - w_young_start;
    
    // 2. Монументальный синапс (Rank 7: 2.0B >> 28 = 7)
    let w_old_start = 2_000_000_000;
    let w_old_new = emulate_gsop_math(w_old_start, 0, true, 0, &nt);
    let delta_old = w_old_new - w_old_start;
    
    // Математическое доказательство (Dampening Effect)
    // Rank 0 inertia = 128. Delta = (80 * 128 * 1) >> 7 = 80.
    // Rank 7 inertia = 16.  Delta = (80 * 16 * 1) >> 7 = 10.
    assert_eq!(delta_young, 80, "Young synapse should have full potentiation");
    assert_eq!(delta_old, 10, "Monumental synapse should have dampened potentiation");
    
    // Старый синапс сопротивляется изменениям в 8 раз сильнее (80 / 10 = 8)
    assert!(delta_young > delta_old * 5, "Inertia dampening is too weak");
}
