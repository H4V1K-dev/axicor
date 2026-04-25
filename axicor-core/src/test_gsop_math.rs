/// CPU emulation and tests for ApplyGSOP kernel math (Spec 1.3)
/// Verifies formulas for potentiation, depression, clamp, spatial cooling, and inertia rank.
use crate::config::blueprints::NeuronType;

/// Full copy of branchless logic from `physics.cu -> apply_gsop_kernel`
fn emulate_gsop_math(
    weight: i32,
    dopamine: i16,
    dist_to_spike: Option<u32>,
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
    let final_dep = raw_dep.max(0);

    // 2. Inertia and bursts
    let rank = (abs_w >> 27) as usize;
    let rank_safe = rank.min(7);
    let inertia = p.inertia_curve[rank_safe] as i32;
    let burst_mult = if burst_count > 0 {
        burst_count as i32
    } else {
        1
    };

    let delta_pot = (raw_pot * inertia * burst_mult) >> 7;
    let delta_dep = (final_dep * inertia * burst_mult) >> 7;

    // 3. Spatial Cooling
    let is_active = dist_to_spike.is_some();
    let min_dist = dist_to_spike.unwrap_or(u32::MAX);
    let cooling_shift = if is_active { (min_dist >> 4) as u32 } else { 0 };

    // 4. Final delta
    let delta = if is_active {
        delta_pot >> cooling_shift
    } else {
        -delta_dep
    };

    // 5. Global Decay
    let decay = 128i32;
    let delta = (delta * decay) >> 7; // [DOD FIX] Single Spatial Cooling

    // 6. Clamp
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
    // weight=100, dopamine=0, active (dist=0), no burst
    let w = emulate_gsop_math(100, 0, Some(0), 0, &nt);
    // delta_pot = (80 * 128 * 1) >> 7 = 80
    // cooling_shift = 0
    // delta = 80 >> 0 = 80
    // decay: (80 * 128) >> 7 = 80
    // new_abs = 100 + 80 = 180
    assert_eq!(w, 180);
}

#[test]
fn test_gsop_depression_basic() {
    let nt = test_neuron();
    // weight=100, dopamine=0, inactive, no burst
    let w = emulate_gsop_math(100, 0, None, 0, &nt);
    // delta_dep = (40 * 128 * 1) >> 7 = 40
    // delta = -40
    // decay: (-40 * 128) >> 7 = -40
    // new_abs = 100 - 40 = 60
    assert_eq!(w, 60);
}

#[test]
fn test_gsop_clamp_max() {
    let nt = test_neuron();
    let w = emulate_gsop_math(2140000000, 0, Some(0), 0, &nt);
    assert_eq!(w, 2140000000);
}

#[test]
fn test_gsop_spatial_cooling() {
    let nt = test_neuron();
    // dist=32 -> cooling_shift = 32 >> 4 = 2
    let w = emulate_gsop_math(1000, 0, Some(32), 0, &nt);
    // delta_pot = (80 * 128) >> 7 = 80
    // cooling = 80 >> 2 = 20
    // decay = (20 * 128) >> (7 + 2) = 2560 >> 9 = 20
    // new_abs = 1000 + 20 = 1020
    assert_eq!(w, 1020);
}
