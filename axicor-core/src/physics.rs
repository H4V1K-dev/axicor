/// Physics constants recalculation pipeline at startup (Spec 01 1.5).
///
/// At startup, the engine takes human-readable values from the config and calculates
/// "raw" GPU constants. The hot loop does not perform multiplications  it operates
/// on pre-calculated numbers from Constant Memory.
///
/// Invariant 1.6: signal_speed_um_tick % segment_length_um == 0.
/// Violation  return Err before any GPU-upload.
///
/// Derived physical constants ready for loading into GPU Constant Memory.
///
/// Calculated once at startup via `compute_derived_physics`.
/// Fields are intentionally flat  direct mapping to C-structure for `cudaMemcpyToSymbol`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedPhysics {
    /// Signal speed in m/tick (1.5 p.1).
    /// GPU adds this to the head position every tick.
    pub signal_speed_um_tick: u32,
    /// Length of one segment in m = voxel_size_um  segment_length_voxels.
    pub segment_length_um: u32,
    /// Discrete speed: segments/tick (1.5 p.2).
    /// `v_seg = signal_speed_um_tick / segment_length_um`  guaranteed integer.
    /// GPU: `axon_head += v_seg` per tick. No floats, no interpolation.
    pub v_seg: u32,
}

/// Calculates `DerivedPhysics` from config values  recalculation pipeline 1.5.
///
/// Returns `Err` if invariant 1.6 is violated (`v_seg` is fractional).
///
/// # Arguments
/// - `signal_speed_um_tick`  from `simulation.signal_speed_um_tick`
/// - `voxel_size_um`         from `simulation.voxel_size_um`
/// - `segment_length_vox`    from `simulation.segment_length_voxels`
///
/// # Example (config from spec)
/// ```
/// # use axicor_core::physics::compute_derived_physics;
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
            "1.6 violation: v_seg ({v_seg_f32:.4}) is not an integer. \
             Physically impossible for Integer Physics engine. \
             Check simulation.signal_speed_m_s ({} m/s) and simulation.voxel_size_um ({} um).",
            signal_speed_m_s, voxel_size_um
        ));
    }

    Ok(DerivedPhysics {
        signal_speed_um_tick: signal_speed_um_tick.round() as u32,
        segment_length_um: segment_length_um.round() as u32,
        v_seg,
    })
}

// ---------------------------------------------------------------------------
// GLIF Dynamics (Spec 03 2, 3)
// ---------------------------------------------------------------------------

/// Calculates new membrane voltage (Integer GLIF).
///
/// `dV = -(V - V_rest) / leak_rate + I_input`
///
/// Integer division (truncation toward zero)  determinism on CPU and GPU.
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
// Homeostasis (Spec 03 3.2  Adaptive Threshold)
// ---------------------------------------------------------------------------

/// Branchless update of adaptive threshold.
///
/// Logic: `max(0, offset - decay) + (is_spiking ? penalty : 0)`
///
/// Uses `wrapping_sub` + arithmetic shift for branchless clamp to 0.
/// No panics in debug builds on underflow.
#[inline(always)]
pub const fn update_homeostasis(
    threshold_offset: i32,
    decay: u16,
    is_spiking: bool,
    penalty: i32,
) -> i32 {
    // wrapping_sub: safe even if offset < decay (yields negative i32)
    let decayed = threshold_offset.wrapping_sub(decay as i32);
    // Arithmetic shift: 0xFFFFFFFF if < 0, else 0x00000000
    let mask = decayed >> 31;
    // Branchless max(0, decayed): zero out if negative
    let clamped = decayed & !mask;

    // Penalty added only on spike
    let spike_penalty = if is_spiking { penalty } else { 0 };
    clamped + spike_penalty
}

// ---------------------------------------------------------------------------
// GSOP Plasticity (Spec 04 2.4, 2.5)
// ---------------------------------------------------------------------------

/// Inertia Rank: split |weight| range (0..2.14B) into 16 ranks.
/// `rank = abs_w >> 27`. This is 1 ALU cycle, no branching.
#[inline(always)]
pub const fn inertia_rank(abs_weight: i32) -> usize {
    let rank = (abs_weight >> 27) as usize; // 2.14B >> 27 = 15
    if rank > 15 {
        15
    } else {
        rank
    }
}

/// Calculates new synaptic weight value according to GSOP (STDP) algorithm.
/// Math is optimized for fixed-point integer calculations.
#[allow(clippy::too_many_arguments)]
pub fn compute_gsop_weight(
    current_weight: i32,
    dopamine: i16,
    d1_affinity: u8,
    d2_affinity: u8,
    gsop_potentiation: u16,
    gsop_depression: u16,
    inertia: u8,
    dist_to_spike: Option<u32>,
    burst_mult: u8,
) -> i32 {
    let sign = if current_weight >= 0 { 1 } else { -1 };
    let abs_w = current_weight.abs();

    // 1. Dopamine modulation (D1 boosts LTP, D2 suppresses LTD on reward)
    // Integer physics: (i16 * u8) >> 7
    let pot_mod = ((dopamine as i32) * (d1_affinity as i32)) >> 7;
    let dep_mod = ((dopamine as i32) * (d2_affinity as i32)) >> 7;

    let raw_pot = (gsop_potentiation as i32) + pot_mod;
    let raw_dep = (gsop_depression as i32) - dep_mod;

    // Protection against negative depression (Dead Zone Guard)
    let final_dep = if raw_dep < 0 { 0 } else { raw_dep };

    // 2. Base delta impulse with inertia and burst strength
    // [BIT ACCURACY] Multiplication BEFORE shift strictly as in CUDA
    let delta_pot = (raw_pot * (inertia as i32) * (burst_mult as i32)) >> 7;
    let delta_dep = (final_dep * (inertia as i32) * (burst_mult as i32)) >> 7;

    // 3. Spatial Cooling
    // Farther spike from synapse = weaker LTP effect
    let is_active = dist_to_spike.is_some();
    let min_dist = dist_to_spike.unwrap_or(u32::MAX);
    let cooling_shift = if is_active { min_dist >> 4 } else { 0 };

    // 4. Final delta
    let delta = if is_active {
        delta_pot >> cooling_shift
    } else {
        -delta_dep
    };

    // 5. Global Decay coefficient
    let decay = 128i32; // 1.0 in Fixed Point 7
    let delta = (delta * decay) >> 7; // [DOD FIX] Single Spatial Cooling

    // 6. Apply change and clamp to i32 limits with Headroom
    // 2.14B ceiling leaves room to i32::MAX to prevent wrap-around.
    let mut new_abs = abs_w + delta;
    new_abs &= !(new_abs >> 31);
    if new_abs > 2140000000 {
        new_abs = 2140000000;
    }
    // 7. Restore sign
    sign * new_abs
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
        assert_eq!(
            // [DOD FIX] burst_mult = 1 (Hardware burst_count, not fixed-point)
            compute_gsop_weight(100, 0, 0, 0, 80, 40, 128, Some(0), 1),
            180
        );
    }

    #[test]
    fn test_gsop_depression_basic() {
        assert_eq!(
            // [DOD FIX] burst_mult = 1
            compute_gsop_weight(100, 0, 0, 0, 80, 40, 128, None, 1),
            60
        );
    }

    #[test]
    fn test_gsop_i32_limits() {
        let w = compute_gsop_weight(-1000000, 0, 0, 0, 80, 40, 128, Some(0), 1);
        assert_eq!(w, -1000080);

        let w2 = compute_gsop_weight(-1000000, 0, 0, 0, 80, 40, 128, None, 1);
        assert_eq!(w2, -999960);
    }

    #[test]
    fn test_gsop_sign_preservation_negative() {
        let w = compute_gsop_weight(-500, 0, 0, 0, 80, 40, 128, None, 1);
        assert!(w <= 0, "Dale's Law violated");
        assert_eq!(w, -460);
    }

    #[test]
    fn test_gsop_sign_preservation_depression_to_zero() {
        let w = compute_gsop_weight(-5, 0, 0, 0, 80, 40, 128, None, 1);
        assert_eq!(w, 0);
    }

    #[test]
    fn test_gsop_clamp_max() {
        let w = compute_gsop_weight(2140000000, 0, 0, 0, 80, 40, 128, Some(0), 1);
        assert_eq!(w, 2140000000);
    }

    #[test]
    fn test_gsop_inertia_rank_calculation() {
        // [DOD FIX] Mass Domain >> 27 (1 rank = 134,217,728)
        assert_eq!(inertia_rank(0), 0);
        assert_eq!(inertia_rank(134_217_727), 0);
        assert_eq!(inertia_rank(134_217_728), 1);
        assert_eq!(inertia_rank(268_435_455), 1);
        assert_eq!(inertia_rank(268_435_456), 2);
        assert_eq!(inertia_rank(1_879_048_192), 14);
        assert_eq!(inertia_rank(2_140_000_000), 15);
    }

    #[test]
    fn test_gsop_cooling_effect() {
        // [DOD FIX] burst_mult = 1. Double Cooling shift removed!
        let w_close = compute_gsop_weight(1000, 0, 0, 0, 128, 40, 128, Some(0), 1);
        let w_far = compute_gsop_weight(1000, 0, 0, 0, 128, 40, 128, Some(64), 1);

        assert!(w_close > w_far);
        assert_eq!(w_close, 1128); // 1000 + 128
        assert_eq!(w_far, 1008); // 1000 + (128 >> 4) = 1000 + 8
    }
}
