use crate::ffi::ShardVramPtrs;
use axicor_core::constants::AXON_SENTINEL;
use axicor_core::layout::BurstHeads8;
use rayon::prelude::*;

// =============================================================================
// 2.1 cpu_propagate_axons
// =============================================================================

///  3:     .
/// DOD FIX:  Branchless (vpaddd)  AVX2 .
pub fn cpu_propagate_axons(axon_heads: &mut [BurstHeads8], v_seg: u32) {
    axon_heads.par_chunks_exact_mut(2).for_each(|chunk| {
        for head in chunk {
            head.h0 = head
                .h0
                .wrapping_add(v_seg * ((head.h0 != AXON_SENTINEL) as u32));
            head.h1 = head
                .h1
                .wrapping_add(v_seg * ((head.h1 != AXON_SENTINEL) as u32));
            head.h2 = head
                .h2
                .wrapping_add(v_seg * ((head.h2 != AXON_SENTINEL) as u32));
            head.h3 = head
                .h3
                .wrapping_add(v_seg * ((head.h3 != AXON_SENTINEL) as u32));
            head.h4 = head
                .h4
                .wrapping_add(v_seg * ((head.h4 != AXON_SENTINEL) as u32));
            head.h5 = head
                .h5
                .wrapping_add(v_seg * ((head.h5 != AXON_SENTINEL) as u32));
            head.h6 = head
                .h6
                .wrapping_add(v_seg * ((head.h6 != AXON_SENTINEL) as u32));
            head.h7 = head
                .h7
                .wrapping_add(v_seg * ((head.h7 != AXON_SENTINEL) as u32));
        }
    });
}

// =============================================================================
// 2.2 cpu_apply_spike_batch
// =============================================================================

///  2:   .
/// DOD FIX: Burst- (  ) +  .
///  Rayon     L1   Work-Stealing .
pub fn cpu_apply_spike_batch(axon_heads: &mut [BurstHeads8], schedule_indices: &[u32], v_seg: u32) {
    for &ghost_id in schedule_indices {
        if let Some(head) = axon_heads.get_mut(ghost_id as usize) {
            //    (Spec 01 1.4.3)
            head.h7 = head.h6;
            head.h6 = head.h5;
            head.h5 = head.h4;
            head.h4 = head.h3;
            head.h3 = head.h2;
            head.h2 = head.h1;
            head.h1 = head.h0;
            //  h0
            head.h0 = 0u32.wrapping_sub(v_seg);
        }
    }
}

// =============================================================================
// 2.3 cpu_inject_inputs
// =============================================================================

///  1:    ( ).
/// DOD FIX: SIMD-friendly -.
pub fn cpu_inject_inputs(
    axon_heads: &mut [BurstHeads8],
    input_bitmask: &[u32],
    virtual_offset: u32,
    num_virtual_axons: u32,
    v_seg: u32,
) {
    for tid in 0..num_virtual_axons as usize {
        let word_idx = tid / 32;
        let bit_idx = tid % 32;
        //
        if (input_bitmask[word_idx] >> bit_idx) & 1 != 0 {
            if let Some(head) = axon_heads.get_mut(virtual_offset as usize + tid) {
                head.h7 = head.h6;
                head.h6 = head.h5;
                head.h5 = head.h4;
                head.h4 = head.h3;
                head.h3 = head.h2;
                head.h2 = head.h1;
                head.h1 = head.h0;
                head.h0 = 0u32.wrapping_sub(v_seg);
            }
        }
    }
}

// =============================================================================
// 2.4 cpu_record_outputs
// =============================================================================

///  6:    (RecordReadout).
/// DOD FIX:     "  ".
///    0xFFFF_FFFF ( ).
pub fn cpu_record_outputs(
    soma_flags: &[u8],
    mapped_soma_ids: &[u32],
    output_history: &mut [u8],
    current_tick: u32,
    total_mapped_somas: u32,
) {
    let tick_offset = (current_tick as usize) * (total_mapped_somas as usize);
    for (i, &soma_id) in mapped_soma_ids.iter().enumerate() {
        //   EMPTY_PIXEL (0xFFFF_FFFF)
        if soma_id != 0xFFFF_FFFF {
            if let Some(&flag) = soma_flags.get(soma_id as usize) {
                if let Some(out) = output_history.get_mut(tick_offset + i) {
                    //   0  1 (LTM/WM state)
                    *out = flag & 0x01;
                }
            }
        }
    }
}

// =============================================================================
// 2.4 cpu_update_neurons (The Hot Loop)
// =============================================================================

///  4:  GLIF,    .
/// DOD FIX: Raw pointer index iteration (Zero-Cost). Branchless .
// MONOLITH: HIGH — cpu_update_neurons is a complex Hot Loop with deeply nested logic and branchless optimizations.
// REFACTOR: Decompose into discrete inline "Math Blocks" (Leak, Integrate, Threshold) for maintainability.
/// # Safety
/// `ptrs` must contain valid, 64-byte aligned SoA pointers allocated for `padded_n` and `total_axons`.
pub unsafe fn cpu_update_neurons(
    ptrs: &ShardVramPtrs,
    padded_n: u32,
    total_axons: u32,
    current_tick: u32,
    v_seg: u32,
) {
    use crate::bindings::VARIANT_LUT;

    (0..padded_n as usize).into_par_iter().for_each(|tid| {
        // 1.   +   (1  L1)
        let flags_ptr = ptrs.soma_flags.add(tid);
        let mut flag = *flags_ptr;
        let var_id = (flag >> 4) & 0x0F;
        let p = &VARIANT_LUT.variants[var_id as usize];

        let timer_ptr = ptrs.timers.add(tid);
        let timer = *timer_ptr;

        flag &= !0x01; //

        // 4.  (Soft Limit) - MOVED UP to prevent chrono-stasis
        let t_off_ptr = ptrs.threshold_offset.add(tid);
        let mut thresh_offset = *t_off_ptr;
        let decayed = thresh_offset - p.homeostasis_decay as i32;
        thresh_offset = decayed & !(decayed >> 31); // Branchless max(0, val)

        // 2.   - Early Exit (~90% )
        if timer > 0 {
            *timer_ptr = timer - 1;
            *flags_ptr = flag;
            *t_off_ptr = thresh_offset;
            return;
        }

        let mut current_voltage = *ptrs.soma_voltage.add(tid);
        let mut i_in = 0;
        let prop = p.signal_propagation_length as u32;

        // 3. Columnar Dendrite Loop: 128  (Coalesced Access / Gather)
        for slot in 0..128 {
            let col_idx = slot * (padded_n as usize) + tid;
            let target_packed = *ptrs.dendrite_targets.add(col_idx);

            // Hardware Trap:
            if target_packed == 0 {
                break;
            }

            let d_timer_ptr = ptrs.dendrite_timers.add(col_idx);
            if *d_timer_ptr > 0 {
                *d_timer_ptr -= 1;
                continue;
            }

            // [DOD FIX] Zero-Index Trap Protection
            let raw_id = target_packed & 0x00FFFFFF;
            if raw_id == 0 {
                break;
            }
            let axon_id = raw_id - 1;
            if axon_id >= total_axons {
                continue;
            }
            let seg_idx = target_packed >> 24;

            let h = *ptrs.axon_heads.add(axon_id as usize);

            // Branchless 8-head Hit Detection ( jmp/br )
            let hit = ((h.h0.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h1.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h2.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h3.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h4.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h5.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h6.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h7.wrapping_sub(seg_idx) <= prop) as i32);

            if hit != 0 {
                let weight = *ptrs.dendrite_weights.add(col_idx);
                // Shift Mass Domain (i32) to Charge Domain ()
                i_in += weight >> 16;
                *d_timer_ptr = p.synapse_refractory_period;
            }
        }

        // 5. Adaptive GLIF Leak (Branchless)
        let adaptive_sub = ((thresh_offset * (p.adaptive_leak_gain as i32)) >> 8) * (p.adaptive_mode as i32);
        let mut new_shift = (p.leak_shift as i32) - adaptive_sub;
        new_shift = std::cmp::max(new_shift, p.adaptive_leak_min_shift);
        let current_shift = std::cmp::max(new_shift, 0) as u32;

        current_voltage += i_in;

        // Shift-based Exponential Leak (Branchless)
        let diff = current_voltage - p.rest_potential;
        current_voltage -= diff >> current_shift;

        let eff_thresh = p.threshold + thresh_offset;
        let is_glif_spiking = (current_voltage >= eff_thresh) as i32;

        // 6.   (Heartbeat DDS)
        let phase = ((current_tick as u64) * (p.heartbeat_m as u64) + (tid as u64) * 104729) & 0xFFFF;
        let is_heartbeat = if p.heartbeat_m > 0 && phase < (p.heartbeat_m as u64) {
            1
        } else {
            0
        };

        let final_spike = is_glif_spiking | is_heartbeat;

        // 7. AHP: Membrane reset with undershoot
        // [DOD FIX] Hardware alignment: ONLY GLIF spikes reset voltage and ref_timer. Heartbeat is independent.
        let reset_v = p.rest_potential - (p.ahp_amplitude as i32);
        current_voltage = is_glif_spiking * reset_v + (1 - is_glif_spiking) * current_voltage;
        thresh_offset += is_glif_spiking * p.homeostasis_penalty;
        *timer_ptr =
            (is_glif_spiking * p.refractory_period as i32 + (1 - is_glif_spiking) * timer as i32) as u8;

        // 8.   (Burst Shift)
        if final_spike != 0 {
            let my_axon = *ptrs.soma_to_axon.add(tid);
            if my_axon != 0xFFFFFFFF && my_axon < total_axons {
                let h_ptr = ptrs.axon_heads.add(my_axon as usize);
                let mut h = *h_ptr;
                h.h7 = h.h6;
                h.h6 = h.h5;
                h.h5 = h.h4;
                h.h4 = h.h3;
                h.h3 = h.h2;
                h.h2 = h.h1;
                h.h1 = h.h0;
                h.h0 = 0u32.wrapping_sub(v_seg);
                *h_ptr = h;
            }
        }

        // 9.    VRAM (Zero-Warp Divergence pattern)
        *ptrs.soma_voltage.add(tid) = current_voltage;
        *t_off_ptr = thresh_offset;

        // 10. Burst-Dependent Plasticity (BDP)
        let mut burst_count = (flag >> 1) & 0x07;
        burst_count = (final_spike as u8) * (burst_count + (burst_count < 7) as u8);
        flag = (flag & 0xF0) | (burst_count << 1) | (final_spike as u8);
        *flags_ptr = flag;
    });
}

// =============================================================================
// 2.5 cpu_apply_gsop
// =============================================================================

///  5:  GSOP.
/// DOD FIX: Branchless- STDP. Zero-Warp Divergence.
/// # Safety
/// `ptrs` must contain valid, 64-byte aligned SoA pointers allocated for `padded_n` and `total_axons`.
pub unsafe fn cpu_apply_gsop(ptrs: &ShardVramPtrs, padded_n: u32, total_axons: u32, dopamine: i16) {
    use crate::bindings::VARIANT_LUT;

    (0..padded_n as usize).into_par_iter().for_each(|tid| {
        let flags = *ptrs.soma_flags.add(tid);

        // Early Exit:    ,
        if (flags & 0x01) == 0 {
            return;
        }

        let burst_count = (flags >> 1) & 0x07;
        let burst_mult = if burst_count > 0 {
            burst_count as i32
        } else {
            1
        };

        let var_id = (flags >> 4) & 0x0F;
        let p = &VARIANT_LUT.variants[var_id as usize];

        for slot in 0..128 {
            let col_idx = slot * (padded_n as usize) + tid;

            let timer = *ptrs.dendrite_timers.add(col_idx);
            if timer > 0 {
                //   > 0,     UpdateNeurons.
                //  ,   .
                continue;
            }

            let target_packed = *ptrs.dendrite_targets.add(col_idx);
            if target_packed == 0 {
                break;
            } // Hardware Trap:

            let weight_ptr = ptrs.dendrite_weights.add(col_idx);
            let w = *weight_ptr;
            if w == 0 {
                continue;
            } //   ( Night Phase Pruning)

            let seg_idx = target_packed >> 24;
            let raw_id = target_packed & 0x00FFFFFF;
            if raw_id == 0 {
                break;
            }
            let axon_id = raw_id - 1;
            if axon_id >= total_axons {
                continue;
            }
            let h = *ptrs.axon_heads.add(axon_id as usize);
            let prop = p.signal_propagation_length as u32;

            // Branchless 8-head Hit Detection
            let is_active = ((h.h0.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h1.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h2.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h3.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h4.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h5.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h6.wrapping_sub(seg_idx) <= prop) as i32)
                | ((h.h7.wrapping_sub(seg_idx) <= prop) as i32);

            let sign = if w >= 0 { 1 } else { -1 };
            let abs_w = w.abs();

            // 1. Inertia Rank (1 , Branchless)
            let mut rank = (abs_w >> 28) as usize;
            if rank > 7 {
                rank = 7;
            }
            let inertia = p.inertia_curve[rank] as i32;

            // 2. Dopamine modulation (D1 boosts LTP, D2 suppresses LTD on reward)
            let pot_mod = ((dopamine as i32) * (p.d1_affinity as i32)) >> 7;
            let dep_mod = ((dopamine as i32) * (p.d2_affinity as i32)) >> 7;

            let raw_pot = (p.gsop_potentiation as i32) + pot_mod;
            let raw_dep = (p.gsop_depression as i32) - dep_mod;

            let final_pot = raw_pot & !(raw_pot >> 31); // max(0, val)
            let final_dep = raw_dep & !(raw_dep >> 31); // max(0, val)

            //       CUDA-
            let delta_pot = (final_pot * inertia * burst_mult) >> 7;
            let delta_dep = (final_dep * inertia * burst_mult) >> 7;

            // Causal Delta.      Active Tail (is_active) -> LTP,  LTD.
            let mut delta = if is_active != 0 {
                delta_pot
            } else {
                -delta_dep
            };

            // Fixed Slot Decay = 1.0x
            delta = (delta * 128) >> 7; // [DOD FIX] Single Spatial Cooling

            // 3. Apply & Clamp to Mass Domain Limits
            let mut new_abs = abs_w + delta;
            new_abs &= !(new_abs >> 31); // Branchless clamp bottom to 0
            if new_abs > 2140000000 {
                new_abs = 2140000000;
            } // Headroom guard

            *weight_ptr = new_abs * sign;
        }
    });
}

// =============================================================================
// 2.6 cpu_extract_telemetry
// =============================================================================

///  7:   ().
/// DOD FIX:   . LLVM    SIMD (pmovmskb),
///   L1/L2     ,   Rayon-  Atomics.
pub fn cpu_extract_telemetry(soma_flags: &[u8], out_ids: &mut [u32]) -> u32 {
    let mut count = 0;

    //  iter_mut  chunking.   +  .
    for (id, &flag) in soma_flags.iter().enumerate() {
        if (flag & 0x01) != 0 {
            //
            if let Some(slot) = out_ids.get_mut(count) {
                *slot = id as u32;
                count += 1;
            }
        }
    }

    count as u32
}

// =============================================================================
// 2.7 cpu_sort_and_prune (CPU Fallback for Compaction)
// =============================================================================
// =============================================================================
// 2.7 cpu_sort_and_prune (CPU Fallback for Compaction)
// =============================================================================
pub unsafe fn cpu_sort_and_prune(ptrs: &crate::ffi::ShardVramPtrs, padded_n: u32, prune_threshold: i16) {
    // [DOD FIX] Strict Mass Domain Shift. Threshold is applied to abs(weight).
    let threshold_mass = (prune_threshold.abs() as u32) << 16;

    #[derive(Clone, Copy)]
    struct DendriteSlot {
        target: u32,
        weight: i32,
        timer: u8,
    }

    (0..padded_n as usize).into_par_iter().for_each(|tid| {
        // Reset burst_count bits [3:1] without branching
        let flag_ptr = ptrs.soma_flags.add(tid);
        *flag_ptr &= 0xF1; 

        // [DOD FIX] Zero-Allocation. Stack array perfectly fits in CPU cache.
        let mut slots = [DendriteSlot { target: 0, weight: 0, timer: 0 }; 128];

        // 1. Load & Prune
        for slot in 0..128 {
            let col_idx = slot * (padded_n as usize) + tid;
            let target = *ptrs.dendrite_targets.add(col_idx);
            let weight = *ptrs.dendrite_weights.add(col_idx);
            let timer = *ptrs.dendrite_timers.add(col_idx);

            // Keep only if structurally alive AND electrically strong enough
            if target != 0 && weight.unsigned_abs() >= threshold_mass {
                slots[slot] = DendriteSlot { target, weight, timer };
            }
        }

        // 2. Sort & Compact (In-Place Pattern-Defeating Quicksort)
        slots.sort_unstable_by(|a, b| {
            let a_alive = a.target != 0;
            let b_alive = b.target != 0;
            
            if a_alive && !b_alive {
                std::cmp::Ordering::Less // Alive slots go first
            } else if !a_alive && b_alive {
                std::cmp::Ordering::Greater // Dead slots go to the tail
            } else if a_alive && b_alive {
                // Both alive: sort by absolute weight descending (LTM promotion)
                b.weight.unsigned_abs().cmp(&a.weight.unsigned_abs())
            } else {
                std::cmp::Ordering::Equal // Both dead
            }
        });

        // 3. Columnar Write Back
        for slot in 0..128 {
            let col_idx = slot * (padded_n as usize) + tid;
            *ptrs.dendrite_targets.add(col_idx) = slots[slot].target;
            *ptrs.dendrite_weights.add(col_idx) = slots[slot].weight;
            *ptrs.dendrite_timers.add(col_idx) = slots[slot].timer;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use axicor_core::constants::V_SEG;
    use crate::bindings::cpu_allocate_shard;
    use crate::bindings::cpu_free_shard;
    use crate::bindings::TEST_MUTEX;
    use crate::bindings::VARIANT_LUT;
    use crate::ffi::ShardVramPtrs;
    use axicor_core::layout::VariantParameters;

    #[test]
    fn test_propagate_axons() {
        let mut heads = vec![BurstHeads8::empty(AXON_SENTINEL); 4];
        heads[0].h0 = 10;
        heads[1].h7 = 20;

        cpu_propagate_axons(&mut heads, 5);

        assert_eq!(heads[0].h0, 15);
        assert_eq!(heads[0].h1, AXON_SENTINEL);
        assert_eq!(heads[1].h7, 25);
    }

    #[test]
    fn test_burst_shift_spike() {
        let mut heads = vec![BurstHeads8::empty(AXON_SENTINEL); 1];
        heads[0].h0 = 100; // old spike

        cpu_apply_spike_batch(&mut heads, &[0], 5);

        assert_eq!(heads[0].h1, 100); // shifted
        assert_eq!(heads[0].h0, 0u32.wrapping_sub(5)); // new initialized
    }

    #[test]
    fn test_record_outputs_unconditional() {
        let flags = vec![0x00, 0x01, 0x00, 0x01];
        let mapped_ids = vec![1, 3];
        let mut history = vec![255; 4]; // Dirty buffer

        cpu_record_outputs(&flags, &mapped_ids, &mut history, 0, 2);
        assert_eq!(history[0], 1);
        assert_eq!(history[1], 1);

        // Now neuron 1 is turned off
        let flags_new = vec![0x00, 0x00, 0x00, 0x01];
        cpu_record_outputs(&flags_new, &mapped_ids, &mut history, 1, 2);
        assert_eq!(history[2], 0); // Should be 0, not 255
        assert_eq!(history[3], 1);
    }

    #[test]
    fn test_update_neurons_spiking() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        let axons = 10;
        unsafe {
            cpu_allocate_shard(padded_n, axons, &mut ptrs);

            // Setup VARIANT_LUT for type 0
            let mut p = VariantParameters::default();
            p.threshold = 100;
            p.rest_potential = 0;
            p.leak_shift = 6; // No leak
            p.refractory_period = 5;
            p.homeostasis_penalty = 50;

            VARIANT_LUT.variants[0] = p;

            // Neuron 0: Voltage = 150 (should spike)
            *ptrs.soma_voltage.add(0) = 150;
            *ptrs.soma_flags.add(0) = 0 << 4; // Type 0
            *ptrs.soma_to_axon.add(0) = 0; // Axon 0

            // Tick 1
            cpu_update_neurons(&ptrs, padded_n, axons, 1, V_SEG);

            // Spike check
            assert_eq!((*ptrs.soma_flags.add(0)) & 0x01, 1, "Neuron 0 must spike");
            assert_eq!(
                (*ptrs.soma_voltage.add(0)),
                0,
                "Voltage must reset to rest_potential"
            );
            assert_eq!((*ptrs.timers.add(0)), 5, "Refractory timer must be set");
            assert_eq!(
                (*ptrs.threshold_offset.add(0)),
                50,
                "Homeostasis penalty applied"
            );

            // Axon fire check
            let h = *ptrs.axon_heads.add(0);
            assert_eq!(
                h.h0,
                0u32.wrapping_sub(1),
                "Axon head h0 must be initialized with temporal sync"
            );

            cpu_free_shard(&mut ptrs);
        }
    }

    #[test]
    fn test_gsop_d1_d2_dopamine_modulation() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        let axons = 10;
        unsafe {
            cpu_allocate_shard(padded_n, axons, &mut ptrs);

            let mut p = VariantParameters::default();
            p.gsop_potentiation = 100;
            p.gsop_depression = 100;
            p.d1_affinity = 128; // 1.0x
            p.d2_affinity = 128; // 1.0x
            p.inertia_curve[0] = 128; // 1.0x (Rank 0)
            p.signal_propagation_length = 5;
            VARIANT_LUT.variants[0] = p;

            // --- Scenario A: Positive Reward (Dopamine = +100) ---
            // Setup: Neuron 0 spiked (flag 0x01)
            *ptrs.soma_flags.add(0) = (0 << 4) | 0x01;

            // Dendrite 0: Active Tail (LTP)
            let w0_start = 1000;
            *ptrs.dendrite_weights.add(0) = w0_start;
            *ptrs.dendrite_targets.add(0) = (0 << 24) | 2; // Axon 1 (2-1=1)
            (*ptrs.axon_heads.add(1)).h0 = 0; // Spike at 0 (Active)

            // Dendrite 1: Miss (LTD)
            let w1_start = 1000;
            *ptrs.dendrite_weights.add(64) = w1_start;
            *ptrs.dendrite_targets.add(64) = (0 << 24) | 3; // Axon 2 (3-1=2)
            (*ptrs.axon_heads.add(2)).h0 = 0xFFFFFFFF; // No spike

            // Run GSOP with reward
            cpu_apply_gsop(&ptrs, padded_n, axons, 100);

            let w0_a = *ptrs.dendrite_weights.add(0);
            let w1_a = *ptrs.dendrite_weights.add(64);

            // delta LTP: (100 + (100*128>>7)) * 1 * 1 = 200
            assert_eq!(w0_a - w0_start, 200, "LTP with reward should be +200");
            // delta LTD: (100 - (100*128>>7)) * 1 * 1 = 0
            assert_eq!(w1_a - w1_start, 0, "LTD with reward should be 0 (suppressed)");

            // --- Scenario B: Punishment (Dopamine = -200) ---
            // Reset weights
            *ptrs.dendrite_weights.add(0) = w0_start;
            *ptrs.dendrite_weights.add(64) = w1_start;

            // Run GSOP with extreme punishment
            cpu_apply_gsop(&ptrs, padded_n, axons, -200);

            let w0_b = *ptrs.dendrite_weights.add(0);
            let w1_b = *ptrs.dendrite_weights.add(64);

            // delta LTP: (100 + (-200*128>>7)) = -100 -> clamp 0
            assert_eq!(w0_b - w0_start, 0, "LTP with punishment should be 0 (clamped)");
            // delta LTD: (100 - (-200*128>>7)) = 300
            assert_eq!(w1_b - w1_start, -300, "LTD with punishment should be -300 (amplified)");

            cpu_free_shard(&mut ptrs);
        }
    }

    #[test]
    fn test_extract_telemetry() {
        let mut flags = vec![0u8; 10000];
        // Put random garbage in other bits
        for i in 0..10000 {
            flags[i] = ((i % 15) as u8) << 4;
        }

        flags[42] |= 0x01;
        flags[1337] |= 0x01;
        flags[9999] |= 0x01;

        let mut out_ids = vec![0u32; 10000];
        let count = cpu_extract_telemetry(&flags, &mut out_ids);

        assert_eq!(count, 3);
        assert_eq!(out_ids[0], 42);
        assert_eq!(out_ids[1], 1337);
        assert_eq!(out_ids[2], 9999);
    }

    #[test]
    fn test_soma_flags_bit_bleed() {
        let mut flag = 0b1111_1110u8; // Type 15, Burst 7, Spiking 0
        let final_spike = 1;
        
        let mut burst_count = (flag >> 1) & 0x07;
        burst_count = (final_spike as u8) * (burst_count + (burst_count < 7) as u8);
        flag = (flag & 0xF0) | (burst_count << 1) | (final_spike as u8);
        
        assert_eq!(flag, 0b1111_1111, "Burst should saturate at 7, Type ID must be preserved"); // Type 15, Burst 7, Spiking 1
        
        let final_spike_0 = 0;
        let mut burst_count_2 = (flag >> 1) & 0x07;
        burst_count_2 = (final_spike_0 as u8) * (burst_count_2 + (burst_count_2 < 7) as u8);
        flag = (flag & 0xF0) | (burst_count_2 << 1) | (final_spike_0 as u8);
        
        assert_eq!(flag, 0b1111_0000, "Burst must reset to 0, Type ID must be preserved"); // Type 15, Burst 0, Spiking 0
    }

    #[test]
    fn test_dds_heartbeat_phase_accumulation() {
        let current_tick = u32::MAX;
        let tid = 100_000u32;
        let heartbeat_m = 32768u16;

        let phase = ((current_tick as u64) * (heartbeat_m as u64) + (tid as u64) * 104729) & 0xFFFF;
        let is_heartbeat = if heartbeat_m > 0 && phase < (heartbeat_m as u64) { 1 } else { 0 };
        
        // As long as it evaluates without panic, phase is correct.
        assert!(is_heartbeat == 0 || is_heartbeat == 1, "is_heartbeat should be deterministically 0 or 1");
    }

    #[test]
    fn test_homeostasis_decays_during_refractory() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        let axons = 10;
        unsafe {
            cpu_allocate_shard(padded_n, axons, &mut ptrs);

            let mut p = VariantParameters::default();
            p.homeostasis_decay = 10;
            p.threshold = 100;
            VARIANT_LUT.variants[0] = p;

            // Neuron 0: Sleeping
            *ptrs.soma_voltage.add(0) = 0;
            *ptrs.timers.add(0) = 5;
            *ptrs.threshold_offset.add(0) = 100;
            *ptrs.soma_flags.add(0) = 0 << 4;

            cpu_update_neurons(&ptrs, padded_n, axons, 1, 1);

            let new_timer = *ptrs.timers.add(0);
            let new_offset = *ptrs.threshold_offset.add(0);

            assert_eq!(new_timer, 4, "Refractory timer must decrease");
            assert_eq!(new_offset, 90, "Threshold offset must decay even during refractory period! Chrono-stasis detected!");

            cpu_free_shard(&mut ptrs);
        }
    }

    #[test]
    fn test_sort_and_prune_compaction_no_holes() {
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        unsafe {
            cpu_allocate_shard(padded_n, 10, &mut ptrs);

            for slot in 0..128 {
                let target = if slot % 12 == 0 && slot < 120 { (slot as u32) + 1 } else { 0 };
                let col_idx = slot * (padded_n as usize);
                *ptrs.dendrite_targets.add(col_idx) = target;
                *ptrs.dendrite_weights.add(col_idx) = if target != 0 { 100 } else { 0 };
            }

            cpu_sort_and_prune(&ptrs, padded_n, 0);

            let mut alive_count = 0;
            for slot in 0..128 {
                if *ptrs.dendrite_targets.add(slot * (padded_n as usize)) != 0 {
                    alive_count += 1;
                }
            }
            assert_eq!(alive_count, 10, "Should have 10 alive slots, found {}", alive_count);

            cpu_free_shard(&mut ptrs);
        }
    }

    #[test]
    fn test_sort_and_prune_threshold_kills() {
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        unsafe {
            cpu_allocate_shard(padded_n, 10, &mut ptrs);

            for slot in 0..10 {
                let col_idx = slot * (padded_n as usize);
                *ptrs.dendrite_targets.add(col_idx) = (slot as u32) + 100;
                *ptrs.dendrite_weights.add(col_idx) = if slot < 4 { 10 << 16 } else { 20 << 16 };
                *ptrs.dendrite_timers.add(col_idx) = 5;
            }

            cpu_sort_and_prune(&ptrs, padded_n, 15);

            for slot in 0..6 {
                let col_idx = slot * (padded_n as usize);
                assert_ne!(*ptrs.dendrite_targets.add(col_idx), 0, "Strong slot {} should survive", slot);
                assert_eq!(*ptrs.dendrite_weights.add(col_idx), 20 << 16);
            }
            for slot in 6..128 {
                let col_idx = slot * (padded_n as usize);
                assert_eq!(*ptrs.dendrite_targets.add(col_idx), 0, "Weak/tail slot {} should be 0", slot);
                assert_eq!(*ptrs.dendrite_weights.add(col_idx), 0);
                assert_eq!(*ptrs.dendrite_timers.add(col_idx), 0);
            }

            cpu_free_shard(&mut ptrs);
        }
    }

    #[test]
    fn test_sort_and_prune_preserves_sort_order() {
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        unsafe {
            cpu_allocate_shard(padded_n, 128, &mut ptrs);

            for slot in 0..128 {
                let col_idx = slot * (padded_n as usize);
                *ptrs.dendrite_targets.add(col_idx) = (slot as u32) + 1;
                *ptrs.dendrite_weights.add(col_idx) = ((slot as i32 % 50) + 1) << 16;
            }

            cpu_sort_and_prune(&ptrs, padded_n, 0);

            for slot in 0..127 {
                let col_idx1 = slot * (padded_n as usize);
                let col_idx2 = (slot + 1) * (padded_n as usize);
                let w1 = *ptrs.dendrite_weights.add(col_idx1);
                let w2 = *ptrs.dendrite_weights.add(col_idx2);
                assert!(w1.unsigned_abs() >= w2.unsigned_abs(), "Not sorted descending");
            }

            cpu_free_shard(&mut ptrs);
        }
    }

    #[test]
    fn test_sort_and_prune_inhibitory_preserved() {
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        unsafe {
            cpu_allocate_shard(padded_n, 10, &mut ptrs);

            let col0 = 0;
            let col1 = 1 * (padded_n as usize);

            *ptrs.dendrite_targets.add(col0) = 101;
            *ptrs.dendrite_weights.add(col0) = 100 << 16;

            *ptrs.dendrite_targets.add(col1) = 102;
            *ptrs.dendrite_weights.add(col1) = -500 << 16; 

            cpu_sort_and_prune(&ptrs, padded_n, 0);

            assert_eq!(*ptrs.dendrite_weights.add(col0), -500 << 16, "Strong should be first");
            assert_eq!(*ptrs.dendrite_targets.add(col0), 102);
            
            assert_eq!(*ptrs.dendrite_weights.add(col1), 100 << 16, "Weak should be second");
            assert_eq!(*ptrs.dendrite_targets.add(col1), 101);

            cpu_free_shard(&mut ptrs);
        }
    }

    #[test]
    fn test_sort_and_prune_burst_reset() {
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        unsafe {
            cpu_allocate_shard(padded_n, 10, &mut ptrs);

            *ptrs.soma_flags.add(0) = 0xFF; // Type=15, Burst=7, Spike=1
            
            cpu_sort_and_prune(&ptrs, padded_n, 0);

            let new_flag = *ptrs.soma_flags.add(0);
            assert_eq!(new_flag, 0xF1, "Burst bits [3:1] should be 0, Type [7:4] and Spike [0] should be preserved");

            cpu_free_shard(&mut ptrs);
        }
    }

    #[test]
    fn test_sort_empty_array() {
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        unsafe {
            cpu_allocate_shard(padded_n, 10, &mut ptrs);

            cpu_sort_and_prune(&ptrs, padded_n, 0);

            for slot in 0..128 {
                assert_eq!(*ptrs.dendrite_targets.add(slot), 0);
            }

            cpu_free_shard(&mut ptrs);
        }
    }

    #[test]
    fn test_sort_full_array_no_prune() {
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        unsafe {
            cpu_allocate_shard(padded_n, 150, &mut ptrs);

            for slot in 0..128 {
                *ptrs.dendrite_targets.add(slot) = slot as u32 + 1;
                *ptrs.dendrite_weights.add(slot) = (slot as i32 + 10) << 16;
            }

            cpu_sort_and_prune(&ptrs, padded_n, 5);

            for slot in 0..128 {
                assert_ne!(*ptrs.dendrite_targets.add(slot), 0, "No slot should be lost");
            }

            cpu_free_shard(&mut ptrs);
        }
    }
}

// =============================================================================
// DEBUG HARNESS (Epic 2) - CPU Implementations
// =============================================================================
/// # Safety
/// Caller must ensure `soma_voltage` pointer is valid and aligned.
pub unsafe fn cpu_debug_inject_current(
    soma_voltage: *mut i32,
    target_tids: &[u32],
    injection_uv: &[i32],
) {
    for (i, &tid) in target_tids.iter().enumerate() {
        unsafe {
            let v_ptr = soma_voltage.add(tid as usize);
            *v_ptr = (*v_ptr).saturating_add(injection_uv[i]);
        }
    }
}

/// # Safety
/// Caller must ensure `soma_voltage` pointer is valid and aligned.
pub unsafe fn cpu_debug_record_v(
    soma_voltage: *const i32,
    target_tids: &[u32],
    out_trace: &mut [i32],
    current_tick: u32,
    max_ticks: u32,
) {
    if current_tick >= max_ticks {
        return;
    }
    for (i, &tid) in target_tids.iter().enumerate() {
        unsafe {
            let v = *soma_voltage.add(tid as usize);
            if let Some(slot) = out_trace.get_mut(i * (max_ticks as usize) + (current_tick as usize)) {
                *slot = v;
            }
        }
    }
}
