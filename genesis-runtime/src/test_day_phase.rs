// genesis-runtime/src/test_day_phase.rs
//
// Day Phase physics pipeline integration test (mock-gpu mode).
// Verifies the 6-kernel BSP chain: InjectInputs → Propagate → UpdateNeurons
//                                  → ApplyGsop → RecordReadout.
//
// In mock mode all kernel calls are no-ops, so assertions verify memory
// layout correctness and allocation safety, not actual physics.

use genesis_core::layout::{VramState, VariantParameters, align_to_warp};
use genesis_core::constants::{MAX_DENDRITE_SLOTS, AXON_SENTINEL};
use genesis_core::config::blueprints::GenesisConstantMemory;
use crate::ffi;
use std::ptr;

#[test]
fn test_day_phase_vram_layout() {
    // Verify that we can allocate and build a VramState that satisifies
    // the genesis_core::layout::VramState contract without panicking.
    unsafe {
        let stream = ptr::null_mut();

        let padded_n    = align_to_warp(4);
        let total_axons = 64usize;

        let d_voltage          = ffi::gpu_malloc(padded_n * 4);
        let d_flags            = ffi::gpu_malloc(padded_n);
        let d_threshold_offset = ffi::gpu_malloc(padded_n * 4);
        let d_refractory_timer = ffi::gpu_malloc(padded_n);
        let d_dendrite_targets = ffi::gpu_malloc(MAX_DENDRITE_SLOTS * padded_n * 4);
        let d_dendrite_weights = ffi::gpu_malloc(MAX_DENDRITE_SLOTS * padded_n * 2);
        let d_dendrite_timers  = ffi::gpu_malloc(MAX_DENDRITE_SLOTS * padded_n);
        let d_axon_heads       = ffi::gpu_malloc(total_axons * 4);

        // Zero axon heads (sentinel init)
        let sentinels = vec![AXON_SENTINEL; total_axons];
        ffi::gpu_memcpy_host_to_device(d_axon_heads, sentinels.as_ptr() as *const _, total_axons * 4);

        let vram = VramState {
            padded_n:          padded_n as u32,
            total_axons:       total_axons as u32,
            voltage:           d_voltage as *mut i32,
            flags:             d_flags as *mut u8,
            threshold_offset:  d_threshold_offset as *mut i32,
            refractory_timer:  d_refractory_timer as *mut u8,
            soma_to_axon:      ptr::null_mut(),
            dendrite_targets:  d_dendrite_targets as *mut u32,
            dendrite_weights:  d_dendrite_weights as *mut i16,
            dendrite_timers:   d_dendrite_timers as *mut u8,
            axon_heads:        d_axon_heads as *mut u32,
            input_bitmask:     ptr::null_mut(),
            output_history:    ptr::null_mut(),
            telemetry_count:   ptr::null_mut(),
            telemetry_spikes:  ptr::null_mut(),
        };

        // Run all 6 kernels — in mock mode these are no-ops but must not segfault.
        ffi::launch_inject_inputs(vram, ptr::null(), 0, 0);
        ffi::launch_apply_spike_batch(vram, ptr::null(), 0);
        ffi::launch_propagate_axons(vram, 1);
        ffi::launch_update_neurons(vram, ptr::null(), 0);
        ffi::launch_apply_gsop(vram);
        ffi::launch_record_readout(vram, ptr::null(), ptr::null_mut(), 0, 0);
        ffi::gpu_stream_synchronize(stream);

        // Verify axon-head layout is sane after no-op kernels.
        let mut out_axons = vec![0u32; total_axons];
        ffi::gpu_memcpy_device_to_host(
            out_axons.as_mut_ptr() as *mut _,
            vram.axon_heads as *const _,
            total_axons * 4,
        );

        // Mock memcpy should have kept the sentinel values intact.
        assert_eq!(out_axons[0], AXON_SENTINEL, "Sentinel must survive no-op kernels");

        ffi::gpu_free(d_voltage);
        ffi::gpu_free(d_flags);
        ffi::gpu_free(d_threshold_offset);
        ffi::gpu_free(d_refractory_timer);
        ffi::gpu_free(d_dendrite_targets);
        ffi::gpu_free(d_dendrite_weights);
        ffi::gpu_free(d_dendrite_timers);
        ffi::gpu_free(d_axon_heads);
    }
}

#[test]
fn test_variant_parameters_layout() {
    // VariantParameters must be exactly 64 bytes per spec (16 × 64B = 1024B CUDA __constant__).
    assert_eq!(std::mem::size_of::<VariantParameters>(), 64);

    // Build a default then override the first variant.
    let mut const_mem: GenesisConstantMemory = unsafe { std::mem::zeroed() };
    const_mem.variants[0] = VariantParameters {
        threshold:               10,
        rest_potential:          0,
        leak_rate:               0,
        homeostasis_penalty:     2,
        homeostasis_decay:       0,
        gsop_potentiation:       100,
        gsop_depression:         10,
        refractory_period:       2,
        synapse_refractory_period: 2,
        slot_decay_ltm:          5,
        slot_decay_wm:           10,
        signal_propagation_length: 5,
        conduction_velocity:     1,
        _padding:                [0; 2],
        inertia_curve:           [128i16; 16],
    };

    assert_eq!(const_mem.variants[0].threshold, 10);
    assert_eq!(const_mem.variants[0].inertia_curve[0], 128);
}

#[test]
fn test_pinned_buffer_allocation() {
    // Verifies that gpu_host_alloc returns a valid non-null pointer,
    // allows read/write as a typed slice, and gpu_host_free does not panic.
    unsafe {
        let len   = 256usize;
        let bytes = len * std::mem::size_of::<u32>();
        let ptr   = ffi::gpu_host_alloc(bytes);

        assert!(!ptr.is_null(), "Pinned (mock) allocation must not return null");

        // Write pattern
        let typed = ptr as *mut u32;
        for i in 0..len {
            *typed.add(i) = i as u32 * 7;
        }

        // Read back and verify
        let slice = std::slice::from_raw_parts(typed, len);
        assert_eq!(slice[0],   0);
        assert_eq!(slice[1],   7);
        assert_eq!(slice[255], 255 * 7);

        ffi::gpu_host_free(ptr);
        // No assert needed — the test passes if Drop doesn't segfault.
    }
}

#[test]
fn test_device_soa_layout() {
    // Verifies that VramState correctly records `padded_n` and `total_axons`.
    unsafe {
        let padded_n    = align_to_warp(100);
        let total_axons = padded_n + 32; // +32 ghost slots

        let d_dummy = ffi::gpu_malloc(4); // Any non-null ptr for unused fields

        let vram = VramState {
            padded_n:         padded_n as u32,
            total_axons:      total_axons as u32,
            voltage:          d_dummy as *mut i32,
            flags:            d_dummy as *mut u8,
            threshold_offset: d_dummy as *mut i32,
            refractory_timer: d_dummy as *mut u8,
            soma_to_axon:     ptr::null_mut(),
            dendrite_targets: d_dummy as *mut u32,
            dendrite_weights: d_dummy as *mut i16,
            dendrite_timers:  d_dummy as *mut u8,
            axon_heads:       d_dummy as *mut u32,
            input_bitmask:    ptr::null_mut(),
            output_history:   ptr::null_mut(),
            telemetry_count:  ptr::null_mut(),
            telemetry_spikes: ptr::null_mut(),
        };

        assert_eq!(vram.padded_n as usize,   padded_n,    "padded_n mismatch in VramState");
        assert_eq!(vram.total_axons as usize, total_axons, "total_axons mismatch in VramState");

        ffi::gpu_free(d_dummy);
    }
}
