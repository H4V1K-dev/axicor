use axicor_core::types::PackedPosition;

#[test]
fn test_sprouting_position_unpacking() {
    let x = 512;
    let y = 256;
    let z = 100;
    let t = 2;
    let packed = PackedPosition::pack_raw(x, y, z, t).0;

    let pos = PackedPosition(packed);
    assert_eq!(pos.x() as u32, x);
    assert_eq!(pos.y() as u32, y);
    assert_eq!(pos.z() as u32, z);
    assert_eq!(pos.type_id(), t);
}

#[test]
fn test_sprouting_rule_of_uniqueness() {
    use axicor_core::constants::MAX_DENDRITE_SLOTS;
    use crate::bake::sprouting::run_sprouting_pass;
    
    let padded_n = 64;
    let mut targets = vec![0u32; MAX_DENDRITE_SLOTS * padded_n];
    let mut weights = vec![0i32; MAX_DENDRITE_SLOTS * padded_n];
    
    // One active soma
    let mut flags = vec![0u8; padded_n];
    flags[0] = 1; // is_spiking = 1 -> triggers sprouting
    
    let ghost_origins = vec![];
    let mut handovers = vec![];
    
    // Axon candidate 1
    let total_axons = 1;
    let mut axon_tips_uvw = vec![PackedPosition::pack_raw(11, 10, 10, 0).0];
    let mut axon_dirs_xyz = vec![0];
    
    let soma_to_axon = vec![u32::MAX; padded_n];
    
    let mut lengths = vec![1u8; total_axons];
    let mut paths = vec![0u32; total_axons * 256];
    paths[0] = axon_tips_uvw[0]; // Active segment nearby
    
    let mut soma_positions = vec![0u32; padded_n];
    soma_positions[0] = PackedPosition::pack_raw(10, 10, 10, 0).0;
    
    // FORCING UNIQUENESS CONFLICT:
    // Slot 0 is already connected to Axon 0
    targets[0 * padded_n + 0] = axicor_core::layout::pack_dendrite_target(0, 0);
    
    let (new_synapses, _, _) = run_sprouting_pass(
        &mut targets,
        &mut weights,
        &flags,
        &ghost_origins,
        &mut handovers,
        0,
        &mut axon_tips_uvw,
        &mut axon_dirs_xyz,
        &soma_to_axon,
        padded_n,
        0,
        0,
        100, 100, 100,
        None,
        0,
        &mut lengths,
        &mut paths,
        &soma_positions,
        42,
        0,
        5, // max_sprouts = 5
        0,
        std::ptr::null_mut()
    );
    
    assert_eq!(new_synapses, 0, "Should not create a synapse if it's already connected to this axon");
}

#[test]
fn test_sprouting_dense_rule_no_gaps() {
    use axicor_core::constants::MAX_DENDRITE_SLOTS;
    use crate::bake::sprouting::run_sprouting_pass;
    
    let padded_n = 64;
    let mut targets = vec![0u32; MAX_DENDRITE_SLOTS * padded_n];
    let mut weights = vec![0i32; MAX_DENDRITE_SLOTS * padded_n];
    
    let mut flags = vec![0u8; padded_n];
    flags[0] = 1; // Active
    
    // Axons candidates 0, 1, 2
    let total_axons = 3;
    let mut axon_tips_uvw = vec![
        PackedPosition::pack_raw(11, 10, 10, 0).0,
        PackedPosition::pack_raw(10, 11, 10, 0).0,
        PackedPosition::pack_raw(10, 10, 11, 0).0,
    ];
    let mut axon_dirs_xyz = vec![0; 3];
    let soma_to_axon = vec![u32::MAX; padded_n];
    
    let mut lengths = vec![1u8; total_axons];
    let mut paths = vec![0u32; total_axons * 256];
    paths[0] = axon_tips_uvw[0];
    paths[256] = axon_tips_uvw[1];
    paths[512] = axon_tips_uvw[2];
    
    let mut soma_positions = vec![0u32; padded_n];
    soma_positions[0] = PackedPosition::pack_raw(10, 10, 10, 0).0;
    
    // Set Slot 0 and Slot 1 as OCCUPIED
    targets[0 * padded_n + 0] = axicor_core::layout::pack_dendrite_target(500, 0); // dummy remote
    targets[1 * padded_n + 0] = axicor_core::layout::pack_dendrite_target(501, 0);
    // Slot 2 and 3 are empty
    
    let (new_synapses, _, _) = run_sprouting_pass(
        &mut targets,
        &mut weights,
        &flags,
        &[],
        &mut [],
        0,
        &mut axon_tips_uvw,
        &mut axon_dirs_xyz,
        &soma_to_axon,
        padded_n,
        0,
        0,
        100, 100, 100,
        None,
        0,
        &mut lengths,
        &mut paths,
        &soma_positions,
        42,
        0,
        2, // ALLOW 2 SPROUTS
        0,
        std::ptr::null_mut()
    );
    
    assert_eq!(new_synapses, 2, "Should create exactly 2 new synapses");
    
    // Verify they were put in slots 2 and 3, not 4
    assert_ne!(targets[2 * padded_n + 0], 0, "Slot 2 must be filled");
    assert_ne!(targets[3 * padded_n + 0], 0, "Slot 3 must be filled");
    assert_eq!(targets[4 * padded_n + 0], 0, "Dense Rule Violation: gaps detected in target array! Slot 4 must be empty");
}

#[test]
fn test_sprouting_doa_and_mass_domain_shift() {
    use axicor_core::constants::MAX_DENDRITE_SLOTS;
    use crate::bake::sprouting::run_sprouting_pass;
    use axicor_core::config::blueprints::{BlueprintsConfig, NeuronType};
    
    let padded_n = 64;
    let mut targets = vec![0u32; MAX_DENDRITE_SLOTS * padded_n];
    let mut weights = vec![0i32; MAX_DENDRITE_SLOTS * padded_n];
    
    let mut flags = vec![0u8; padded_n];
    flags[0] = 1; // Active
    
    let total_axons = 1;
    let mut axon_tips_uvw = vec![PackedPosition::pack_raw(11, 10, 10, 0).0];
    let mut axon_dirs_xyz = vec![0];
    let soma_to_axon = vec![u32::MAX; padded_n];
    let mut lengths = vec![1u8; total_axons];
    let mut paths = vec![0u32; total_axons * 256];
    paths[0] = axon_tips_uvw[0];
    let mut soma_positions = vec![0u32; padded_n];
    soma_positions[0] = PackedPosition::pack_raw(10, 10, 10, 0).0;
    
    // Mock BlueprintsConfig where initial_synapse_weight = 10
    let mut nt = NeuronType::default();
    nt.initial_synapse_weight = 10;
    let bp = BlueprintsConfig {
        neuron_types: vec![nt],
    };
    
    let prune_threshold = 15;
    
    run_sprouting_pass(
        &mut targets,
        &mut weights,
        &flags,
        &[],
        &mut [],
        0,
        &mut axon_tips_uvw,
        &mut axon_dirs_xyz,
        &soma_to_axon,
        padded_n,
        0,
        0,
        100, 100, 100,
        Some(&bp),
        0,
        &mut lengths,
        &mut paths,
        &soma_positions,
        42,
        0,
        5,
        prune_threshold,
        std::ptr::null_mut()
    );
    
    let w = weights[0];
    let prune_i32 = (prune_threshold as i32) << 16;
    
    assert!(w > prune_i32, "Weight should get survival capital if it's below prune_threshold");
    assert_ne!(w, 10, "Weight should be shifted to Mass Domain, not just 10");
}

#[test]
fn test_sprouting_dales_law_strict_sign() {
    use axicor_core::constants::MAX_DENDRITE_SLOTS;
    use crate::bake::sprouting::run_sprouting_pass;
    use axicor_core::config::blueprints::{BlueprintsConfig, NeuronType};
    
    let padded_n = 64;
    let mut targets = vec![0u32; MAX_DENDRITE_SLOTS * padded_n];
    let mut weights = vec![0i32; MAX_DENDRITE_SLOTS * padded_n];
    
    let mut flags = vec![0u8; padded_n];
    flags[0] = 1; // Active
    
    let total_axons = 2;
    // Axon 0 is type 0, Axon 1 is type 1
    let mut axon_tips_uvw = vec![
        PackedPosition::pack_raw(11, 10, 10, 0).0,
        PackedPosition::pack_raw(10, 11, 10, 1).0,
    ];
    let mut axon_dirs_xyz = vec![0; 2];
    let soma_to_axon = vec![u32::MAX; padded_n];
    let mut lengths = vec![1u8; total_axons];
    let mut paths = vec![0u32; total_axons * 256];
    paths[0] = axon_tips_uvw[0];
    paths[256] = axon_tips_uvw[1];
    
    let mut soma_positions = vec![0u32; padded_n];
    soma_positions[0] = PackedPosition::pack_raw(10, 10, 10, 0).0;
    
    let mut nt0 = NeuronType::default();
    nt0.is_inhibitory = false;
    nt0.initial_synapse_weight = 50;
    
    let mut nt1 = NeuronType::default();
    nt1.is_inhibitory = true;
    nt1.initial_synapse_weight = 50;
    
    let bp = BlueprintsConfig {
        neuron_types: vec![nt0, nt1],
    };
    
    run_sprouting_pass(
        &mut targets,
        &mut weights,
        &flags,
        &[],
        &mut [],
        0,
        &mut axon_tips_uvw,
        &mut axon_dirs_xyz,
        &soma_to_axon,
        padded_n,
        0,
        0,
        100, 100, 100,
        Some(&bp),
        0,
        &mut lengths,
        &mut paths,
        &soma_positions,
        42,
        0,
        2, // max_sprouts = 2
        0,
        std::ptr::null_mut()
    );
    
    let mut negative_found = false;
    
    for i in 0..MAX_DENDRITE_SLOTS {
        let w = weights[i * padded_n];
        if w < 0 { negative_found = true; }
    }
    
    assert!(negative_found, "Must have a strictly negative weight for inhibitory axon (Type 1)");
    }

    #[test]
    fn test_full_night_cycle_dense_rule() {
    use axicor_compute::ffi::ShardVramPtrs;
    use axicor_compute::cpu::physics::cpu_sort_and_prune;
    use crate::bake::sprouting::run_sprouting_pass;
    use axicor_core::constants::MAX_DENDRITE_SLOTS;
    use axicor_core::types::PackedPosition;

    let padded_n = 64;
    let mut targets = vec![0u32; MAX_DENDRITE_SLOTS * padded_n];
    let mut weights = vec![0i32; MAX_DENDRITE_SLOTS * padded_n];
    let mut timers = vec![0u8; MAX_DENDRITE_SLOTS * padded_n];
    let mut flags = vec![0u8; padded_n];
    
    // Setup for Tid=0
    let tid = 0;
    
    // Slot 0: Live (target=100, weight=50<<16)
    let col0 = 0 * (padded_n as usize) + tid;
    targets[col0] = 100;
    weights[col0] = 50 << 16;
    
    // Slot 1: Dead (target=0, weight=10<<16)
    let col1 = 1 * (padded_n as usize) + tid;
    targets[col1] = 0;
    weights[col1] = 10 << 16;
    
    // Slot 2: Live (target=101, weight=60<<16)
    let col2 = 2 * (padded_n as usize) + tid;
    targets[col2] = 101;
    weights[col2] = 60 << 16;
    
    flags[tid] = 1;

    let ptrs = ShardVramPtrs {
        dendrite_targets: targets.as_mut_ptr(),
        dendrite_weights: weights.as_mut_ptr(),
        dendrite_timers: timers.as_mut_ptr(),
        soma_flags: flags.as_mut_ptr(),
        ..unsafe { std::mem::zeroed() }
    };

    // 1. Compaction
    unsafe { cpu_sort_and_prune(&ptrs, padded_n as u32, 15); }
    
    // Expectations: Slot 0=60, Slot 1=50, Slot 2=0 (targets shifted)
    
    // 2. Sprouting
    let mut axon_tips_uvw = vec![PackedPosition::pack_raw(10, 10, 11, 0).0];
    let mut axon_dirs_xyz = vec![0];
    let mut lengths = vec![1u8; 1];
    let mut paths = vec![0u32; 256];
    paths[0] = axon_tips_uvw[0];
    let mut soma_positions = vec![0u32; padded_n];
    soma_positions[tid] = PackedPosition::pack_raw(10, 10, 10, 0).0;
    let mut soma_to_axon = vec![u32::MAX; padded_n];

    run_sprouting_pass(
        &mut targets,
        &mut weights,
        &flags,
        &[],
        &mut [],
        0,
        &mut axon_tips_uvw,
        &mut axon_dirs_xyz,
        &mut soma_to_axon,
        padded_n as usize,
        0,
        0,
        100, 100, 100,
        None,
        0,
        &mut lengths,
        &mut paths,
        &soma_positions,
        42,
        0,
        1, // Max 1 sprout
        0,
        std::ptr::null_mut()
    );

    // 3. Paranoid Asserts
    let w0 = weights[0 * (padded_n as usize) + tid];
    let w1 = weights[1 * (padded_n as usize) + tid];
    let w2 = weights[2 * (padded_n as usize) + tid];
    assert!(w0 != 0 && w1 != 0 && w2 != 0, "All 3 slots should be filled, weights: {}, {}, {}", w0, w1, w2);
    assert_eq!(weights[3 * (padded_n as usize) + tid], 0); // Empty

    // Early Exit Proof
    let mut sum1 = 0;
    for slot in 0..128 {
        let t = targets[slot * (padded_n as usize) + tid];
        if t == 0 { break; }
        sum1 += weights[slot * (padded_n as usize) + tid];
    }
    
    let mut sum2 = 0;
    for slot in 0..128 {
        sum2 += weights[slot * (padded_n as usize) + tid];
    }
    assert_eq!(sum1, sum2, "Early Exit sum mismatch");
}
