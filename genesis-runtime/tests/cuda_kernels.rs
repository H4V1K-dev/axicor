#[path = "mock.rs"]
pub mod mock;

use mock::MockBakerBuilder;
use genesis_runtime::{Runtime, VariantParameters, GenesisConstantMemory};
use genesis_runtime::memory::VramState;

fn setup_constants() -> GenesisConstantMemory {
    let mut constants = GenesisConstantMemory::default();
    // Default variant 0
    constants.variants[0] = VariantParameters {
        threshold: 100,
        rest_potential: 0,
        leak: 2,
        homeostasis_penalty: 10,
        homeostasis_decay: 1,
        gsop_potentiation: 100,
        gsop_depression: 4000,
        refractory_period: 2,
        synapse_refractory: 5,
        slot_decay_ltm: 10,
        slot_decay_wm: 2,
        _padding: [0; 4],
    };
    // Inertia LUT (example: higher weights -> less inertia, simplified)
    for i in 0..16 {
        constants.inertia_lut[i] = (16 - i) as u8;
    }
    constants
}

#[test]
fn test_propagate_axons() {
    let consts = setup_constants();
    Runtime::init_constants(&consts);

    let mut builder = MockBakerBuilder::new(1, 2);
    builder.axon_heads[0] = 50; // Active axon
    // builder.axon_heads[1] is 0x80000000 by default (sentinel)

    let (state_bytes, axons_bytes) = builder.build();
    let vram = VramState::load_shard(&state_bytes, &axons_bytes).unwrap();
    let mut runtime = Runtime::new(vram, 3); // v_seg = 3

    runtime.tick();
    runtime.synchronize();

    let axon_heads = runtime.vram.download_axon_head_index().unwrap();

    // Verify propagation
    assert_eq!(axon_heads[0], 53, "Active axon should advance by v_seg (3)");
    assert_eq!(axon_heads[1], 0x80000000, "Sentinel axon must not advance");
}

#[test]
fn test_update_neurons() {
    let consts = setup_constants();
    Runtime::init_constants(&consts);

    let mut builder = MockBakerBuilder::new(2, 1);
    
    // Neuron 0 setup: should leak
    builder.voltages[0] = 50;
    
    // Neuron 1 setup: exact hit from axon 0 on segment 10, weight 60
    builder.voltages[1] = 45; // Below threshold (100)
    // builder.flags[1] is 0 (variant 0)
    
    // Assuming v_seg = 1, in propagate it will become 10. So we need segment 10.
    builder.axon_heads[0] = 9; // Propagates to 10
    
    // Set dendrite slot 0 for neuron 1
    // builder.set_dendrite(nid, slot, axon_id, segment, weight)
    builder.set_dendrite(1, 0, 0, 10, 60);

    let (state_bytes, axons_bytes) = builder.build();
    let vram = VramState::load_shard(&state_bytes, &axons_bytes).unwrap();
    let mut runtime = Runtime::new(vram, 1);

    runtime.tick();
    runtime.synchronize();

    let voltages = runtime.vram.download_voltage().unwrap();
    let flags = runtime.vram.download_flags().unwrap();

    // Neuron 0 only leaked by 2
    assert_eq!(voltages[0], 48, "Neuron 0 should have leaked 2 voltage");

    // Neuron 1 received 60 dendrite sum, starting from 45. Peak 105. Spikes!
    assert_eq!(voltages[1], consts.variants[0].rest_potential, "Neuron 1 should have reset to rest_potential after spiking");
    assert_eq!(flags[1] & 1, 1, "Neuron 1 should have the spiked flag set");
}

#[test]
fn test_apply_gsop() {
    let consts = setup_constants();
    Runtime::init_constants(&consts);

    // Testing plasticity:
    // Neuron 0 spikes, dendrite weight should increase if timer > 0 (potentiation)
    // Neuron 1 spikes, dendrite weight should decrease if timer == 0 (depression)
    
    let mut builder = MockBakerBuilder::new(2, 2);
    builder.voltages[0] = 200; // Will definitely spike
    builder.voltages[1] = 200; // Will definitely spike

    // Both get target assigned on slot 0
    builder.set_dendrite(0, 0, 0, 10, 100);
    builder.set_dendrite(1, 0, 1, 10, 100);

    // Neuron 0's dendrite has a timer > 0
    builder.dendrite_timers[0] = 3;
    
    // Neuron 1's dendrite timer == 0
    builder.dendrite_timers[1] = 0;

    let (state_bytes, axons_bytes) = builder.build();
    let vram = VramState::load_shard(&state_bytes, &axons_bytes).unwrap();
    let mut runtime = Runtime::new(vram, 1);

    runtime.tick();
    runtime.synchronize();

    let weights = runtime.vram.download_dendrite_weights().unwrap();
    // weights[0] is slot 0 nid 0
    // weights[1] is slot 0 nid 1
    
    let new_w0 = weights[0];
    let new_w1 = weights[1];

    assert!(new_w0 > 100, "Weight 0 should be potentiated, was {} expected > 100", new_w0);
    assert!(new_w1 < 100, "Weight 1 should be depressed, was {} expected < 100", new_w1);
}

use genesis_runtime::network::{SpikeEvent, bsp::BspBarrier};
use genesis_runtime::orchestrator::day_phase::DayPhase;

#[test]
fn test_orchestrator_day_phase() {
    let consts = setup_constants();
    Runtime::init_constants(&consts);

    let mut builder = MockBakerBuilder::new(1, 2);
    // Axon 0 is Active Local Axon
    builder.axon_heads[0] = 10;
    
    // Axon 1 is Ghost Axon (receives network spikes). Let's say network spike resets it to 0.
    builder.axon_heads[1] = 0x80000000; // start as Sentinel

    let (state_bytes, axons_bytes) = builder.build();
    let vram = VramState::load_shard(&state_bytes, &axons_bytes).unwrap();
    let mut runtime = Runtime::new(vram, 2); // v_seg = 2

    // 100 ticks per batch
    let mut barrier = BspBarrier::new(100);

    // Simulate incoming network traffic from previous Night Phase / Barrier
    let incoming_spikes = vec![
        SpikeEvent { receiver_ghost_id: 1, tick_offset: 5, _pad: [0; 3] }, // Arrives at tick 5
    ];
    barrier.ingest_spike_batch(&incoming_spikes);
    
    // Swap barrier (read what we just ingested)
    barrier.sync_and_swap();

    // The Orchestrator expects schedule.buffer to be on the GPU! Let's memcopy it.
    let schedule = barrier.get_active_schedule();
    let schedule_size = schedule.buffer.len() * std::mem::size_of::<u32>();
    let gpu_schedule_buffer = unsafe { genesis_runtime::ffi::gpu_malloc(schedule_size) };
    unsafe {
        genesis_runtime::ffi::gpu_memcpy_host_to_device(
            gpu_schedule_buffer,
            schedule.buffer.as_ptr() as *const std::ffi::c_void,
            schedule_size
        );
    }
    
    // Temporarily replace the schedule buffer inside the barrier just for the CUDA pointer!
    // We can't mutate barrier, so we change day_phase to take a raw pointer or do it here. 
    // Wait, in `day_phase.rs` we did `schedule.buffer[offset..].as_ptr() as *mut c_void`
    // THIS IS ILLEGAL! We passed a Host Pointer to `launch_apply_spike_batch_impl`!
    // Since this is just a test/stub, let's fix DayPhase to accept a device pointer.

    // Run the Day Phase with the copied device pointer!
    DayPhase::run_batch(&mut runtime, &barrier, gpu_schedule_buffer);
    runtime.synchronize();

    let axon_heads = runtime.vram.download_axon_head_index().unwrap();

    // Free the test buffer
    unsafe { genesis_runtime::ffi::gpu_free(gpu_schedule_buffer); }

    // Verification:
    // Local Axon: moved v_seg(2) * 100 ticks = 200. Initial was 10. Final = 210.
    assert_eq!(axon_heads[0], 210, "Local axon should advance for 100 ticks");

    // Ghost Axon:
    // Started at Sentinel.
    // Ignored for ticks 0, 1, 2, 3, 4. (Sentinel + 0 = Sentinel, theoretically, our Propagate just ignores Sentinel)
    // At tick 5: apply_spike_batch resets it to 0!
    // Then it moves for ticks 5, 6... 99 (95 ticks total of movement).
    // 95 ticks * v_seg(2) = 190.
    assert_eq!(axon_heads[1], 190, "Ghost axon should have been injected at tick 5 and propagated 95 times");
}

#[test]
fn test_record_outputs() {
    use std::ffi::c_void;
    let consts = setup_constants();
    Runtime::init_constants(&consts);

    let mut builder = MockBakerBuilder::new(3, 1);
    
    // We want neuron 0 and neuron 2 to fire.
    // Neuron 0: Voltage = 150, Threshold = 100 -> FIRES
    builder.voltages[0] = 150;
    
    // Neuron 1: Voltage = 0, Threshold = 100 -> DOES NOT FIRE
    builder.voltages[1] = 0;
    
    // Neuron 2: Voltage = 150, Threshold = 100 -> FIRES
    builder.voltages[2] = 150;
    
    // Ensure they aren't refractory
    builder.refractory_timers[0] = 0;
    builder.refractory_timers[1] = 0;
    builder.refractory_timers[2] = 0;

    let (state_bytes, axons_bytes) = builder.build();
    let vram = VramState::load_shard(&state_bytes, &axons_bytes).unwrap();
    let mut runtime = Runtime::new(vram, 1);

    // We only need 1 tick to test firing
    let barrier = BspBarrier::new(1);
    let schedule_size = 1024 * std::mem::size_of::<u32>();
    let gpu_schedule_buffer = unsafe { genesis_runtime::ffi::gpu_malloc(schedule_size) };
    let zero: u32 = 0;
    unsafe {
        genesis_runtime::ffi::gpu_memcpy_host_to_device(
            gpu_schedule_buffer,
            &zero as *const _ as *const c_void, // just zero out the first 4 bytes is enough since count is 0
            4
        );
    }

    DayPhase::run_batch(&mut runtime, &barrier, gpu_schedule_buffer);
    runtime.synchronize();

    let count = runtime.vram.download_outbound_spikes_count().unwrap();
    assert_eq!(count, 2, "Expected exactly 2 neurons to fire");

    let spikes = runtime.vram.download_outbound_spikes_buffer(count as usize).unwrap();
    
    // Which ones fired? We expect index 0 and 2.
    // Note: atomicAdd on the GPU doesn't guarantee strict ordering, so they might be [0, 2] or [2, 0]
    assert!(spikes.contains(&0), "Expected neuron 0 in outbound spikes");
    assert!(spikes.contains(&2), "Expected neuron 2 in outbound spikes");
    assert!(!spikes.contains(&1), "Neuron 1 should not have spiked");

    unsafe { genesis_runtime::ffi::gpu_free(gpu_schedule_buffer); }
}

