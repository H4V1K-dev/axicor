#![cfg(not(feature = "mock-gpu"))]

use genesis_runtime::zone_runtime::ZoneRuntime;
use genesis_runtime::Runtime;
use genesis_runtime::network::bsp::BspBarrier;
use genesis_runtime::network::router::SpikeRouter;
use genesis_runtime::network::channel::Channel;
use genesis_runtime::network::intra_gpu::IntraGpuChannel;
use genesis_core::config::{SimulationConfig, BlueprintsConfig, InstanceConfig};
use std::time::Instant;
use std::ffi::c_void;

use genesis_runtime::memory::VramState;
use genesis_core::layout::padded_n;
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use genesis_runtime::ffi;

/// Helper function to build a fake V1 Zone with N neurons for benchmarking the true GPU load
fn build_benchmark_zone(name: &str, num_neurons: usize) -> ZoneRuntime {
    let blueprints = genesis_core::config::BlueprintsConfig {
        neuron_types: vec![genesis_core::config::blueprints::NeuronType::default()],
    };
    let const_mem = ZoneRuntime::build_constant_memory(&blueprints);
    
    let pa = padded_n(num_neurons);
    let dc = MAX_DENDRITE_SLOTS * pa;
    
    // Allocate RAW VRAM on the GPU 
    let allocate_zeros = |count: usize, size: usize| -> *mut c_void {
        let bytes = count * size;
        let ptr = unsafe { ffi::gpu_malloc(bytes) };
        // We do not zero it on device to save init time; benchmarking physics doesn't care about noise initially
        // But let's actually cudaMemset or copy host zeros so it's deterministic.
        let zeros = vec![0u8; bytes];
        unsafe {
            ffi::gpu_memcpy_host_to_device(ptr, zeros.as_ptr() as *const c_void, bytes);
        }
        ptr
    };

    let voltage = allocate_zeros(pa, 4);
    let flags = allocate_zeros(pa, 1);
    let threshold_offset = allocate_zeros(pa, 4);
    let refractory_timer = allocate_zeros(pa, 1);
    let soma_to_axon = allocate_zeros(pa, 4);
    
    let dendrite_targets = allocate_zeros(dc, 4);
    let dendrite_weights = allocate_zeros(dc, 2);
    let dendrite_refractory = allocate_zeros(dc, 1);

    // Give it 1% of neurons as virtual sensors
    let num_pixels = (num_neurons / 100).max(1) as u32;
    let _map_pixel_to_axon = allocate_zeros(num_pixels as usize, 4);
    let _input_bitmask_buffer = allocate_zeros(((num_pixels as usize + 31) / 32) * 1000, 4);

    // Give it 1% of neurons as readouts
    let num_mapped_somas = num_pixels;
    let _mapped_soma_ids = allocate_zeros(num_mapped_somas as usize, 4);
    let _output_history = allocate_zeros((num_mapped_somas as usize) * 1000, 1);
    
    // Axons
    let max_ghost_axons = 1000;
    let total_axons = pa + max_ghost_axons;
    let axon_head_index = allocate_zeros(total_axons, 4);
    // Init to Sentinels
    let sentinels = vec![0x80000000u32; total_axons];
    unsafe {
        ffi::gpu_memcpy_host_to_device(axon_head_index, sentinels.as_ptr() as *const c_void, total_axons * 4);
    }

    let vram = VramState {
        padded_n: pa,
        voltage,
        threshold_offset,
        refractory_timer,
        flags,
        total_axons,
        ghost_axons_allocated: 0,
        max_ghost_axons,
        base_axons: pa,
        axon_head_index,
        soma_to_axon,
        dendrite_targets,
        dendrite_weights,
        dendrite_refractory,
        num_pixels,
        map_pixel_to_axon: _map_pixel_to_axon,
        input_bitmask_buffer: _input_bitmask_buffer,
        num_mapped_somas,
        readout_batch_ticks: 1000,
        mapped_soma_ids: _mapped_soma_ids,
        output_history: std::ptr::null_mut(), 
        telemetry_spikes: std::ptr::null_mut(),
        telemetry_count: std::ptr::null_mut(),
        input_matrices: Vec::new(),
        input_stride: 1,
    };

    let runtime = Runtime::new(vram, 256, 42, None);
    let config = genesis_core::config::InstanceConfig {
        zone_id: "0".to_string(),
        world_offset: genesis_core::config::instance::Coordinate { x: 0, y: 0, z: 0 },
        dimensions: genesis_core::config::instance::Dimensions { w: 1, d: 1, h: 1 },
        neighbors: genesis_core::config::instance::Neighbors { x_plus: None, x_minus: None, y_plus: None, y_minus: None },
    };

    ZoneRuntime {
        name: name.to_string(),
        runtime,
        const_mem,
        config,
        prune_threshold: -50,
        is_sleeping: false,
        sleep_requested: false,
    }
}

#[tokio::test]
#[cfg(not(feature = "mock-gpu"))]
// WARNING: This test requires an NVIDIA GPU and proper CUDA toolkit installation.
// It executes the TRUE C++ CUDA Kernels (physics.cu, inject_inputs.cu, etc.)
async fn test_e2e_hardware_metrics() {
    let scales = vec![1_000, 10_000, 100_000, 1_000_000];
    
    for neurons in scales {
        println!("\n==============================================");
        println!("🚀 Running Full GPU Data Flow E2E ({} neurons)", neurons);
        println!("==============================================");
        
        // Build Zone
        let mut zone = build_benchmark_zone("TestZone", neurons);
        let mut zones = vec![zone];
        
        // Build Router and Channel
        let mut channel = IntraGpuChannel::new(vec![]);
        let mut barrier = BspBarrier::new(100); // 100 ticks per batch
        let mut router = SpikeRouter::new();
        let mut gpu_schedule_buffer = vec![0u8; 100 * 1024 * 4];
        
        let start_time = Instant::now();
        
        // Run 10 batches (1000 ticks)
        for i in 0..10 {
            genesis_runtime::orchestrator::day_phase::DayPhase::run_batch(
                &mut zones,
                &mut channel,
                &mut barrier,
                &mut router,
                gpu_schedule_buffer.as_mut_ptr() as *mut c_void,
                i,
                None
            ).await.expect("Failed batch");
        }
        
        let elapsed = start_time.elapsed();
        let ticks_per_sec = 1000.0 / elapsed.as_secs_f64();
        
        println!("Total GPU Compute Time for 1000 Ticks: {:?}", elapsed);
        println!("Speed: {:.2} Ticks/sec (Realtime is 10,000 tick/sec for 100µs resolution)", ticks_per_sec);
        
        // Teardown GPU memory to prevent OOM on next scale
        let z = &zones[0].runtime.vram;
        unsafe {
            ffi::gpu_free(z.voltage);
            ffi::gpu_free(z.threshold_offset);
            ffi::gpu_free(z.refractory_timer);
            ffi::gpu_free(z.flags);
            ffi::gpu_free(z.soma_to_axon);
            ffi::gpu_free(z.dendrite_targets);
            ffi::gpu_free(z.dendrite_weights);
            ffi::gpu_free(z.dendrite_refractory);
            ffi::gpu_free(z.axon_head_index);
            ffi::gpu_free(z.map_pixel_to_axon);
            ffi::gpu_free(z.input_bitmask_buffer);
            ffi::gpu_free(z.mapped_soma_ids);
            ffi::gpu_free(z.output_history);
        }
    }
}
