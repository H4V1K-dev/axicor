use crate::network::router::SpikeRouter;
use crate::network::channel::Channel;
use crate::zone_runtime::ZoneRuntime;
use std::ffi::c_void;
use crate::Runtime;
use crate::ffi;

pub fn execute_day_batch(
    zone: &mut ZoneRuntime, 
    batch_ticks: u32, 
    stream: crate::ffi::CudaStream,
    telemetry_tx: Option<&tokio::sync::broadcast::Sender<crate::network::telemetry::TelemetryPayload>>,
    total_ticks: u64
) {

    // Set Constant Memory
    Runtime::init_constants(&zone.const_mem);

    let layout_vram = genesis_core::layout::VramState {
        voltage: zone.runtime.vram.voltage as *mut i32,
        flags: zone.runtime.vram.flags as *mut u8,
        threshold_offset: zone.runtime.vram.threshold_offset as *mut i32,
        refractory_timer: zone.runtime.vram.refractory_timer as *mut u8,
        soma_to_axon: zone.runtime.vram.soma_to_axon as *mut u32,
        dendrite_targets: zone.runtime.vram.dendrite_targets as *mut u32,
        dendrite_weights: zone.runtime.vram.dendrite_weights as *mut i16,
        dendrite_timers: zone.runtime.vram.dendrite_refractory as *mut u8,
        axon_heads: zone.runtime.vram.axon_head_index as *mut u32,
        input_bitmask: zone.runtime.vram.input_bitmask_buffer as *mut u32,
        output_history: zone.runtime.vram.output_history as *mut u8,
        telemetry_spikes: zone.runtime.vram.telemetry_spikes as *mut u32,
        telemetry_count: zone.runtime.vram.telemetry_count as *mut u32,
    };
    let vram_ptr = &layout_vram as *const genesis_core::layout::VramState;

    unsafe { ffi::gpu_reset_telemetry_count(vram_ptr, stream); }

    let padded_n = zone.runtime.vram.padded_n as u32;
    let total_axons = zone.runtime.vram.total_axons as u32;
    
    let virtual_offset = 0u32;
    let total_virtual_axons = zone.runtime.vram.num_pixels;
    let input_stride = zone.runtime.vram.input_stride as u8;
    
    let v_seg = zone.runtime.v_seg;
    let num_output_channels = zone.runtime.vram.num_mapped_somas;
    let mapped_soma_ids = zone.runtime.vram.mapped_soma_ids as *const u32;

    let schedule_buffer = if zone.ping_pong.reading_from_a.load(std::sync::atomic::Ordering::Relaxed) {
        zone.ping_pong.buffer_a
    } else {
        zone.ping_pong.buffer_b
    };
    
    let counts_buffer = if zone.ping_pong.reading_from_a.load(std::sync::atomic::Ordering::Relaxed) {
        zone.ping_pong.counts_a
    } else {
        zone.ping_pong.counts_b
    };
    
    let max_spikes_per_tick = zone.ping_pong.max_spikes_per_tick as u32;

    for current_tick in 0..batch_ticks {
        unsafe {
            if total_virtual_axons > 0 {
                ffi::launch_inject_inputs(
                    vram_ptr, virtual_offset, current_tick as u32, input_stride, total_virtual_axons, stream
                );
            }

            let is_sleeping = zone.is_sleeping.load(std::sync::atomic::Ordering::Acquire);
            if !is_sleeping {
                ffi::launch_apply_spike_batch(vram_ptr, schedule_buffer, counts_buffer, current_tick as u32, max_spikes_per_tick, stream);
                ffi::launch_propagate_axons(vram_ptr, total_axons, v_seg, stream);
                ffi::launch_update_neurons(vram_ptr, padded_n, stream);
                ffi::launch_apply_gsop(vram_ptr, padded_n, stream);
            }

            if num_output_channels > 0 && !mapped_soma_ids.is_null() {
                ffi::launch_record_readout(
                    vram_ptr, mapped_soma_ids, num_output_channels, current_tick as u32, stream
                );
            }
        }
    }

    zone.runtime.synchronize();
    
    if let Some(tx) = telemetry_tx {
        let mut h_count: u32 = 0;
        unsafe {
            ffi::gpu_memcpy_device_to_host_async(
                &mut h_count as *mut _ as *mut std::ffi::c_void,
                zone.runtime.vram.telemetry_count as *const _,
                4,
                stream,
            );
            ffi::gpu_stream_synchronize(stream); // Ждем счетчик
        }

        let safe_count = std::cmp::min(h_count, 500_000);

        if safe_count > 0 {
            unsafe {
                ffi::gpu_memcpy_device_to_host_async(
                    zone.runtime.vram.telemetry_spikes_host,
                    zone.runtime.vram.telemetry_spikes as *const _,
                    (safe_count as usize) * 4,
                    stream,
                );
                ffi::gpu_stream_synchronize(stream);
            }

            let spikes = unsafe {
                std::slice::from_raw_parts(
                    zone.runtime.vram.telemetry_spikes_host as *const u32,
                    safe_count as usize,
                ).to_vec()
            };

            let _ = tx.send(crate::network::telemetry::TelemetryPayload {
                tick: total_ticks, 
                active_spikes: spikes,
            });
        }
    }
}
