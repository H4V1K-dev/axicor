use crate::Runtime;
use crate::network::bsp::BspBarrier;
use crate::network::router::SpikeRouter;
use crate::network::channel::Channel;
use crate::ffi;
use std::ffi::c_void;
use crate::zone_runtime::ZoneRuntime;

pub struct DayPhase;

impl DayPhase {
    /// Runs the main GPU compute loop for one full synchronization batch across all zones.
    pub async fn run_batch(
        zones: &mut [ZoneRuntime],
        channel: &mut dyn Channel,
        barrier: &mut BspBarrier, 
        router: &mut SpikeRouter, 
        gpu_schedule_buffer: *mut c_void, 
        batch_id: u32,
        telemetry_tx: Option<&tokio::sync::broadcast::Sender<crate::network::telemetry::TelemetryPayload>>
    ) -> anyhow::Result<()> {
        let schedule = barrier.get_active_schedule();
        let batch_ticks = schedule.sync_batch_ticks;

        for current_tick in 0..batch_ticks {
            for zone in zones.iter_mut() {
                // Set the Constant Memory specifically for this zone's blueprints
                Runtime::init_constants(&zone.const_mem);

                // 0. Inject Sensory Inputs (Virtual Axons)
                if zone.runtime.vram.num_virtual > 0 {
                    let u32s_per_tick = (zone.runtime.vram.num_virtual as usize + 31) / 32;
                    let byte_offset = (current_tick as usize) * u32s_per_tick * 4;
                    unsafe {
                        let ptr = (zone.runtime.vram.input_bitmask_buffer as *mut u8).add(byte_offset) as *const c_void;
                        ffi::launch_inject_inputs(
                            zone.runtime.vram.axon_head_index,
                            ptr,
                            zone.runtime.vram.virtual_offset,
                            zone.runtime.vram.num_virtual,
                            std::ptr::null_mut(),
                        );
                    }
                }

                // 1. Process Network Spikes for this specific tick
                let num_spikes = schedule.counts[current_tick];
                if num_spikes > 0 {
                    let element_offset = current_tick * 1024; // MAX_SPIKES_PER_TICK
                    let byte_offset = element_offset * std::mem::size_of::<u32>();
                    
                    unsafe {
                        let ptr = (gpu_schedule_buffer as *mut u8).add(byte_offset) as *mut c_void;
                        ffi::launch_apply_spike_batch_impl(
                            num_spikes,
                            ptr,
                            zone.runtime.vram.axon_head_index,
                            std::ptr::null_mut(),
                        );
                    }
                }

                // 2. Propagate Axons, Update Neurons, Apply GSOP
                zone.runtime.tick();

                // 3. Record Outgoing Spikes
                unsafe {
                    let zero: u32 = 0;
                    ffi::gpu_memcpy_host_to_device(
                        zone.runtime.vram.outbound_spikes_count,
                        &zero as *const _ as *const c_void,
                        4
                    );

                    ffi::launch_record_outputs(
                        zone.runtime.vram.padded_n as u32,
                        zone.runtime.vram.flags,
                        zone.runtime.vram.outbound_spikes_buffer,
                        zone.runtime.vram.outbound_spikes_count,
                        std::ptr::null_mut(),
                    );
                    
                    let mut host_count: u32 = 0;
                    ffi::gpu_memcpy_device_to_host(
                        &mut host_count as *mut _ as *mut c_void,
                        zone.runtime.vram.outbound_spikes_count,
                        4
                    );

                    if host_count > 0 {
                        let mut host_spikes = vec![0u32; host_count as usize];
                        ffi::gpu_memcpy_device_to_host(
                            host_spikes.as_mut_ptr() as *mut c_void,
                            zone.runtime.vram.outbound_spikes_buffer,
                            (host_count as usize) * 4
                        );

                        router.route_spikes(&host_spikes, current_tick as u32);
                        
                        if let Some(tx) = telemetry_tx {
                            let _ = tx.send(crate::network::telemetry::TelemetryPayload {
                                tick: (batch_id as u64) * (batch_ticks as u64) + (current_tick as u64),
                                active_spikes: host_spikes,
                            });
                        }
                    }
                }
            }

            // Sync Ghost Axons between local zones
            channel.sync_spikes(zones);
        }

        // Wait for all GPU streams to finish before network barrier
        for zone in zones.iter() {
            zone.runtime.synchronize();
        }

        // Flush outbound router queues and run the BSP Barrier Sync
        let outgoing = router.flush_outgoing();
        barrier.sync_and_swap(outgoing, batch_id).await?;
        
        Ok(())
    }
}
