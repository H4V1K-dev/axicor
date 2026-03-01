#[cfg(test)]
mod tests {
    use crate::network::intra_gpu::{IntraGpuChannel, GhostLink};
    use crate::network::channel::Channel;
    use crate::zone_runtime::ZoneRuntime;
    use crate::memory::VramState;
    use crate::{Runtime, GenesisConstantMemory};
    use genesis_core::config::instance::InstanceConfig;
    use std::ffi::c_void;

    // --- Helper Functions ---

    /// Creates a fake VramState using `gpu_malloc` (which maps to `libc::malloc` via mock_ffi).
    fn make_test_vram(num_axons: usize) -> VramState {
        let axon_head_size = num_axons * 4;
        let axon_head_index = unsafe { crate::ffi::gpu_malloc(axon_head_size) };

        // Zero out memory just in case (libc::malloc doesn't zero)
        unsafe {
            std::ptr::write_bytes(axon_head_index as *mut u8, 0, axon_head_size);
        }

        VramState {
            padded_n: 128, // Arbitrary
            
            // Unused in tests
            voltage: std::ptr::null_mut(),
            threshold_offset: std::ptr::null_mut(),
            refractory_timer: std::ptr::null_mut(),
            flags: std::ptr::null_mut(),
            dendrite_targets: std::ptr::null_mut(),
            dendrite_weights: std::ptr::null_mut(),
            dendrite_refractory: std::ptr::null_mut(),
            num_pixels: 0,
            map_pixel_to_axon: std::ptr::null_mut(),
            input_bitmask_buffer: std::ptr::null_mut(),
            num_mapped_somas: 0,
            readout_batch_ticks: 0,
            mapped_soma_ids: std::ptr::null_mut(),
            output_history: std::ptr::null_mut(),
            telemetry_spikes: std::ptr::null_mut(),
            telemetry_count: std::ptr::null_mut(),
            soma_to_axon: std::ptr::null_mut(),
            input_matrices: Vec::new(),
            input_stride: 1,

            // Axon specific
            total_axons: num_axons,
            ghost_axons_allocated: 0,
            max_ghost_axons: num_axons / 2, // Arbitrary slice
            base_axons: num_axons / 2,      // Half are real, half ghost
            axon_head_index,
        }
    }

    /// Creates a fake ZoneRuntime.
    fn make_test_zone(name: &str, num_axons: usize) -> ZoneRuntime {
        let vram = make_test_vram(num_axons);
        
        // Mock geometry receiver/baker channel
        let (_, dummy_rx) = tokio::sync::mpsc::channel(1);

        let runtime = Runtime { 
            vram,
            v_seg: 256,
            master_seed: 0,
            shard_data_path: None,
            baker_client: None,
            geometry_receiver: Some(dummy_rx),
            sentinel: crate::sentinel::SentinelManager::new(),
        };
        
        let config = InstanceConfig {
            zone_id: "0".to_string(),
            world_offset: genesis_core::config::instance::Coordinate { x: 0, y: 0, z: 0 },
            dimensions: genesis_core::config::instance::Dimensions { w: 1, d: 1, h: 1 },
            neighbors: genesis_core::config::instance::Neighbors {
                x_plus: None, x_minus: None, y_plus: None, y_minus: None
            }
        };

        ZoneRuntime {
            name: name.to_string(),
            runtime,
            const_mem: GenesisConstantMemory {
                // total_axons was removed from GenesisConstantMemory, we just use default
                ..GenesisConstantMemory::default()
            },
            config,
            prune_threshold: -50,
            is_sleeping: false,
            sleep_requested: false,
        }
    }

    /// Sets the value of a specific axon in VRAM directly (simulating GPU execution).
    fn set_head(zone: &mut ZoneRuntime, axon_index: u32, value: u32) {
        unsafe {
            let ptr = zone.runtime.vram.axon_head_index.add(axon_index as usize * 4) as *mut u32;
            std::ptr::write(ptr, value);
        }
    }

    /// Reads the value of a specific axon in VRAM directly.
    fn get_head(zone: &ZoneRuntime, axon_index: u32) -> u32 {
        unsafe {
            let ptr = zone.runtime.vram.axon_head_index.add(axon_index as usize * 4) as *const u32;
            std::ptr::read(ptr)
        }
    }

    // --- Tests ---

    #[test]
    fn test_basic_spike_transfer() {
        let mut zones = vec![
            make_test_zone("V1", 100),
            make_test_zone("V2", 100),
        ];

        let link = GhostLink {
            src_zone_idx: 0,
            src_axon_id: 10,  // Axon 10 in V1
            dst_zone_idx: 1,
            dst_ghost_id: 60, // Ghost slot 60 in V2
        };

        let mut channel = IntraGpuChannel::new(vec![link]);

        // Simulate V1 Axon 10 firing
        set_head(&mut zones[0], 10, 42);

        // Sync network
        channel.sync_spikes(&mut zones);

        // Assert V2 Ghost 60 received 42
        assert_eq!(get_head(&zones[1], 60), 42);
        
        // Assert other slots didn't change
        assert_eq!(get_head(&zones[1], 61), 0);
    }

    #[test]
    fn test_fanout_one_to_many() {
        let mut zones = vec![
            make_test_zone("V1", 100),
            make_test_zone("V2", 100),
        ];

        let links = vec![
            GhostLink { src_zone_idx: 0, src_axon_id: 5, dst_zone_idx: 1, dst_ghost_id: 50 },
            GhostLink { src_zone_idx: 0, src_axon_id: 5, dst_zone_idx: 1, dst_ghost_id: 51 },
            GhostLink { src_zone_idx: 0, src_axon_id: 5, dst_zone_idx: 1, dst_ghost_id: 52 },
        ];

        let mut channel = IntraGpuChannel::new(links);
        set_head(&mut zones[0], 5, 99);
        channel.sync_spikes(&mut zones);

        assert_eq!(get_head(&zones[1], 50), 99);
        assert_eq!(get_head(&zones[1], 51), 99);
        assert_eq!(get_head(&zones[1], 52), 99);
    }

    #[test]
    fn test_bidirectional() {
        let mut zones = vec![
            make_test_zone("V1", 100),
            make_test_zone("V2", 100),
        ];

        let links = vec![
            GhostLink { src_zone_idx: 0, src_axon_id: 1, dst_zone_idx: 1, dst_ghost_id: 99 },
            GhostLink { src_zone_idx: 1, src_axon_id: 2, dst_zone_idx: 0, dst_ghost_id: 98 },
        ];

        let mut channel = IntraGpuChannel::new(links);
        set_head(&mut zones[0], 1, 111); // V1 -> V2
        set_head(&mut zones[1], 2, 222); // V2 -> V1
        
        channel.sync_spikes(&mut zones);

        assert_eq!(get_head(&zones[1], 99), 111);
        assert_eq!(get_head(&zones[0], 98), 222);
    }

    #[test]
    fn test_empty_channel() {
        let mut zones = vec![make_test_zone("V1", 100)];
        let mut channel = IntraGpuChannel::new(vec![]);
        
        set_head(&mut zones[0], 10, 42);
        channel.sync_spikes(&mut zones); // Should not panic or do anything
        
        assert_eq!(get_head(&zones[0], 10), 42);
    }

    #[test]
    fn test_repeated_sync() {
        let mut zones = vec![make_test_zone("V1", 100), make_test_zone("V2", 100)];
        let link = GhostLink { src_zone_idx: 0, src_axon_id: 10, dst_zone_idx: 1, dst_ghost_id: 60 };
        let mut channel = IntraGpuChannel::new(vec![link]);

        set_head(&mut zones[0], 10, 42);
        channel.sync_spikes(&mut zones);
        assert_eq!(get_head(&zones[1], 60), 42);

        // V1 head resets (e.g. decayed or spike finished)
        set_head(&mut zones[0], 10, 0);
        channel.sync_spikes(&mut zones);
        // Ghost should reflect the new 0 state
        assert_eq!(get_head(&zones[1], 60), 0);
    }

    #[test]
    fn test_sentinel_propagation() {
        let mut zones = vec![make_test_zone("V1", 100), make_test_zone("V2", 100)];
        let link = GhostLink { src_zone_idx: 0, src_axon_id: 10, dst_zone_idx: 1, dst_ghost_id: 60 };
        let mut channel = IntraGpuChannel::new(vec![link]);

        // 0x80000000 is the Pruned Sentinel
        let sentinel = 0x80000000;
        set_head(&mut zones[0], 10, sentinel);
        
        channel.sync_spikes(&mut zones);
        
        assert_eq!(get_head(&zones[1], 60), sentinel);
    }
}
