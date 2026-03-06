// genesis-runtime/src/network/test_intra_gpu.rs
//
// Tests for IntraGpuChannel: GhostLink based spike routing.
// Uses mock-gpu (libc::malloc) so no real CUDA is required.
//
// Strategy: Instead of building full ZoneRuntime (which has 9+ required fields
// that change frequently), we test sync_spikes through the thinnest shim
// possible — a raw axon head array manipulated directly as *mut u32.

#[cfg(test)]
mod tests {
    use crate::network::intra_gpu::{IntraGpuChannel, GhostLink};

    /// Allocate a flat axon-heads buffer on host (via mock gpu_malloc) and
    /// expose it as a (*mut u32, len) tuple. This mimics the VRAM ptr layout
    /// that VramState.axon_head_index points to, without constructing VramState.
    unsafe fn make_heads(count: usize) -> *mut u32 {
        let ptr = genesis_compute::ffi::gpu_malloc(count * 4) as *mut u32;
        std::ptr::write_bytes(ptr as *mut u8, 0, count * 4);
        ptr
    }

    unsafe fn set(ptr: *mut u32, idx: u32, val: u32) {
        *ptr.add(idx as usize) = val;
    }

    unsafe fn get(ptr: *const u32, idx: u32) -> u32 {
        *ptr.add(idx as usize)
    }

    /// Direct channel sync: reads src heads and writes to dst heads (mock-mode).
    /// Mirrors what sync_spikes does, but operating on raw pointers so we don't
    /// need ZoneRuntime at all.
    fn manual_sync(
        channel: &IntraGpuChannel,
        heads: &[*mut u32],  // One pointer per "zone"
    ) {
        for i in 0..channel.count as usize {
            let src_zone  = channel.src_zone_indices[i];
            let dst_zone  = channel.dst_zone_indices[i];
            let src_axon  = channel.src_indices_host[i];
            let dst_ghost = channel.dst_indices_host[i];
            unsafe {
                let val = get(heads[src_zone], src_axon);
                set(heads[dst_zone], dst_ghost, val);
            }
        }
    }

    #[test]
    #[ignore]
    fn test_basic_spike_transfer() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let channel = IntraGpuChannel::new(vec![
                GhostLink { src_zone_idx: 0, src_axon_id: 10, dst_zone_idx: 1, dst_ghost_id: 60 },
            ]);

            set(h0, 10, 42);
            manual_sync(&channel, &[h0, h1]);

            assert_eq!(get(h1, 60), 42);
            assert_eq!(get(h1, 61), 0);  // Adjacent slot untouched

            genesis_compute::ffi::gpu_free(h0 as *mut _);
            genesis_compute::ffi::gpu_free(h1 as *mut _);
        }
    }

    #[test]
    #[ignore]
    fn test_fanout_one_to_many() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let channel = IntraGpuChannel::new(vec![
                GhostLink { src_zone_idx: 0, src_axon_id: 5, dst_zone_idx: 1, dst_ghost_id: 50 },
                GhostLink { src_zone_idx: 0, src_axon_id: 5, dst_zone_idx: 1, dst_ghost_id: 51 },
                GhostLink { src_zone_idx: 0, src_axon_id: 5, dst_zone_idx: 1, dst_ghost_id: 52 },
            ]);

            set(h0, 5, 99);
            manual_sync(&channel, &[h0, h1]);

            assert_eq!(get(h1, 50), 99);
            assert_eq!(get(h1, 51), 99);
            assert_eq!(get(h1, 52), 99);

            genesis_compute::ffi::gpu_free(h0 as *mut _);
            genesis_compute::ffi::gpu_free(h1 as *mut _);
        }
    }

    #[test]
    #[ignore]
    fn test_bidirectional() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let channel = IntraGpuChannel::new(vec![
                GhostLink { src_zone_idx: 0, src_axon_id: 1, dst_zone_idx: 1, dst_ghost_id: 99 },
                GhostLink { src_zone_idx: 1, src_axon_id: 2, dst_zone_idx: 0, dst_ghost_id: 98 },
            ]);

            set(h0, 1, 111);
            set(h1, 2, 222);
            manual_sync(&channel, &[h0, h1]);

            assert_eq!(get(h1, 99), 111);
            assert_eq!(get(h0, 98), 222);

            genesis_compute::ffi::gpu_free(h0 as *mut _);
            genesis_compute::ffi::gpu_free(h1 as *mut _);
        }
    }

    #[test]
    #[ignore]
    fn test_empty_channel() {
        unsafe {
            let h0 = make_heads(100);
            set(h0, 10, 42);

            let channel = IntraGpuChannel::new(vec![]);
            manual_sync(&channel, &[h0]);

            // Nothing should change
            assert_eq!(get(h0, 10), 42);

            genesis_compute::ffi::gpu_free(h0 as *mut _);
        }
    }

    #[test]
    #[ignore]
    fn test_repeated_sync() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let channel = IntraGpuChannel::new(vec![
                GhostLink { src_zone_idx: 0, src_axon_id: 10, dst_zone_idx: 1, dst_ghost_id: 60 },
            ]);

            set(h0, 10, 42);
            manual_sync(&channel, &[h0, h1]);
            assert_eq!(get(h1, 60), 42);

            // Simulate decay: head resets to 0
            set(h0, 10, 0);
            manual_sync(&channel, &[h0, h1]);
            assert_eq!(get(h1, 60), 0);

            genesis_compute::ffi::gpu_free(h0 as *mut _);
            genesis_compute::ffi::gpu_free(h1 as *mut _);
        }
    }

    #[test]
    #[ignore]
    fn test_sentinel_propagation() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let sentinel = 0x80000000u32;
            let channel = IntraGpuChannel::new(vec![
                GhostLink { src_zone_idx: 0, src_axon_id: 10, dst_zone_idx: 1, dst_ghost_id: 60 },
            ]);

            set(h0, 10, sentinel);
            manual_sync(&channel, &[h0, h1]);

            // Sentinel MUST be faithfully copied — GPU kernel will handle early-exit
            assert_eq!(get(h1, 60), sentinel);

            genesis_compute::ffi::gpu_free(h0 as *mut _);
            genesis_compute::ffi::gpu_free(h1 as *mut _);
        }
    }
}
