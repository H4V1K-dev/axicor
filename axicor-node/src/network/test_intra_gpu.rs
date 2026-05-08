// axicor-runtime/src/network/test_intra_gpu.rs
//
// Tests for IntraGpuChannel: Spike routing verification.
// Uses mock-gpu (libc::malloc) so no real CUDA is required.

#[cfg(test)]
mod tests {
    use crate::network::intra_gpu::IntraGpuChannel;

    /// Allocate a flat axon-heads buffer on host (via mock gpu_malloc) and
    /// expose it as a *mut u32.
    unsafe fn make_heads(count: usize) -> *mut u32 {
        let ptr = axicor_compute::ffi::gpu_host_alloc(count * 4) as *mut u32;
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
    fn manual_sync(channel: &IntraGpuChannel, src_heads: *const u32, dst_heads: *mut u32) {
        for i in 0..channel.count as usize {
            let src_axon = channel.src_indices_host[i];
            let dst_ghost = channel.dst_indices_host[i];
            unsafe {
                let val = get(src_heads, src_axon);
                set(dst_heads, dst_ghost, val);
            }
        }
    }

    #[test]
    fn test_basic_spike_transfer() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let channel = IntraGpuChannel::from_slices(0, 1, &[8], &[9], 10, 128, 128);

            set(h0, 10, 42);
            manual_sync(&channel, h0, h1);

            assert_eq!(get(h1, 60), 42);
            assert_eq!(get(h1, 61), 0); // Adjacent slot untouched

            axicor_compute::ffi::gpu_free(h0 as *mut _);
            axicor_compute::ffi::gpu_free(h1 as *mut _);
        }
    }

    #[test]
    fn test_fanout_one_to_many() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let channel = IntraGpuChannel::from_slices(0, 1, &[10, 10, 10], &[11, 12, 13], 10, 128, 128);

            set(h0, 5, 99);
            manual_sync(&channel, h0, h1);

            assert_eq!(get(h1, 50), 99);
            assert_eq!(get(h1, 51), 99);
            assert_eq!(get(h1, 52), 99);

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }

    #[test]
    fn test_bidirectional() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            // Forward channel: 0 -> 1
            let ch_fwd = IntraGpuChannel::from_slices(0, 1, &[14], &[15], 10, 128, 128);
            // Backward channel: 1 -> 0
            let ch_bwd = IntraGpuChannel::from_slices(1, 0, &[1], &[16], 10, 128, 128);

            set(h0, 1, 111);
            set(h1, 2, 222);

            manual_sync(&ch_fwd, h0, h1);
            manual_sync(&ch_bwd, h1, h0);

            assert_eq!(get(h1, 99), 111);
            assert_eq!(get(h0, 98), 222);

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }

    #[test]
    fn test_empty_channel() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);
            set(h0, 10, 42);

            let channel = IntraGpuChannel::from_slices(0, 1, &[], &[], 10, 128, 128);
            manual_sync(&channel, h0, h1);

            // Nothing should change
            assert_eq!(get(h0, 10), 42);
            assert_eq!(get(h1, 10), 0);

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }

    #[test]
    fn test_repeated_sync() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let mut channel = IntraGpuChannel::from_slices(0, 1, &[8], &[9], 10, 128, 128);

            set(h0, 10, 42);
            manual_sync(&channel, h0, h1);
            assert_eq!(get(h1, 60), 42);

            // Simulate decay: head resets to 0
            set(h0, 10, 0);
            manual_sync(&channel, h0, h1);
            assert_eq!(get(h1, 60), 0);

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }

    #[test]
    fn test_sentinel_propagation() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let sentinel = 0x80000000u32;
            let mut channel = IntraGpuChannel::from_slices(0, 1, &[8], &[9], 10, 128, 128);

            set(h0, 10, sentinel);
            manual_sync(&channel, h0, h1);

            // Sentinel MUST be faithfully copied  GPU kernel will handle early-exit
            assert_eq!(get(h1, 60), sentinel);

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }
}
