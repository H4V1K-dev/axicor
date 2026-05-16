// axicor-runtime/src/network/test_intra_gpu.rs
//
// Tests for IntraGpuChannel: Spike routing verification.
// Uses mock-gpu (libc::malloc) so no real CUDA is required.

#[cfg(test)]
mod tests {
    use crate::network::intra_gpu::IntraGpuChannel;

    use axicor_core::layout::BurstHeads8;

    unsafe fn make_heads(count: usize) -> *mut BurstHeads8 {
        let ptr = axicor_compute::ffi::gpu_host_alloc(count * std::mem::size_of::<BurstHeads8>()) as *mut BurstHeads8;
        for i in 0..count {
            *ptr.add(i) = BurstHeads8::empty(0x80000000);
        }
        ptr
    }

    unsafe fn set(ptr: *mut BurstHeads8, idx: u32, val: BurstHeads8) {
        *ptr.add(idx as usize) = val;
    }

    unsafe fn get(ptr: *const BurstHeads8, idx: u32) -> BurstHeads8 {
        *ptr.add(idx as usize)
    }

    fn manual_sync(channel: &IntraGpuChannel, src_heads: *const BurstHeads8, dst_heads: *mut BurstHeads8) {
        unsafe {
            channel.sync_ghosts(src_heads, dst_heads, 100, 256, std::ptr::null_mut());
            // [DOD FIX] Hardware Barrier. CPU must wait for async GPU kernel to complete.
            axicor_compute::ffi::gpu_device_synchronize();
        }
    }

    #[test]
    fn test_basic_spike_transfer() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let channel = IntraGpuChannel::from_slices(0, 1, &[8], &[9], 10, 128, 128);

            let mut spike = BurstHeads8::empty(0x80000000);
            spike.h0 = 10 * 256;
            set(h0, 8, spike);
            manual_sync(&channel, h0, h1);

            let out = get(h1, 9);
            assert_eq!(out.h0, (10u32 * 256u32).wrapping_sub(100 * 256)); // Shifted by batch_shift
            assert_eq!(out.h1, 0x80000000);

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

            let mut spike = BurstHeads8::empty(0x80000000);
            spike.h0 = 42 * 256;
            set(h0, 10, spike);
            manual_sync(&channel, h0, h1);

            assert_eq!(get(h1, 11).h0, (42u32 * 256u32).wrapping_sub(100 * 256));
            assert_eq!(get(h1, 12).h0, (42u32 * 256u32).wrapping_sub(100 * 256));
            assert_eq!(get(h1, 13).h0, (42u32 * 256u32).wrapping_sub(100 * 256));

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }

    #[test]
    fn test_bidirectional() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let ch_fwd = IntraGpuChannel::from_slices(0, 1, &[14], &[15], 10, 128, 128);
            let ch_bwd = IntraGpuChannel::from_slices(1, 0, &[1], &[16], 10, 128, 128);

            let mut s1 = BurstHeads8::empty(0x80000000); s1.h0 = 111 * 256;
            let mut s2 = BurstHeads8::empty(0x80000000); s2.h0 = 222 * 256;
            set(h0, 14, s1);
            set(h1, 1, s2);

            manual_sync(&ch_fwd, h0, h1);
            manual_sync(&ch_bwd, h1, h0);

            assert_eq!(get(h1, 15).h0, (111u32 * 256u32).wrapping_sub(100 * 256));
            assert_eq!(get(h0, 16).h0, (222u32 * 256u32).wrapping_sub(100 * 256));

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }

    #[test]
    fn test_empty_channel() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);
            
            let mut s1 = BurstHeads8::empty(0x80000000); s1.h0 = 42;
            set(h0, 10, s1);

            let channel = IntraGpuChannel::from_slices(0, 1, &[], &[], 10, 128, 128);
            manual_sync(&channel, h0, h1);

            assert_eq!(get(h0, 10).h0, 42);
            assert_eq!(get(h1, 10).h0, 0x80000000);

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }

    #[test]
    fn test_repeated_sync() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let channel = IntraGpuChannel::from_slices(0, 1, &[8], &[9], 10, 128, 128);

            let mut s1 = BurstHeads8::empty(0x80000000); s1.h0 = 42 * 256;
            set(h0, 8, s1);
            manual_sync(&channel, h0, h1);
            assert_eq!(get(h1, 9).h0, (42u32 * 256u32).wrapping_sub(100 * 256));

            set(h0, 8, BurstHeads8::empty(0x80000000));
            manual_sync(&channel, h0, h1);
            assert_eq!(get(h1, 9).h0, 0x80000000);

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }

    #[test]
    fn test_sentinel_propagation() {
        unsafe {
            let h0 = make_heads(100);
            let h1 = make_heads(100);

            let sentinel = BurstHeads8::empty(0x80000000);
            let channel = IntraGpuChannel::from_slices(0, 1, &[8], &[9], 10, 128, 128);

            set(h0, 8, sentinel);
            manual_sync(&channel, h0, h1);

            assert_eq!(get(h1, 9).h0, 0x80000000);

            axicor_compute::ffi::gpu_host_free(h0 as *mut _);
            axicor_compute::ffi::gpu_host_free(h1 as *mut _);
        }
    }
}
