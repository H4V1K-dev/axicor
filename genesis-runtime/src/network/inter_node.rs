use crate::ffi;
use crate::network::SpikeEvent;
use std::ptr;

pub struct InterNodeChannel {
    pub target_zone_hash: u32,
    pub src_indices_d: *mut u32,
    pub dst_ghost_ids_d: *mut u32,
    pub count: u32,
    
    // Zero-Copy Pinned RAM (доступен и GPU, и CPU)
    pub out_events_pinned: *mut SpikeEvent,
    pub out_count_pinned: *mut u32,
}

impl InterNodeChannel {
    pub unsafe fn new(target_zone_hash: u32, src_indices: &[u32], dst_ghost_ids: &[u32]) -> Self {
        let count = src_indices.len() as u32;
        
        let src_d = ffi::gpu_malloc((count as usize) * 4) as *mut u32;
        let dst_d = ffi::gpu_malloc((count as usize) * 4) as *mut u32;
        ffi::gpu_memcpy_host_to_device_async(src_d as *mut _, src_indices.as_ptr() as *const _, (count as usize) * 4, ptr::null_mut());
        ffi::gpu_memcpy_host_to_device_async(dst_d as *mut _, dst_ghost_ids.as_ptr() as *const _, (count as usize) * 4, ptr::null_mut());

        // Максимум 1 спайк на аксон за батч
        // Используем 8 байт (SpikeEvent pack layout)
        let events_size = (count as usize) * std::mem::size_of::<SpikeEvent>();
        
        Self {
            target_zone_hash,
            src_indices_d: src_d,
            dst_ghost_ids_d: dst_d,
            count,
            out_events_pinned: ffi::gpu_host_alloc(events_size) as *mut SpikeEvent,
            out_count_pinned: ffi::gpu_host_alloc(4) as *mut u32,
        }
    }

    pub unsafe fn extract_spikes(&self, axon_heads: *const u32, sync_batch_ticks: u32, stream: ffi::CudaStream) {
        if self.count == 0 { return; }
        crate::ffi::launch_extract_outgoing_spikes(
            axon_heads,
            self.src_indices_d,
            self.dst_ghost_ids_d,
            self.count,
            sync_batch_ticks,
            self.out_events_pinned as *mut std::ffi::c_void,
            self.out_count_pinned,
            stream
        );
    }
}
