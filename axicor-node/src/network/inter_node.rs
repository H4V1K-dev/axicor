use crate::network::SpikeEvent;
use std::ptr;

pub struct InterNodeChannel {
    pub target_zone_hash: u32,
    pub src_zone_hash: u32,
    pub capacity: u32, // [DOD FIX] Physical VRAM limit
    pub count: u32,    // [DOD FIX] Current number of active links

    pub src_indices_host: Vec<u32>,
    pub dst_ghost_ids_host: Vec<u32>, // [DOD FIX] Mirror for Swap-and-Pop

    pub src_indices_d: *mut u32,
    pub dst_ghost_ids_d: *mut u32,

    // Zero-Copy Pinned RAM (accessible by both GPU and CPU)
    pub out_events_pinned: *mut SpikeEvent,
    pub out_count_pinned: *mut u32,
}

unsafe impl Send for InterNodeChannel {}
unsafe impl Sync for InterNodeChannel {}

impl Drop for InterNodeChannel {
    fn drop(&mut self) {
        unsafe {
            axicor_compute::ffi::gpu_free(self.src_indices_d as *mut _);
            axicor_compute::ffi::gpu_free(self.dst_ghost_ids_d as *mut _);
            axicor_compute::ffi::gpu_host_free(self.out_events_pinned as *mut _);
            axicor_compute::ffi::gpu_host_free(self.out_count_pinned as *mut _);
        }
    }
}

impl InterNodeChannel {
    /// # Safety
    /// Pointers must be valid and aligned.
    pub unsafe fn new(
        src_zone_hash: u32,
        target_zone_hash: u32,
        src_indices: &[u32],
        dst_ghost_ids: &[u32],
        capacity: u32,
    ) -> Self {
        let count = src_indices.len() as u32;
        debug_assert!(
            count <= capacity,
            "FATAL: Initial connections exceed capacity"
        );
        if count > capacity {
            tracing::error!(
                "CRITICAL: Initial connections (count={}) exceed capacity ({}). Link dropped.",
                count,
                capacity
            );
            // In constructor, we can't easily return a dummy Self without breaking everything.
            // But we should at least clamp it for safety.
        }
        let count = std::cmp::min(count, capacity);

        // Allocate for MAX CAPACITY, not current count
        let bytes_capacity = (capacity as usize) * 4;
        let src_d = axicor_compute::ffi::gpu_malloc(bytes_capacity) as *mut u32;
        let dst_d = axicor_compute::ffi::gpu_malloc(bytes_capacity) as *mut u32;

        // Copy only REAL data
        if count > 0 {
            let bytes_active = (count as usize) * 4;
            axicor_compute::ffi::gpu_memcpy_host_to_device_async(
                src_d as *mut _,
                src_indices.as_ptr() as *const _,
                bytes_active,
                ptr::null_mut(),
            );
            axicor_compute::ffi::gpu_memcpy_host_to_device_async(
                dst_d as *mut _,
                dst_ghost_ids.as_ptr() as *const _,
                bytes_active,
                ptr::null_mut(),
            );
        }

        // Max 8 spikes per axon per batch. Limit derived from Capacity.
        let events_size = (capacity as usize) * 8 * std::mem::size_of::<SpikeEvent>();

        // Host-mirroring with preallocation
        let mut src_host = Vec::with_capacity(capacity as usize);
        src_host.extend_from_slice(src_indices);
        let mut dst_host = Vec::with_capacity(capacity as usize);
        dst_host.extend_from_slice(dst_ghost_ids);

        Self {
            target_zone_hash,
            src_zone_hash,
            capacity,
            count,
            src_indices_host: src_host,
            dst_ghost_ids_host: dst_host,
            src_indices_d: src_d,
            dst_ghost_ids_d: dst_d,
            out_events_pinned: axicor_compute::ffi::gpu_host_alloc(events_size) as *mut SpikeEvent,
            out_count_pinned: axicor_compute::ffi::gpu_host_alloc(4) as *mut u32,
        }
    }

    /// # Safety
    /// Caller must ensure CUDA pointers and sizes are valid.
    pub unsafe fn extract_spikes(
        &self,
        axon_heads: *const axicor_core::layout::BurstHeads8,
        sync_batch_ticks: u32,
        v_seg: u32,
        stream: axicor_compute::ffi::CudaStream,
    ) {
        if self.count == 0 {
            return;
        }
        axicor_compute::ffi::launch_extract_outgoing_spikes(
            axon_heads,
            self.src_indices_d,
            self.dst_ghost_ids_d,
            self.count, // Kernel reads only count, garbage at VRAM tail is ignored
            sync_batch_ticks,
            v_seg,
            self.out_events_pinned as *mut std::ffi::c_void,
            self.out_count_pinned,
            stream,
        );
    }

    /// # Safety
    /// Caller must ensure CUDA stream and routing table indices are valid.
    pub unsafe fn push_route(
        &mut self,
        src_axon: u32,
        dst_ghost: u32,
        stream: axicor_compute::ffi::CudaStream,
    ) {
        debug_assert!(
            self.count < self.capacity,
            "FATAL: Routing capacity exceeded. Increase ghost_capacity."
        );
        if self.count >= self.capacity {
            tracing::error!(
                "CRITICAL: Routing capacity exceeded ({}). Link 0x{:08X} -> 0x{:08X} dropped.",
                self.capacity, src_axon, dst_ghost
            );
            return;
        }
        let idx = self.count as usize;

        self.src_indices_host.push(src_axon);
        self.dst_ghost_ids_host.push(dst_ghost);

        let src_ptr = self.src_indices_host.as_ptr().add(idx);
        let dst_ptr = self.dst_ghost_ids_host.as_ptr().add(idx);

        // Micro-DMA (4 bytes) directly into free tail of VRAM array
        axicor_compute::ffi::gpu_memcpy_host_to_device_async(
            self.src_indices_d.add(idx) as *mut _,
            src_ptr as *const _,
            4,
            stream,
        );
        axicor_compute::ffi::gpu_memcpy_host_to_device_async(
            self.dst_ghost_ids_d.add(idx) as *mut _,
            dst_ptr as *const _,
            4,
            stream,
        );

        self.count += 1;
    }

    /// O(1) link removal via Swap-and-Pop
    /// # Safety
    /// Caller must ensure CUDA stream is valid.
    pub unsafe fn prune_route(
        &mut self,
        target_ghost_id: u32,
        stream: axicor_compute::ffi::CudaStream,
    ) {
        // O(N) host search is acceptable as it runs only in Night Phase
        let Some(idx) = self
            .dst_ghost_ids_host
            .iter()
            .position(|&g| g == target_ghost_id)
        else {
            return; // Link already removed
        };

        let last_idx = (self.count - 1) as usize;

        // 1. Swap on host
        self.src_indices_host.swap(idx, last_idx);
        self.dst_ghost_ids_host.swap(idx, last_idx);

        // 2. Pop tail
        self.src_indices_host.pop();
        self.dst_ghost_ids_host.pop();

        // 3. If removed element was not last, patch the resulting hole in VRAM
        if idx != last_idx {
            let src_ptr = self.src_indices_host.as_ptr().add(idx);
            let dst_ptr = self.dst_ghost_ids_host.as_ptr().add(idx);

            axicor_compute::ffi::gpu_memcpy_host_to_device_async(
                self.src_indices_d.add(idx) as *mut _,
                src_ptr as *const _,
                4,
                stream,
            );
            axicor_compute::ffi::gpu_memcpy_host_to_device_async(
                self.dst_ghost_ids_d.add(idx) as *mut _,
                dst_ptr as *const _,
                4,
                stream,
            );
        }

        self.count -= 1;
    }
}
