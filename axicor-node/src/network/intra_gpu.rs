use axicor_compute::ffi;
use std::ptr;

/// Synchronization channel between zones on single GPU.
///
/// Stores two parallel tables (src -> dst) in VRAM.
/// Supports both usage modes:
///   - Legacy: pass two `&[u32]` slices directly.
///   - New: pass `Vec<GhostLink>` and get automatic splitting.
pub struct IntraGpuChannel {
    pub src_zone_hash: u32,    // [DOD FIX] Strict binding to source
    pub target_zone_hash: u32, // [DOD FIX] Strict binding to destination
    pub capacity: u32,
    pub count: u32,
    pub dst_total_axons: u32, // [DOD FIX] Guard for OOB Ghost Sync

    pub src_indices_host: Vec<u32>,
    pub dst_indices_host: Vec<u32>,

    pub src_indices_d: *mut u32,
    pub dst_indices_d: *mut u32,
}

unsafe impl Send for IntraGpuChannel {}
unsafe impl Sync for IntraGpuChannel {}

impl Drop for IntraGpuChannel {
    fn drop(&mut self) {
        unsafe {
            axicor_compute::ffi::gpu_free(self.src_indices_d as *mut _);
            axicor_compute::ffi::gpu_free(self.dst_indices_d as *mut _);
        }
    }
}

impl IntraGpuChannel {
    /// # Safety
    /// Caller must ensure CUDA context is initialized and that capacity is sufficient for the connection count.
    pub unsafe fn from_slices(
        src_zone_hash: u32,
        target_zone_hash: u32,
        src_indices: &[u32],
        dst_indices: &[u32],
        capacity: u32,
        dst_total_axons: u32,
    ) -> Self {
        assert_eq!(src_indices.len(), dst_indices.len());
        let count = src_indices.len() as u32;
        debug_assert!(
            count <= capacity,
            "FATAL: Initial connections exceed capacity"
        );
        if count > capacity {
            tracing::error!(
                "CRITICAL: Initial IntraGPU connections (count={}) exceed capacity ({}). Clamping.",
                count,
                capacity
            );
        }
        let count = std::cmp::min(count, capacity);

        let bytes_capacity = (capacity as usize) * 4;
        let src_d = axicor_compute::ffi::gpu_malloc(bytes_capacity) as *mut u32;
        let dst_d = axicor_compute::ffi::gpu_malloc(bytes_capacity) as *mut u32;

        let stream = ptr::null_mut();
        if count > 0 {
            let bytes_active = (count as usize) * 4;
            axicor_compute::ffi::gpu_memcpy_host_to_device_async(
                src_d as *mut _,
                src_indices.as_ptr() as *const _,
                bytes_active,
                stream,
            );
            axicor_compute::ffi::gpu_memcpy_host_to_device_async(
                dst_d as *mut _,
                dst_indices.as_ptr() as *const _,
                bytes_active,
                stream,
            );
        }

        let mut src_host = Vec::with_capacity(capacity as usize);
        src_host.extend_from_slice(src_indices);
        let mut dst_host = Vec::with_capacity(capacity as usize);
        dst_host.extend_from_slice(dst_indices);

        Self {
            src_zone_hash,
            target_zone_hash,
            capacity,
            count,
            src_indices_host: src_host,
            dst_indices_host: dst_host,
            src_indices_d: src_d,
            dst_indices_d: dst_d,
            dst_total_axons,
        }
    }

    /// CPU-side ghost sync. Used in mock mode and intra-host tests.
    /// In production zones pass axon_head_index via ShardEngine, not ZoneRuntime.
    pub fn sync_spikes(&mut self, _zones: &mut [crate::zone_runtime::ZoneRuntime]) {
        // Legacy path removed  ZoneRuntime no longer owns raw VRAM.
        // In production GPU sync is done via sync_ghosts() with raw device pointers.
        unimplemented!("sync_spikes is not available after axicor-compute split");
    }

    /// # Safety
    /// Pointers must be aligned to 32 bytes (`BurstHeads8`).
    pub unsafe fn sync_ghosts(
        &self,
        src_heads: *const axicor_core::layout::BurstHeads8,
        dst_heads: *mut axicor_core::layout::BurstHeads8,
        sync_batch_ticks: u32,
        v_seg: u32,
        stream: ffi::CudaStream,
    ) {
        if self.count == 0 {
            return;
        }
        ffi::launch_ghost_sync(
            src_heads,
            dst_heads,
            self.src_indices_d,
            self.dst_indices_d,
            self.count,
            self.dst_total_axons, // [DOD FIX] Hardware VRAM Guard
            sync_batch_ticks,
            v_seg,
            stream,
        );
    }

    // Drop src_zone_idx and dst_zone_idx from push_route
    /// # Safety
    /// Caller must ensure CUDA stream and routing indices are valid.
    pub unsafe fn push_route(
        &mut self,
        src_axon: u32,
        dst_ghost: u32,
        stream: axicor_compute::ffi::CudaStream,
    ) {
        debug_assert!(
            self.count < self.capacity,
            "FATAL: IntraGPU Routing capacity exceeded."
        );
        if self.count >= self.capacity {
            tracing::error!(
                "CRITICAL: IntraGPU Routing capacity exceeded ({}). Link 0x{:08X} -> 0x{:08X} dropped.",
                self.capacity, src_axon, dst_ghost
            );
            return;
        }
        let idx = self.count as usize;

        self.src_indices_host.push(src_axon);
        self.dst_indices_host.push(dst_ghost);

        let src_ptr = self.src_indices_host.as_ptr().add(idx);
        let dst_ptr = self.dst_indices_host.as_ptr().add(idx);

        axicor_compute::ffi::gpu_memcpy_host_to_device_async(
            self.src_indices_d.add(idx) as *mut _,
            src_ptr as *const _,
            4,
            stream,
        );
        axicor_compute::ffi::gpu_memcpy_host_to_device_async(
            self.dst_indices_d.add(idx) as *mut _,
            dst_ptr as *const _,
            4,
            stream,
        );

        self.count += 1;
    }

    /// # Safety
    /// Caller must ensure CUDA stream is valid.
    pub unsafe fn prune_route(
        &mut self,
        target_ghost_id: u32,
        stream: axicor_compute::ffi::CudaStream,
    ) {
        let Some(idx) = self
            .dst_indices_host
            .iter()
            .position(|&g| g == target_ghost_id)
        else {
            return;
        };
        let last_idx = (self.count - 1) as usize;

        self.src_indices_host.swap(idx, last_idx);
        self.dst_indices_host.swap(idx, last_idx);

        self.src_indices_host.pop();
        self.dst_indices_host.pop();

        if idx != last_idx {
            let src_ptr = self.src_indices_host.as_ptr().add(idx);
            let dst_ptr = self.dst_indices_host.as_ptr().add(idx);

            axicor_compute::ffi::gpu_memcpy_host_to_device_async(
                self.src_indices_d.add(idx) as *mut _,
                src_ptr as *const _,
                4,
                stream,
            );
            axicor_compute::ffi::gpu_memcpy_host_to_device_async(
                self.dst_indices_d.add(idx) as *mut _,
                dst_ptr as *const _,
                4,
                stream,
            );
        }

        self.count -= 1;
    }
}
