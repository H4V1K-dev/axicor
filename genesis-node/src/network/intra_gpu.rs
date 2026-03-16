use genesis_compute::ffi;
use std::ptr;

/// Канал синхронизации между зонами на одном GPU.
///
/// Хранит две параллельных таблицы (src → dst) в VRAM.
/// Поддерживает оба режима использования:
///   - Legacy: передать два среза `&[u32]` напрямую.
///   - Новый: передать `Vec<GhostLink>` и получить автоматическую разбивку.
pub struct IntraGpuChannel {
    pub src_zone_hash: u32,    // [DOD FIX] Жесткая привязка к источнику
    pub target_zone_hash: u32, // [DOD FIX] Жесткая привязка к приемнику
    pub capacity: u32,
    pub count: u32,

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
            genesis_compute::ffi::gpu_free(self.src_indices_d as *mut _);
            genesis_compute::ffi::gpu_free(self.dst_indices_d as *mut _);
        }
    }
}

impl IntraGpuChannel {
    pub unsafe fn from_slices(
        src_zone_hash: u32, 
        target_zone_hash: u32, 
        src_indices: &[u32], 
        dst_indices: &[u32], 
        capacity: u32
    ) -> Self {
        assert_eq!(src_indices.len(), dst_indices.len());
        let count = src_indices.len() as u32;
        assert!(count <= capacity, "FATAL: Initial connections exceed capacity");

        let bytes_capacity = (capacity as usize) * 4;
        let src_d = genesis_compute::ffi::gpu_malloc(bytes_capacity) as *mut u32;
        let dst_d = genesis_compute::ffi::gpu_malloc(bytes_capacity) as *mut u32;

        let stream = ptr::null_mut();
        if count > 0 {
            let bytes_active = (count as usize) * 4;
            genesis_compute::ffi::gpu_memcpy_host_to_device_async(src_d as *mut _, src_indices.as_ptr() as *const _, bytes_active, stream);
            genesis_compute::ffi::gpu_memcpy_host_to_device_async(dst_d as *mut _, dst_indices.as_ptr() as *const _, bytes_active, stream);
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
        }
    }

    /// GPU-side sync: вызов CUDA kernel launch_ghost_sync (production).
    pub unsafe fn sync_ghosts(&self, src_heads: *const genesis_core::layout::BurstHeads8, dst_heads: *mut genesis_core::layout::BurstHeads8, stream: ffi::CudaStream) {
        if self.count == 0 { return; }
        ffi::launch_ghost_sync(
            src_heads,
            dst_heads,
            self.src_indices_d,
            self.dst_indices_d,
            self.count,
            stream
        );
    }

    // Выкидываем src_zone_idx и dst_zone_idx из push_route
    pub unsafe fn push_route(&mut self, src_axon: u32, dst_ghost: u32, stream: genesis_compute::ffi::CudaStream) {
        assert!(self.count < self.capacity, "FATAL: IntraGPU Routing capacity exceeded.");
        let idx = self.count as usize;

        self.src_indices_host.push(src_axon);
        self.dst_indices_host.push(dst_ghost);

        let src_ptr = self.src_indices_host.as_ptr().add(idx);
        let dst_ptr = self.dst_indices_host.as_ptr().add(idx);

        genesis_compute::ffi::gpu_memcpy_host_to_device_async(self.src_indices_d.add(idx) as *mut _, src_ptr as *const _, 4, stream);
        genesis_compute::ffi::gpu_memcpy_host_to_device_async(self.dst_indices_d.add(idx) as *mut _, dst_ptr as *const _, 4, stream);

        self.count += 1;
    }

    pub unsafe fn prune_route(&mut self, target_ghost_id: u32, stream: genesis_compute::ffi::CudaStream) {
        let Some(idx) = self.dst_indices_host.iter().position(|&g| g == target_ghost_id) else { return; };
        let last_idx = (self.count - 1) as usize;

        self.src_indices_host.swap(idx, last_idx);
        self.dst_indices_host.swap(idx, last_idx);

        self.src_indices_host.pop();
        self.dst_indices_host.pop();

        if idx != last_idx {
            let src_ptr = self.src_indices_host.as_ptr().add(idx);
            let dst_ptr = self.dst_indices_host.as_ptr().add(idx);

            genesis_compute::ffi::gpu_memcpy_host_to_device_async(self.src_indices_d.add(idx) as *mut _, src_ptr as *const _, 4, stream);
            genesis_compute::ffi::gpu_memcpy_host_to_device_async(self.dst_indices_d.add(idx) as *mut _, dst_ptr as *const _, 4, stream);
        }

        self.count -= 1;
    }
}
