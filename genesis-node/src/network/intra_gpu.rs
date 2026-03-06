use genesis_compute::ffi;
use std::ptr;

/// Описание одной связи: аксон зоны-источника → слот ghost-аксона зоны назначения.
/// Используется для построения IntraGpuChannel из .ghosts файла Baker'а.
pub struct GhostLink {
    pub src_zone_idx: usize,
    pub src_axon_id: u32,
    pub dst_zone_idx: usize,
    pub dst_ghost_id: u32,
}

/// Канал синхронизации между зонами на одном GPU.
///
/// Хранит две параллельных таблицы (src → dst) в VRAM.
/// Поддерживает оба режима использования:
///   - Legacy: передать два среза `&[u32]` напрямую.
///   - Новый: передать `Vec<GhostLink>` и получить автоматическую разбивку.
pub struct IntraGpuChannel {
    pub src_indices_host: Vec<u32>,
    pub dst_indices_host: Vec<u32>,
    pub src_zone_indices: Vec<usize>,  // Индекс зоны-источника (per link)
    pub dst_zone_indices: Vec<usize>,  // Индекс зоны-назначения (per link)
    pub src_indices_d: *mut u32,       // VRAM
    pub dst_indices_d: *mut u32,       // VRAM
    pub count: u32,
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
    /// Новый API: строит канал из списка `GhostLink`.
    /// Caller после этого вызывает `sync_spikes(&mut zones)` на барьере BSP.
    pub unsafe fn new(links: Vec<GhostLink>) -> Self {
        let src_axon_ids:  Vec<u32>   = links.iter().map(|l| l.src_axon_id).collect();
        let dst_ghost_ids: Vec<u32>   = links.iter().map(|l| l.dst_ghost_id).collect();
        let src_zone_ids:  Vec<usize> = links.iter().map(|l| l.src_zone_idx).collect();
        let dst_zone_ids:  Vec<usize> = links.iter().map(|l| l.dst_zone_idx).collect();

        let count = src_axon_ids.len() as u32;
        let bytes = (count as usize) * 4;

        let src_d = if bytes > 0 { ffi::gpu_malloc(bytes) as *mut u32 } else { ptr::null_mut() };
        let dst_d = if bytes > 0 { ffi::gpu_malloc(bytes) as *mut u32 } else { ptr::null_mut() };

        if bytes > 0 {
            let stream = ptr::null_mut();
            ffi::gpu_memcpy_host_to_device_async(src_d as *mut _, src_axon_ids.as_ptr() as *const _, bytes, stream);
            ffi::gpu_memcpy_host_to_device_async(dst_d as *mut _, dst_ghost_ids.as_ptr() as *const _, bytes, stream);
            ffi::gpu_stream_synchronize(stream);
        }

        Self {
            src_indices_host: src_axon_ids,
            dst_indices_host: dst_ghost_ids,
            src_zone_indices: src_zone_ids,
            dst_zone_indices: dst_zone_ids,
            src_indices_d: src_d,
            dst_indices_d: dst_d,
            count,
        }
    }

    /// Legacy API: raw index slices (kein GhostLink Overhead).
    pub unsafe fn from_slices(src_indices: &[u32], dst_indices: &[u32]) -> Self {
        assert_eq!(src_indices.len(), dst_indices.len());
        let count = src_indices.len() as u32;
        let bytes = (count as usize) * 4;

        let src_d = ffi::gpu_malloc(bytes) as *mut u32;
        let dst_d = ffi::gpu_malloc(bytes) as *mut u32;

        let stream = ptr::null_mut();
        ffi::gpu_memcpy_host_to_device_async(src_d as *mut _, src_indices.as_ptr() as *const _, bytes, stream);
        ffi::gpu_memcpy_host_to_device_async(dst_d as *mut _, dst_indices.as_ptr() as *const _, bytes, stream);
        ffi::gpu_stream_synchronize(stream);

        Self {
            src_indices_host:  src_indices.to_vec(),
            dst_indices_host:  dst_indices.to_vec(),
            src_zone_indices:  (0..count as usize).map(|_| 0).collect(),
            dst_zone_indices:  (0..count as usize).map(|_| 1).collect(),
            src_indices_d: src_d,
            dst_indices_d: dst_d,
            count,
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
}
