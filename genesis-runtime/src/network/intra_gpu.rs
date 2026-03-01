use crate::zone_runtime::ZoneRuntime;
use crate::network::channel::Channel;
use crate::ffi;
use std::ptr;

/// Канал синхронизации между зонами на одном GPU.
/// Индексы хранятся в VRAM, чтобы CPU вообще не участвовал в передаче спайков.
pub struct IntraGpuChannel {
    pub src_indices_d: *mut u32, // Указатель VRAM
    pub dst_indices_d: *mut u32, // Указатель VRAM
    pub count: u32,
}

impl IntraGpuChannel {
    /// Инициализация канала на базе маппинга (сгенерированного в .ghosts файле Baker'ом)
    pub unsafe fn new(src_indices_host: &[u32], dst_indices_host: &[u32]) -> Self {
        assert_eq!(src_indices_host.len(), dst_indices_host.len());
        let count = src_indices_host.len() as u32;
        let bytes = (count as usize) * 4;

        let src_d = ffi::gpu_malloc(bytes) as *mut u32;
        let dst_d = ffi::gpu_malloc(bytes) as *mut u32;

        let stream = ptr::null_mut();
        ffi::gpu_memcpy_host_to_device_async(src_d as *mut _, src_indices_host.as_ptr() as *const _, bytes, stream);
        ffi::gpu_memcpy_host_to_device_async(dst_d as *mut _, dst_indices_host.as_ptr() as *const _, bytes, stream);
        ffi::gpu_stream_synchronize(stream);

        Self {
            src_indices_d: src_d,
            dst_indices_d: dst_d,
            count,
        }
    }

    /// Вызывается во время BSP Барьера (Strict BSP)
    pub unsafe fn sync_ghosts(&self, src_heads: *const u32, dst_heads: *mut u32, stream: ffi::CudaStream) {
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
