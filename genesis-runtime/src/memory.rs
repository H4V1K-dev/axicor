use genesis_core::layout::{padded_n, StateFileHeader, AxonsFileHeader};
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use std::ffi::c_void;
use crate::ffi;

/// Per-matrix input metadata for handling multiple input maps
#[derive(Clone, Debug)]
pub struct InputMatrixInfo {
    pub pixel_offset: u32,   // Starting index in axon_ids array
    pub num_pixels: u32,     // width * height
    pub stride: u8,          // Injection frequency
}

/// Typesafe wrapper over device pointers for the GPU SoA layout.

pub struct VramState {
    pub padded_n: usize,
    
    // Soma State
    pub voltage: *mut c_void,
    pub threshold_offset: *mut c_void,
    pub refractory_timer: *mut c_void,
    pub flags: *mut c_void,

    // Axon State (total_axons length, not padded_n)
    pub total_axons: usize,
    pub ghost_axons_allocated: usize,
    pub max_ghost_axons: usize,
    pub base_axons: usize,
    pub axon_head_index: *mut c_void,
    pub soma_to_axon: *mut c_void,

    // Dendrite Columns (MAX_DENDRITE_SLOTS * padded_n length)
    pub dendrite_targets: *mut c_void,
    pub dendrite_weights: *mut c_void,
    pub dendrite_refractory: *mut c_void,

    pub pinned_host_targets: *mut c_void,
    pub pinned_host_weights: *mut c_void,

    // Virtual Axons (InjectInputs)
    pub num_pixels: u32,
    pub map_pixel_to_axon: *mut c_void,
    pub input_bitmask_buffer: *mut c_void,
    pub input_matrices: Vec<InputMatrixInfo>,  // Per-matrix stride and offset info
    pub input_stride: u32,  // Legacy: default stride if no matrices defined

    // Outbound Spikes (Per-Tick, MAX_SPIKES_PER_TICK length) - REMOVED

    // Readout Interface (Output §3)
    pub num_mapped_somas: u32,
    pub readout_batch_ticks: u32,
    pub mapped_soma_ids: *mut c_void,   // [total_mapped_somas] u32
    pub output_history: *mut c_void,     // [batch_ticks × total_mapped_somas] u8
    pub telemetry_spikes: *mut c_void,
    pub telemetry_count: *mut c_void,
    pub telemetry_spikes_host: *mut c_void,
}

impl VramState {
    pub fn load_shard(state_bytes: &[u8], axons_bytes: &[u8], gxi: Option<&crate::input::GxiFile>, gxo: Option<&crate::output::GxoFile>, readout_batch_ticks: u32, input_stride: u32, required_ghost_slots: usize) -> anyhow::Result<Self> {
        let axons_header = AxonsFileHeader::from_bytes(axons_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        let num_axons = axons_header.total_axons as usize;

        let pa = padded_n(num_axons);

        let state_header = StateFileHeader::from_bytes(state_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        
        let pn = state_header.padded_n as usize;
        let dc = MAX_DENDRITE_SLOTS * pn;

        let mut offset = state_header.header_size as usize;
        // --- Inject Inputs Context ---
        let mut map_pixel_to_axon = std::ptr::null_mut();
        let mut num_pixels = 0;
        let mut bitmask_buffer = std::ptr::null_mut();
        let mut input_matrices = Vec::new();
        
        let batch_size_ticks = readout_batch_ticks as usize;

        if let Some(g) = gxi {
            num_pixels = g.axon_ids.len() as u32;
            if num_pixels > 0 {
                // Build per-matrix metadata
                let mut pixel_offset = 0;
                for matrix_desc in &g.matrices {
                    let num_pix = (matrix_desc.width * matrix_desc.height) as u32;
                    input_matrices.push(InputMatrixInfo {
                        pixel_offset,
                        num_pixels: num_pix,
                        stride: matrix_desc.stride,
                    });
                    pixel_offset += num_pix;
                }
                
                unsafe {
                    let bytes = (num_pixels as usize) * 4;
                    map_pixel_to_axon = ffi::gpu_malloc(bytes);
                    if map_pixel_to_axon.is_null() {
                        anyhow::bail!("gpu_malloc failed for map_pixel_to_axon ({} bytes)", bytes);
                    }
                    let success = ffi::gpu_memcpy_host_to_device(
                        map_pixel_to_axon,
                        g.axon_ids.as_ptr() as *const c_void,
                        bytes,
                    );
                    if !success {
                        anyhow::bail!("Failed to upload map_pixel_to_axon to GPU");
                    }

                    let bitmask_bytes = ((num_pixels as usize + 31) / 32) * 4 * batch_size_ticks;
                    bitmask_buffer = ffi::gpu_malloc(bitmask_bytes);
                    if bitmask_buffer.is_null() {
                        anyhow::bail!("gpu_malloc failed for input_bitmask_buffer ({} bytes)", bitmask_bytes);
                    }
                }
            }
        }
        let mut allocate_and_copy = |slice_len: usize| -> anyhow::Result<*mut c_void> {
            let ptr = unsafe { ffi::gpu_malloc(slice_len) };
            if ptr.is_null() {
                anyhow::bail!("gpu_malloc failed for size {}", slice_len);
            }
            let success = unsafe {
                ffi::gpu_memcpy_host_to_device(
                    ptr,
                    state_bytes[offset..offset + slice_len].as_ptr() as *const c_void,
                    slice_len,
                )
            };
            if !success {
                anyhow::bail!("gpu_memcpy_host_to_device failed for size {}", slice_len);
            }
            offset += slice_len;
            Ok(ptr)
        };

        let voltage = allocate_and_copy(pn * 4)?;
        let flags = allocate_and_copy(pn * 1)?;
        let threshold_offset = allocate_and_copy(pn * 4)?;
        let refractory_timer = allocate_and_copy(pn * 1)?;
        let soma_to_axon = allocate_and_copy(pn * 4)?;
        let dendrite_targets = allocate_and_copy(dc * 4)?;
        let dendrite_weights = allocate_and_copy(dc * 2)?;
        let dendrite_refractory = allocate_and_copy(dc * 1)?;

        let pinned_host_targets = unsafe { ffi::gpu_host_alloc(dc * 4) };
        if pinned_host_targets.is_null() { anyhow::bail!("alloc failed for pinned_host_targets"); }
        let pinned_host_weights = unsafe { ffi::gpu_host_alloc(dc * 2) };
        if pinned_host_weights.is_null() { anyhow::bail!("alloc failed for pinned_host_weights"); }
        
        
        // Телеметрия
        let telemetry_count = unsafe { ffi::gpu_malloc(4) };
        if telemetry_count.is_null() { anyhow::bail!("alloc failed for telemetry_count"); }

        let telemetry_spikes = unsafe { ffi::gpu_malloc(500_000 * 4) }; 
        if telemetry_spikes.is_null() { anyhow::bail!("alloc failed for telemetry_spikes"); }
        
        let telemetry_spikes_host = unsafe { ffi::gpu_host_alloc(500_000 * 4) };
        if telemetry_spikes_host.is_null() { anyhow::bail!("alloc failed for telemetry_spikes_host"); }
        
        // Axon Heads: Base + Pre-allocate Ghost Axons (dynamic based on actual connections)
        let max_ghost_axons = required_ghost_slots;
        let total_axons = pa + max_ghost_axons;
        println!("       Allocating {} base + {} ghost = {} axon slots", pa, max_ghost_axons, total_axons);
        let axon_head_index = unsafe { ffi::gpu_malloc(total_axons * 4) };
        if axon_head_index.is_null() { anyhow::bail!("alloc failed for axon heads"); }
        
        // Copy base axons
        unsafe {
            ffi::gpu_memcpy_host_to_device(
                axon_head_index,
                state_bytes[offset..offset + pa * 4].as_ptr() as *const c_void,
                pa * 4,
            );
        }
        // No need to increment offset here, it's the end of axons

        // Init spare Ghost Axons to AXON_SENTINEL
        let sentinels = vec![0x80000000u32; max_ghost_axons];
        unsafe {
            ffi::gpu_memcpy_host_to_device(
                (axon_head_index as *mut u32).add(pa) as *mut c_void,
                sentinels.as_ptr() as *const c_void,
                max_ghost_axons * 4,
            );
        }

        // Readout Buffer Allocation
        let mut mapped_soma_ids = std::ptr::null_mut();
        let mut output_history = std::ptr::null_mut();
        let mut num_mapped_somas = 0;
        if let Some(o) = gxo {
            num_mapped_somas = o.soma_ids.len() as u32;
            if num_mapped_somas > 0 && readout_batch_ticks > 0 {
                unsafe {
                    let somas_bytes = (num_mapped_somas as usize) * 4;
                    mapped_soma_ids = ffi::gpu_malloc(somas_bytes);
                    if mapped_soma_ids.is_null() {
                        anyhow::bail!("gpu_malloc failed for mapped_soma_ids ({} bytes)", somas_bytes);
                    }
                    if !ffi::gpu_memcpy_host_to_device(mapped_soma_ids, o.soma_ids.as_ptr() as *const c_void, somas_bytes) {
                        anyhow::bail!("Failed to upload mapped_soma_ids to GPU");
                    }

                    // output_history buffer (u8 per tick per soma)
                    let history_bytes = (num_mapped_somas as usize) * (readout_batch_ticks as usize);
                    output_history = ffi::gpu_malloc(history_bytes);
                    if output_history.is_null() {
                        anyhow::bail!("gpu_malloc failed for output_history ({} bytes)", history_bytes);
                    }
                    // It's good practice to zero it out, though the kernel writes absolutely every byte
                    // we'll leave it uninitialized on GPU to save time, it will be fully overwritten over the batch ticks.
                }
            }
        }



        println!("Zone VRAM Load! num_pixels={}, bitmask_buffer is null: {}", num_pixels, bitmask_buffer.is_null());

        Ok(VramState {
            padded_n: pn,
            total_axons,
            ghost_axons_allocated: 0,
            max_ghost_axons,
            base_axons: pa,
            num_pixels,
            map_pixel_to_axon,
            input_bitmask_buffer: bitmask_buffer,
            input_matrices,
            voltage,
            threshold_offset,
            refractory_timer,
            flags,
            soma_to_axon,
            axon_head_index,
            dendrite_targets,
            dendrite_weights,
            dendrite_refractory,
            pinned_host_targets,
            pinned_host_weights,
            num_mapped_somas,
            readout_batch_ticks,
            mapped_soma_ids,
            output_history,
            telemetry_spikes,
            telemetry_count,
            telemetry_spikes_host,
            input_stride,
        })
    }

    /// Extracted helper to launch the Night Phase Sort & Prune kernel
    pub fn run_sort_and_prune(&self, prune_threshold: i16) {
        unsafe {
            ffi::launch_sort_and_prune(
                self.padded_n as u32,
                self.dendrite_targets,
                self.dendrite_weights,
                self.dendrite_refractory, // Using timers
                prune_threshold,
                std::ptr::null_mut()
            );
        }
    }

    /// Downloads a generic slice of data from the GPU.
    fn download_generic<T: Clone + Default>(&self, ptr: *mut c_void, count: usize) -> anyhow::Result<Vec<T>> {
        let size = count * std::mem::size_of::<T>();
        let mut host_data = vec![T::default(); count];
        
        let success = unsafe {
            ffi::gpu_memcpy_device_to_host(
                host_data.as_mut_ptr() as *mut c_void,
                ptr as *const c_void,
                size,
            )
        };

        if !success {
            anyhow::bail!("gpu_memcpy_device_to_host failed for size {}", size);
        }

        Ok(host_data)
    }

    pub fn download_telemetry(&self, _stream: crate::ffi::CudaStream) -> anyhow::Result<Vec<u32>> {
        // Obsolete function, replaced by raw memory copies in orchestrator/day_phase.rs Zero-Copy logic.
        Ok(Vec::new())
    }

    pub fn download_voltage(&self) -> anyhow::Result<Vec<i32>> {
        self.download_generic(self.voltage, self.padded_n)
    }

    pub fn download_flags(&self) -> anyhow::Result<Vec<u8>> {
        self.download_generic(self.flags, self.padded_n)
    }

    pub fn download_threshold_offset(&self) -> anyhow::Result<Vec<i32>> {
        self.download_generic(self.threshold_offset, self.padded_n)
    }

    pub fn download_refractory_timer(&self) -> anyhow::Result<Vec<u8>> {
        self.download_generic(self.refractory_timer, self.padded_n)
    }

    pub fn download_axon_head_index(&self) -> anyhow::Result<Vec<u32>> {
        self.download_generic(self.axon_head_index, self.total_axons)
    }

    pub fn download_dendrite_weights(&self) -> anyhow::Result<Vec<i16>> {
        self.download_generic(self.dendrite_weights, self.padded_n * MAX_DENDRITE_SLOTS)
    }

    pub fn download_dendrite_targets(&self) -> anyhow::Result<Vec<u32>> {
        self.download_generic(self.dendrite_targets, self.padded_n * MAX_DENDRITE_SLOTS)
    }

    pub fn upload_dendrite_weights(&self, host_data: &[i16]) -> anyhow::Result<()> {
        let expected_len = self.padded_n * MAX_DENDRITE_SLOTS;
        if host_data.len() != expected_len {
            anyhow::bail!("Invalid length: expected {}, got {}", expected_len, host_data.len());
        }
        let size = expected_len * std::mem::size_of::<i16>();
        let success = unsafe {
            ffi::gpu_memcpy_host_to_device(
                self.dendrite_weights,
                host_data.as_ptr() as *const std::ffi::c_void,
                size,
            )
        };
        if !success {
            anyhow::bail!("gpu_memcpy_host_to_device failed for dendrite weights");
        }
        Ok(())
    }

    pub fn upload_dendrite_targets(&self, host_data: &[u32]) -> anyhow::Result<()> {
        let expected_len = self.padded_n * MAX_DENDRITE_SLOTS;
        if host_data.len() != expected_len {
            anyhow::bail!("Invalid length: expected {}, got {}", expected_len, host_data.len());
        }
        let size = expected_len * std::mem::size_of::<u32>();
        let success = unsafe {
            ffi::gpu_memcpy_host_to_device(
                self.dendrite_targets,
                host_data.as_ptr() as *const std::ffi::c_void,
                size,
            )
        };
        if !success {
            anyhow::bail!("gpu_memcpy_host_to_device failed for dendrite targets");
        }
        Ok(())
    }

    pub fn download_dendrite_timers(&self) -> anyhow::Result<Vec<u8>> {
        self.download_generic(self.dendrite_refractory, self.padded_n * MAX_DENDRITE_SLOTS)
    }

    pub fn download_output_history(&self) -> anyhow::Result<Vec<u8>> {
        if self.num_mapped_somas == 0 || self.readout_batch_ticks == 0 {
            return Ok(Vec::new());
        }
        let total_bytes = (self.num_mapped_somas as usize) * (self.readout_batch_ticks as usize);
        self.download_generic(self.output_history, total_bytes)
    }

    /// Uploads a bitmask array to GPU memory. Used for External Virtual Axons.
    /// Bitmask must be accurately sized: ((num_pixels + 31)/32) u32s times batch size.
    pub fn upload_input_bitmask(&self, bitmask: &[u32], num_ticks: usize) -> anyhow::Result<()> {
        if self.num_pixels == 0 {
            return Ok(());
        }
        let max_ticks = self.readout_batch_ticks as usize;
        if num_ticks > max_ticks {
            anyhow::bail!("Batch size too large: {} (max readout_batch_ticks {})", num_ticks, max_ticks);
        }
        
        let words_per_tick = (self.num_pixels as usize + 31) / 32;
        let total_words = words_per_tick * num_ticks;
        
        if bitmask.len() < total_words {
            anyhow::bail!("Bitmask len {} is less than required {} for {} ticks", bitmask.len(), total_words, num_ticks);
        }

        let bytes = total_words * std::mem::size_of::<u32>();
        let success = unsafe {
            ffi::gpu_memcpy_host_to_device(
                self.input_bitmask_buffer,
                bitmask.as_ptr() as *const c_void,
                bytes,
            )
        };
        if !success {
            anyhow::bail!("Failed to upload input bitmask to GPU");
        }

        Ok(())
    }

    pub fn allocate_ghost_axon(&mut self) -> Option<u32> {
        if self.ghost_axons_allocated < self.max_ghost_axons {
            let id = (self.base_axons + self.ghost_axons_allocated) as u32;
            self.ghost_axons_allocated += 1;
            Some(id)
        } else {
            None
        }
    }

    pub fn free_ghost_axon(&mut self, ghost_id: u32) {
        if (ghost_id as usize) >= self.base_axons && (ghost_id as usize) < self.base_axons + self.ghost_axons_allocated {
            let sentinel: u32 = 0x80000000;
            let offset = ghost_id as usize;
            unsafe {
                ffi::gpu_memcpy_host_to_device(
                    (self.axon_head_index as *mut u32).add(offset) as *mut std::ffi::c_void,
                    &sentinel as *const _ as *const std::ffi::c_void,
                    4,
                );
            }
        }
    }

    /// Zero-cost загрузка состояния шарда.
    /// Читает сырые байты с диска и заливает их в VRAM.
    /// Никакой десериализации. Скорость ограничена только PCIe x16.
    ///
    /// # Safety
    /// Порядок полей в .state ОБЯЗАН совпадать с порядком дампа в ShardSoA::dump_to_disk:
    ///   voltage | flags | threshold_offset | refractory_timer |
    ///   dendrite_targets | dendrite_weights | dendrite_timers
    /// Нарушение этого контракта → Silent Data Corruption в VRAM.
    pub unsafe fn load_from_disk(
        &mut self,
        state_path: &std::path::Path,
        axons_path: &std::path::Path,
        stream: *mut std::ffi::c_void, // opaque cudaStream_t; зарезервирован для async API
    ) {
        let state_bytes = std::fs::read(state_path).expect("Fatal: Failed to read .state file");
        let axons_bytes = std::fs::read(axons_path).expect("Fatal: Failed to read .axons file");

        let pn = self.padded_n;
        let dc = MAX_DENDRITE_SLOTS * pn;

        // Хард-валидация: размер файла должен совпадать байт в байт с ожидаемой SoA-раскладкой.
        // Если не совпадает — это несовпадение версий бейкера и рантайма. Segfault лучше не допускать.
        let expected_state_size =
            pn * 4   // voltage (i32)
          + pn * 1   // flags (u8)
          + pn * 4   // threshold_offset (i32)
          + pn * 1   // refractory_timer (u8)
          + dc * 4   // dendrite_targets (u32)
          + dc * 2   // dendrite_weights (i16)
          + dc * 1;  // dendrite_timers (u8)

        let expected_axons_size = self.base_axons * 4; // axon_heads (u32), только base

        assert_eq!(
            state_bytes.len(),
            expected_state_size,
            "VRAM Layout mismatch! .state size {} != expected {} (padded_n={})",
            state_bytes.len(), expected_state_size, pn
        );
        assert_eq!(
            axons_bytes.len(),
            expected_axons_size,
            "VRAM Layout mismatch! .axons size {} != expected {} (base_axons={})",
            axons_bytes.len(), expected_axons_size, self.base_axons
        );

        // DMA-трансферы в VRAM: по одному на каждое SoA-поле.
        // Порядок зеркалит ShardSoA::dump_to_disk.
        let mut offset = 0usize;

        macro_rules! copy_field {
            ($dst:expr, $bytes:expr) => {{
                crate::ffi::gpu_memcpy_host_to_device_async(
                    $dst as *mut std::ffi::c_void,
                    state_bytes[offset..offset + $bytes].as_ptr() as *const std::ffi::c_void,
                    $bytes,
                    stream as crate::ffi::CudaStream,
                );
                offset += $bytes;
            }};
        }

        copy_field!(self.voltage,             pn * 4);
        copy_field!(self.flags,               pn * 1);
        copy_field!(self.threshold_offset,    pn * 4);
        copy_field!(self.refractory_timer,    pn * 1);
        copy_field!(self.dendrite_targets,    dc * 4);
        copy_field!(self.dendrite_weights,    dc * 2);
        copy_field!(self.dendrite_refractory, dc * 1);

        // Аксоны — отдельный файл, только base (ghost slots уже инициализированы SENTINEL'ами при аллокации)
        crate::ffi::gpu_memcpy_host_to_device_async(
            self.axon_head_index as *mut std::ffi::c_void,
            axons_bytes.as_ptr() as *const std::ffi::c_void,
            expected_axons_size,
            stream as crate::ffi::CudaStream,
        );

        crate::ffi::gpu_stream_synchronize(stream as crate::ffi::CudaStream);
    }
}


impl Drop for VramState {
    fn drop(&mut self) {
        unsafe {
            if !self.voltage.is_null() { ffi::gpu_free(self.voltage); }
            if !self.threshold_offset.is_null() { ffi::gpu_free(self.threshold_offset); }
            if !self.refractory_timer.is_null() { ffi::gpu_free(self.refractory_timer); }
            if !self.flags.is_null() { ffi::gpu_free(self.flags); }

            if !self.axon_head_index.is_null() { ffi::gpu_free(self.axon_head_index); }
            if !self.soma_to_axon.is_null() { ffi::gpu_free(self.soma_to_axon); }

            if !self.dendrite_targets.is_null() { ffi::gpu_free(self.dendrite_targets); }
            if !self.dendrite_weights.is_null() { ffi::gpu_free(self.dendrite_weights); }
            if !self.dendrite_refractory.is_null() { ffi::gpu_free(self.dendrite_refractory); }

            if !self.mapped_soma_ids.is_null() {
                ffi::gpu_free(self.mapped_soma_ids);
            }
            if !self.output_history.is_null() {
                ffi::gpu_free(self.output_history);
            }

            if !self.map_pixel_to_axon.is_null() {
                ffi::gpu_free(self.map_pixel_to_axon);
            }
            if !self.input_bitmask_buffer.is_null() {
                ffi::gpu_free(self.input_bitmask_buffer);
            }

            if !self.telemetry_count.is_null() { ffi::gpu_free(self.telemetry_count); }
            if !self.telemetry_spikes.is_null() { ffi::gpu_free(self.telemetry_spikes); }
            if !self.telemetry_spikes_host.is_null() { ffi::gpu_host_free(self.telemetry_spikes_host); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axon_allocation_with_ghosts() {
        // Test that total_axons = padded_n(base_axons) + required_ghost_slots
        let base_axons = 1024usize;
        let required_ghosts = 5usize;
        
        let base_padded = padded_n(base_axons);
        let total_expected = base_padded + required_ghosts;
        
        // With required_ghost_slots = 5, we should allocate base_padded + 5 slots
        assert_eq!(total_expected, base_padded + required_ghosts);
        
        // Verify padded_n rounds up to multiple of 64
        assert_eq!(base_padded % 64, 0);
    }

    #[test]
    fn test_small_required_ghosts() {
        let required_ghosts = 0usize;
        let base_axons = 100usize;
        let base_padded = padded_n(base_axons);
        
        // With 0 required ghosts, still allocate the padded base
        let total = base_padded + required_ghosts;
        assert!(total > 0);
    }

    #[test]
    fn test_large_required_ghosts() {
        let required_ghosts = 10000usize;
        let base_axons = 5598usize;
        let base_padded = padded_n(base_axons);
        
        let total = base_padded + required_ghosts;
        assert_eq!(total, base_padded + required_ghosts);
    }
}
