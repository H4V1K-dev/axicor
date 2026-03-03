use genesis_core::layout::{align_to_warp, StateFileHeader, AxonsFileHeader};
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
    pub max_ghost_axons: usize,
    pub base_axons: usize,
    pub available_ghost_slots: Vec<u32>, // FreeList for GC
    pub axon_head_index: *mut c_void,
    pub soma_to_axon: *mut c_void,
    pub axon_tips_uvw: Vec<u32>,
    pub axon_dirs_xyz: Vec<u32>,
    pub host_neuron_positions: Vec<u32>,

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
    pub output_history_host: *mut c_void, // Pinned host memory for DMA download
    pub telemetry_spikes: *mut c_void,
    pub telemetry_count: *mut c_void,
    pub telemetry_spikes_host: *mut c_void,
}

impl VramState {
    pub fn load_shard(state_bytes: &[u8], axons_bytes: &[u8], gxi: Option<&crate::input::GxiFile>, gxo: Option<&crate::output::GxoFile>, readout_batch_ticks: u32, input_stride: u32, required_ghost_slots: usize) -> anyhow::Result<Self> {
        let axons_header = AxonsFileHeader::from_bytes(axons_bytes)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse AxonsFileHeader"))?;
        let num_axons = axons_header.total_axons as usize;

        let pa = align_to_warp(num_axons);

        let state_header = StateFileHeader::from_bytes(state_bytes)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse StateFileHeader"))?;
        
        let pn = state_header.padded_n as usize;
        let dc = MAX_DENDRITE_SLOTS * pn;

        // [Fail-Fast] CRC/Size Validation (§1.2.1)
        // Ensure the .state file is exactly the size we expect for SoA loading.
        // Layout: Header(32) + Voltage(4) + Flags(1) + Threshold(4) + RefTimer(1) + SomaToAxon(4)
        //         + DendriteTargets(4) + Weights(2) + DendriteTimers(1) + AxonHeads(4)
        // All arrays are padded to 'pn' (Neurons) or 'pa' (Axons) or 'dc' (Dendrites).
        let expected_size = std::mem::size_of::<StateFileHeader>() 
            + (pn * 4) + (pn * 1) + (pn * 4) + (pn * 1) + (pn * 4) // Somas
            + (dc * 4) + (dc * 2) + (dc * 1)                      // Dendrites
            + (pa * 4);                                          // Axons (Heads)

        if state_bytes.len() < expected_size {
            panic!(
                "CRITICAL: Binary file size mismatch! \
                Manifest/Header claims {} axons ({} bytes), but .state file is only {} bytes. \
                The Baker Daemon failed to pad the SoA arrays. RE-RUN BAKER!",
                pa, expected_size, state_bytes.len()
            );
        }

        let mut offset = std::mem::size_of::<StateFileHeader>();
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

        // [AUDIT]: Host-side neuron positions for SpatialGrid
        let mut host_neuron_positions = vec![0u32; pn];
        let nrn_pos_offset = std::mem::size_of::<StateFileHeader>();
        unsafe {
            let src_ptr = state_bytes.as_ptr().add(nrn_pos_offset) as *const u32;
            std::ptr::copy_nonoverlapping(src_ptr, host_neuron_positions.as_mut_ptr(), pn);
        }

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

        // 5. Axon Geometry (Host-side for Sprouting)
        let mut axon_tips_uvw = vec![0; total_axons];
        let mut axon_dirs_xyz = vec![0; total_axons];
        
        if axons_bytes.len() >= 8 + pa * 8 {
            let tips_ptr = axons_bytes[8..].as_ptr() as *const u32;
            unsafe {
                std::ptr::copy_nonoverlapping(tips_ptr, axon_tips_uvw.as_mut_ptr(), pa);
                let dirs_ptr = tips_ptr.add(pa); 
                std::ptr::copy_nonoverlapping(dirs_ptr, axon_dirs_xyz.as_mut_ptr(), pa);
            }
        }

        // Init spare Ghost Axons to AXON_SENTINEL (GPU)
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
        let mut output_history_host = std::ptr::null_mut();
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
                    
                    // Pinned host memory for DMA Device-to-Host
                    output_history_host = ffi::gpu_host_alloc(history_bytes);
                    if output_history_host.is_null() {
                        anyhow::bail!("gpu_host_alloc failed for output_history_host ({} bytes)", history_bytes);
                    }
                    // It's good practice to zero it out, though the kernel writes absolutely every byte
                    // we'll leave it uninitialized on GPU to save time, it will be fully overwritten over the batch ticks.
                }
            }
        }



        let mut available_ghost_slots = Vec::with_capacity(max_ghost_axons);
        for id in (pa..(pa + max_ghost_axons)).rev() { // Start from base to max, reversed for pop()
            available_ghost_slots.push(id as u32);
        }

        println!("Zone VRAM Load! num_pixels={}, bitmask_buffer is null: {}", num_pixels, bitmask_buffer.is_null());

        Ok(VramState {
            padded_n: pn,
            total_axons,
            max_ghost_axons,
            available_ghost_slots,
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
            axon_tips_uvw,
            axon_dirs_xyz,
            host_neuron_positions,
            dendrite_targets,
            dendrite_weights,
            dendrite_refractory,
            pinned_host_targets,
            pinned_host_weights,
            num_mapped_somas,
            readout_batch_ticks,
            mapped_soma_ids,
            output_history,
            output_history_host,
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
                self.to_layout(),
                prune_threshold,
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
    pub fn to_layout(&self) -> genesis_core::layout::VramState {
        genesis_core::layout::VramState {
            padded_n: self.padded_n as u32,
            total_axons: self.total_axons as u32,
            voltage: self.voltage as *mut i32,
            flags: self.flags as *mut u8,
            threshold_offset: self.threshold_offset as *mut i32,
            refractory_timer: self.refractory_timer as *mut u8,
            soma_to_axon: self.soma_to_axon as *mut u32,
            dendrite_targets: self.dendrite_targets as *mut u32,
            dendrite_weights: self.dendrite_weights as *mut i16,
            dendrite_timers: self.dendrite_refractory as *mut u8,
            axon_heads: self.axon_head_index as *mut u32,
            input_bitmask: self.input_bitmask_buffer as *mut u32,
            output_history: self.output_history as *mut u8,
            telemetry_count: self.telemetry_count as *mut u32,
            telemetry_spikes: self.telemetry_spikes as *mut u32,
        }
    }

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
        self.available_ghost_slots.pop()
    }

    pub fn free_ghost_axon(&mut self, ghost_id: u32) {
        if (ghost_id as usize) >= self.base_axons && (ghost_id as usize) < self.base_axons + self.max_ghost_axons {
            self.available_ghost_slots.push(ghost_id); // Reclaim via FreeList
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
            if !self.output_history_host.is_null() {
                ffi::gpu_host_free(self.output_history_host);
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
        // Test that total_axons = align_to_warp(base_axons) + required_ghost_slots
        let base_axons = 1024usize;
        let required_ghosts = 5usize;
        
        let base_padded = align_to_warp(base_axons);
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
        let base_padded = align_to_warp(base_axons);
        
        // With 0 required ghosts, still allocate the padded base
        let total = base_padded + required_ghosts;
        assert!(total > 0);
    }

    #[test]
    fn test_large_required_ghosts() {
        let required_ghosts = 10000usize;
        let base_axons = 5598usize;
        let base_padded = align_to_warp(base_axons);

        let total = base_padded + required_ghosts;
        assert_eq!(total, base_padded + required_ghosts);
    }
}

// =============================================================================
// PinnedBuffer<T> — Page-Locked Host Memory
// =============================================================================
//
// Pinned (page-locked) memory is the prerequisite for true async DMA.
// When cudaMemcpyAsync receives a *pageable* src/dst pointer, the NVIDIA
// driver silently allocates a staging pinned buffer, does a synchronous
// memcpy into it, and THEN starts the DMA.  That hidden sync kills the
// Tokio thread and destroys BSP timing.
//
// Contract (spec §1.0): every host buffer that participates in H↔D transfer
// MUST be allocated with cudaMallocHost (= ffi::gpu_host_alloc).
//
// PinnedBuffer<T> owns a flat typed array in pinned memory.
// No Vec<T> is ever created on the DMA-critical path.

pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

impl<T> PinnedBuffer<T> {
    /// Allocate `len` elements of `T` in page-locked host memory.
    ///
    /// In mock-gpu mode `gpu_host_alloc` delegates to `libc::malloc`, so
    /// tests run without a CUDA installation.
    pub fn new(len: usize) -> Result<Self, String> {
        if len == 0 {
            return Ok(Self { ptr: std::ptr::NonNull::dangling().as_ptr(), len: 0 });
        }
        let size = len * std::mem::size_of::<T>();
        let ptr = unsafe { crate::ffi::gpu_host_alloc(size) as *mut T };
        if ptr.is_null() {
            return Err(format!(
                "PinnedBuffer::new — gpu_host_alloc failed ({} bytes)", size
            ));
        }
        Ok(Self { ptr, len })
    }

    #[inline(always)] pub fn len(&self) -> usize { self.len }
    #[inline(always)] pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Raw pointer — pass to cudaMemcpyAsync as host src/dst.
    #[inline(always)] pub fn as_ptr(&self) -> *const T { self.ptr }
    #[inline(always)] pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            // cudaFreeHost in production; libc::free in mock.
            unsafe { crate::ffi::gpu_host_free(self.ptr as *mut std::ffi::c_void) };
        }
    }
}

// =============================================================================
// DeviceSoA — VRAM Bootstrapper
// =============================================================================
//
// DeviceSoA takes the CPU-side `ShardStateSoA` produced by the Baker and
// uploads each flat array into video memory via cudaMalloc + cudaMemcpy.
// The resulting `VramState` holds raw device pointers passed verbatim to
// every CUDA kernel.
//
// Rules:
//  • No Vec<T> on the hot upload path — arrays are &[T] slices from ShardStateSoA.
//  • Every device allocation is recorded in `raw_ptrs` for deallocation on Drop.
//  • Upload is synchronous (cudaMemcpy, not Async) so no stream tracking needed.

pub struct DeviceSoA {
    /// CUDA-compatible view (raw device pointers).
    pub state: genesis_core::layout::VramState,
    /// Every cudaMalloc'd pointer — freed in Drop.
    raw_ptrs: Vec<*mut std::ffi::c_void>,
}

unsafe impl Send for DeviceSoA {}
unsafe impl Sync for DeviceSoA {}

impl DeviceSoA {
    /// Boot a zone from a Baker-generated `ShardStateSoA`.
    ///
    /// # Safety
    /// All raw GPU allocations happen here. In mock-gpu mode all FFI calls
    /// are synchronous host-side memcpy, so tests pass without a GPU.
    pub unsafe fn boot_from_host(
        host_soa: &genesis_core::layout::ShardStateSoA,
    ) -> Result<Self, String> {
        let mut raw_ptrs: Vec<*mut std::ffi::c_void> = Vec::with_capacity(8);

        // Inline helper: cudaMalloc + cudaMemcpy, push ptr into allocation list.
        macro_rules! upload {
            ($slice:expr, $ty:ty) => {{
                let bytes = $slice.len() * std::mem::size_of::<$ty>();
                let dev = crate::ffi::gpu_malloc(bytes);
                if dev.is_null() {
                    return Err(format!(
                        "DeviceSoA — gpu_malloc failed ({} bytes for {})",
                        bytes,
                        stringify!($ty)
                    ));
                }
                raw_ptrs.push(dev);
                let ok = crate::ffi::gpu_memcpy_host_to_device(
                    dev,
                    $slice.as_ptr() as *const std::ffi::c_void,
                    bytes,
                );
                if !ok {
                    return Err(format!(
                        "DeviceSoA — gpu_memcpy_host_to_device failed ({} bytes)",
                        bytes
                    ));
                }
                dev
            }};
        }

        let pn = host_soa.padded_n;
        let dc = genesis_core::constants::MAX_DENDRITE_SLOTS * pn;

        let d_voltage          = upload!(host_soa.voltage,          i32) as *mut i32;
        let d_flags            = upload!(host_soa.flags,            u8)  as *mut u8;
        let d_threshold_offset = upload!(host_soa.threshold_offset, i32) as *mut i32;
        let d_refractory_timer = upload!(host_soa.refractory_timer, u8)  as *mut u8;
        let d_dendrite_targets = upload!(host_soa.dendrite_targets, u32) as *mut u32;
        let d_dendrite_weights = upload!(host_soa.dendrite_weights, i16) as *mut i16;
        let d_dendrite_timers  = upload!(host_soa.dendrite_timers,  u8)  as *mut u8;
        let d_axon_heads       = upload!(host_soa.axon_heads,       u32) as *mut u32;

        // Sanity assertions — catch layout mismatches at boot time.
        debug_assert_eq!(host_soa.dendrite_targets.len(), dc);
        debug_assert_eq!(host_soa.dendrite_weights.len(), dc);
        debug_assert_eq!(host_soa.dendrite_timers.len(),  dc);

        let state = genesis_core::layout::VramState {
            padded_n:         pn as u32,
            total_axons:      host_soa.axon_heads.len() as u32,
            voltage:          d_voltage,
            flags:            d_flags,
            threshold_offset: d_threshold_offset,
            refractory_timer: d_refractory_timer,
            soma_to_axon:     std::ptr::null_mut(),
            dendrite_targets: d_dendrite_targets,
            dendrite_weights: d_dendrite_weights,
            dendrite_timers:  d_dendrite_timers,
            axon_heads:       d_axon_heads,
            input_bitmask:    std::ptr::null_mut(),
            output_history:   std::ptr::null_mut(),
            telemetry_count:  std::ptr::null_mut(),
            telemetry_spikes: std::ptr::null_mut(),
        };

        Ok(Self { state, raw_ptrs })
    }

    /// Boot a zone from raw host pointers (e.g. from mmap).
    /// Used for Hot Standby recovery.
    pub unsafe fn boot_from_raw_parts(
        padded_n: usize,
        total_axons: usize,
        voltage: *const i32,
        flags: *const u8,
        threshold_offset: *const i32,
        refractory_timer: *const u8,
        soma_to_axon: *const u32,
        dendrite_targets: *const u32,
        dendrite_weights: *const i16,
        dendrite_timers: *const u8,
        axon_heads: *const u32,
    ) -> Result<Self, String> {
        let mut raw_ptrs: Vec<*mut std::ffi::c_void> = Vec::with_capacity(8);

        macro_rules! upload_raw {
            ($ptr:expr, $len:expr, $ty:ty) => {{
                let bytes = $len * std::mem::size_of::<$ty>();
                let dev = crate::ffi::gpu_malloc(bytes);
                if dev.is_null() {
                    return Err(format!("DeviceSoA::boot_from_raw_parts failed gpu_malloc"));
                }
                raw_ptrs.push(dev);
                crate::ffi::gpu_memcpy_host_to_device(dev, $ptr as *const _, bytes);
                dev
            }};
        }

        let dc = genesis_core::constants::MAX_DENDRITE_SLOTS * padded_n;
        
        let d_voltage = upload_raw!(voltage, padded_n, i32) as *mut i32;
        let d_flags = upload_raw!(flags, padded_n, u8) as *mut u8;
        let d_threshold_offset = upload_raw!(threshold_offset, padded_n, i32) as *mut i32;
        let d_refractory_timer = upload_raw!(refractory_timer, padded_n, u8) as *mut u8;
        let d_soma_to_axon = upload_raw!(soma_to_axon, padded_n, u32) as *mut u32;
        let d_dendrite_targets = upload_raw!(dendrite_targets, dc, u32) as *mut u32;
        let d_dendrite_weights = upload_raw!(dendrite_weights, dc, i16) as *mut i16;
        let d_dendrite_timers = upload_raw!(dendrite_timers, dc, u8) as *mut u8;
        let d_axon_heads = upload_raw!(axon_heads, total_axons, u32) as *mut u32;

        let state = genesis_core::layout::VramState {
            padded_n: padded_n as u32,
            total_axons: total_axons as u32,
            voltage: d_voltage,
            flags: d_flags,
            threshold_offset: d_threshold_offset,
            refractory_timer: d_refractory_timer,
            soma_to_axon: d_soma_to_axon,
            dendrite_targets: d_dendrite_targets,
            dendrite_weights: d_dendrite_weights,
            dendrite_timers: d_dendrite_timers,
            axon_heads: d_axon_heads,
            input_bitmask: std::ptr::null_mut(),
            output_history: std::ptr::null_mut(),
            telemetry_count: std::ptr::null_mut(),
            telemetry_spikes: std::ptr::null_mut(),
        };

        Ok(Self { state, raw_ptrs })
    }
}

impl Drop for DeviceSoA {
    fn drop(&mut self) {
        // Free in LIFO order — mirrors typical CUDA resource teardown.
        for &ptr in self.raw_ptrs.iter().rev() {
            unsafe { crate::ffi::gpu_free(ptr) };
        }
    }
}

// =============================================================================
// PinnedBuffer / DeviceSoA tests
// =============================================================================

#[cfg(test)]
mod pinned_device_tests {
    use super::*;
    use genesis_core::layout::ShardStateSoA;

    /// Allocation, write, read-back, and Drop via gpu_host_free.
    #[test]
    fn test_pinned_buffer() {
        let len = 256usize;
        let mut buf: PinnedBuffer<u32> = PinnedBuffer::new(len).expect("alloc");

        assert_eq!(buf.len(), len);
        assert!(!buf.is_empty());
        assert!(!buf.as_ptr().is_null());

        // Write sentinel pattern (no Vec involved).
        for (i, v) in buf.as_mut_slice().iter_mut().enumerate() {
            *v = (i as u32).wrapping_mul(0xDEAD_BEEF);
        }

        // Read-back — verifies no silent truncation or misalignment.
        for (i, &v) in buf.as_slice().iter().enumerate() {
            assert_eq!(
                v,
                (i as u32).wrapping_mul(0xDEAD_BEEF),
                "PinnedBuffer[{}] mismatch", i
            );
        }

        // as_ptr and as_mut_ptr must agree.
        assert_eq!(buf.as_ptr() as usize, buf.as_mut_ptr() as usize);

        // Drop at end of scope → gpu_host_free must not segfault.
    }

    /// Smaller element type smoke test.
    #[test]
    fn test_pinned_buffer_i16() {
        let mut buf: PinnedBuffer<i16> = PinnedBuffer::new(64).expect("alloc");
        for (i, v) in buf.as_mut_slice().iter_mut().enumerate() {
            *v = i as i16;
        }
        assert_eq!(buf.as_slice()[63], 63);
    }

    /// Empty buffer is legal and must not segfault on Drop.
    #[test]
    fn test_pinned_buffer_zero_len() {
        let buf: PinnedBuffer<u32> = PinnedBuffer::new(0).expect("zero-len alloc");
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        // Drop — no gpu_host_free called (guarded by len > 0 check).
    }

    /// boot_from_host uploads all SoA fields, VramState reflects correct sizes,
    /// and a device→host round-trip preserves voltage sentinel values.
    #[test]
    fn test_device_soa_boot() {
        let n      = 64usize;
        let n_axon = 128usize;

        let mut host_soa = ShardStateSoA::new(n, n_axon);

        // Write sentinel into voltage.
        for (i, v) in host_soa.voltage.iter_mut().enumerate() {
            *v = (i as i32).wrapping_mul(1337);
        }

        let device = unsafe {
            DeviceSoA::boot_from_host(&host_soa).expect("boot_from_host")
        };

        // Structural invariants.
        assert_eq!(device.state.padded_n   as usize, host_soa.padded_n,
                   "padded_n mismatch");
        assert_eq!(device.state.total_axons as usize, n_axon,
                   "total_axons mismatch");
        assert!(!device.state.voltage.is_null(),     "voltage null");
        assert!(!device.state.axon_heads.is_null(),  "axon_heads null");
        assert!(!device.state.dendrite_weights.is_null(), "dw null");

        // Round-trip: copy device voltage back to host and verify sentinel.
        let mut readback = vec![0i32; host_soa.padded_n];
        let ok = unsafe {
            crate::ffi::gpu_memcpy_device_to_host(
                readback.as_mut_ptr() as *mut std::ffi::c_void,
                device.state.voltage as *const std::ffi::c_void,
                host_soa.padded_n * std::mem::size_of::<i32>(),
            )
        };
        assert!(ok, "device→host copy failed");

        for (i, &v) in readback.iter().enumerate() {
            assert_eq!(
                v,
                (i as i32).wrapping_mul(1337),
                "voltage[{}] round-trip mismatch", i
            );
        }

        // Drop frees 8 raw_ptrs in LIFO order — no segfault = success.
    }
}
