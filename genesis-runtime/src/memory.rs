use crate::ffi;
use std::ffi::c_void;
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use genesis_core::layout::padded_n;
use genesis_baker::bake::neuron_placement::PlacedNeuron;
use genesis_baker::bake::axon_growth::GrownAxon;

/// Десериализует `shard.positions` (4 байта * N: packed_pos u32) →
/// `Vec<PlacedNeuron>`.
/// Формат: [packed_pos: u32; N] без заголовка.
pub fn deserialize_positions(bytes: &[u8]) -> anyhow::Result<Vec<PlacedNeuron>> {
    if bytes.len() % 4 != 0 {
        anyhow::bail!("shard.positions size {} is not divisible by 4", bytes.len());
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let packed = u32::from_le_bytes(bytes[i*4..i*4+4].try_into().unwrap());
        // PlacedNeuron stores packed_pos directly; type_idx comes from bits 28..31
        let type_idx = ((packed >> 28) & 0xF) as usize;
        out.push(PlacedNeuron { position: packed, type_idx, layer_name: String::new() });
    }
    Ok(out)
}

/// Десериализует `shard.axons` → `Vec<GrownAxon>`.
///
/// Формат (зеркало serialize_axons из baker/src/main.rs):
/// ```
/// [count: u32]
/// per-axon {
///   tip_x:   u16
///   tip_y:   u16
///   tip_z:   u16
///   length:  u32  (число сегментов)
///   segments: [u32; length]  (PackedPositions)
/// }
/// ```
/// soma_idx и type_idx извлекаются где возможно (type из первого сегмента bits 28..31).
pub fn deserialize_axons(bytes: &[u8]) -> anyhow::Result<Vec<GrownAxon>> {
    if bytes.len() < 4 {
        anyhow::bail!("shard.axons too small: {} bytes", bytes.len());
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut axons = Vec::with_capacity(count);
    let mut offset = 4usize;

    for soma_idx in 0..count {
        if offset + 10 > bytes.len() {
            anyhow::bail!("shard.axons truncated at axon {}", soma_idx);
        }
        let tip_x = u16::from_le_bytes(bytes[offset..offset+2].try_into().unwrap()) as u32;
        let tip_y = u16::from_le_bytes(bytes[offset+2..offset+4].try_into().unwrap()) as u32;
        let tip_z = u16::from_le_bytes(bytes[offset+4..offset+6].try_into().unwrap()) as u32;
        let seg_count = u32::from_le_bytes(bytes[offset+6..offset+10].try_into().unwrap()) as usize;
        offset += 10;

        if offset + seg_count * 4 > bytes.len() {
            anyhow::bail!("shard.axons: axon {} claims {} segs but bytes run out", soma_idx, seg_count);
        }
        let mut segments = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            let seg = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
            segments.push(seg);
            offset += 4;
        }

        // Extract type from the type bits of the first segment (bits 28..31)
        let type_idx = segments.first().map(|&s| ((s >> 28) & 0xF) as usize).unwrap_or(0);

        axons.push(GrownAxon {
            soma_idx,
            type_idx,
            tip_x,
            tip_y,
            tip_z,
            length_segments: seg_count as u32,
            segments,
        });
    }
    Ok(axons)
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

    // Virtual Axons (InjectInputs)
    pub num_virtual: u32,
    pub virtual_offset: u32,
    pub input_bitmask_buffer: *mut c_void,

    // Outbound Spikes (Per-Tick, MAX_SPIKES_PER_TICK length)
    pub outbound_spikes_buffer: *mut c_void,
    pub outbound_spikes_count: *mut c_void,
}

impl VramState {
    /// Loads the raw binary `.state` and `.axons` blobs from baker and 
    /// zero-copy migrates them into GPU VRAM (SoA layout).
    pub fn load_shard(state_bytes: &[u8], axons_bytes: &[u8], num_virtual_axons: u32) -> anyhow::Result<Self> {
        if axons_bytes.len() < 4 {
            anyhow::bail!("axons_bytes too small");
        }
        let num_axons = u32::from_le_bytes(axons_bytes[0..4].try_into().unwrap()) as usize;
        let virtual_offset = num_axons.saturating_sub(num_virtual_axons as usize) as u32;
        let pa = padded_n(num_axons);
        
        // Equation from byte_size(): pn * 4 + pn + pn * 4 + pn + pn * 4 + (pn * 128) * (4 + 2 + 1) + pa * 4 
        // = pn * 14 + pn * 896 + pa * 4 = pn * 910 + pa * 4
        let base_len = state_bytes.len().checked_sub(pa * 4)
            .ok_or_else(|| anyhow::anyhow!("State file too small for axons"))?;
            
        if base_len % 910 != 0 {
            anyhow::bail!("State file size mismatch: {} % 910 != 0", base_len);
        }
        let pn = base_len / 910;
        let dc = MAX_DENDRITE_SLOTS * pn;

        let mut offset = 0;
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
        
        // Axon Heads: Base + Pre-allocate 10000 Ghost Axons
        let max_ghost_axons = 10000;
        let total_axons = pa + max_ghost_axons;
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
        offset += pa * 4;

        // Init spare Ghost Axons to AXON_SENTINEL
        let sentinels = vec![0x80000000u32; max_ghost_axons];
        unsafe {
            ffi::gpu_memcpy_host_to_device(
                (axon_head_index as *mut u32).add(pa) as *mut c_void,
                sentinels.as_ptr() as *const c_void,
                max_ghost_axons * 4,
            );
        }

        // Output buffer for spikes. Max 1024 spikes per tick.
        let outbound_spikes_buffer = unsafe { ffi::gpu_malloc(1024 * 4) };
        let outbound_spikes_count = unsafe { ffi::gpu_malloc(4) };

        // Initialize count to 0
        let zero: u32 = 0;
        unsafe { ffi::gpu_memcpy_host_to_device(outbound_spikes_count, &zero as *const _ as *const c_void, 4) };

        // Virtual Axons placeholder allocation (default up to 8192 u32s = 32KB = 262k bits)
        let input_bitmask_buffer = unsafe { ffi::gpu_malloc(32768) };
        let zero_vec = vec![0u8; 32768];
        unsafe { ffi::gpu_memcpy_host_to_device(input_bitmask_buffer, zero_vec.as_ptr() as *const c_void, 32768) };

        Ok(VramState {
            padded_n: pn,
            total_axons,
            ghost_axons_allocated: 0,
            max_ghost_axons,
            base_axons: pa,
            num_virtual: num_virtual_axons,
            virtual_offset,
            voltage,
            threshold_offset,
            refractory_timer,
            flags,
            soma_to_axon,
            axon_head_index,
            dendrite_targets,
            dendrite_weights,
            dendrite_refractory,
            outbound_spikes_buffer,
            outbound_spikes_count,
            input_bitmask_buffer,
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

    pub fn download_outbound_spikes_count(&self) -> anyhow::Result<u32> {
        let mut count: u32 = 0;
        let success = unsafe {
            ffi::gpu_memcpy_device_to_host(
                &mut count as *mut _ as *mut c_void,
                self.outbound_spikes_count,
                4,
            )
        };
        if !success { anyhow::bail!("Failed to download outbound spikes count") }
        Ok(count)
    }

    pub fn download_outbound_spikes_buffer(&self, count: usize) -> anyhow::Result<Vec<u32>> {
        self.download_generic(self.outbound_spikes_buffer, count)
    }

    pub fn upload_input_bitmask(&self, host_mask: &[u32]) -> anyhow::Result<()> {
        let size = std::mem::size_of_val(host_mask);
        let success = unsafe {
            ffi::gpu_memcpy_host_to_device(
                self.input_bitmask_buffer,
                host_mask.as_ptr() as *const std::ffi::c_void,
                size,
            )
        };
        if !success { anyhow::bail!("Failed to upload input bitmask") }
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
}

impl Drop for VramState {
    fn drop(&mut self) {
        // Free GPU memory implicitly when the VramState goes out of scope
        unsafe {
            ffi::gpu_free(self.voltage);
            ffi::gpu_free(self.threshold_offset);
            ffi::gpu_free(self.refractory_timer);
            ffi::gpu_free(self.flags);

            ffi::gpu_free(self.axon_head_index);
            ffi::gpu_free(self.soma_to_axon);

            ffi::gpu_free(self.dendrite_targets);
            ffi::gpu_free(self.dendrite_weights);
            ffi::gpu_free(self.dendrite_refractory);

            ffi::gpu_free(self.outbound_spikes_buffer);
            ffi::gpu_free(self.outbound_spikes_count);
            ffi::gpu_free(self.input_bitmask_buffer);
        }
    }
}
