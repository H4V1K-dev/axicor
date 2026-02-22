use crate::ffi;
use std::ffi::c_void;
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use genesis_core::layout::padded_n;

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
    pub axon_head_index: *mut c_void,
    pub soma_to_axon: *mut c_void,

    // Dendrite Columns (MAX_DENDRITE_SLOTS * padded_n length)
    pub dendrite_targets: *mut c_void,
    pub dendrite_weights: *mut c_void,
    pub dendrite_refractory: *mut c_void,

    // Outbound Spikes (Per-Tick, MAX_SPIKES_PER_TICK length)
    pub outbound_spikes_buffer: *mut c_void,
    pub outbound_spikes_count: *mut c_void,
}

impl VramState {
    /// Loads the raw binary `.state` and `.axons` blobs from baker and 
    /// zero-copy migrates them into GPU VRAM (SoA layout).
    pub fn load_shard(state_bytes: &[u8], axons_bytes: &[u8]) -> anyhow::Result<Self> {
        let num_axons = axons_bytes.len() / 10;
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
        let axon_head_index = allocate_and_copy(pa * 4)?;

        // Output buffer for spikes. Max 1024 spikes per tick.
        let outbound_spikes_buffer = unsafe { ffi::gpu_malloc(1024 * 4) };
        let outbound_spikes_count = unsafe { ffi::gpu_malloc(4) };

        // Initialize count to 0
        let zero: u32 = 0;
        unsafe { ffi::gpu_memcpy_host_to_device(outbound_spikes_count, &zero as *const _ as *const c_void, 4) };

        Ok(VramState {
            padded_n: pn,
            total_axons: pa,
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
        }
    }
}
