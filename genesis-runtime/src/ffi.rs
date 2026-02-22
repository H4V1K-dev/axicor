use std::ffi::c_void;

extern "C" {
    // Basic Memory Ops
    pub fn gpu_malloc(size: usize) -> *mut c_void;
    pub fn gpu_free(dev_ptr: *mut c_void);
    pub fn gpu_memcpy_host_to_device(
        dst_dev: *mut c_void,
        src_host: *const c_void,
        size: usize,
    ) -> bool;
    pub fn gpu_memcpy_device_to_host(
        dst_host: *mut c_void,
        src_dev: *const c_void,
        size: usize,
    ) -> bool;
    pub fn gpu_device_synchronize();

    pub fn upload_constant_memory(host_ptr: *const c_void) -> bool;
    pub fn launch_propagate_axons(
        total_axons: u32,
        axon_heads: *mut c_void,
        v_seg: u32,
        stream: *mut c_void,
    );
    pub fn launch_update_neurons(
        padded_n: u32,
        voltage: *mut c_void,
        threshold_offset: *mut c_void,
        refractory_timer: *mut c_void,
        flags: *mut c_void,
        soma_to_axon: *mut c_void,
        dendrite_targets: *mut c_void,
        dendrite_weights: *mut c_void,
        axon_heads: *mut c_void,
        stream: *mut c_void,
    );
    pub fn launch_apply_gsop(
        padded_n: u32,
        flags: *mut c_void,
        dendrite_targets: *mut c_void,
        dendrite_weights: *mut c_void,
        dendrite_timers: *mut c_void,
        axon_heads: *mut c_void,
        stream: *mut c_void,
    );
    pub fn launch_apply_spike_batch_impl(
        num_spikes: u32,
        schedule_indices: *mut c_void,
        axon_heads: *mut c_void,
        stream: *mut c_void,
    );
    pub fn launch_record_outputs(
        padded_n: u32,
        flags: *mut c_void,
        outbound_spikes_buffer: *mut c_void,
        outbound_spikes_count: *mut c_void,
        stream: *mut c_void,
    );
    pub fn launch_sort_and_prune(
        padded_n: u32,
        dendrite_targets: *mut c_void,
        dendrite_weights: *mut c_void,
        dendrite_timers: *mut c_void,
        prune_threshold: i16,
        stream: *mut c_void,
    );
}
