use crate::zone_runtime::ZoneRuntime;
use crate::network::channel::Channel;
use crate::ffi;
use std::ffi::c_void;

/// Represents a single projecting Ghost Axon connection between two zones.
#[derive(Clone, Debug)]
pub struct GhostLink {
    /// Index in the `Vec<ZoneRuntime>` of the zone that *originates* the axon.
    pub src_zone_idx: usize,
    /// The local Axon ID within the originating zone's VRAM buffer.
    pub src_axon_id: u32,
    
    /// Index in the `Vec<ZoneRuntime>` of the zone that *receives* the signal via a Ghost Axon slot.
    pub dst_zone_idx: usize,
    /// The Ghost Axon slot ID within the receiving zone's VRAM buffer (`axon_heads[]`).
    pub dst_ghost_id: u32,
}

/// The Zero-Copy IntraGPU Channel.
/// 
/// Since both the source and destination zones reside in the same device VRAM,
/// spikes are synchronized by simply copying the 4-byte `head` state directly
/// from the source `axon_heads` array into the destination `axon_heads` array.
/// 
/// GPU kernels NEVER execute out-of-bounds cross-zone memory accesses.
/// Instead, the CPU orchestrates these Memcpys between the isolated arrays.
pub struct IntraGpuChannel {
    pub links: Vec<GhostLink>,
}

impl IntraGpuChannel {
    pub fn new(links: Vec<GhostLink>) -> Self {
        Self { links }
    }
}

impl Channel for IntraGpuChannel {
    fn sync_spikes(&mut self, zones: &mut [ZoneRuntime]) {
        if self.links.is_empty() {
            return;
        }
        
        // MVP: Individual D2H -> H2D memcpys. 
        // TODO: In the future, this should be optimized into a single Batched D2D pointer copy
        // via a custom CUDA kernel or batched API if thousands of links exist.
        for link in &self.links {
            let src_zone = &zones[link.src_zone_idx];
            let dst_zone = &zones[link.dst_zone_idx];
            
            let mut head_val: u32 = 0;
            
            unsafe {
                // 1. Read source head
                ffi::gpu_memcpy_device_to_host(
                    &mut head_val as *mut _ as *mut c_void,
                    src_zone.runtime.vram.axon_head_index.add((link.src_axon_id as usize) * 4) as *const c_void,
                    4
                );

                // 2. Write to destination ghost slot
                ffi::gpu_memcpy_host_to_device(
                    dst_zone.runtime.vram.axon_head_index.add((link.dst_ghost_id as usize) * 4) as *mut c_void,
                    &head_val as *const _ as *const c_void,
                    4
                );
            }
        }
    }

    fn sync_geometry(&mut self, _zones: &mut [ZoneRuntime]) {
        // Night phase structural updates (sprouting/pruning).
        // For now, connections are static from brain.toml MVP.
    }
}
