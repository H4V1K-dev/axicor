pub mod network;
pub mod memory;
pub mod config;
pub mod output;
pub mod input;
pub mod ipc;
pub mod ffi;
pub mod tui;

#[cfg(feature = "mock-gpu")]
pub mod mock_ffi;
pub mod orchestrator;
pub mod zone_runtime;
pub mod sentinel;

use memory::VramState;
use std::ptr;
use std::ffi::c_void;
use tokio::sync::{mpsc, oneshot};
use crate::network::slow_path::{GeometryRequest, GeometryResponse};

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct VariantParameters {
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak: i32,
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: i32,
    pub gsop_potentiation: u16,
    pub gsop_depression: u16,
    pub refractory_period: u8,
    pub synapse_refractory: u8,
    pub slot_decay_ltm: u8,
    pub slot_decay_wm: u8,
    pub propagation_length: u8,
    pub ltm_slot_count: u8,
    pub inertia_curve: [u8; 16],
    pub _padding: [u8; 14],
}

impl Default for VariantParameters {
    fn default() -> Self {
        Self {
            threshold: 0,
            rest_potential: 0,
            leak: 0,
            homeostasis_penalty: 0,
            homeostasis_decay: 0,
            gsop_potentiation: 0,
            gsop_depression: 0,
            refractory_period: 0,
            synapse_refractory: 0,
            slot_decay_ltm: 0,
            slot_decay_wm: 0,
            propagation_length: 0,
            ltm_slot_count: 80,
            inertia_curve: [0; 16],
            _padding: [0; 14],
        }
    }
}

#[repr(C, align(128))]
#[derive(Clone, Copy)]
pub struct GenesisConstantMemory {
    pub variants: [VariantParameters; 16],
    // Total size of GenesisConstantMemory: 64 * 16 = 1024 bytes (well within 64KB constant memory limit)
}

impl Default for GenesisConstantMemory {
    fn default() -> Self {
        Self {
            variants: [VariantParameters::default(); 16],
        }
    }
}

pub struct Runtime {
    pub vram: VramState,
    pub v_seg: u32,
    pub master_seed: u64,
    /// Path to the shard data directory (for Night Phase IPC with baker subprocess)
    pub shard_data_path: Option<std::path::PathBuf>,
    /// IPC client for baker daemon (None if baker not configured)
    pub baker_client: Option<crate::ipc::BakerClient>,
    pub geometry_receiver: Option<mpsc::Receiver<(GeometryRequest, oneshot::Sender<GeometryResponse>)>>,
    pub sentinel: crate::sentinel::SentinelManager,
}

impl Runtime {
    pub fn new(
        vram: VramState,
        v_seg: u32,
        master_seed: u64,
        shard_data_path: Option<std::path::PathBuf>,
    ) -> Self {
        Self { 
            vram, 
            v_seg, 
            master_seed, 
            shard_data_path, 
            baker_client: None, 
            geometry_receiver: None,
            sentinel: crate::sentinel::SentinelManager::new() 
        }
    }

    pub fn init_constants(constants: &GenesisConstantMemory) -> bool {
        unsafe { ffi::gpu_load_constants(constants as *const _ as *const c_void); true }
    }



    pub fn synchronize(&self) {
        unsafe { ffi::gpu_device_synchronize() };
    }
}


#[cfg(test)]
mod test_day_phase;
