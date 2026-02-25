use crate::zone_runtime::ZoneRuntime;

pub trait Channel {
    /// Fast Path: Called every `sync_batch_ticks` to push dense spike arrays
    /// to their corresponding Ghost Axon slots in receiving zones.
    fn sync_spikes(&mut self, zones: &mut [ZoneRuntime]);

    /// Slow Path: Handover of structurally modified Ghost Axons (Night Phase).
    fn sync_geometry(&mut self, zones: &mut [ZoneRuntime]);
}
