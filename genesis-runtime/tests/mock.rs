use genesis_core::constants::MAX_DENDRITE_SLOTS;
use genesis_core::layout::padded_n;

pub struct MockBakerBuilder {
    pub num_neurons: usize,
    pub num_axons: usize,
    
    pub pn: usize,
    pub pa: usize,

    pub voltages: Vec<i32>,
    pub flags: Vec<u8>,
    pub threshold_offsets: Vec<i32>,
    pub refractory_timers: Vec<u8>,
    pub soma_to_axons: Vec<u32>,
    
    pub dendrite_targets: Vec<u32>,
    pub dendrite_weights: Vec<i16>,
    pub dendrite_timers: Vec<u8>,
    
    pub axon_heads: Vec<u32>,
}

impl MockBakerBuilder {
    pub fn new(num_neurons: usize, num_axons: usize) -> Self {
        let pn = padded_n(num_neurons);
        let pa = padded_n(num_axons);
        let dc = pn * MAX_DENDRITE_SLOTS;

        Self {
            num_neurons,
            num_axons,
            pn,
            pa,
            voltages: vec![0; pn],
            flags: vec![0; pn],
            threshold_offsets: vec![0; pn],
            refractory_timers: vec![0; pn],
            soma_to_axons: vec![0xFFFFFFFF; pn],
            dendrite_targets: vec![0; dc],
            dendrite_weights: vec![0; dc],
            dendrite_timers: vec![0; dc],
            axon_heads: vec![0x80000000; pa], // AXON_SENTINEL
        }
    }

    /// Helper to set dendrite target for a specific neuron and slot
    pub fn set_dendrite(&mut self, nid: usize, slot: usize, axon_id: u32, segment: u8, weight: i16) {
        let idx = slot * self.pn + nid;
        self.dendrite_targets[idx] = (axon_id << 8) | (segment as u32);
        self.dendrite_weights[idx] = weight;
    }

    /// Serializes arrays into the byte blobs expected by VramState::load_shard.
    /// Returns (state_bytes, axons_bytes)
    pub fn build(self) -> (Vec<u8>, Vec<u8>) {
        let mut state_bytes = Vec::new();

        // Helper macro to append slices as bytes
        macro_rules! append_bytes {
            ($vec:expr) => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        $vec.as_ptr() as *const u8,
                        $vec.len() * std::mem::size_of_val(&$vec[0])
                    )
                };
                state_bytes.extend_from_slice(bytes);
            };
        }

        // Layout matching memory.rs allocate_and_copy sequence:
        // voltage, threshold_offset, refractory_timer, flags, soma_to_axon, targets, weights, refractory(timers), axon_heads
        append_bytes!(self.voltages);
        append_bytes!(self.threshold_offsets);
        append_bytes!(self.refractory_timers);
        append_bytes!(self.flags);
        append_bytes!(self.soma_to_axons);
        append_bytes!(self.dendrite_targets);
        append_bytes!(self.dendrite_weights);
        append_bytes!(self.dendrite_timers);
        // Axon heads at the end
        append_bytes!(self.axon_heads);

        // axons_bytes must have len == num_axons * 10
        let axons_bytes = vec![0u8; self.num_axons * 10];

        (state_bytes, axons_bytes)
    }
}
