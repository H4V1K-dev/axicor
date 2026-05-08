#[cfg(test)]
mod tests {
    use crate::memory::{compute_state_offsets, MAX_DENDRITES};

    #[test]
    fn test_mathematical_1166_byte_invariant() {
        // Soma fields: voltage(4) + flags(1) + threshold_offset(4) + timers(1) + soma_to_axon(4) = 14
        let soma_size = 4 + 1 + 4 + 1 + 4;
        assert_eq!(soma_size, 14, "Soma size invariant violated");

        // Dendrites per neuron: 128 * (target(4) + weight(4) + timer(1)) = 128 * 9 = 1152
        let dendrite_size = MAX_DENDRITES * (4 + 4 + 1);
        assert_eq!(dendrite_size, 1152, "Dendrite size invariant violated");

        assert_eq!(soma_size + dendrite_size, 1166, "Total 1166-byte invariant violated");
    }

    #[test]
    fn test_soa_64byte_alignment() {
        let offsets = compute_state_offsets(64);
        
        let checked_offsets = [
            offsets.soma_voltage,
            offsets.soma_flags,
            offsets.threshold_offset,
            offsets.timers,
            offsets.soma_to_axon,
            offsets.dendrite_targets,
            offsets.dendrite_weights,
            offsets.dendrite_timers,
        ];

        for &off in &checked_offsets {
            assert_eq!(off % 64, 0, "Alignment violation at offset {}", off);
        }
    }
}
