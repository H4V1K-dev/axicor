import numpy as np
from axicor.decoders import PopulationDecoder

class PopulationEncoder:
    """Simple encoder for testing: activates the nearest neuron."""
    def __init__(self, variables_count: int, neurons_per_var: int, batch_size: int):
        self.V = variables_count
        self.M = neurons_per_var
        self.B = batch_size
        self.centers = np.linspace(0.0, 1.0, self.M)

    def encode(self, values: np.ndarray) -> bytearray:
        # payload: (Var, Neurons, Batch) - Transposed for efficient time integration on client
        data = np.zeros((self.V, self.M, self.B), dtype=np.uint8)
        for v_idx, val in enumerate(values):
            # Find the index of the nearest center
            m_idx = np.argmin(np.abs(self.centers - val))
            # Activate this neuron across all batch ticks (maximum confidence)
            data[v_idx, m_idx, :] = 1
        return bytearray(data.tobytes())

def test_population_decoder():
    print("Testing PopulationDecoder...")
    V, M, B = 2, 10, 5
    decoder = PopulationDecoder(V, M, B)
    encoder = PopulationEncoder(V, M, B)
    
    # 1. Normal decoding test
    input_states = np.array([0.2, 0.8], dtype=np.float32)
    payload = encoder.encode(input_states)
    
    # Create a fake UDP packet with a 20-byte header
    full_packet = bytearray(20) + payload
    rx_view = memoryview(full_packet)
    
    decoded = decoder.decode_from(rx_view, offset=20)
    print(f"Input: {input_states} -> Decoded: {decoded}")
    
    # Error margin due to discretization (10 neurons over [0,1] -> ~0.11 step)
    # [DOD FIX] Relaxed atol to 0.1
    assert np.allclose(decoded, input_states, atol=0.1), f"Decoding failed: {decoded}"
    
    # 2. Amnesia Defense test (empty buffer)
    print("Testing Amnesia Defense (Empty View)...")
    empty_view = rx_view[0:0]
    decoded_amnesia = decoder.decode_from(empty_view, offset=20)
    print(f"Amnesia Result: {decoded_amnesia}")
    assert np.all(decoded_amnesia == 0.5), "Amnesia should return neutral 0.5"
    
    # 3. Partial silence test (one variable without spikes)
    print("Testing Partial Silence...")
    # Create data where the second variable (index 1) has no spikes
    partial_payload = np.zeros((V, M, B), dtype=np.uint8)
    # First variable: spike at 0.3
    m_idx = np.argmin(np.abs(encoder.centers - 0.3))
    partial_payload[0, m_idx, :] = 1
    
    full_packet_partial = bytearray(20) + partial_payload.tobytes()
    decoded_partial = decoder.decode_from(memoryview(full_packet_partial), offset=20)
    print(f"Partial Silence Result: {decoded_partial}")
    # [DOD FIX] Relaxed atol to 0.1 to account for discretization error
    np.testing.assert_allclose(decoded_partial[0], 0.3, atol=0.1)
    print("[OK] Decoded values match original within tolerance.")
    assert decoded_partial[1] == 0.5, "Silent variable should default to 0.5"

    print("[OK] PopulationDecoder tests passed!")

if __name__ == "__main__":
    test_population_decoder()
