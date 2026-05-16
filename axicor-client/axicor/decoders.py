import numpy as np

class PwmDecoder:
    """
    Temporal PWM Decoding (Rate Coding) for motor cortex.
    Converts the binary spike history (Output_History) of a batch
    into a dense f16 array of analog efforts (Duty Cycle / 0.0 - 1.0).
    """
    def __init__(self, num_outputs: int, batch_size: int):
        self.N = num_outputs
        self.B = batch_size
        
        # Payload size: B ticks * N motors (1 byte = 1 spike flag)
        self.payload_size = self.N * self.B
        self._inv_b = np.float32(1.0 / self.B)
        
        # Preallocation for HFT cycle (Zero-Garbage)
        self._sum_buffer = np.zeros(self.N, dtype=np.float32)
        self._out_buffer = np.zeros(self.N, dtype=np.float32)
        
        # Pre-calculated reshape view (N, B) -> Project standard: [Channel, Batch]
        self._raw_bytes = np.zeros(self.payload_size, dtype=np.uint8)
        self._spikes_view = self._raw_bytes.reshape((self.N, self.B)) # [DOD FIX] Server sends [Pixel][Tick]

    def decode_from(self, rx_view: memoryview, offset: int = 0) -> np.ndarray:
        """
        Extracts data from raw UDP buffer without memory copying.
        rx_view: socket memoryview (header ALREADY stripped in client.step)
        """
        # Amnesia Defense: If no data, return zero effort
        if len(rx_view) == 0:
            self._out_buffer.fill(0.0)
            return self._out_buffer

        # [DOD FIX] Zero-Slice Rule (Invariant 20): Use offset instead of slicing
        # This prevents accidental sub-array allocation during HFT loop
        view = np.ndarray(self.payload_size, dtype=np.uint8, buffer=rx_view, offset=offset)
        self._raw_bytes[:] = view
        
        # 3. Vectorized sum across ticks axis (axis=1). Written directly into preallocated buffer!
        np.sum(self._spikes_view, axis=1, dtype=np.float32, out=self._sum_buffer)
        
        # 4. Normalize to [0.0, 1.0] range (In-place)
        np.multiply(self._sum_buffer, self._inv_b, out=self._out_buffer)
        
        # Return reference to internal buffer. Data valid until next decode_from call.
        return self._out_buffer

class PopulationDecoder:
    """
    Population Decoder (Center of Mass) for extracting continuous float values
    from neuron receptive field activity.
    """
    def __init__(self, variables_count: int, neurons_per_var: int, batch_size: int):
        self.V = variables_count
        self.M = neurons_per_var
        self.N = self.V * self.M
        self.B = batch_size
        self.payload_size = self.N * self.B
        
        # Vector of receptive field centers [0.0 ... 1.0]
        self.centers = np.linspace(0.0, 1.0, self.M, dtype=np.float32)
        
        # Zero-Allocation Buffers
        self._sum_buffer = np.zeros((self.V, self.M), dtype=np.float32)
        self._mass_buffer = np.zeros(self.V, dtype=np.float32)
        self._out_buffer = np.zeros(self.V, dtype=np.float32)
        self._silence_mask = np.zeros(self.V, dtype=bool)

        # [DOD FIX] Data layout is [Var, Neuron, Batch]
        self._raw_bytes = np.zeros(self.payload_size, dtype=np.uint8)
        self._spikes_view = self._raw_bytes.reshape((self.V, self.M, self.B))

    def decode_from(self, rx_view: memoryview, offset: int = 0) -> np.ndarray:
        # Amnesia Defense: Return neutral state (0.5)
        if len(rx_view) == 0:
            self._out_buffer.fill(0.5)
            return self._out_buffer

        # [DOD FIX] Invariant 20: Zero-Slice Rule
        try:
            view = np.ndarray(self.payload_size, dtype=np.uint8, buffer=rx_view, offset=offset)
            self._raw_bytes[:] = view
        except Exception as e:
            print(f"[ERROR] [PopulationDecoder] Buffer mapping failed: {e}")
            self._out_buffer.fill(0.5)
            return self._out_buffer
        
        # 3. Sum spikes across ticks (Time Integration, axis=2)
        np.sum(self._spikes_view, axis=2, dtype=np.float32, out=self._sum_buffer) # [DOD FIX] Sum by Batch dimension
        
        # 4. Find total spike mass for each variable
        np.sum(self._sum_buffer, axis=1, out=self._mass_buffer)

        # 5. Weight activity by field centers (Strictly In-place)
        np.multiply(self._sum_buffer, self.centers, out=self._sum_buffer)

        # 6. Sum weighted values directly into _out_buffer
        np.sum(self._sum_buffer, axis=1, out=self._out_buffer)

        # 7. Center of Mass: Sum(spikes * centers) / Sum(spikes)
        # _out_buffer acts as both numerator and output
        np.divide(self._out_buffer, self._mass_buffer, out=self._out_buffer, where=self._mass_buffer != 0)

        # 8. Silence protection (Zero-Allocation mask)
        np.equal(self._mass_buffer, 0, out=self._silence_mask)
        np.copyto(self._out_buffer, 0.5, where=self._silence_mask)

        return self._out_buffer
