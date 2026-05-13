import math
import numpy as np

class PwmEncoder:
    """
    Temporal PWM Encoding (Rate Coding) for continuous analog signals.
    Spreads spikes across the batch via phase shifting, preventing Burst Gating.
    """
    def __init__(self, num_sensors: int, batch_size: int):
        self.N = num_sensors
        self.B = batch_size
        
        # GPU expects an array of u32 (32 virtual axons per word).
        # Each tick row must be a multiple of 4 bytes (32 bits).
        self.padded_N = math.ceil(self.N / 64) * 64
        self.bytes_per_tick = self.padded_N // 8
        self.total_bytes = self.bytes_per_tick * self.B
        
        # Temporal axis and phase shift (Golden Ratio Dither)
        t = np.linspace(0, 1, self.B, endpoint=False, dtype=np.float32)[:, None]
        phase = (np.arange(self.N, dtype=np.float32) * 0.618033) % 1.0
        self.pwm_wave = (t + phase) % 1.0
        
        # Preallocated buffer to avoid heap allocations in the Hot Loop
        self._bool_buffer = np.zeros((self.B, self.padded_N), dtype=np.bool_)
        # Preallocated packed buffer
        self._packed_buffer = np.zeros((self.B, self.bytes_per_tick), dtype=np.uint8)
        self._packed_view = self._packed_buffer.ravel()

        # [DOD FIX] Deep Zero-Malloc (Numpy C-Backend)
        self._powers = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
        self._bits_view = self._bool_buffer.view(np.uint8).reshape(self.B, self.bytes_per_tick, 8)
        self._mul_temp = np.zeros((self.B, self.bytes_per_tick, 8), dtype=np.uint8)

    def encode_into(self, sensors_f16: np.ndarray, tx_view: memoryview) -> int:
        """
        Broadcasting comparison and Zero-Copy write to the network buffer.
        Returns the number of bytes written.
        """
        # [DOD FIX] Strict Zero-Allocation comparison.
        # No temporary arrays! The result is written directly into _bool_buffer.
        np.less(self.pwm_wave, sensors_f16, out=self._bool_buffer[:, :self.N])

        # To be truly Zero-GC, we need a way to packbits without allocation.
        self._manual_packbits()

        # Copy flat byte array directly into the UDP socket buffer
        tx_view[:self.total_bytes] = self._packed_view
        return self.total_bytes

    def _manual_packbits(self):
        # [DOD FIX] Elimination of temporary arrays in C-backend
        np.multiply(self._bits_view, self._powers, out=self._mul_temp)
        np.sum(self._mul_temp, axis=2, out=self._packed_buffer)

class PopulationEncoder:
    """
    Spatial encoding (Gaussian Receptive Fields).
    Expands 1 float variable into a population of M neurons.
    """
    def __init__(self, variables_count: int, neurons_per_var: int, batch_size: int, sigma: float = 0.15):
        self.V = variables_count
        self.M = neurons_per_var
        self.N = self.V * self.M
        self.B = batch_size
        
        self.padded_N = math.ceil(self.N / 64) * 64
        self.bytes_per_tick = self.padded_N // 8
        self.total_bytes = self.bytes_per_tick * self.B
        
        # Precalculate activation radius for Gaussian. prob > 0.5 is equivalent to abs(dist) < R
        self.sigma = sigma
        self.radius = self.sigma * math.sqrt(-2.0 * math.log(0.5))
        
        # [DOD FIX] Preallocation of buffers for in-place calculations (Zero-Garbage)
        self.centers = np.linspace(0.0, 1.0, self.M, dtype=np.float32)
        self._expanded_buffer = np.zeros(self.N, dtype=np.float32)
        self._expanded_view = self._expanded_buffer.reshape(self.V, self.M)
        
        self._bool_buffer = np.zeros(self.padded_N, dtype=np.bool_)
        self._batch_bool_buffer = np.zeros((self.B, self.padded_N), dtype=np.bool_)
        self._packed_buffer = np.zeros((self.B, self.bytes_per_tick), dtype=np.uint8)
        self._packed_view = self._packed_buffer.ravel()
        
        self._powers = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

        # [DOD FIX] Deep Zero-Malloc (Numpy C-Backend)
        self._bits_view = self._batch_bool_buffer.view(np.uint8).reshape(self.B, self.bytes_per_tick, 8)
        self._mul_temp = np.zeros((self.B, self.bytes_per_tick, 8), dtype=np.uint8)

    def encode_into(self, states_f16: np.ndarray, tx_view: memoryview) -> int:
        """
        states_f16: array of normalized [0..1] values (size V)
        """
        # [DOD FIX] Zero-Allocation math pipeline
        self._expanded_view[:] = states_f16[:, None]

        # Vectorized subtraction of centers In-Place
        np.subtract(self._expanded_view, self.centers, out=self._expanded_view)
        # Use out=self._expanded_view to keep everything in the same buffer
        np.abs(self._expanded_view, out=self._expanded_view)

        # Threshold activation In-Place
        np.less(self._expanded_buffer, self.radius, out=self._bool_buffer[:self.N])

        # [DOD Task 1] Single-Tick Pulse
        self._batch_bool_buffer[0, :] = self._bool_buffer

        self._manual_packbits()

        tx_view[:self.total_bytes] = self._packed_view
        return self.total_bytes

    def _manual_packbits(self):
        # [DOD FIX] Elimination of temporary arrays in C-backend
        np.multiply(self._bits_view, self._powers, out=self._mul_temp)
        np.sum(self._mul_temp, axis=2, out=self._packed_buffer)
