import socket
import struct
import ctypes
import numpy as np
from typing import List, Dict

# C-ABI Constants (Strictly from 08_io_matrix.md)
MAX_UDP_PAYLOAD = 65507
HEADER_SIZE = 20
HEADER_FMT = "<IIIIhH"  # 20 bytes: magic, zone_hash, matrix_hash, size, reward, pad
GSIO_MAGIC = 0x4F495347
GSOO_MAGIC = 0x4F4F5347  # Axicor Standard Output

class ExternalIoHeader(ctypes.Structure):
    """
    Strict C-ABI Header for Data Plane UDP packets.
    Must be exactly 20 bytes without CPython padding.
    """
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),         # 0x4F495347 ("GSIO") or 0x4F4F5347 ("GSOO")
        ("zone_hash", ctypes.c_uint32),     # FNV-1a hash of the zone
        ("matrix_hash", ctypes.c_uint32),   # FNV-1a hash of the I/O matrix
        ("payload_size", ctypes.c_uint32),  # Size of bitmask payload (excluding header)
        ("global_reward", ctypes.c_int16),  # R-STDP Dopamine Modulator
        ("_padding", ctypes.c_uint16),      # Alignment to 20 bytes
    ]

class AxicorMultiClient:
    def __init__(self, addr: tuple[str, int], matrices: List[Dict[str, int]], rx_layout: list[dict] = None, timeout: float = 2.0, bind_addr: tuple[str, int] = None):
        """
        :param addr: Node address (ip, port) (External UDP In)
        :param matrices: List of dictionaries [{'zone_hash': int, 'matrix_hash': int, 'payload_size': int}]
        :param rx_layout: List of expected response chunks [{'matrix_hash': int, 'size': int}]
        :param timeout: Response timeout (Biological Amnesia)
        :param bind_addr: Optional local (ip, port) to bind before first send. Ensures node replies to fixed port.
        """
        self.addr = addr
        self.num_chunks = len(matrices)
        self.rx_layout = rx_layout or []
        self.expected_chunks = len(self.rx_layout)
        
        # 1. Single Memory Arena for the entire TX payload
        total_tx_size = sum(HEADER_SIZE + m['payload_size'] for m in matrices)
        self._tx_arena = bytearray(total_tx_size)
        
        # 2. Response Arena (RX)  Zero-Copy Assembler
        # Total size pre-calculated from rx_layout
        total_rx_size = sum(m['size'] for m in self.rx_layout)
        self._rx_arena = bytearray(total_rx_size)
        self._rx_view = memoryview(self._rx_arena)
        
        # Mapping hash -> (offset, size) for instantaneous assembly
        self._rx_map = {}
        offset = 0
        for m in self.rx_layout:
            self._rx_map[m['matrix_hash']] = (offset, m['size'])
            offset += m['size']

        # Buffer for receiving a single UDP packet
        self._udp_buf = bytearray(MAX_UDP_PAYLOAD)
        self._udp_view = memoryview(self._udp_buf)
        # Pre-cast header for zero-allocation parsing
        self._header_np = np.frombuffer(self._udp_buf, dtype=np.uint32, count=4)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # [DOD FIX] Expand OS receive buffer to 8 MB to handle burst L7 chunks
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.sock.settimeout(timeout)
        # [DOD FIX] Bind before any sendto so node replies to our fixed port.
        if bind_addr is not None:
            self.sock.bind(bind_addr)

        self.payload_views = []
        self._tx_packets = []
        
        # 3. Slicing TX arena (Zero-Copy)
        offset = 0
        for m in matrices:
            size = m['payload_size']
            assert size <= MAX_UDP_PAYLOAD - HEADER_SIZE, "Chunk exceeds UDP MTU!"
            
            # Pack static header once
            struct.pack_into(
                HEADER_FMT, self._tx_arena, offset,
                GSIO_MAGIC, m['zone_hash'], m['matrix_hash'], size, 0, 0
            )
            
            # View of the entire packet (Header + Payload) for the socket
            packet_view = memoryview(self._tx_arena)[offset : offset + HEADER_SIZE + size]
            self._tx_packets.append(packet_view)
            
            # View of payload only, mapped to NumPy
            payload_view = packet_view[HEADER_SIZE:]
            np_view = np.ndarray((size,), dtype=np.uint8, buffer=payload_view)
            self.payload_views.append(np_view)
            
            offset += HEADER_SIZE + size

    def step(self, reward: int = 0, expected_rx_hash: int = None) -> memoryview:
        """
        Hot Loop. Zero-Copy L7 Assembler.
        """
        # 1. TX: Rapid-fire burst
        if self.num_chunks > 0:
            struct.pack_into("<h", self._tx_arena, 16, reward)

        for packet in self._tx_packets:
            self.sock.sendto(packet, self.addr)

        # 2. RX: Assembler
        if self.expected_chunks == 0 and expected_rx_hash is None:
            return self._rx_view[0:0]

        chunks_received = 0
        target_received = False
        
        # [DOD FIX] Limit search to 1 target if hash is provided, or all chunks in layout
        limit = 1 if expected_rx_hash is not None else self.expected_chunks

        try:
            while (expected_rx_hash is not None and not target_received) or (expected_rx_hash is None and chunks_received < limit):
                size, _ = self.sock.recvfrom_into(self._udp_buf, MAX_UDP_PAYLOAD)
                if size < HEADER_SIZE: continue

                # Parse L7 GSOO header using pre-cast NumPy view (Zero-Allocation)
                # [magic, z_hash, m_hash, pld_size]
                magic = self._header_np[0]
                m_hash = self._header_np[2]
                
                # Strict validation: client only accepts OUTPUTS (GSOO_MAGIC) from the node
                if magic != GSOO_MAGIC: continue

                # [DOD FIX] If explicit hash requested, ignore everything else
                if expected_rx_hash is not None:
                    if m_hash != expected_rx_hash:
                        continue
                    
                    # Target found. If it's in our map, copy it there. 
                    # If not (dynamic request), we'd need a different buffer, but here we stick to layout.
                    if m_hash in self._rx_map:
                        offset, expected_size = self._rx_map[m_hash]
                        self._rx_view[offset : offset + expected_size] = self._udp_view[HEADER_SIZE : HEADER_SIZE + expected_size]
                        return self._rx_view[offset : offset + expected_size]
                    else:
                        # Return the raw packet payload view if not in layout
                        return self._udp_view[HEADER_SIZE : size]

                # Direct assembly into the arena via mapping (Standard Layout Mode)
                if m_hash in self._rx_map:
                    offset, expected_size = self._rx_map[m_hash]
                    self._rx_view[offset : offset + expected_size] = self._udp_view[HEADER_SIZE : HEADER_SIZE + expected_size]
                    chunks_received += 1
            
            return self._rx_view
            
        except (socket.timeout, TimeoutError):
            if expected_rx_hash is not None:
                print(f"[WARN] [AxicorClient] UDP Timeout waiting for hash {expected_rx_hash}")
            else:
                print(f"[WARN] [AxicorClient] UDP Timeout. Received {chunks_received}/{self.expected_chunks} chunks.")
            return self._rx_view[0:0]
        except ConnectionResetError as e:
            print(f"[WARN] [AxicorClient] UDP ConnectionResetError: {e}")
            return self._rx_view[0:0]
        except Exception as e:
            print(f"[WARN] [AxicorClient] UDP Unexpected Error: {e}")
            return self._rx_view[0:0]
