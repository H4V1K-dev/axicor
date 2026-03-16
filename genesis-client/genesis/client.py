import socket
import struct
import numpy as np
from typing import List, Dict

# C-ABI Constants (Строго из 08_io_matrix.md)
MAX_UDP_PAYLOAD = 65507
HEADER_SIZE = 20
HEADER_FMT = "<IIIIhH"  # 20 bytes: magic, zone_hash, matrix_hash, size, reward, pad
GSIO_MAGIC = 0x4F495347

class GenesisMultiClient:
    def __init__(self, addr: tuple[str, int], matrices: List[Dict[str, int]]):
        """
        :param addr: (ip, port) ноды (External UDP In)
        :param matrices: Список словарей [{'zone_hash': int, 'matrix_hash': int, 'payload_size': int}]
        """
        self.addr = addr
        self.num_chunks = len(matrices)
        
        # 1. Единая Memory Arena под весь "Пирог"
        total_tx_size = sum(HEADER_SIZE + m['payload_size'] for m in matrices)
        self._tx_arena = bytearray(total_tx_size)
        
        # 2. Арена под ответ (Output_History)
        self._rx_buf = bytearray(MAX_UDP_PAYLOAD)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(True) # ЖЕСТКИЙ БАРЬЕР (Lockstep)
        
        self.payload_views = []
        self._tx_packets = []
        
        # 3. Нарезка арены (Zero-Copy)
        offset = 0
        for m in matrices:
            size = m['payload_size']
            assert size <= MAX_UDP_PAYLOAD - HEADER_SIZE, "Chunk exceeds UDP MTU!"
            
            # Запекаем статический заголовок один раз
            struct.pack_into(
                HEADER_FMT, self._tx_arena, offset,
                GSIO_MAGIC, m['zone_hash'], m['matrix_hash'], size, 0, 0
            )
            
            # View на весь пакет (Header + Payload) для сокета
            packet_view = memoryview(self._tx_arena)[offset : offset + HEADER_SIZE + size]
            self._tx_packets.append(packet_view)
            
            # View только на payload, натянутый на NumPy. 
            # Юзер будет писать прямо в эту память.
            payload_view = packet_view[HEADER_SIZE:]
            np_view = np.ndarray((size,), dtype=np.uint8, buffer=payload_view)
            self.payload_views.append(np_view)
            
            offset += HEADER_SIZE + size

    def step(self, reward: int = 0, expected_rx_hash: int = None) -> memoryview:
        """
        Hot Loop. Выполняется 100+ раз в секунду.
        Пользователь УЖЕ записал биты в массивы self.payload_views[...].
        """
        # Инъекция дофамина. Пишем reward только в первый чанк (ноде этого хватит), 
        # смещение 16 байт = позиция `global_reward` в `ExternalIoHeader`
        if self.num_chunks > 0:
            struct.pack_into("<h", self._tx_arena, 16, reward)

        # Пулеметная очередь в сеть без аллокаций
        for packet in self._tx_packets:
            self.sock.sendto(packet, self.addr)

        # Барьер синхронизации: ждем ответ от моторов
        while True:
            size, _ = self.sock.recvfrom_into(self._rx_buf, MAX_UDP_PAYLOAD)
            
            if expected_rx_hash is not None:
                magic, z_hash, m_hash, pld_size, r, p = struct.unpack_from(HEADER_FMT, self._rx_buf, 0)
                if m_hash != expected_rx_hash:
                    continue  # Игнорируем пакет и ждем нужный (например, motor_out)
                    
            # Возвращаем Zero-Copy срез ответа (без 20-байтового заголовка GSOO)
            return memoryview(self._rx_buf)[HEADER_SIZE:size]
