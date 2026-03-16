use std::net::UdpSocket;

// --- C-ABI Контракты (Strict 20 bytes) ---
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ExternalIoHeader {
    magic: u32,
    zone_hash: u32,
    matrix_hash: u32,
    payload_size: u32,
    global_reward: i16,
    _padding: u16,
}

const GSIO_MAGIC: u32 = 0x4F495347;
const GSOO_MAGIC: u32 = 0x4F4F5347;

// --- Детерминированный Хэш (FNV-1a) ---
const fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9dc5;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u32;
        hash = hash.wrapping_mul(0x01000193);
        i += 1;
    }
    hash
}

const ZONE_HASH_IN: u32 = fnv1a_32(b"SensoryCortex");
const ZONE_HASH_OUT: u32 = fnv1a_32(b"MotorCortex");
const MATRIX_IN: u32 = fnv1a_32(b"cartpole_sensors");
const MATRIX_OUT: u32 = fnv1a_32(b"motor_actions");

// --- Математика CartPole (Euler Integration) ---
struct CartPole {
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
}

impl CartPole {
    fn new() -> Self {
        Self { x: 0.0, x_dot: 0.0, theta: 0.0, theta_dot: 0.0 }
    }

    fn step(&mut self, action: u8) -> bool {
        let gravity = 9.8;
        let masscart = 1.0;
        let masspole = 0.1;
        let total_mass = masscart + masspole;
        let length = 0.5; // half the pole's length
        let polemass_length = masspole * length;
        let force = if action == 1 { 10.0 } else { -10.0 };
        let tau = 0.02; // seconds between state updates

        let costheta = self.theta.cos();
        let sintheta = self.theta.sin();

        let temp = (force + polemass_length * self.theta_dot * self.theta_dot * sintheta) / total_mass;
        let thetaacc = (gravity * sintheta - costheta * temp) / 
                       (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        let xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        self.x += tau * self.x_dot;
        self.x_dot += tau * xacc;
        self.theta += tau * self.theta_dot;
        self.theta_dot += tau * thetaacc;

        // Палка упала или тележка уехала
        self.x.abs() > 2.4 || self.theta.abs() > 0.20943951 // 12 degrees
    }
}

// --- Population Coding (Без аллокаций) ---
fn encode_variable(val: f32, min: f32, max: f32, bit_offset: usize, out_mask: &mut [u8]) {
    let norm = ((val - min) / (max - min)).clamp(0.0, 1.0);
    for i in 0..16 {
        let center = i as f32 / 15.0;
        let dist = norm - center;
        let prob = (-dist * dist / (2.0 * 0.15 * 0.15)).exp();
        
        if prob > 0.5 {
            let abs_bit = bit_offset + i;
            out_mask[abs_bit / 8] |= 1 << (abs_bit % 8);
        }
    }
}

fn main() {
    // [DOD FIX] UDP Lockstep. 
    // Слушаем на 8092 (куда Оркестратор шлет Output_History)
    // Шлем на 8081 (входной порт Оркестратора)
    let sock = UdpSocket::bind("0.0.0.0:8092").expect("Failed to bind UDP 8092");
    let node_addr = "127.0.0.1:8081";
    let dashboard_addr = "127.0.0.1:8100";
    // НЕ connect()! Egress-тред ноды шлёт с эфемерного порта, а не c 8081.
    // Connected-сокет дропает пакеты с неизвестного source — вызывая таймаут.
    sock.set_read_timeout(Some(std::time::Duration::from_secs(2))).unwrap();

    println!("🚀 HFT CartPole Environment Online (Rust).");
    println!("Hashes: ZoneIn={:08X}, ZoneOut={:08X}, In={:08X}, Out={:08X}", ZONE_HASH_IN, ZONE_HASH_OUT, MATRIX_IN, MATRIX_OUT);

    let mut env = CartPole::new();
    let mut episodes = 0u32;
    let mut score = 0u32;
    let mut tps_start = std::time::Instant::now();
    let mut total_ticks_for_tps = 0u32;
    
    // Преаллоцированные буферы для Zero-Cost цикла
    let mut rx_buf = [0u8; 65535];
    let mut tx_buf = [0u8; 28]; // 20 bytes header + 8 bytes payload (64 bits)

    // Константа задержки нашего конвейера
    const CAUSAL_DELAY_BATCHES: u32 = 3;

    loop {
        // 1. Кодирование состояния (4 переменные по 16 нейронов = 64 бита = 8 байт)
        let mut bitmask = [0u8; 8];
        encode_variable(env.x, -2.4, 2.4, 0, &mut bitmask);
        encode_variable(env.x_dot, -3.0, 3.0, 16, &mut bitmask);
        encode_variable(env.theta, -0.209, 0.209, 32, &mut bitmask);
        encode_variable(env.theta_dot, -2.0, 2.0, 48, &mut bitmask);

        // 2. Расчет нейромодулятора (Dopamine)
        let angle_penalty = (env.theta.abs() / 0.209) * 100.0;
        let vel_penalty = (env.theta_dot.abs() * 20.0).min(50.0);
        
        // Базовая линия = 0. Наказываем за отклонения.
        let mut dopamine = (10.0 - angle_penalty * 1.5 - vel_penalty) as i16;
        dopamine = dopamine.clamp(-100, 50); // Не даем слишком много позитива

        // 3. Формирование пакета
        let header = ExternalIoHeader {
            magic: GSIO_MAGIC,
            zone_hash: ZONE_HASH_IN,
            matrix_hash: MATRIX_IN,
            payload_size: 8,
            global_reward: dopamine,
            _padding: 0,
        };

        let mut left_spikes_accum = 0;
        let mut right_spikes_accum = 0;

        // [DOD FIX] Action Repeat Loop (Ждем пока сигнал пробьет 3 зоны)
        for step in 0..CAUSAL_DELAY_BATCHES {
            unsafe {
                std::ptr::copy_nonoverlapping(&header as *const _ as *const u8, tx_buf.as_mut_ptr(), 20);
                // Шлем сенсоры только в первый батч (вспышка), остальные батчи шлем нули (тишина), 
                // чтобы не перестимулировать L4 и дать хвосту сигнала пройти дальше.
                if step == 0 {
                    std::ptr::copy_nonoverlapping(bitmask.as_ptr(), tx_buf.as_mut_ptr().add(20), 8);
                } else {
                    std::ptr::write_bytes(tx_buf.as_mut_ptr().add(20), 0, 8);
                }
            }

            sock.send_to(&tx_buf, node_addr).unwrap();

            // Strict Lockstep: ждём ответа (Оркестратор работает на частоте сети)
            let payload_bytes: Vec<u8>;
            loop {
                match sock.recv_from(&mut rx_buf) {
                    Ok((size, _src)) => {
                        if size < 20 { continue; }
                        let hdr = unsafe { &*(rx_buf.as_ptr() as *const ExternalIoHeader) };
                        if hdr.magic == GSOO_MAGIC && hdr.zone_hash == ZONE_HASH_OUT && hdr.matrix_hash == MATRIX_OUT {
                            payload_bytes = rx_buf[20..size].to_vec();
                            break;
                        }
                    }
                    Err(e) => {
                        if e.kind() == std::io::ErrorKind::WouldBlock {
                            println!("⚠️ Node timeout. Waiting for Genesis...");
                            let _ = sock.send_to(&tx_buf, node_addr);
                        } else {
                            panic!("Socket error: {}", e);
                        }
                    }
                }
            }

            // Аккумулируем спайки за все 3 батча задержки
            let mid = payload_bytes.len() / 2;
            left_spikes_accum += payload_bytes[..mid].iter().map(|&b| b as u32).sum::<u32>();
            right_spikes_accum += payload_bytes[mid..].iter().map(|&b| b as u32).sum::<u32>();
        }

        // 6. Population Decoding по итогам 3 батчей
        let action = if left_spikes_accum > right_spikes_accum { 0 } else { 1 };

        // 7. Делаем шаг физики РАЗ В 30 мс
        let done = env.step(action);
        score += 1;

        if done {
            total_ticks_for_tps += score;
            let elapsed = tps_start.elapsed().as_secs_f32();
            let tps = if elapsed > 0.0 { total_ticks_for_tps as f32 / elapsed } else { 0.0 };
            
            let mut dash_buf = [0u8; 16];
            dash_buf[0..4].copy_from_slice(&episodes.to_le_bytes());
            dash_buf[4..8].copy_from_slice(&(score as f32).to_le_bytes());
            dash_buf[8..12].copy_from_slice(&tps.to_le_bytes());
            dash_buf[12..16].copy_from_slice(&(1.0f32).to_le_bytes()); // is_done = 1.0
            let _ = sock.send_to(&dash_buf, dashboard_addr);

            if episodes % 20 == 0 {
                tps_start = std::time::Instant::now();
                total_ticks_for_tps = 0;
            }

            // Выжигаем пути
            let mut death_hdr = header;
            death_hdr.global_reward = -20; // Штраф за падение палки
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &death_hdr as *const _ as *const u8,
                    tx_buf.as_mut_ptr(),
                    20
                );
            }
            let _ = sock.send_to(&tx_buf, node_addr);

            episodes += 1;
            if episodes % 10 == 0 {
                println!("Episode {:>4} | Score: {}", episodes, score);
            }
            
            env = CartPole::new();
            score = 0;
        }
    }
}