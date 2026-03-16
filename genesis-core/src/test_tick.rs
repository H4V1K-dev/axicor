#![cfg(test)]

use crate::config::blueprints::NeuronType;
use crate::constants::{AXON_SENTINEL, V_SEG};

/// Эмулятор ядра PropagateAxons (Spec §1.6)
fn emulate_propagate_axons(heads: &mut [u32], v_seg: u32) {
    for head in heads.iter_mut() {
        if *head != AXON_SENTINEL {
            *head += v_seg;
        }
    }
}

/// Состояние одной сомы для эмулятора UpdateNeurons
#[derive(Debug, Clone, Default)]
struct SomaState {
    pub voltage: i32,
    pub threshold_offset: i32,
    pub refractory_timer: u8,
    pub flags: u8, // bit 0: is_spiking
}

/// Состояние одного дендрита
#[derive(Debug, Clone, Default)]
struct DendriteState {
    pub target_packed: u32,
    pub weight: i16,
    pub timer: u8,
}

/// Эмулятор ядра UpdateNeurons (Spec §1.5)
/// Для тестов упрощен до работы с одной сомой и списком её дендритов
fn emulate_update_neuron(
    soma: &mut SomaState,
    dendrites: &mut [DendriteState],
    my_axon_idx: Option<usize>,
    axon_heads: &mut [u32],
    p: &NeuronType,
) {
    // 2. Homeostasis decay
    let mut t_off = soma.threshold_offset;
    let decayed = t_off - p.homeostasis_decay as i32;
    // max(0, decayed)
    t_off = if decayed > 0 { decayed } else { 0 };

    // 3. Soma Refractory
    let mut s_ref = soma.refractory_timer;
    if s_ref > 0 {
        soma.refractory_timer = s_ref - 1;
        soma.threshold_offset = t_off;
        soma.flags &= !0x1;
        return; // Early exit
    }

    // 4. GLIF Leak
    let mut v = soma.voltage;
    let leaked = v - p.leak_rate;
    // max(rest, leaked)
    v = if leaked > p.rest_potential { leaked } else { p.rest_potential };
    println!("DEBUG: v after leak={}, leaked={}, rest={}", v, leaked, p.rest_potential);

    // 5. Dendrite loop
    for d in dendrites.iter_mut() {
        if d.timer > 0 {
            d.timer -= 1;
            continue;
        }

        if d.target_packed == 0 {
            break; // Empty slot invariant
        }

        let axon_id = (d.target_packed & 0x00FF_FFFF).saturating_sub(1);
        let seg_idx = d.target_packed >> 24;
        let head = axon_heads[axon_id as usize];
        let dist = head.wrapping_sub(seg_idx);

        println!("DEBUG: checking slot. head={}, seg={}, dist={}, p_len={}", head, seg_idx, dist, p.signal_propagation_length);

        // Инвариант §1.6 (exclusive, синхронизировано с CUDA в этой итерации)
        if dist < p.signal_propagation_length as u32 {
            // sign baked in
            v += d.weight as i32;
            d.timer = p.synapse_refractory_period;
            println!("DEBUG: active! added weight. new v={}", v);
        }
    }

    // 6. Threshold & Fire
    let eff_threshold = p.threshold + t_off;
    let is_spiking = v >= eff_threshold;

    if is_spiking {
        v = p.rest_potential;
        s_ref = p.refractory_period;
        t_off += p.homeostasis_penalty;
        soma.flags |= 0x1;
    } else {
        soma.flags &= !0x1;
    }

    // 7. Reset axon
    if is_spiking {
        if let Some(axon) = my_axon_idx {
            axon_heads[axon] = 0;
        }
    }

    soma.voltage = v;
    soma.threshold_offset = t_off;
    soma.refractory_timer = s_ref;
}

fn test_neuron() -> NeuronType {
    NeuronType {
        name: "Test".to_string(),
        threshold: 200,
        rest_potential: 0,
        leak_rate: 10,
        homeostasis_penalty: 50,
        homeostasis_decay: 5,
        refractory_period: 10,
        synapse_refractory_period: 5,
        signal_propagation_length: 3,
        ..Default::default()
    }
}

// ============================================================================
// Тесты PropagateAxons
// ============================================================================

#[test]
fn test_propagate_advances_head() {
    let mut heads = [0, 5, 10];
    emulate_propagate_axons(&mut heads, V_SEG);
    assert_eq!(heads, [V_SEG, 5 + V_SEG, 10 + V_SEG]);
}

#[test]
fn test_propagate_sentinel_stays() {
    let mut heads = [0, AXON_SENTINEL, 10];
    emulate_propagate_axons(&mut heads, V_SEG);
    assert_eq!(heads[1], AXON_SENTINEL);
    assert_eq!(heads[0], V_SEG);
}

// ============================================================================
// Тесты UpdateNeurons
// ============================================================================

#[test]
fn test_glif_leak_clamps_at_rest() {
    let p = test_neuron();
    let mut soma = SomaState { voltage: 5, ..Default::default() };
    let mut dendrites = [];
    let mut heads = [];
    
    // leak=10, voltage=5. Expected: max(0, 5-10) = 0
    emulate_update_neuron(&mut soma, &mut dendrites, None, &mut heads, &p);
    assert_eq!(soma.voltage, 0);

    // leak=10, voltage=25. Expected: max(0, 25-10) = 15
    soma.voltage = 25;
    emulate_update_neuron(&mut soma, &mut dendrites, None, &mut heads, &p);
    assert_eq!(soma.voltage, 15);
}

#[test]
fn test_refractory_blocks_integration() {
    let p = test_neuron();
    let mut soma = SomaState { voltage: 100, refractory_timer: 2, ..Default::default() };
    let mut dendrites = [];
    let mut heads = [];
    
    // While refractory > 0, voltage doesn't leak or integrate
    emulate_update_neuron(&mut soma, &mut dendrites, None, &mut heads, &p);
    assert_eq!(soma.refractory_timer, 1);
    assert_eq!(soma.voltage, 100);
    assert_eq!(soma.flags & 1, 0);
}

#[test]
fn test_homeostasis_decay_clamps_zero() {
    let p = test_neuron();
    let mut soma = SomaState { threshold_offset: 2, ..Default::default() };
    let mut dendrites = [];
    let mut heads = [];
    
    // decay = 5. offset = 2. Expected: max(0, 2-5) = 0
    emulate_update_neuron(&mut soma, &mut dendrites, None, &mut heads, &p);
    assert_eq!(soma.threshold_offset, 0);
}

#[test]
fn test_active_tail_triggers_voltage() {
    let p = test_neuron();
    // propagation_length = 3
    // Leak = 10. Если начать с 0, leak сделает напряжение 0 (clamp rest).
    // Ставим начальное 10, leak сделает 0, dendrite добавит 150 -> итог 150.
    let mut soma = SomaState { voltage: 10, ..Default::default() };
    let mut dendrites = [
        // axon 1, seg 0 -> target_packed = (0 << 24) | (1 + 1) = 2
        DendriteState { target_packed: 2, weight: 150, timer: 0 }
    ];
    let mut heads = [0, 2]; // axon 0 (dummy), axon 1 (head=2)
    
    emulate_update_neuron(&mut soma, &mut dendrites, None, &mut heads, &p);
    assert_eq!(soma.voltage, 150); 
    assert_eq!(dendrites[0].timer, p.synapse_refractory_period);
}

#[test]
fn test_dendrite_refractory_skips_slot() {
    let p = test_neuron();
    let mut soma = SomaState { voltage: 0, ..Default::default() };
    let mut dendrites = [
        DendriteState { target_packed: 0, weight: 150, timer: 1 } // Refractory!
    ];
    let mut heads = [2]; // dist < 3
    
    emulate_update_neuron(&mut soma, &mut dendrites, None, &mut heads, &p);
    
    // Weight should NOT be added
    assert_eq!(soma.voltage, 0);
    // Timer should tick down
    assert_eq!(dendrites[0].timer, 0);
}

#[test]
fn test_fire_resets_head_to_zero_and_applies_penalties() {
    let p = test_neuron(); // thresh=200, rest=0, h_pen=50
    let mut soma = SomaState { voltage: 250, ..Default::default() }; // will fire
    let mut dendrites = [];
    let mut heads = [50];
    
    emulate_update_neuron(&mut soma, &mut dendrites, Some(0), &mut heads, &p);
    
    assert_eq!(soma.flags & 1, 1); // Spiking
    assert_eq!(soma.voltage, 0);   // Reset to rest
    assert_eq!(soma.threshold_offset, 50); // Penalty
    assert_eq!(soma.refractory_timer, p.refractory_period);
    assert_eq!(heads[0], 0); // Axon head reset
}

#[test]
fn test_full_spike_cycle() {
    let p = test_neuron(); // thresh=200, prop_len=3
    let mut soma = SomaState::default();
    let mut dendrites = [
        DendriteState { target_packed: (5 << 24) | (1 + 1) /* axon 1, seg 5 */, weight: 210, timer: 0 }
    ];
    let mut heads = [AXON_SENTINEL, 4]; // My axon is 0, target is 1
    
    // Tick 1: head=4, seg=5 -> dist wraps -> inactive
    emulate_update_neuron(&mut soma, &mut dendrites, Some(0), &mut heads, &p);
    assert_eq!(soma.voltage, 0);
    
    // Propagate
    emulate_propagate_axons(&mut heads, V_SEG); // heads = [AXON_SENTINEL, 5]
    
    // Tick 2: head=5, seg=5 -> dist=0 < 3 -> ACTIVE!
    emulate_update_neuron(&mut soma, &mut dendrites, Some(0), &mut heads, &p);
    // v = 0 -> leak to 0 -> +210 = 210. 210 >= 200 -> SPK!
    assert_eq!(soma.flags & 1, 1);
    assert_eq!(soma.voltage, 0); // Reset
    assert_eq!(heads[0], 0); // My axon sprouted
    assert_eq!(dendrites[0].timer, p.synapse_refractory_period);
}
