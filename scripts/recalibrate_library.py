import os
import glob
import re
import math

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Parse required parameters
    inh_match = re.search(r'is_inhibitory\s*=\s*(true|false)', content)
    penalty_match = re.search(r'homeostasis_penalty\s*=\s*(\d+)', content)
    leak_match = re.search(r'leak_rate\s*=\s*(\d+)', content)
    decay_match = re.search(r'homeostasis_decay\s*=\s*(\d+)', content)

    if not (inh_match and penalty_match and leak_match and decay_match):
        print(f"Skipping {filepath} - Missing required parameters.")
        return False

    is_inhibitory = inh_match.group(1) == 'true'
    homeostasis_penalty = float(penalty_match.group(1))
    leak_rate = float(leak_match.group(1))
    homeostasis_decay = float(decay_match.group(1))

    # Formula 1: Potentiation
    penalty_norm = clamp((homeostasis_penalty - 1000.0) / 19000.0, 0.0, 1.0)
    if is_inhibitory:
        original_pot = int(100.0 + 50.0 * penalty_norm)
    else:
        original_pot = int(60.0 + 70.0 * penalty_norm)
        
    gsop_potentiation = max(5, original_pot // 5)

    # Formula 2: Depression (Entropy Erosion)
    gsop_depression = int(gsop_potentiation * 1.2)

    # Formula 3: Inertia Curve with Dead Zone Guard
    # delta_pot не имеет права обнуляться из-за сдвига >> 7
    min_inertia = math.ceil(128.0 / gsop_potentiation)
    
    decay_norm = clamp(homeostasis_decay / 50.0, 0.0, 1.0)
    steepness = 0.5 + (1.5 * decay_norm)

    inertia_curve = []
    for i in range(16):
        progress = (15 - i) / 15.0
        val = min_inertia + (128 - min_inertia) * (progress ** steepness)
        inertia_curve.append(max(min_inertia, int(round(val))))

    prune_threshold = 100
    initial_synapse_weight = 1500 if is_inhibitory else 1000

    # Regex replacements to modify the file content in-place
    content = re.sub(r'(?m)^gsop_potentiation\s*=.*$', f'gsop_potentiation = {gsop_potentiation}', content)
    content = re.sub(r'(?m)^gsop_depression\s*=.*$', f'gsop_depression = {gsop_depression}', content)
    content = re.sub(r'(?m)^inertia_curve\s*=.*$', f'inertia_curve = {inertia_curve}', content)
    content = re.sub(r'(?m)^prune_threshold\s*=.*$', f'prune_threshold = {prune_threshold}', content)
    content = re.sub(r'(?m)^initial_synapse_weight\s*=.*$', f'initial_synapse_weight = {initial_synapse_weight}', content)

    with open(filepath, 'w') as f:
        f.write(content)

    return True

processed = 0
for root, _, files in os.walk('GNM-Library'):
    for file in files:
        if file.endswith('.toml'):
            if process_file(os.path.join(root, file)):
                processed += 1

print(f"Successfully recalibrated {processed} TOML files in GNM-Library.")
