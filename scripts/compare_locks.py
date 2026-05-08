import sys

def parse_lock(path):
    packages = {}
    current_name = None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('name = '):
                    current_name = line.split('=')[1].strip().strip('"')
                elif line.startswith('version = ') and current_name:
                    packages[current_name] = line.split('=')[1].strip().strip('"')
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return packages

def main():
    current_pkgs = parse_lock('Cargo.lock')
    old_pkgs = parse_lock(r'C:\Users\alex\Downloads\Cargo.lock')

    diffs = {}
    all_names = sorted(set(current_pkgs.keys()) | set(old_pkgs.keys()))

    for name in all_names:
        v_old = old_pkgs.get(name)
        v_curr = current_pkgs.get(name)

        # Выводим только если пакет существует в обоих файлах и версии отличаются
        if v_old is not None and v_curr is not None and v_old != v_curr:
            diffs[name] = (v_old, v_curr)

    print(f"{'Package':<30} | {'Old Version':<15} | {'Current Version':<15}")
    print("-" * 65)
    
    if not diffs:
        print("No version differences found.")
    else:
        for name, (o, c) in diffs.items():
            print(f"{name:<30} | {str(o):<15} | {str(c):<15}")

if __name__ == '__main__':
    main()
