#!/usr/bin/env bash
# Axicor Linux Bootstrap (Debian Trixie / Hermetic Python 3.11)
set -e

echo "[BOOT] Bootstrapping Axicor Linux Native Environment..."

# [DOD FIX] Updated C-ABI dependencies for Debian 13 (Trixie) GLVND architecture
echo "[BOOT] Installing C-ABI dependencies and build tools..."
sudo apt-get update
sudo apt-get install -y \
    build-essential curl git \
    libosmesa6-dev libgl1 libgl-dev libglfw3-dev libglew-dev patchelf

# 2. Rust Toolchain (MSRV 1.75+)
if ! command -v cargo &> /dev/null; then
    echo "[BOOT] Rust not found. Installing toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[BOOT] Rust toolchain detected."
fi

# 3. Hermetic Python Workspace (Zero-Build via uv)
echo "[BOOT] Provisioning Hermetic Python Workspace..."
if ! command -v uv &> /dev/null; then
    echo "[BOOT] uv not found. Installing Rust-based Python manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
    # [DOD FIX] Скачиваем предкомпилированный CPython 3.11. Изолируем от сломанного ABI Debian 13.
    uv venv --python 3.11 .venv
fi
source .venv/bin/activate

echo "[BOOT] Resolving wheels (Zero-Build)..."
uv pip install --upgrade pip
uv pip install "numpy>=1.26,<2.0" toml opencv-python "gymnasium[mujoco]==0.29.1" mujoco==2.3.7 pygame==2.5.2 optuna==3.6.1
uv pip install -e ./axicor-client

# 4. IPC / POSIX Shared Memory Verification
echo "[BOOT] Verifying POSIX Shared Memory capacity..."
SHM_AVAIL_KB=$(df -k /dev/shm | awk 'NR==2 {print $4}')
REQ_KB=2097152 # Require at least 2GB

if [ "$SHM_AVAIL_KB" -lt "$REQ_KB" ]; then
    echo -e "[WARN] /dev/shm capacity is critically low ($((SHM_AVAIL_KB / 1024)) MB)."
    echo -e "[WARN] Day/Night IPC will crash on macro-topologies with SIGBUS."
    echo -e "[WARN] Fix this by running: sudo mount -o remount,size=4G /dev/shm"
else
    echo -e "[OK] POSIX Shared Memory capacity is sufficient ($((SHM_AVAIL_KB / 1024)) MB)."
fi

# 5. Dual-Backend Hardware Detection & Compilation
echo "[BOOT] Compiling Axicor Engine (Release Profile)..."
if command -v nvcc &> /dev/null; then
    echo "[BOOT] NVIDIA CUDA detected. Compiling NVCC kernels..."
    cargo build --release -p axicor-node -p axicor-baker
elif command -v hipcc &> /dev/null; then
    echo "[BOOT] AMD ROCm/HIP detected. Compiling HIPCC kernels..."
    cargo build --release -p axicor-node -p axicor-baker --features amd
else
    echo "[BOOT] No hardware accelerators found. Building CPU Fallback (mock-gpu)..."
    cargo build --release -p axicor-node -p axicor-baker --features axicor-compute/mock-gpu
fi

echo "[OK] Axicor Linux Environment Ready."
