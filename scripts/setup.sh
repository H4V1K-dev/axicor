#!/bin/bash
# Axicor Alpha — Bootstrap Script
# https://github.com/H4V1K-dev/axicor
set -e

# ─────────────────────────────────────────────
# Цвета
# ─────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}\033[38;5;196m"
echo "   █████╗ ██╗  ██╗██╗ ██████╗ ██████╗ ██████╗ "
echo "  ██╔══██╗╚██╗██╔╝██║██╔════╝██╔═══██╗██╔══██╗"
echo "  ███████║ ╚███╔╝ ██║██║     ██║   ██║██████╔╝"
echo "  ██╔══██║ ██╔██╗ ██║██║     ██║   ██║██╔══██╗"
echo "  ██║  ██║██╔╝ ██╗██║╚██████╗╚██████╔╝██║  ██║"
echo "  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝"
echo -e "${NC}"
echo -e "  ${BOLD}Embodied AI Engine — Alpha Bootstrap${NC}"
echo ""

# ─────────────────────────────────────────────
# 1. Обязательные зависимости
# ─────────────────────────────────────────────
echo -e "${CYAN}[1/5] Checking dependencies...${NC}"

MISSING=0

check_cmd() {
    if command -v "$1" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $1 found"
    else
        echo -e "  ${RED}✗${NC} $1 not found — $2"
        MISSING=1
    fi
}

check_cmd python3  "install from https://python.org"
check_cmd cargo   "install from https://rustup.rs"
check_cmd git     "install via your package manager"

if [ $MISSING -eq 1 ]; then
    echo ""
    echo -e "${RED}❌ Missing required dependencies. Please install them and re-run.${NC}"
    exit 1
fi

# ─────────────────────────────────────────────
# 2. GPU Detection
# ─────────────────────────────────────────────
echo ""
echo -e "${CYAN}[2/5] Detecting GPU backend...${NC}"

GPU_FEATURES=""
GPU_FOUND=0

if command -v nvcc >/dev/null 2>&1; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "  ${GREEN}✓${NC} NVIDIA CUDA found (nvcc ${NVCC_VER})"
    GPU_FEATURES="--features cuda"
    GPU_FOUND=1
elif command -v hipcc >/dev/null 2>&1; then
    HIPCC_VER=$(hipcc --version 2>&1 | head -1)
    echo -e "  ${GREEN}✓${NC} AMD ROCm found (${HIPCC_VER})"
    GPU_FEATURES="--features rocm"
    GPU_FOUND=1
else
    echo -e "  ${YELLOW}⚠${NC}  No GPU toolkit detected (nvcc / hipcc not found)"
fi

if [ $GPU_FOUND -eq 0 ]; then
    echo ""
    echo -e "  ${YELLOW}Genesis requires CUDA or ROCm for full performance.${NC}"
    echo -e "  A ${BOLD}CPU mock mode${NC} is available for testing without GPU."
    echo ""
    read -p "  Enable Mock-GPU mode (CPU-only simulation)? [y/N]: " MOCK_CHOICE
    case "$MOCK_CHOICE" in
        y|Y)
            GPU_FEATURES="--features mock-gpu"
            echo -e "  ${YELLOW}⚡ Mock-GPU enabled. Performance will be limited.${NC}"
            ;;
        *)
            echo -e "  ${RED}Aborting. Install CUDA or ROCm and re-run.${NC}"
            echo -e "  CUDA:  https://developer.nvidia.com/cuda-downloads"
            echo -e "  ROCm:  https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
            exit 1
            ;;
    esac
fi

# ─────────────────────────────────────────────
# 3. Python venv
# ─────────────────────────────────────────────
echo ""
echo -e "${CYAN}[3/5] Setting up Python environment...${NC}"

if [ ! -d ".venv" ]; then
    echo -e "  Creating virtual environment..."
    python3 -m venv .venv
    echo -e "  ${GREEN}✓${NC} .venv created"
else
    echo -e "  ${GREEN}✓${NC} .venv already exists"
fi

source .venv/bin/activate

echo -e "  Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q numpy gymnasium pygame optuna

echo -e "  ${GREEN}✓${NC} Python dependencies installed"

# ─────────────────────────────────────────────
# 4. Rust build
# ─────────────────────────────────────────────
echo ""
echo -e "${CYAN}[4/5] Building Genesis Node (release)...${NC}"
echo -e "  Features: ${BOLD}${GPU_FEATURES:-default}${NC}"
echo ""

cargo build --release -p genesis-node -p genesis-baker $GPU_FEATURES

echo -e "  ${GREEN}✓${NC} Build complete"

# ─────────────────────────────────────────────
# 5. Проверка
# ─────────────────────────────────────────────
echo ""
echo -e "${CYAN}[5/5] Verifying installation...${NC}"

NODE_BIN="./target/release/genesis-node"
BAKER_BIN="./target/release/baker"

[ -f "$NODE_BIN" ]  && echo -e "  ${GREEN}✓${NC} genesis-node binary found" \
                    || echo -e "  ${RED}✗${NC} genesis-node binary missing"

[ -f "$BAKER_BIN" ] && echo -e "  ${GREEN}✓${NC} baker binary found" \
                    || echo -e "  ${RED}✗${NC} baker binary missing"

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  ✅ Axicor is ready.${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo ""
echo -e "  ${CYAN}1.${NC} Bake the brain:"
echo -e "     ${BOLD}python3 examples/cartpole/build_brain.py${NC}"
echo ""
echo -e "  ${CYAN}2.${NC} Start the node:"
echo -e "     ${BOLD}cargo run --release -p genesis-node -- --brain CartPoleAgent${NC}"
echo ""
echo -e "  ${CYAN}3.${NC} Run the agent:"
echo -e "     ${BOLD}python3 examples/cartpole/agent.py${NC}"
echo ""
echo -e "  Docs: ${CYAN}https://github.com/H4V1K-dev/axicor${NC}"
echo ""