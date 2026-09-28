#!/usr/bin/env bash
# Install the CUDA 12.8 environment used by the public ReSPO launchers.
#
# This script deliberately does not install an NVIDIA driver, CUDA toolkit, or
# system development headers. Install those as described in ENVIRONMENT.md,
# then run this script on a GPU node.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: install_environment.sh [--backend fsdp|megatron] [--venv PATH]

  --backend fsdp       Environment for Qwen3-1.7B (default)
  --backend megatron   Adds Transformer Engine, Megatron-Core, MBridge,
                       NVIDIA Apex, and DeepEP for Qwen3-30B-A3B
  --venv PATH          Virtual environment path (default: REPO/.venv)

Run this script from an x86_64 Linux GPU node with CUDA 12.8 and Python 3.12.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
BACKEND="fsdp"
VENV_DIR="$REPO_ROOT/.venv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="${2:?--backend requires fsdp or megatron}"
            shift 2
            ;;
        --venv)
            VENV_DIR="${2:?--venv requires a path}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$BACKEND" in
    fsdp|megatron) ;;
    *) echo "--backend must be fsdp or megatron" >&2; exit 2 ;;
esac

[[ "$(uname -m)" == "x86_64" ]] || {
    echo "The pinned FlashAttention wheel requires x86_64 Linux." >&2
    exit 2
}

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || {
    echo "CUDA compiler not found at $CUDA_HOME/bin/nvcc." >&2
    echo "Install CUDA toolkit 12.8 or set CUDA_HOME explicitly." >&2
    exit 2
}
if ! "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12\.8'; then
    echo "Expected CUDA toolkit 12.8 at CUDA_HOME=$CUDA_HOME." >&2
    "$CUDA_HOME/bin/nvcc" --version >&2
    exit 2
fi
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is unavailable; run the installer on a GPU node." >&2
    exit 2
}

export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export CUTLASS_NVCC_ARCHS="${CUTLASS_NVCC_ARCHS:-90}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv installation failed" >&2; exit 1; }

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    uv venv --python 3.12 "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing ReSPO $BACKEND environment in $VENV_DIR"
uv pip install --upgrade pip setuptools wheel packaging ninja cmake

# Install torch first so every compiled extension sees the same ABI.
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0

# vLLM 0.11.0 is the rollout engine used by both paper configurations.
uv pip install \
    vllm==0.11.0 \
    torch-memory-saver \
    "transformers[hf_xet]==4.56.1" \
    accelerate datasets peft hf-transfer \
    "pyarrow>=19.0.0" pandas \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    "ray[default]>=2.41.0" codetiming hydra-core pylatexenc \
    wandb dill pybind11 liger-kernel mathruler math-verify tensorboard \
    nvidia-ml-py "fastapi[standard]>=0.115.0" optree pydantic grpcio \
    pytest pre-commit ruff scipy numba

FA_WHEEL="flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/$FA_WHEEL"
uv pip install flashinfer-python==0.3.1

if [[ "$BACKEND" == "megatron" ]]; then
    [[ -f /usr/include/infiniband/mlx5dv.h ]] || {
        echo "Megatron/DeepEP needs InfiniBand development headers." >&2
        echo "On Ubuntu: sudo apt-get install libibverbs-dev librdmacm-dev" >&2
        exit 2
    }

    # Install cuDNN before building Transformer Engine. Explicit cu12 package
    # names prevent pip from silently selecting the CUDA 13 distribution.
    uv pip install nvidia-cudnn-cu12==9.10.2.21 onnxscript==0.3.1
    NVIDIA_PKG=$(python -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])' | xargs dirname)
    export CPLUS_INCLUDE_PATH="$NVIDIA_PKG/cudnn/include:$NVIDIA_PKG/nccl/include:$NVIDIA_PKG/nvtx/include:${CPLUS_INCLUDE_PATH:-}"
    export C_INCLUDE_PATH="$NVIDIA_PKG/nvtx/include:${C_INCLUDE_PATH:-}"
    export LIBRARY_PATH="$NVIDIA_PKG/cudnn/lib:$NVIDIA_PKG/nccl/lib:${LIBRARY_PATH:-}"
    export CUDNN_PATH="$NVIDIA_PKG/cudnn"

    NVTE_FRAMEWORK=pytorch uv pip install --no-build-isolation \
        --extra-index-url https://pypi.nvidia.com \
        transformer_engine_cu12==2.14.0 \
        transformer_engine_torch==2.14.0 \
        transformer_engine==2.14.0

    # These pins are intentional. Megatron main has broken MBridge/verl
    # compatibility in the past; MBridge 0.15.1 supports Qwen3-MoE.
    uv pip install --no-deps \
        "git+https://github.com/NVIDIA/Megatron-LM.git@55ac7082517c3878ae653c07c09c534b8aed49f6" \
        "git+https://github.com/ISEEKYAN/mbridge.git@0cd4ae23f2425da77a80cb3f517828452fa8e984"

    BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/respo-gpu-build.XXXXXX")
    cleanup() { rm -rf -- "$BUILD_DIR"; }
    trap cleanup EXIT

    echo "Building NVIDIA Apex (this can take several minutes)..."
    git clone --quiet https://github.com/NVIDIA/apex.git "$BUILD_DIR/apex"
    git -C "$BUILD_DIR/apex" checkout --quiet 4bdecd06b3c4b2c0a8fb6603829a8f9f05a42b49
    (
        cd "$BUILD_DIR/apex"
        APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install --no-build-isolation .
    )

    echo "Building DeepEP v1.2.1..."
    uv pip install nvidia-nvshmem-cu12
    mkdir -p "$VENV_DIR/include"
    [[ -e "$VENV_DIR/include/infiniband" ]] || ln -s /usr/include/infiniband "$VENV_DIR/include/infiniband"
    [[ -e "$VENV_DIR/include/rdma" ]] || ln -s /usr/include/rdma "$VENV_DIR/include/rdma"
    git clone --quiet --depth 1 --branch v1.2.1 https://github.com/deepseek-ai/DeepEP.git "$BUILD_DIR/deepep"
    (
        cd "$BUILD_DIR/deepep"
        # Never add /usr/include to CPLUS_INCLUDE_PATH. Doing so breaks the
        # libstdc++ include_next chain (for example, cmath -> math.h).
        uv pip install --no-build-isolation .
    )
fi

# The repository is installed without dependency resolution because the
# tested CUDA stack requires NumPy 2.2.6; install that pin last so vLLM's
# transitive dependencies cannot replace it.
uv pip install --editable "$REPO_ROOT" --no-deps
uv pip install numpy==2.2.6

echo "Running import and CUDA checks..."
RESPO_BACKEND="$BACKEND" python - <<'PY'
import importlib
import os
import sys

import torch

modules = ["verl", "vllm", "flash_attn", "ray", "datasets", "scipy", "numba"]
if os.environ["RESPO_BACKEND"] == "megatron":
    modules += ["apex", "transformer_engine", "megatron.core", "mbridge", "deep_ep"]

errors = []
for name in modules:
    try:
        importlib.import_module(name)
        print(f"OK   {name}")
    except Exception as exc:
        errors.append((name, exc))
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")

print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}; devices: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    errors.append(("CUDA", RuntimeError("torch.cuda.is_available() is false")))
if errors:
    sys.exit(1)
PY

echo
echo "Environment ready. Activate it with:"
echo "  source $VENV_DIR/bin/activate"
