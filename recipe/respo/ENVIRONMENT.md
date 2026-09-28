# ReSPO environment installation

This guide installs the two environments needed by the public training
launchers. The 1.7B run uses FSDP; the 30B-A3B run uses Megatron-Core and has a
substantially more delicate build.

The reference stack is:

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 or 24.04, x86_64 |
| Python | 3.12 |
| NVIDIA driver | CUDA 12.8-capable (570 series or newer) |
| CUDA toolkit | 12.8 |
| PyTorch | 2.8.0+cu128 |
| vLLM | 0.11.0 |
| FlashAttention | 2.8.1, torch 2.8, CXX11 ABI build |
| Transformer Engine | 2.14.0, CUDA 12 build |
| Megatron-Core | `core_v0.16.1` |
| MBridge | post-0.15.1 commit `641a5a0` |
| DeepEP | 1.2.1 |

Do not replace the pinned Megatron or MBridge revisions with `main`. These
projects evolve together, and an apparently successful installation can fail
only when the Ray workers construct the model.

## 1. Hardware and storage

Run installation and training on a GPU node, not a CPU-only login node. CUDA
extensions detect the local toolkit and GPU architecture while compiling.

- Qwen3-1.7B: the launcher targets one node with 8 H100/H200-class GPUs.
- Qwen3-30B-A3B: the paper configuration targets one node with 8 H200 141 GB
  GPUs. The unmodified configuration does not fit on 80 GB H100s.
- Reserve at least 50 GB for the environment and package/build caches.
- The 30B optimizer-inclusive checkpoints can be roughly 430 GB each. Use a
  multi-terabyte checkpoint volume, keep only one checkpoint, or use the
  model-only option described below.

The launchers accept `NNODES` and standard Ray/cluster configuration, but the
published configurations were written for eight GPUs on one node. Establish a
working single-node run before adapting them to multiple nodes.

## 2. System prerequisites

Install an NVIDIA driver and the full CUDA 12.8 **toolkit**, including `nvcc`.
The smaller CUDA runtime package is not enough because Apex, Transformer
Engine, and DeepEP compile CUDA extensions.

On Ubuntu, install the non-Python build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential git curl wget pkg-config ccache ninja-build cmake \
  libibverbs-dev librdmacm-dev
```

Then check the driver and compiler independently:

```bash
nvidia-smi
/usr/local/cuda-12.8/bin/nvcc --version
```

`nvidia-smi` may display a newer maximum CUDA version; that is fine. The
important requirement is that `nvcc --version` reports CUDA 12.8 and that the
driver can run CUDA 12.8 binaries.

If CUDA is installed elsewhere, export its location before installing:

```bash
export CUDA_HOME=/path/to/cuda-12.8
```

## 3. Automated installation

The installer uses `uv`, creates an isolated Python 3.12 environment, pins the
brittle CUDA packages, installs this repository in editable mode, and finishes
with import and CUDA checks. It does not modify the system driver or toolkit.

For the 1.7B FSDP experiment:

```bash
bash recipe/respo/scripts/install_environment.sh --backend fsdp
source .venv/bin/activate
```

For the 30B-A3B Megatron experiment:

```bash
bash recipe/respo/scripts/install_environment.sh --backend megatron
source .venv/bin/activate
```

The Megatron path includes the FSDP dependencies, so a Megatron environment
can run either launcher. To put the environment on a larger local volume:

```bash
bash recipe/respo/scripts/install_environment.sh \
  --backend megatron --venv /local_nvme/respo-venv
source /local_nvme/respo-venv/bin/activate
```

Source builds commonly take 20--60 minutes. `MAX_JOBS` controls compilation
parallelism; reduce it if the node runs out of host memory:

```bash
MAX_JOBS=16 bash recipe/respo/scripts/install_environment.sh --backend megatron
```

## 4. Verify before launching

The installer performs these checks, but it is useful to repeat them in a new
shell or batch allocation:

```bash
source .venv/bin/activate
python - <<'PY'
import torch
import vllm
import flash_attn
import verl

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0))
PY
```

For the Megatron environment, also run:

```bash
python - <<'PY'
import apex
import deep_ep
import megatron.core
import mbridge
import transformer_engine

print("Megatron stack imports successfully")
PY
```

Finally, confirm that the package versions did not drift:

```bash
python -m pip show torch vllm flash-attn transformer-engine megatron-core mbridge
python -c 'import numpy; print(numpy.__version__)'
```

The expected NumPy version is `2.2.6`. Run these checks inside the same batch
job or container that will launch training; cluster modules can change
`PATH`, `LD_LIBRARY_PATH`, or `CUDA_HOME` between shells.

## 5. Prepare data and start training

```bash
python recipe/respo/scripts/prepare_dapo_math.py

bash recipe/respo/scripts/train_qwen3_1_7b.sh --n 8
bash recipe/respo/scripts/train_qwen3_30b_a3b.sh --n 8
```

Repeat with `--n 16` and `--n 32` for the other reuse ratios. Model weights are
downloaded from Hugging Face unless `MODEL_PATH` points to a local directory.
Set `HF_HOME` to a shared or sufficiently large cache before launching, for
example:

```bash
export HF_HOME=/local_nvme/huggingface
```

The launchers use console-only logging by default. To opt into Weights &
Biases, log in first and set:

```bash
export TRAINER_LOGGER="['console','wandb']"
```

## 6. Checkpoint storage choices

The launchers save model, optimizer, and extra state by default. This is the
right choice for an exact optimizer-state resume, but it is expensive for the
30B model. Limit retention without changing checkpoint contents:

```bash
bash recipe/respo/scripts/train_qwen3_30b_a3b.sh --n 8 \
  trainer.max_actor_ckpt_to_keep=1
```

If disk space matters more than exact Adam-state resume, save model and extra
state only:

```bash
bash recipe/respo/scripts/train_qwen3_30b_a3b.sh --n 8 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]' \
  trainer.max_actor_ckpt_to_keep=1
```

This reduces a checkpoint to approximately the model size, but resuming will
not restore optimizer moments. Use a fresh `CHECKPOINT_DIR` when switching
checkpoint formats.

## 7. Megatron troubleshooting

### Transformer Engine selects CUDA 13

Symptoms include missing CUDA 13 libraries or a cuBLAS link error even though
PyTorch reports CUDA 12.8. The generic `transformer-engine[pytorch]` package can
resolve a CUDA 13 binary. The installer names all three CUDA 12 packages
explicitly. Recreate the environment rather than mixing cu12 and cu13 wheels.

### MBridge fails in `isinstance`

An error such as `isinstance() arg 2 must be a type` usually means Megatron
`main` was installed. Later Megatron versions changed custom FSDP symbols that
MBridge inspects. Reinstall the pinned `core_v0.16.1` commit and MBridge
`641a5a0`, or recreate the environment with the supplied script.

### DeepEP cannot find `math.h`

Do **not** prepend `/usr/include` to `CPLUS_INCLUDE_PATH`. That changes GCC's
`#include_next` search order and can make `<cmath>` fail to locate `math.h`.
Install `libibverbs-dev` and let `/usr/include` remain a compiler default. The
installer exposes only the needed `infiniband` and `rdma` directories through
the virtual environment.

### DeepEP cannot find `nccl_device.h`

This normally means DeepEP `main` was installed. The pip NCCL package does not
ship that experimental header. Use the pinned DeepEP 1.2.1 tag.

### vLLM or Transformers fails after installation

Check NumPy directly:

```bash
python - <<'PY'
import numpy, scipy.optimize, numba
print(numpy.__version__)
PY
```

NumPy newer than 2.2 can break Numba; an old NumPy can be incompatible with the
resolved SciPy and surface as a misleading Transformers lazy-import failure.
The installer pins NumPy 2.2.6 as its final package operation.

### vLLM fails with an expandable-segments assertion

Do not set:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Expandable segments are incompatible with the memory pool used by vLLM sleep
mode. Clear the variable before training:

```bash
unset PYTORCH_CUDA_ALLOC_CONF
```

If memory is fragmented, reduce rollout GPU utilization or the per-GPU token
budget instead.

### Both CUDA and ROCm visibility variables are set

Some schedulers export `ROCR_VISIBLE_DEVICES` on NVIDIA nodes. Clear the ROCm
variables before starting Ray workers:

```bash
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
```

### 30B runs out of memory on 80 GB GPUs

The distributed optimizer, weights, gradients, Transformer Engine workspace,
and vLLM cache share the same devices. Optimizer offload between rollout and
training phases does not reduce the optimizer's update-phase peak. Use 141 GB
H200s for the published configuration; lowering only vLLM utilization is not
enough to make the unmodified run fit on 80 GB H100s.

## 8. Clean recovery

GPU Python stacks are easier to reproduce from a new environment than to
repair after several conflicting installs. To retry safely, choose a new venv
path rather than editing the failed one in place:

```bash
bash recipe/respo/scripts/install_environment.sh \
  --backend megatron --venv .venv-megatron-clean
source .venv-megatron-clean/bin/activate
```

Keep the failed environment until the replacement passes verification, then
remove it yourself if it is no longer needed.
