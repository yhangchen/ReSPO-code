# ReSPO

Official training code for **ReSPO: Reshaped Sequence Policy Optimization**.

- Project page: <https://yhangchen.github.io/ReSPO> (coming soon)
- Framework: [verl](https://github.com/verl-project/verl)
- Training data: [DAPO-MATH-17k](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)

This repository contains the `verl/` runtime package plus the ReSPO policy loss
and paper configurations. Upstream CI, Docker images, examples, general
documentation, and unrelated tests are intentionally omitted. Evaluation
results, W diagnostics, response-length analysis, experiment logs, and
checkpoints are not part of the release.

## Reproduce the training runs

Clone the repository, then install the environment for the model you want to
train:

```bash
git clone git@github.com:yhangchen/ReSPO-code.git
cd ReSPO-code

# Qwen3-1.7B / FSDP
bash recipe/respo/scripts/install_environment.sh --backend fsdp

# Or Qwen3-30B-A3B / Megatron (also supports the 1.7B run)
bash recipe/respo/scripts/install_environment.sh --backend megatron

source .venv/bin/activate
python recipe/respo/scripts/prepare_dapo_math.py
```

Run each model with rollout-reuse ratio `N` in `{8, 16, 32}`:

```bash
bash recipe/respo/scripts/train_qwen3_1_7b.sh --N 8
bash recipe/respo/scripts/train_qwen3_30b_a3b.sh --N 8
```

Repeat the commands with `--N 16` and `--N 32` to reproduce all six runs.
Installation—especially the Megatron build—is covered step by step in
[recipe/respo/ENVIRONMENT.md](recipe/respo/ENVIRONMENT.md).

## Experiment configuration

| Setting | Qwen3-1.7B-Base | Qwen3-30B-A3B-Base |
|---|---:|---:|
| Training backend | FSDP | Megatron-Core |
| GPUs | 8 × H100/H200 class | 8 × H200 141 GB |
| Response limit | 15,360 | 8,192 |
| Actor parallelism | Ulysses SP 2 | TP 2, EP 8 |
| vLLM tensor parallelism | 2 | 4 |
| Responses per prompt | 8 | 8 |
| PPO mini-batch size | 32 | 32 |
| Learning rate | `1e-6` | `1e-6` |
| Trainer iterations | `1024 / N` | `1024 / N` |
| KL penalty | none | none |

Each trainer iteration reuses the generated batch for `N` mini-batch updates,
giving 1,024 optimizer mini-batch updates per run. Both models use a global
prompt batch of `32 × N`, token-mean loss aggregation, Adam-family
optimization, 5% learning-rate warmup, weight decay 0.1, and gradient clipping
at 1.0. The 30B run applies the paper's linear overlong penalty from 4,096 to
8,192 response tokens.

The fixed ReSPO kernel is:

| Advantage branch | alpha | beta | lambda |
|---|---:|---:|---:|
| Positive | 2 | 0.5 | 2 |
| Negative | 1 | 0.5 | 2 |

## Layout

- `verl/`: the retained verl runtime used by the training entry points.
- `recipe/respo/core_algos.py`: ReSPO sequence-level policy loss.
- `recipe/respo/main_ppo.py`: minimal verl training entry point.
- `recipe/respo/workers.py`: worker registration and FSDP reload support.
- `recipe/respo/scripts/prepare_dapo_math.py`: training-data preparation.
- `recipe/respo/scripts/train_qwen3_1_7b.sh`: paper FSDP launcher.
- `recipe/respo/scripts/train_qwen3_30b_a3b.sh`: paper Megatron launcher.
- `recipe/respo/ENVIRONMENT.md`: detailed CUDA and Megatron installation guide.

The launchers follow the original ReSPO script layout: defaults and argument
parsing first, followed by grouped actor, rollout, reward, and trainer configs.
They accept Hydra overrides after the launcher flags. Common environment
overrides are `TRAIN_FILE`, `VAL_FILE`, `CHECKPOINT_DIR`, `GPUS_PER_NODE`,
`NNODES`, `PYTHON_BIN`, and `TRAINER_LOGGER`; use `--model` to select a local
or Hugging Face model path. For example:

```bash
CHECKPOINT_DIR=/checkpoints/respo-1.7b-n16 \
  bash recipe/respo/scripts/train_qwen3_1_7b.sh --N 16 \
  --model /models/Qwen3-1.7B-Base
```

In-training validation is disabled so the public path contains training only;
this does not change the optimizer updates used by the experiments.

## Checks

CPU-side checks can be run without starting a training job:

```bash
python -m pytest -q tests/recipe/respo
bash -n recipe/respo/scripts/install_environment.sh
bash -n recipe/respo/scripts/train_qwen3_1_7b.sh
bash -n recipe/respo/scripts/train_qwen3_30b_a3b.sh
```

## License and acknowledgement

Released under the Apache License 2.0. ReSPO is implemented on top of
[verl](https://github.com/verl-project/verl); see [Notice.txt](Notice.txt) and
the retained upstream source headers for attribution.
