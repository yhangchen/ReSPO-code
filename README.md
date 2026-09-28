# ReSPO

Official training code for **ReSPO: Reshaped Sequence Policy Optimization**.

This repository is intentionally small. It contains the ReSPO policy loss and
the launch scripts for the Qwen3-1.7B-Base and Qwen3-30B-A3B-Base experiments.
Evaluation, W diagnostics, and response-length analyses are not included.

- Project page: <https://respo.github.io> (coming soon)
- Base framework: [verl](https://github.com/verl-project/verl)
- Training data: [DAPO-MATH-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)

## Setup

The code is a verl submodule and is expected at `recipe/respo/code`. It is
tested with verl commit `9713583cad81e4118ac21c108986ad1f7b6db4b8`.

```bash
git clone --recursive git@github.com:yhangchen/verl.git
cd verl
git submodule update --init --recursive
```

The ReSPO extension was developed and tested against verl base commit
`9713583cad81e4118ac21c108986ad1f7b6db4b8`.

Install verl's NVIDIA/vLLM environment following the upstream installation
guide. The 30B run additionally needs verl's Megatron, MBridge, DeepEP, and
Transformer Engine dependencies. The reported configurations used eight GPUs:
H100-class GPUs for 1.7B and 141 GB H200 GPUs for 30B-A3B.

Prepare the training split once:

```bash
python recipe/respo/code/scripts/prepare_dapo_math.py
```

This writes `data/dapo_math/train.parquet`. You can instead point the launchers
at an existing compatible parquet file with `TRAIN_FILE=/path/to/train.parquet`.

## Training

Run each model with rollout-reuse ratio `N` in `{8, 16, 32}`:

```bash
bash recipe/respo/code/scripts/train_qwen3_1_7b.sh --n 8
bash recipe/respo/code/scripts/train_qwen3_30b_a3b.sh --n 8
```

Repeat with `--n 16` and `--n 32` to reproduce the six ReSPO training runs.
Each run uses 1,024 policy updates, mini-batch size 32, eight responses per
prompt, learning rate `1e-6`, token-mean aggregation, and no KL penalty.

The launchers accept ordinary Hydra overrides after `--n`. Common environment
overrides are `MODEL_PATH`, `TRAIN_FILE`, `CHECKPOINT_DIR`, `GPUS_PER_NODE`,
`NNODES`, `PYTHON_BIN`, and `TRAINER_LOGGER`. For example:

```bash
TRAINER_LOGGER="['console','wandb']" \
  bash recipe/respo/code/scripts/train_qwen3_1_7b.sh --n 16
```

The scripts disable in-training validation so that this release contains only
the training path. This does not change the optimizer updates used in the
reported experiments.

## Method configuration

The published runs use one fixed two-branch kernel:

| Branch | alpha | beta | lambda |
|---|---:|---:|---:|
| Positive advantage | 2 | 0.5 | 2 |
| Negative advantage | 1 | 0.5 | 2 |

Qwen3-1.7B uses FSDP, a 15,360-token response cap, and rollout tensor
parallelism 2. Qwen3-30B-A3B uses Megatron TP=2/EP=8, an 8,192-token response
cap, rollout tensor parallelism 4, and the paper's linear overlong penalty from
4,096 to 8,192 tokens.

## Tests

```bash
python -m pytest recipe/respo/code/tests
bash -n recipe/respo/code/scripts/train_qwen3_1_7b.sh
bash -n recipe/respo/code/scripts/train_qwen3_30b_a3b.sh
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
