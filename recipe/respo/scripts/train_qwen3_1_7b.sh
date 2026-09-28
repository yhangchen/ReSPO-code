#!/usr/bin/env bash

set -xeuo pipefail

# --------------------------------- Usage ----------------------------------
# bash recipe/respo/scripts/train_qwen3_1_7b.sh [--N N] [--model PATH] [extra hydra overrides...]
#
# FSDP configuration for the paper's Qwen3-1.7B-Base experiments.
# Flags:
#   --N N        Set train_batch_size = ppo_mini_batch_size * N (default: 8)
#                and total_training_steps = 1024 / N
#   --model PATH Model path (default: Qwen/Qwen3-1.7B-Base)
# Example: bash recipe/respo/scripts/train_qwen3_1_7b.sh --N 16
# --------------------------------------------------------------------------

train_batch_n=8
model_path="Qwen/Qwen3-1.7B-Base"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --N) train_batch_n="${2:?--N requires a value}"; shift 2 ;;
        --model) model_path="${2:?--model requires a value}"; shift 2 ;;
        *) break ;;
    esac
done

case "$train_batch_n" in
    8|16|32) ;;
    *) echo "Error: N must be 8, 16, or 32" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
DATA_ROOT="${DATA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
export NNODES

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR=1

echo "Using $NNODES nodes for Qwen3-1.7B ReSPO training with N=$train_batch_n..."

# ------------------------------------- Setup xp params ---------------------------------------
project_name="respo"
dataset_name="dapo_math"
loss_mode="respo"
adv_estimator="grpo"
loss_agg_mode="token-mean"
MODEL_PATH="${model_path}"
offload=false
rollout_engine="vllm"
rollout_mode="async"
return_raw_chat="True"
shuffle_dataset=true

test_freq=-1
save_freq=$((256 / train_batch_n))
total_epochs=100
val_before_train=false

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

ppo_mini_batch_size=32
train_batch_size=$((ppo_mini_batch_size * train_batch_n))
total_training_steps=$((1024 / train_batch_n))
n_resp_per_prompt=8
warmup_steps_ratio=0.05

max_prompt_length=$((1024))
max_response_length=$((1024 * 15))

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1

# Performance related parameters
sp_size=2
use_dynamic_bsz=true
entropy_checkpointing=true
gpu_memory_utilization=0.75
gen_tp=2
ppo_micro_batch_size_per_gpu=4
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

# Paths and namings
SFT_MODEL=$(basename "$MODEL_PATH")
exp_name="${loss_mode}_${dataset_name}-${SFT_MODEL}-RL_N_${train_batch_n}"
CKPTS_DIR="${CHECKPOINT_DIR:-$DATA_ROOT/checkpoints/respo/qwen3_1_7b_n${train_batch_n}}"
train_files="${TRAIN_FILE:-$DATA_ROOT/data/$dataset_name/train.parquet}"
test_files="${VAL_FILE:-$train_files}"
trainer_logger="${TRAINER_LOGGER:-['console']}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$train_files" ]]; then
    echo "Training data not found: $train_files" >&2
    exit 2
fi

# ===================================== Actor =====================================
ACTOR_CONFIG="
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${warmup_steps_ratio} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}"

# ===================================== Rollout =====================================
ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k}"

# ===================================== Reward =====================================
REWARD_CONFIG="
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.max_resp_len=${max_response_length}"

# ===================================== Trainer =====================================
TRAINER_CONFIG="
    trainer.logger=${trainer_logger} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.resume_mode=auto"

# ------------------------------------- Launch training ---------------------------------------
cd "$DATA_ROOT"
"$PYTHON_BIN" -m recipe.respo.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=${shuffle_dataset} \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.return_raw_chat=${return_raw_chat} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    ${ACTOR_CONFIG} \
    ${ROLLOUT_CONFIG} \
    ${REWARD_CONFIG} \
    ${TRAINER_CONFIG} \
    "$@"
