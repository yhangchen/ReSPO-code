#!/usr/bin/env bash

set -xeuo pipefail

# --------------------------------- Usage ----------------------------------
# bash recipe/respo/scripts/train_qwen3_30b_a3b.sh [--N N] [parallelism flags] [extra hydra overrides...]
#
# Megatron configuration for the paper's Qwen3-30B-A3B-Base experiments.
# Flags:
#   --N N        Set train_batch_size = ppo_mini_batch_size * N (default: 8)
#                and total_training_steps = 1024 / N
#   --model PATH Model path (default: Qwen/Qwen3-30B-A3B-Base)
#   --tp N       Megatron tensor parallel size (default: 2)
#   --cp N       Megatron context parallel size (default: 1)
#   --pp N       Megatron pipeline parallel size (default: 1)
#   --vpp N      Megatron virtual pipeline size (default: null)
#   --ep N       Megatron expert parallel size (default: 8)
#   --etp N      Megatron expert tensor parallel size (default: 1)
#   --infer_tp N vLLM tensor parallel size (default: 4)
#   --infer_dp N vLLM data parallel size (default: 1)
#   --infer_ep N vLLM expert parallel size (default: 1)
# Example: bash recipe/respo/scripts/train_qwen3_30b_a3b.sh --N 16 --tp 2 --ep 8
# --------------------------------------------------------------------------

train_batch_n=8
model_path="Qwen/Qwen3-30B-A3B-Base"

# Megatron parallelism defaults (8 x H200)
megatron_tp=2
megatron_cp=1
megatron_pp=1
megatron_vpp=null
megatron_ep=8
megatron_etp=1

# Offload (disabled for the paper's 8 x H200 configuration)
offload=False

# Inference parallelism defaults
infer_tp=4
infer_dp=1
infer_ep=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --N) train_batch_n="${2:?--N requires a value}"; shift 2 ;;
        --model) model_path="${2:?--model requires a value}"; shift 2 ;;
        --tp) megatron_tp="${2:?--tp requires a value}"; shift 2 ;;
        --cp) megatron_cp="${2:?--cp requires a value}"; shift 2 ;;
        --pp) megatron_pp="${2:?--pp requires a value}"; shift 2 ;;
        --vpp) megatron_vpp="${2:?--vpp requires a value}"; shift 2 ;;
        --ep) megatron_ep="${2:?--ep requires a value}"; shift 2 ;;
        --etp) megatron_etp="${2:?--etp requires a value}"; shift 2 ;;
        --infer_tp) infer_tp="${2:?--infer_tp requires a value}"; shift 2 ;;
        --infer_dp) infer_dp="${2:?--infer_dp requires a value}"; shift 2 ;;
        --infer_ep) infer_ep="${2:?--infer_ep requires a value}"; shift 2 ;;
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

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR=1

echo "Using $NNODES nodes for Qwen3-30B-A3B ReSPO training with N=$train_batch_n (Megatron backend)..."

# ------------------------------------- Setup xp params ---------------------------------------
project_name="respo"
dataset_name="dapo_math"
loss_mode="respo"
adv_estimator="grpo"
loss_agg_mode="token-mean"
MODEL_PATH="${model_path}"
rollout_engine="vllm"
rollout_mode="async"
return_raw_chat="True"
gpu_memory_utilization=0.85
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

max_prompt_length=$((1024))
max_response_length=$((1024 * 8))
enable_overlong_buffer=true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1

# Performance related parameters
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

# Paths and namings
SFT_MODEL=$(basename "$MODEL_PATH")
exp_name="megatron_${loss_mode}_${dataset_name}-${SFT_MODEL}-RL_N_${train_batch_n}"
CKPTS_DIR="${CHECKPOINT_DIR:-$DATA_ROOT/checkpoints/respo/qwen3_30b_a3b_n${train_batch_n}}"
train_files="${TRAIN_FILE:-$DATA_ROOT/data/$dataset_name/train.parquet}"
test_files="${VAL_FILE:-$train_files}"
trainer_logger="${TRAINER_LOGGER:-['console']}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$train_files" ]]; then
    echo "Training data not found: $train_files" >&2
    exit 2
fi

# ===================================== Actor =====================================

# Megatron parallelism config
ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${megatron_tp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${megatron_cp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${megatron_pp} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${megatron_vpp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${megatron_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${megatron_etp} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True"

# Ref model Megatron config (mirrors actor parallelism)
REF_MEGATRON_CONFIG="
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${megatron_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${megatron_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${megatron_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${megatron_etp} \
    actor_rollout_ref.ref.megatron.param_offload=${offload}"

# Actor model config
ACTOR_CONFIG="
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    $ACTOR_MEGATRON_CONFIG \
    $REF_MEGATRON_CONFIG"

# ===================================== Rollout =====================================
ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.data_parallel_size=${infer_dp} \
    actor_rollout_ref.rollout.expert_parallel_size=${infer_ep} \
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
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=false \
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
    --config-name=ppo_megatron_trainer \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=${shuffle_dataset} \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=64 \
    data.return_raw_chat=${return_raw_chat} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    ${ACTOR_CONFIG} \
    ${ROLLOUT_CONFIG} \
    ${REWARD_CONFIG} \
    ${TRAINER_CONFIG} \
    "$@"
