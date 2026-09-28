#!/usr/bin/env bash
set -euo pipefail

# Paper configuration for Qwen3-30B-A3B-Base: Megatron, 8,192-token responses.
N="${N:-8}"
if [[ "${1:-}" == "--n" ]]; then
    N="${2:?--n requires 8, 16, or 32}"
    shift 2
fi
case "$N" in
    8|16|32) ;;
    *) echo "N must be 8, 16, or 32" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
VERL_ROOT="${VERL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TRAIN_FILE="${TRAIN_FILE:-$VERL_ROOT/data/dapo_math/train.parquet}"
[[ -f "$TRAIN_FILE" ]] || { echo "Training data not found: $TRAIN_FILE" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$VERL_ROOT/checkpoints/respo/qwen3_30b_a3b_n${N}}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR=1

cd "$VERL_ROOT"
"$PYTHON_BIN" -m recipe.respo.main_ppo \
    --config-name=ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TRAIN_FILE" \
    data.shuffle=true \
    data.prompt_key=prompt \
    data.truncation=error \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=64 \
    data.return_raw_chat=true \
    data.train_batch_size=$((32 * N)) \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=respo \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=false \
    actor_rollout_ref.actor.megatron.grad_offload=false \
    actor_rollout_ref.actor.megatron.optimizer_offload=false \
    actor_rollout_ref.actor.megatron.use_mbridge=true \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=true \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=true \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=true \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=true \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=true \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18432 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.param_offload=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.expert_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18432 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=true \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward.reward_kwargs.max_resp_len=8192 \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name=respo \
    trainer.experiment_name="qwen3_30b_a3b_n${N}" \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=$((256 / N)) \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$((1024 / N)) \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.resume_mode=auto \
    "$@"
