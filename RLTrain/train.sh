set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
MODEL_PATH=${MODEL_PATH:-/path/to/your/sft-checkpoint}
TRAIN_FILE=${TRAIN_FILE:-/path/to/your/train.parquet}
VAL_FILE=${VAL_FILE:-$TRAIN_FILE}
ENGINE=${2:-vllm}
rm -rf /tmp/ray

export CUDA_LAUNCH_BLOCKING=1
n_gpus=$(nvidia-smi -L | wc -l)

train_batch_size=16
ppo_mini_batch_size=$((2 * n_gpus))
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent=5

tensor_model_parallel_size=4
val_before_train=False
search_url=${SEARCH_URL:-"http://127.0.0.1:9001/search"}
rm_url=${RM_URL:-"http://127.0.0.1:8003/eval"}
max_turns=10
project_name="RAG-RL"
experiment_name="SFT_w_crop_${n_gpus}_gpus_${max_turns}_maxturns_${n_agent}_ngroups"

export RAY_memory_usage_threshold=0.995
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=40000 \
    data.max_response_length=1024 \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=$n_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager='rm' \
    reward_model.rm_url=$rm_url \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=600 \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    trainer.val_before_train=$val_before_train \
    retriever.url=$search_url \
    max_turns=$max_turns $@
