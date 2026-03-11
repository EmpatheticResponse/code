#!/usr/bin/env bash
set -euo pipefail


# 让本地回环不走代理（非常重要）
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

# NCCL 保守参数，避免握手/通道问题
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

# CUDA_VISIBLE_DEVICES=7 \
# trl vllm-serve \
#   --model /data/wjq/code/new_les/GRPO_qwen_2.5/outputs_sft_qwen2.5-7b_t/Qwen2.5-7B-sft-merged \
#   --dtype bfloat16 \
#   --max-model-len 4096 \
#   --host 127.0.0.1 --port 8000 \
#   --gpu-memory-utilization 0.8 \
#   --tensor-parallel-size 1

# VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
# CUDA_VISIBLE_DEVICES=5 \
# python -m vllm.entrypoints.openai.api_server \
#   --model /data/wjq/code/new_les/Qwen2.5-7B-unsloth-bnb-4bit \
#   --served-model-name qwen-sft \
#   --enable-lora \
#   --max_lora_rank 128 \
#   --max-model-len 4096 \
#   --lora-modules A=/data/wjq/code/new_les/GRPO_qwen_2.5/outputs_sft_qwen2.5-7b_tmp/best-45000 \
#   --chat-template /data/wjq/code/new_les/GRPO_qwen_2.5/qwen_chat_template.jinja \
#   --host 127.0.0.1 \
#   --port 8001 \
#   --gpu-memory-utilization 0.8 \

VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
CUDA_VISIBLE_DEVICES=2 \
python -m vllm.entrypoints.openai.api_server \
  --model /data/wjq/code/new_les/GRPO_qwen_2.5/outputs_sft_qwen2.5-7b_tmp/Qwen2.5-7B-sft-merged \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --enable-lora \
    --max_lora_rank 128 \
    --port 8002
