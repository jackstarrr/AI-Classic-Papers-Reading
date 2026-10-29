#!/usr/bin/env bash
# Qwen3-VL-8B LoRA SFT（ms-swift）
# 默认单张 5090 32G 可跑；若用双卡：CUDA_VISIBLE_DEVICES=0,1 并追加 --deepspeed zero2
set -e

MODEL_PATH=${MODEL_PATH:-/mnt/data_sda/qwenvl-8B-instruct}   # 服务器上 8B 权重路径
DATA=${DATA:-sft/sft_rationale_train.jsonl}                # 短答案变体改成 sft/sft_short_train.jsonl
VAL_DATA=${VAL_DATA:-sft/sft_rationale_val.jsonl}          # 对应改 sft_short_val.jsonl
OUT=${OUT:-ckpt/qwen3vl8b_lora_rationale}                  # 短答案变体改 ..._short
GPU=${GPU:-0}                                             # 指定空闲显卡，如 GPU=1
BATCH=${BATCH:-2}                                         # 显存不足时 BATCH=1 GA=32
GA=${GA:-16}                                              # 保持有效 batch = BATCH * GA = 32

CUDA_VISIBLE_DEVICES="$GPU" swift sft \
  --model "$MODEL_PATH" \
  --model_type qwen3_vl \
  --tuner_type lora \
  --dataset "$DATA" \
  --val_dataset "$VAL_DATA" \
  --eval_strategy steps \
  --lora_rank 32 --lora_alpha 64 \
  --target_modules all-linear \
  --freeze_vit false --freeze_aligner false \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --per_device_train_batch_size "$BATCH" \
  --per_device_eval_batch_size "$BATCH" \
  --gradient_accumulation_steps "$GA" \
  --max_length 3072 \
  --model_kwargs '{"IMAGE_MIN_TOKEN_NUM":196,"IMAGE_MAX_TOKEN_NUM":392}' \
  --gradient_checkpointing true \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --dataset_num_proc 8 --dataloader_num_workers 4 \
  --logging_steps 10 --eval_steps 200 --save_steps 200 --save_total_limit 3 \
  --output_dir "$OUT"

# ---- 说明 ----
# 1) Qwen3-VL 原生按图像 token 数控制分辨率：196/392 对应
#    min_pixels=196*32*32=200704，max_pixels=392*32*32=401408，与生成/评估服务一致。
#    drug 等小物体类别若效果差，可把 IMAGE_MAX_TOKEN_NUM 调到 784（802816 pixels）再训一版对比。
# 2) freeze_vit/freeze_aligner=false 表示 ViT 与 projector 也挂 LoRA；
#    若 OOM 或想做消融（只调 LLM 侧），改 true 即可，记得改 OUT 区分。
# 3) 训练完成后合并 LoRA，得到可按普通模型 serve 的完整权重：
#      swift export --model "$MODEL_PATH" --model_type qwen3_vl \
#        --adapters "$OUT"/<best_ckpt> \
#        --merge_lora true --output_dir merged/qwen3vl8b_sft_rationale
#    <best_ckpt> 形如 "$OUT"/v0-2026xxxx-xxxx/checkpoint-800，取 eval 最优或最后一个。
# 4) 全参数微调（可选，慢）：--tuner_type full --deepspeed zero3_offload，双卡，
#    lr 降到 1e-5~2e-5，batch 1 + GA 32。
