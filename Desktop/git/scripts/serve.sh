#!/usr/bin/env bash
# vLLM 服务启动命令（路径/端口已按服务器实际情况填写）
# 用法：bash scripts/serve.sh <场景>   场景： gen32b | base8b | sft8b | stop
# 日志在 logs/ 下；启动后用 curl http://127.0.0.1:8000/v1/models 确认就绪。

SCENE=${1:-gen32b}
mkdir -p logs

# 同环境装有 ms-swift 时，它注册的 vllm 插件可能与当前 vllm 版本不兼容
# （如 import qwen3_5 失败）导致 vllm CLI 启动即崩；置空可跳过所有通用插件，
# 不影响 Qwen3-VL 正常 serve。
export VLLM_PLUGINS=""

case "$SCENE" in

gen32b)
  # 数据生成：Qwen3-VL-32B FP8，双卡 TP=2，占满两张 5090
  # （FP8 权重约 32G 分跨两卡，剩余显存给 KV cache；此场景下不要同时跑训练）
  CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve /mnt/data_sda/qwen3-vl-32B-instruct-FP8 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":401408,"min_pixels":200704}' \
    --port 8000 > logs/vllm_32b.log 2>&1 &
  echo "32B FP8 启动中，日志 logs/vllm_32b.log"
  ;;

base8b)
  # 对照组评估：冻结 base 8B，单卡
  CUDA_VISIBLE_DEVICES=0 nohup vllm serve /mnt/data_sda/qwenvl-8B-instruct \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":401408,"min_pixels":200704}' \
    --port 8000 > logs/vllm_8b_base.log 2>&1 &
  echo "base 8B 启动中，日志 logs/vllm_8b_base.log"
  ;;

sft8b)
  # SFT 模型评估：合并后的权重，单卡，绑 GPU1 + 端口 8001（可同时用 GPU0 训练）
  # eval 时加 --api-base http://127.0.0.1:8001/v1
  MERGED=${2:-merged/qwen3vl8b_sft_rationale}
  CUDA_VISIBLE_DEVICES=1 nohup vllm serve "$MERGED" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":401408,"min_pixels":200704}' \
    --port 8001 > logs/vllm_8b_sft.log 2>&1 &
  echo "SFT 8B ($MERGED) 启动中，端口 8001，日志 logs/vllm_8b_sft.log"
  ;;

stop)
  pkill -f "vllm serve" && echo "已停止所有 vllm 服务" || echo "没有运行中的 vllm 服务"
  ;;

*)
  echo "未知场景: $SCENE (可选: gen32b | base8b | sft8b | stop)"
  exit 1
  ;;
esac

# ---- 启动后检查 ----
# curl http://127.0.0.1:8000/v1/models
# 冒烟测试（纯文本）：
# curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
#   "model": "'"/mnt/data_sda/qwen3-vl-32B-instruct-FP8"'",
#   "messages": [{"role":"user","content":"Reply with the word OK."}],
#   "max_tokens": 8, "temperature": 0
# }'
