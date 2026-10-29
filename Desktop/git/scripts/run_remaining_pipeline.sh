#!/usr/bin/env bash
# 接续当前 rationale 训练：等待完成 -> 训练 short -> 选最优 LoRA -> 合并 -> 4 组评估。
# 在 sft 环境中启动；vLLM 默认取独立的 vllm conda 环境，可用 VLLM_BIN 覆盖。
# 现在即可后台启动，等待期间不占用额外 GPU：
#   mkdir -p logs && nohup bash scripts/run_remaining_pipeline.sh > logs/remaining_pipeline.log 2>&1 &
# 若 ckpt/merged 已迁移到大磁盘：
#   CKPT_ROOT=/mnt/data_sda/xujie/ckpt MERGED_ROOT=/mnt/data_sda/xujie/merged \
#   nohup bash scripts/run_remaining_pipeline.sh > logs/remaining_pipeline_eval.log 2>&1 &
set -euo pipefail

GPU=${GPU:-1}
BATCH=${BATCH:-1}
GA=${GA:-32}
MODEL_PATH=${MODEL_PATH:-/mnt/data_sda/qwenvl-8B-instruct}
PYTHON=${PYTHON:-python}
SWIFT=${SWIFT:-swift}
PORT=${PORT:-8000}
CONCURRENCY=${CONCURRENCY:-64}
TIMING_SUBSET=${TIMING_SUBSET:-200}
LIMIT=${LIMIT:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}
RUN_EVAL=${RUN_EVAL:-1}
SKIP_SHORT=${SKIP_SHORT:-0}
WAIT_FOR_RATIONALE=${WAIT_FOR_RATIONALE:-1}
WAIT_PATTERN=${WAIT_PATTERN:-'swift/cli/sft.py.*--output_dir[[:space:]]+.*ckpt/qwen3vl8b_lora_rationale'}

CKPT_ROOT=${CKPT_ROOT:-ckpt}
MERGED_ROOT=${MERGED_ROOT:-merged}
RATIONALE_OUT="$CKPT_ROOT/qwen3vl8b_lora_rationale"
SHORT_OUT="$CKPT_ROOT/qwen3vl8b_lora_short"
RATIONALE_MERGED="$MERGED_ROOT/qwen3vl8b_sft_rationale"
SHORT_MERGED="$MERGED_ROOT/qwen3vl8b_sft_short"
SERVER_PID=""
SERVER_GROUP=0

mkdir -p logs results "$MERGED_ROOT"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

if [[ -z "${VLLM_BIN:-}" ]]; then
  if [[ -x /opt/miniconda3/envs/vllm/bin/vllm ]]; then
    VLLM_BIN=/opt/miniconda3/envs/vllm/bin/vllm
  else
    VLLM_BIN=$(command -v vllm || true)
  fi
fi

# 在等待训练前先检查环境，避免数小时后才暴露路径问题。
for cmd in "$PYTHON" "$SWIFT" curl; do
  command -v "$cmd" >/dev/null 2>&1 || { log "找不到命令: $cmd"; exit 1; }
done
if [[ "$RUN_EVAL" == 1 && ( -z "$VLLM_BIN" || ! -x "$VLLM_BIN" ) ]]; then
  log "找不到可执行的 vllm。请用 VLLM_BIN=/path/to/vllm 指定独立 serving 环境。"
  exit 1
fi
for file in "$MODEL_PATH/config.json" data/manifest.csv \
            sft/sft_short_train.jsonl sft/sft_short_val.jsonl \
            configs/prompts_t4.yaml configs/prompts_t0.yaml; do
  [[ -f "$file" ]] || { log "缺少文件: $file（请从 VLM 根目录启动）"; exit 1; }
done
log "环境检查通过；VLLM_BIN=${VLLM_BIN:-未启用评估}"

stop_server() {
  local pid=${SERVER_PID:-}
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    if [[ "$SERVER_GROUP" == 1 ]]; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 2
    done
    if kill -0 "$pid" 2>/dev/null; then
      if [[ "$SERVER_GROUP" == 1 ]]; then
        kill -KILL -- "-$pid" 2>/dev/null || true
      else
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
  fi
  wait "$pid" 2>/dev/null || true
  SERVER_PID=""
  SERVER_GROUP=0
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
  local model=$1
  local name=$2
  local logfile="logs/vllm_eval_${name}.log"
  local url="http://127.0.0.1:${PORT}/v1/models"
  if curl -fsS "$url" >/dev/null 2>&1; then
    log "端口 $PORT 已有服务，未动它；请先停止你自己的旧服务或改用其他 PORT。"
    return 1
  fi
  log "启动 $name，GPU=$GPU，日志=$logfile"
  if command -v setsid >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$GPU" VLLM_PLUGINS="" setsid "$VLLM_BIN" serve "$model" \
      --tensor-parallel-size 1 --max-model-len 8192 \
      --max-num-seqs "$CONCURRENCY" --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --mm-processor-kwargs '{"max_pixels":401408,"min_pixels":200704}' \
      --port "$PORT" > "$logfile" 2>&1 &
    SERVER_GROUP=1
  else
    CUDA_VISIBLE_DEVICES="$GPU" VLLM_PLUGINS="" "$VLLM_BIN" serve "$model" \
      --tensor-parallel-size 1 --max-model-len 8192 \
      --max-num-seqs "$CONCURRENCY" --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --limit-mm-per-prompt '{"image":1,"video":0}' \
      --mm-processor-kwargs '{"max_pixels":401408,"min_pixels":200704}' \
      --port "$PORT" > "$logfile" 2>&1 &
    SERVER_GROUP=0
  fi
  SERVER_PID=$!

  local waited=0
  while (( waited < STARTUP_TIMEOUT )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "$name 服务已就绪"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "$name 启动失败，请查看 $logfile"
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  log "$name 启动超时，请查看 $logfile"
  return 1
}

merge_lora() {
  local ckpt=$1 output=$2
  if [[ -f "$output/config.json" ]] && \
     { [[ -f "$output/model.safetensors" ]] || [[ -f "$output/model.safetensors.index.json" ]]; }; then
    log "合并权重已存在，跳过: $output"
    return 0
  fi
  if [[ -e "$output" ]]; then
    log "$output 已存在但不完整；为避免覆盖，流水线停止。请检查后移走该目录再重跑。"
    return 1
  fi
  log "合并 $ckpt -> $output"
  CUDA_VISIBLE_DEVICES="$GPU" "$SWIFT" export \
    --model "$MODEL_PATH" --model_type qwen3_vl \
    --adapters "$ckpt" --merge_lora true --output_dir "$output"
}

run_eval_only() {
  local model=$1 name=$2 prompts=$3 out=$4
  if [[ -f "$out" ]]; then
    local rows
    rows=$("$PYTHON" -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], encoding="utf-8"))))' "$out")
    if (( rows >= 9 )); then
      log "$name 已有完整 CSV（$rows 行），跳过；文件=$out"
      return 0
    fi
    if (( rows > 0 )); then
      log "$out 只有 $rows 行，直接重跑会产生重复记录。请先移走不完整 CSV 后再启动流水线。"
      return 1
    fi
  fi
  log "开始评估 $name -> $out"
  "$PYTHON" scripts/eval_vlm.py \
    --model "$model" --model-name "$name" --prompts "$prompts" \
    --out "$out" --concurrency "$CONCURRENCY" \
    --timing-subset "$TIMING_SUBSET" --limit "$LIMIT"
}

eval_with_server() {
  local model=$1 name=$2 out=$3
  start_server "$model" "$name"
  run_eval_only "$model" "$name" configs/prompts_t4.yaml "$out"
  stop_server
}

# 1) 等待当前 rationale 训练；pgrep 不占用 GPU。
if [[ "$WAIT_FOR_RATIONALE" == 1 ]] && pgrep -f "$WAIT_PATTERN" >/dev/null; then
  log "检测到 rationale 正在训练，每 60 秒检查一次，等待其结束。"
  while pgrep -f "$WAIT_PATTERN" >/dev/null; do sleep 60; done
fi
log "选择 rationale 最优 checkpoint（要求训练已完成）。"
RATIONALE_CKPT=$("$PYTHON" scripts/select_best_checkpoint.py "$RATIONALE_OUT")

# 2) short 使用与当前 rationale 相同的 GPU/BATCH/GA；已完成则自动跳过。
if [[ "$SKIP_SHORT" != 1 ]]; then
  if "$PYTHON" scripts/select_best_checkpoint.py "$SHORT_OUT" >/dev/null 2>&1; then
    log "检测到已完成的 short 训练，跳过训练阶段。"
  elif [[ -e "$SHORT_OUT" ]]; then
    log "$SHORT_OUT 已存在但没有完整训练结果；为避免重复训练，请先检查并恢复该训练。"
    exit 1
  else
    log "开始 short 训练：GPU=$GPU BATCH=$BATCH GA=$GA"
    GPU="$GPU" BATCH="$BATCH" GA="$GA" \
      DATA=sft/sft_short_train.jsonl VAL_DATA=sft/sft_short_val.jsonl \
      OUT="$SHORT_OUT" bash scripts/train_sft.sh
  fi
fi
SHORT_CKPT=$("$PYTHON" scripts/select_best_checkpoint.py "$SHORT_OUT")

# 3) 合并两个 LoRA；已完整合并则跳过。
merge_lora "$RATIONALE_CKPT" "$RATIONALE_MERGED"
merge_lora "$SHORT_CKPT" "$SHORT_MERGED"

# ms-swift 4.6 可能将 extra_special_tokens 写成 list，导致 vLLM 的
# Transformers 4.57 加载失败；仅在检测到该情况时修复并保留备份。
"$PYTHON" scripts/fix_merged_tokenizer.py \
  "$MODEL_PATH" "$RATIONALE_MERGED" "$SHORT_MERGED"

# 4) 主实验的 4 组评估；可选 frozen-32B 需双卡，不在此自动执行。
if [[ "$RUN_EVAL" == 1 ]]; then
  start_server "$MODEL_PATH" frozen-8B
  run_eval_only "$MODEL_PATH" frozen-8B-T4 configs/prompts_t4.yaml results/eval_frozen_8b_t4.csv
  run_eval_only "$MODEL_PATH" frozen-8B-T0 configs/prompts_t0.yaml results/eval_frozen_8b_t0.csv
  stop_server

  eval_with_server "$RATIONALE_MERGED" SFT-rationale-8B results/eval_sft_rationale.csv
  eval_with_server "$SHORT_MERGED" SFT-short-8B results/eval_sft_short.csv
fi

log "流水线完成。"
