#!/usr/bin/env bash
# 全量生成 SFT 数据（9 个类别循环；断点续跑安全，可反复执行）
# 用法:
#   bash scripts/run_gen_all.sh                          # 全量（train 2000/侧, val 250/侧）
#   bash scripts/run_gen_all.sh --limit 30               # 试点
#   MODEL=/mnt/data_sda/qwen3-vl-32B-instruct-FP8 bash scripts/run_gen_all.sh
set -e
MODEL=${MODEL:-/mnt/data_sda/qwen3-vl-32B-instruct-FP8}   # 与 curl /v1/models 查到的名字一致

for CAT in accident drug fire gambling horror political porn robbery weapon; do
  echo "===== $CAT ====="
  python scripts/gen_sft_data.py --categories "$CAT" --model "$MODEL" "$@"
done
