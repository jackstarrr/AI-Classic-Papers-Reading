# 方向一：SFT 微调 VLM Refiner —— 代码包说明

目标：把 SynerTCD 中"冻结 VLM + 手工 prompt"的第二阶段，替换为 **LoRA-SFT 的 Qwen3-VL-8B**，
与 prompt 方案做严格对照（**同数据划分、同 prompt 原文、同测试集、同指标**）。

数据构造：服务器上的 **Qwen3-VL-32B（FP8）** 在 train/val 集上做 rejection sampling
（STaR 式自蒸馏），保留判定正确的"推理 + YES/NO"作为 8B 的 SFT 目标。
**test 集全程不参与生成与训练。**

## 目录结构

```
configs/
  prompts_t4.yaml           # 论文最终版 prompt（T4=角色+判定要素+保守偏置）逐类别原文，
                            #   提取自 run_fastvlm_all.py 的 PROMPTS_T4_ROLE_ELEMENTS_SAFETY
  prompts_t0.yaml           # 论文 T1（仅类别名提问），对照实验的 prompt 下界
  prompt_rationalize.txt    # STaR rationalize 前缀（拼在原始 prompt 之前）
scripts/
  vlm_common.py             # 公共工具（API 调用、verdict 解析、prompt 加载）
  prepare_manifest.py       # 第 0 步：生成数据清单 manifest.csv
  gen_sft_data.py           # 第 1 步：32B rejection sampling（断点续跑）
  build_sft_dataset.py      # 第 2 步：汇总为 ms-swift SFT 数据集
  train_sft.sh              # 第 3 步：LoRA SFT（含 merge 说明）
  serve.sh                  # vLLM 部署参考命令（已填真实路径）
  eval_vlm.py               # 第 4 步：测试集评估（Table 2 口径）
  run_gen_all.sh            # 一键循环 9 类生成
requirements-server.txt
```

**输出契约**：论文 prompt 要求"最后一行只含 YES 或 NO"（无 `Decision:` 前缀）。
解析、SFT 目标格式均已按此实现：rationale 版 assistant = 推理 + 尾行裸 `YES`/`NO`；
short 版 assistant = 仅 `YES`/`NO`。

## 服务器信息（已写入脚本默认值）

- 数据集根目录：`/mnt/data_sda/xujie/BTCD_xujie`，布局 `<类别>/<split>/`（BTCD/ 目录与 .py 文件自动忽略）
- 8B 权重：`/mnt/data_sda/qwenvl-8B-instruct`
- 32B 权重：`/mnt/data_sda/qwen3-vl-32B-instruct-FP8`（FP8 仅用于离线数据生成，不进最终系统）

## 第 0 步：环境 + 数据清单

```bash
conda create -n sft python=3.10 -y && conda activate sft
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "ms-swift[llm]>=3.8" qwen-vl-utils openai pyyaml tqdm

# 直接使用磁盘上已有的 train/val/test 划分（与历史实验一致）
python scripts/prepare_manifest.py --root /mnt/data_sda/xujie/BTCD_xujie --mode from-dirs
```

生成后抽查 `data/manifest.csv` 几行路径。若发现与 CNN/论文实验的划分不一致，
用原划分名单替换该文件（三列：image_path,label,split）。

## 第 1 步：生成 SFT 数据（32B FP8，vLLM OpenAI 接口）

```bash
# 启动 32B 服务（serve.sh 第 1 段），确认模型名
curl http://127.0.0.1:8000/v1/models

# 试点：每类每侧 30 条，检查接受率、输出格式、耗时（约 10-20 分钟）
bash scripts/run_gen_all.sh --limit 30

# 全量：默认每类 train 2000正+2000负、val 250正+250负（9 类共约 36k+4.5k 条）
bash scripts/run_gen_all.sh
```

- 预计 4-8 小时。**中断后重跑同一命令即可续跑**；提高 `--target` 再跑可无缝扩量。
- 服务端用 `--mm-processor-kwargs '{"max_pixels":401408,"min_pixels":200704}'`
  封顶图像 token（与训练一致）；不封顶大图会拖慢 prefill 数倍，且与训练分辨率不一致。
- 若试点实测仍 >20s/条，可把 `--target` 降到 1200-1500（LoRA SFT 每类 2.4k-3k 条足够）。
- 输出 `gen/gen_<category>.jsonl`，每条含 prompt/response/parsed/accepted/latency 完整信息。
- 首判错误的样本自动用 rationalize 前缀补量（STaR）；消融可用 `--no-rationalize`
  与 build 脚本的 `--exclude-rationalized`。

## 第 2 步：构建 SFT 数据集

```bash
python scripts/build_sft_dataset.py
# 产出 sft/ 下：统一模型版(rationale/short × train/val)、per_class/ 单类版、stats.csv
# rationalized 记录里的 GT 提示前缀只用于教师生成；build 会自动恢复为原始 T4 prompt，
# 不会把“已知答案”写进 8B 的训练输入。
```

## 第 3 步：训练 + 合并

```bash
# rationale 版（路径已填好；GPU=1 表示使用第 2 张卡）
GPU=1 bash scripts/train_sft.sh

# 若可用显存只有约 24G，可减小单步 batch；有效 batch 仍为 32
GPU=1 BATCH=1 GA=32 bash scripts/train_sft.sh

# short 版（延迟变体）
DATA=sft/sft_short_train.jsonl VAL_DATA=sft/sft_short_val.jsonl \
OUT=ckpt/qwen3vl8b_lora_short bash scripts/train_sft.sh

# 合并 LoRA（训练完按脚本末尾注释执行）
swift export --model /mnt/data_sda/qwenvl-8B-instruct \
  --model_type qwen3_vl \
  --adapters ckpt/qwen3vl8b_lora_rationale/<best_ckpt> \
  --merge_lora true --output_dir merged/qwen3vl8b_sft_rationale
```

36k 条、LoRA r32、单卡 5090，预计 3-6 小时/版。注意 32B（TP=2）要占满双卡，
所以数据生成全部完成后再开始训练；训练（GPU0）期间可以用 GPU1 起 8B 服务做评估
（`bash scripts/serve.sh sft8b`，端口 8001）。

### 自动接续训练与评估（可选）

当前 rationale 仍在训练时即可启动守候流水线；它默认沿用 `GPU=1 BATCH=1 GA=32`，
等待 rationale 完成后依次执行：short 训练 → 按验证集 loss 选 checkpoint → 合并两个
LoRA → frozen-8B-T4/T0 与 SFT-rationale/short 四组评估。只停止本脚本启动的服务。

```bash
mkdir -p logs
nohup bash scripts/run_remaining_pipeline.sh > logs/remaining_pipeline.log 2>&1 &
# 查看进度：tail -f logs/remaining_pipeline.log

# 若 ckpt/merged 已迁移到大磁盘（例如 /mnt/data_sda/xujie）：
CKPT_ROOT=/mnt/data_sda/xujie/ckpt MERGED_ROOT=/mnt/data_sda/xujie/merged \
nohup bash scripts/run_remaining_pipeline.sh > logs/remaining_pipeline_eval.log 2>&1 &
```

已完成的 short 训练、合并权重和 9 类完整评估 CSV 会自动跳过。显存不足导致 vLLM
无法启动时，可在别人释放显存后重跑；也可用 `GPU_MEM_UTIL=0.7` 降低 KV cache 预留。
可选 frozen-32B 对照需双卡，不包含在自动流水线中。

## 第 4 步：评估（产出 Table 2 格式的 CSV）

```bash
# 先 serve 被评模型（serve.sh 第 2/3 段），然后：

# SFT 模型
python scripts/eval_vlm.py --model merged/qwen3vl8b_sft_rationale \
  --model-name SFT-rationale-8B --out results/eval_sft_rationale.csv

# ★ 对照组 1（必须）：冻结 base 8B + 论文 T4 原文（重新跑原始提示词作为基础）
python scripts/eval_vlm.py --model /mnt/data_sda/qwenvl-8B-instruct \
  --model-name frozen-8B-T4 --out results/eval_frozen_8b_t4.csv

# 对照组 2（建议）：冻结 base 8B + T0（论文 T1，prompt 下界）
python scripts/eval_vlm.py --model /mnt/data_sda/qwenvl-8B-instruct \
  --model-name frozen-8B-T0 --prompts configs/prompts_t0.yaml \
  --out results/eval_frozen_8b_t0.csv

# 对照组 3（可选）：冻结 32B + T4（验证蒸馏差距；需先启动 32B 服务）
python scripts/eval_vlm.py --model /mnt/data_sda/qwen3-vl-32B-instruct-FP8 \
  --model-name frozen-32B-T4 --out results/eval_frozen_32b_t4.csv
```

评估协议与论文一致：逐样本串行计时子集测延迟；ACC/P/R/F1 按二分类混淆矩阵；
原始预测存 `results/preds/<model-name>/` 供错误分析。

## 论文实验矩阵（建议）

| # | 模型 | prompt / 适配 | 说明 |
|---|------|--------------|------|
| 1 | 冻结 8B | T0（configs/prompts_t0.yaml）| prompt 下界 |
| 2 | 冻结 8B | T4（configs/prompts_t4.yaml）| 论文方案重跑，主对照 |
| 3 | 冻结 32B | T4 | 蒸馏上界参考 |
| 4 | SFT-8B | 同 T4（训练已内化）| 主结果：rationale 版 |
| 5 | SFT-8B | 同 T4 | short 版（延迟变体）|
| 6 | 同 4 | 去掉 rationalized 样本 | 消融 |
| 7 | 同 4 | per_class 单类专家 | 消融：统一 vs 专家 |
| 8 | 同 4 | freeze_vit=true | 消融：LoRA 作用范围 |
| 9 | SFT-8B 塞回 SynerTCD 级联 | — | 重跑路由，看 VCR/延迟/F1 变化 |

## 注意事项

1. **prompt 一致性**：gen、训练、eval 全部使用 `configs/prompts_t4.yaml` 的同一份原文，
   中途改动会让对照失效（改了就必须清空 gen/ 重来）。
2. **T4 与论文热图**：本流程的 T4 = 脚本 `PROMPTS_T4_ROLE_ELEMENTS_SAFETY`
   （角色+要素+保守偏置），不含 CNN LayerCAM 热图位置描述；若论文的完整版还含热图
   组件，对照组重跑数字可能与论文 Table 2 有小幅差异，属预期（实现细节里注明即可）。
3. **数据泄漏**：gen 只读 train/val；test 只出现在 eval。manifest 错了全部重来，先核对。
4. **rationalized 样本**可能含"凑答案"的伪推理；`sft/stats.csv` 的各类接受率可做附录
   分析，political/horror 接受率预计最低，正好印证这两类最难。
5. **解析口径**：输出契约是裸 YES/NO 尾行；解析取全文最后一个全大写 YES/NO 整词，
   失败计入 parse_fail（eval CSV 里有该列，正常应在千分级以下）。
6. 耗时估计（2×5090）：数据生成 4-8h，训练 3-6h/版，评估 ~1-2h/模型（全量 9 类）。
