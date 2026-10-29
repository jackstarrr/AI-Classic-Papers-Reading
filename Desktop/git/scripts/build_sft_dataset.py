"""把 gen/*.jsonl 汇总成 ms-swift SFT 数据集。

输出（--out-dir 下）：
  sft_rationale_{train,val}.jsonl      统一模型版：9 类混合，assistant = 推理 + 尾行 YES/NO
  sft_short_{train,val}.jsonl          短答案版：assistant = 仅 YES/NO（低延迟变体）
  per_class/<cat>_{rationale,short}_{train,val}.jsonl   单类专家版（消融用）
  stats.csv                            各类接受率统计（论文附录可用）

ms-swift 数据格式：{"messages": [...], "images": ["绝对路径"]}
assistant 输出契约与论文 prompt 一致：最后一行只含 YES 或 NO（无 Decision: 前缀）。
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import TOXIC_CLASSES, iter_jsonl, load_prompts

_BARE_TAIL = re.compile(r"^\*{0,2}(YES|NO)\*{0,2}[.!]?$")


def normalize_rationale(text: str, verdict: str) -> str:
    """保证 assistant 目标以裸 YES/NO 尾行结束（与 prompt 输出契约一致）。"""
    t = (text or "").rstrip()
    lines = t.splitlines()
    if lines and _BARE_TAIL.match(lines[-1].strip()):
        lines[-1] = verdict
        return "\n".join(lines)
    return (t + "\n" + verdict) if t else verdict


def to_row(prompt, response, image_path):
    return {"messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": response}],
            "images": [image_path]}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="gen")
    ap.add_argument("--out-dir", default="sft")
    ap.add_argument("--prompts", default="configs/prompts_t4.yaml",
                    help="必须与生成时一致；rationalized 样本也只保留其中的原始 prompt")
    ap.add_argument("--target", type=int, default=2000,
                    help="train 每类每侧最多取多少条（0=不截断）")
    ap.add_argument("--val-target", type=int, default=250)
    ap.add_argument("--exclude-rationalized", action="store_true",
                    help="只用第一遍采样接受的样本（消融用）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "per_class").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    prompts = load_prompts(args.prompts)
    unified = {"rationale": {"train": [], "val": []},
               "short": {"train": [], "val": []}}
    stats_rows = []

    for cat in TOXIC_CLASSES:
        assert cat in prompts, f"{args.prompts} 缺少类别 prompt: {cat}"
        f = Path(args.gen_dir) / f"gen_{cat}.jsonl"
        if not f.exists():
            print(f"[warn] 缺少 {f}，跳过")
            continue
        recs = [r for r in iter_jsonl(f) if r.get("status") == "ok"]
        sampled = [r for r in recs if r.get("source", "sampled") == "sampled"]
        rat = [r for r in recs if r.get("source") == "rationalized"]
        accepted = [r for r in recs if r.get("accepted")]
        if args.exclude_rationalized:
            accepted = [r for r in accepted if r.get("source", "sampled") == "sampled"]

        buckets = defaultdict(list)
        for r in accepted:
            buckets[(r["split"], r["side"])].append(r)

        per_class = {"rationale": {"train": [], "val": []},
                     "short": {"train": [], "val": []}}
        used = defaultdict(int)
        for (split, side), rs in sorted(buckets.items()):
            tgt = args.val_target if split == "val" else args.target
            if tgt > 0:
                rs = rs[:tgt]
            for r in rs:
                rat_text = normalize_rationale(r["response"], r["parsed"])
                # rationalized 记录的生成 prompt 含 GT 提示前缀，只用于引导教师生成；
                # SFT 输入必须恢复为原始 T4 prompt，避免答案泄漏并保持训练/评估一致。
                per_class["rationale"][split].append(
                    to_row(prompts[cat], rat_text, r["image_path"]))
                per_class["short"][split].append(
                    to_row(prompts[cat], r["parsed"], r["image_path"]))
                used[(split, side)] += 1

        for variant in ("rationale", "short"):
            for split in ("train", "val"):
                rows_v = per_class[variant][split]
                rng.shuffle(rows_v)
                write_jsonl(out / "per_class" / f"{cat}_{variant}_{split}.jsonl", rows_v)
                unified[variant][split].extend(rows_v)

        att_s = len(sampled)
        acc_s = sum(1 for r in sampled if r.get("accepted"))
        stats_rows.append({
            "category": cat,
            "attempted_sampled": att_s,
            "accepted_sampled": acc_s,
            "accept_rate": f"{acc_s / att_s * 100:.1f}" if att_s else "NA",
            "rationalized_accepted": sum(1 for r in rat if r.get("accepted")),
            "train_pos": used[("train", "pos")], "train_neg": used[("train", "neg")],
            "val_pos": used[("val", "pos")], "val_neg": used[("val", "neg")],
        })
        print(f"{cat:10s} 接受率 {stats_rows[-1]['accept_rate']}%  "
              f"train pos/neg={used[('train','pos')]}/{used[('train','neg')]}  "
              f"val pos/neg={used[('val','pos')]}/{used[('val','neg')]}")

    for variant in ("rationale", "short"):
        for split in ("train", "val"):
            rows_v = unified[variant][split]
            rng.shuffle(rows_v)
            path = out / f"sft_{variant}_{split}.jsonl"
            write_jsonl(path, rows_v)
            print(f"写出 {path}  ({len(rows_v)} 条)")

    with open(out / "stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"统计已写出: {out / 'stats.csv'}")


if __name__ == "__main__":
    main()
