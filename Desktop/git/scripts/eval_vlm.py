"""测试集评估：复现论文 Table 2 的二分类协议与指标。

每个类别：正例 = 该类 test 样本，负例 = normal test 样本。
  Pass 1  并发跑全部样本 → ACC / Precision / Recall / F1
          （解析失败自动重试 --parse-retries 次，最终失败计为判错并计入 parse_fail）；
  Pass 2  在 --timing-subset 个样本上【串行】重测平均延迟，
          与论文"sample-by-sample inference"的计时口径一致。

结果追加写 CSV（每个类别一行），并把原始预测存成 jsonl 供错误分析。
"""
import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (TOXIC_CLASSES, chat_with_image, load_prompts,
                        make_client, parse_verdict)


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def judge(client, args, image_path, prompt):
    """返回 (verdict, latency_ms, completion_tokens, parse_fail)。"""
    last_lat, last_tok = 0.0, 0
    for _ in range(1 + args.parse_retries):
        try:
            r = await chat_with_image(client, args.model, prompt, image_path,
                                      temperature=args.temperature,
                                      top_p=args.top_p,
                                      max_tokens=args.max_tokens)
        except Exception:  # noqa: BLE001
            continue
        last_lat, last_tok = r["latency_ms"], r["usage"].get("completion_tokens") or 0
        v = parse_verdict(r["text"])
        if v is not None:
            return v, last_lat, last_tok, False
    return None, last_lat, last_tok, True


def calc_metrics(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    acc = (tp + tn) / n * 100 if n else 0.0
    p = tp / (tp + fp) * 100 if tp + fp else 0.0
    r = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return acc, p, r, f1


async def eval_category(cat, args, client, prompts, rows, sem, pred_dir):
    prompt = prompts[cat]  # 与数据生成/训练一致的逐类别 prompt 原文
    pos = [r["image_path"] for r in rows if r["label"] == cat and r["split"] == "test"]
    neg = [r["image_path"] for r in rows if r["label"] == "normal" and r["split"] == "test"]
    if args.limit:
        pos, neg = pos[:args.limit], neg[:args.limit]

    async def run(items, gt):
        async def one(p):
            async with sem:
                v, lat, tok, pf = await judge(client, args, p, prompt)
                return {"path": p, "gt": gt, "pred": v, "lat": lat,
                        "tok": tok, "pf": pf}
        out, chunk = [], args.concurrency * 2
        for i in range(0, len(items), chunk):
            out.extend(await asyncio.gather(*(one(p) for p in items[i:i + chunk])))
            print(f"  [{cat}] pass1 {min(i + chunk, len(items))}/{len(items)}",
                  flush=True)
        return out

    res = await run(pos, "YES") + await run(neg, "NO")
    tp = sum(1 for r in res if r["gt"] == "YES" and r["pred"] == "YES")
    fn = sum(1 for r in res if r["gt"] == "YES" and r["pred"] != "YES")
    tn = sum(1 for r in res if r["gt"] == "NO" and r["pred"] == "NO")
    fp = sum(1 for r in res if r["gt"] == "NO" and r["pred"] != "NO")
    parse_fail = sum(1 for r in res if r["pf"])
    acc, p, r_, f1 = calc_metrics(tp, fp, tn, fn)
    avg_tok = sum(r["tok"] for r in res) / max(1, len(res))

    # ---- Pass 2：串行计时子集（此时服务器无并发负载，接近论文口径）----
    k = args.timing_subset
    tsub = pos[:k // 2] + neg[:k - k // 2]
    lats = []
    for p_ in tsub:
        _, lat, _, _ = await judge(client, args, p_, prompt)
        lats.append(lat)
    avg_t = sum(lats) / max(1, len(lats))

    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / f"preds_{cat}.jsonl", "w", encoding="utf-8") as f:
        for r0 in res:
            f.write(json.dumps(r0, ensure_ascii=False) + "\n")

    return {"model": args.model_name, "category": cat,
            "n_pos": len(pos), "n_neg": len(neg),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "ACC": round(acc, 2), "Precision": round(p, 2),
            "Recall": round(r_, 2), "F1": round(f1, 2),
            "avg_time_ms": round(avg_t, 1),
            "avg_completion_tokens": round(avg_tok, 1),
            "parse_fail": parse_fail}


async def amain(args):
    cats = TOXIC_CLASSES if args.categories == "all" else \
        [c.strip().lower() for c in args.categories.split(",")]
    rows = load_manifest(args.manifest)
    prompts = load_prompts(args.prompts)
    missing = [c for c in cats if c not in prompts]
    assert not missing, f"{args.prompts} 缺少类别 prompt: {missing}"
    client = make_client(args.api_base)
    sem = asyncio.Semaphore(args.concurrency)
    pred_dir = Path(args.pred_dir) / args.model_name

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    results = []
    for cat in cats:
        print(f"===== 评估类别: {cat} =====", flush=True)
        row = await eval_category(cat, args, client, prompts, rows, sem, pred_dir)
        results.append(row)
        print(f"  -> ACC={row['ACC']}  F1={row['F1']}  "
              f"time={row['avg_time_ms']}ms  parse_fail={row['parse_fail']}",
              flush=True)
        with out_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(row)

    if results:
        for m in ("ACC", "Precision", "Recall", "F1", "avg_time_ms"):
            avg = sum(r[m] for r in results) / len(results)
            print(f"宏平均 {m}: {avg:.2f}")
    print(f"结果已追加写入: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--categories", default="all")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="vLLM 服务的模型名/路径")
    ap.add_argument("--model-name", default=None,
                    help="写入 CSV 的模型标识，默认取 --model 末段")
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--parse-retries", type=int, default=2)
    ap.add_argument("--timing-subset", type=int, default=200,
                    help="串行计时子集大小（pos/neg 各半）")
    ap.add_argument("--limit", type=int, default=0, help="试点：每类每侧最多 N 条")
    ap.add_argument("--out", default="results/eval.csv")
    ap.add_argument("--pred-dir", default="results/preds")
    ap.add_argument("--prompts", default="configs/prompts_t4.yaml",
                    help="逐类别 prompt 配置；对照实验可换 configs/prompts_t0.yaml")
    args = ap.parse_args()
    if args.model_name is None:
        args.model_name = args.model.rstrip("/").split("/")[-1]
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
