"""SFT 数据生成：调用服务器上部署的 Qwen3-VL-32B（vLLM OpenAI 接口）做 rejection sampling。

流程（每个类别、每个 split、每一侧 pos/neg）：
  1. 从 manifest 取确定性采样池（target * oversample），低温（默认 0.02）生成
     "逐步分析 + Decision: YES/NO"，解析 verdict 与 GT 一致 → accepted；
  2. 被拒样本用 rationalize 模板（告知 GT）在稍高温度重生成（STaR 式），通过 →
     accepted(source=rationalized)；
  3. 结果逐行追加写 <out-dir>/gen_<category>.jsonl；中断后重跑自动跳过已完成样本，
     采样池种子固定，提高 --target 后重跑可无缝扩量。

用法见 README.md 与 run_gen_all.sh。
"""
import argparse
import asyncio
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (TOXIC_CLASSES, chat_with_image, iter_jsonl, load_prompts,
                        load_template, make_client, parse_verdict, stable_seed)

_SPLIT_OFF = {"train": 0, "val": 1, "test": 2}


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_pool(rows, category, split, target, oversample, limit):
    """返回 (pos_pool, neg_pool)；种子固定 => 多次运行池顺序一致。"""
    rng = random.Random(stable_seed(category) + _SPLIT_OFF[split])
    pos = [r["image_path"] for r in rows
           if r["label"] == category and r["split"] == split]
    neg = [r["image_path"] for r in rows
           if r["label"] == "normal" and r["split"] == split]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if limit:
        return pos[:limit], neg[:limit]
    n = max(1, int(target * oversample))
    return pos[:n], neg[:n]


def load_state(out_path):
    """从已有生成文件恢复：done 键、接受计数、被拒样本列表。"""
    done, counters, rejected = set(), defaultdict(int), []
    for r in iter_jsonl(out_path):
        if r.get("status") != "ok":
            continue  # error 记录不标记完成，重跑时会再试
        done.add((r["image_path"], r["split"], r.get("source", "sampled")))
        if r.get("accepted"):
            counters[(r["split"], r["side"])] += 1
        elif r.get("source", "sampled") == "sampled":
            rejected.append({"image_path": r["image_path"],
                             "split": r["split"], "side": r["side"]})
    return {"done": done, "counters": counters, "rejected": rejected}


def make_record(image_path, cat, side, split, prompt, resp, gt_verdict,
                source, model, temperature, err=None):
    rec = {"image_path": image_path, "category": cat, "side": side, "split": split,
           "gt_verdict": gt_verdict, "source": source, "model": model,
           "temperature": temperature, "prompt": prompt, "ts": int(time.time())}
    if err is not None:
        rec.update(status="error", error=str(err)[:500], accepted=False,
                   parsed=None, response=None, latency_ms=None, completion_tokens=None)
        return rec
    verdict = parse_verdict(resp["text"])
    rec.update(status="ok", error=None, response=resp["text"], parsed=verdict,
               accepted=(verdict == gt_verdict),
               latency_ms=resp["latency_ms"],
               completion_tokens=resp["usage"].get("completion_tokens"))
    return rec


async def fill_side(client, args, cat, split, side, pool, gt_verdict, prompt,
                    source, target, state, fout, temperature):
    """按 pool 顺序处理，直到该侧 accepted 数达到 target（或池耗尽）。"""
    done, counters = state["done"], state["counters"]
    key = (split, side)
    sem = asyncio.Semaphore(args.concurrency)
    rejected, cursor = [], 0
    chunk = args.concurrency * 2

    async def bound(p):
        async with sem:
            try:
                resp = await chat_with_image(client, args.model, prompt, p,
                                             temperature=temperature,
                                             top_p=args.top_p,
                                             max_tokens=args.max_tokens)
                return resp, None
            except Exception as e:  # noqa: BLE001
                return None, e

    while counters[key] < target and cursor < len(pool):
        batch = []
        while len(batch) < chunk and cursor < len(pool):
            p = pool[cursor]
            cursor += 1
            if (p, split, source) not in done:
                batch.append(p)
        if not batch:
            break
        results = await asyncio.gather(*(bound(p) for p in batch))
        for p, (resp, err) in zip(batch, results):
            rec = make_record(p, cat, side, split, prompt, resp, gt_verdict,
                              source, args.model, temperature, err=err)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["status"] == "ok":
                done.add((p, split, source))
                if rec["accepted"]:
                    counters[key] += 1
                elif source == "sampled":
                    rejected.append({"image_path": p, "split": split, "side": side})
        fout.flush()
        print(f"  [{cat}|{split}|{side}|{source}] "
              f"accepted={counters[key]}/{target}  pool={cursor}/{len(pool)}",
              flush=True)
    return rejected


async def run_category(cat, args, client, prompts, rat_prefix, rows):
    out_path = Path(args.out_dir) / f"gen_{cat}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(out_path)
    prompt = prompts[cat]  # 论文冻结的逐类别 T4 原文
    splits = [("train", args.target), ("val", args.val_target)]
    t0 = time.time()

    with out_path.open("a", encoding="utf-8") as fout:
        # ---- 第一遍：正常采样 ----
        for split, tgt in splits:
            if args.limit:
                tgt = args.limit
            pos_pool, neg_pool = sample_pool(rows, cat, split, tgt,
                                             args.oversample, args.limit)
            for side, pool, gt in (("pos", pos_pool, "YES"), ("neg", neg_pool, "NO")):
                rej = await fill_side(client, args, cat, split, side, pool, gt,
                                      prompt, "sampled", tgt, state, fout,
                                      args.temperature)
                state["rejected"].extend(rej)

        # ---- 第二遍：rationalize 补量（STaR，前缀告知 GT + 原始 prompt）----
        if not args.no_rationalize:
            for split, tgt in splits:
                if args.limit:
                    tgt = args.limit
                for side, gt in (("pos", "YES"), ("neg", "NO")):
                    need = tgt - state["counters"][(split, side)]
                    if need <= 0:
                        continue
                    cands = [r["image_path"] for r in state["rejected"]
                             if r["split"] == split and r["side"] == side
                             and (r["image_path"], split, "rationalized")
                             not in state["done"]]
                    if not cands:
                        continue
                    pool = cands[:int(need * args.oversample) + 1]
                    rat_prompt = rat_prefix.replace("{gt_answer}", gt) + "\n\n" + prompt
                    print(f"  [{cat}|{split}|{side}] rationalize 补量 {len(pool)} 条",
                          flush=True)
                    await fill_side(client, args, cat, split, side, pool, gt,
                                    rat_prompt, "rationalized", tgt, state, fout,
                                    args.rationalize_temperature)

    c = state["counters"]
    print(f"[{cat}] 完成  train pos/neg = {c[('train','pos')]}/{c[('train','neg')]}  "
          f"val pos/neg = {c[('val','pos')]}/{c[('val','neg')]}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


async def amain(args):
    cats = TOXIC_CLASSES if args.categories == "all" else \
        [c.strip().lower() for c in args.categories.split(",")]
    for c in cats:
        assert c in TOXIC_CLASSES, f"未知类别: {c}"
    rows = load_manifest(args.manifest)
    prompts = load_prompts(args.prompts)
    missing = [c for c in cats if c not in prompts]
    assert not missing, f"{args.prompts} 缺少类别 prompt: {missing}"
    rat_prefix = load_template(args.rationalize_template)
    client = make_client(args.api_base)
    t0 = time.time()
    for cat in cats:
        print(f"===== 生成类别: {cat} =====", flush=True)
        await run_category(cat, args, client, prompts, rat_prefix, rows)
    print(f"全部完成，总耗时 {(time.time()-t0)/3600:.2f} h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--categories", default="all", help="逗号分隔，或 all")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True,
                    help="vLLM 服务的模型名（curl /v1/models 可查），如 Qwen3-VL-32B-Instruct")
    ap.add_argument("--target", type=int, default=2000,
                    help="train 集每类每侧(pos/neg)接受目标数")
    ap.add_argument("--val-target", type=int, default=250)
    ap.add_argument("--oversample", type=float, default=1.4,
                    help="采样池 = target * oversample，用于抵消拒样损失")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--rationalize-temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="rationale 实测 100-250 token，768 余量充足；过小会截断导致解析失败")
    ap.add_argument("--no-rationalize", action="store_true")
    ap.add_argument("--out-dir", default="gen")
    ap.add_argument("--prompts", default="configs/prompts_t4.yaml",
                    help="逐类别 prompt 配置（论文冻结原文）")
    ap.add_argument("--rationalize-template", default="configs/prompt_rationalize.txt")
    ap.add_argument("--limit", type=int, default=0,
                    help="试点模式：每类每侧最多 N 条（忽略 target/oversample）")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
