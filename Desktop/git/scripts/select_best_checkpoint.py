"""选择 ms-swift 训练中验证集 loss 最低的 LoRA checkpoint。

默认输出最近一个已完成训练中的最佳 checkpoint；没有 eval_loss 时取最后一步。
用法：python scripts/select_best_checkpoint.py ckpt/qwen3vl8b_lora_rationale
"""
import argparse
import json
import re
import sys
from pathlib import Path

_CKPT_RE = re.compile(r"checkpoint-(\d+)$")


def load_state(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def checkpoints(run_dir):
    out = []
    for p in run_dir.glob("checkpoint-*"):
        m = _CKPT_RE.fullmatch(p.name)
        if not p.is_dir() or not m or not (p / "adapter_config.json").exists():
            continue
        if not any((p / name).exists() for name in
                   ("adapter_model.safetensors", "adapter_model.bin")):
            continue
        out.append((int(m.group(1)), p))
    return sorted(out)


def is_complete(run_dir, ckpts):
    states = [load_state(run_dir / "trainer_state.json")]
    if ckpts:
        states.append(load_state(ckpts[-1][1] / "trainer_state.json"))
    for state in states:
        if not state:
            continue
        history = state.get("log_history", [])
        if any("train_runtime" in entry for entry in history):
            return True
        step, max_steps = state.get("global_step", 0), state.get("max_steps", 0)
        if max_steps > 0 and step >= max_steps:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", help="训练 OUT 目录或其下的 v0-* 目录")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="允许从异常中断的训练中选择 checkpoint（默认不允许）")
    args = ap.parse_args()

    root = Path(args.run_root)
    candidates = [root] + sorted(p for p in root.glob("v*") if p.is_dir())
    usable = []
    for run_dir in candidates:
        ckpts = checkpoints(run_dir)
        if ckpts and (args.allow_incomplete or is_complete(run_dir, ckpts)):
            usable.append((max(p.stat().st_mtime for _, p in ckpts), run_dir, ckpts))
    if not usable:
        raise SystemExit(f"没有找到已完成且含 LoRA 权重的训练: {root}\n"
                         "若训练异常中断，请先恢复训练；确认要使用中断权重时再加 --allow-incomplete。")

    _, run_dir, ckpts = max(usable, key=lambda x: x[0])
    scored = []
    for step, ckpt in ckpts:
        state = load_state(ckpt / "trainer_state.json") or {}
        losses = [h["eval_loss"] for h in state.get("log_history", [])
                  if isinstance(h.get("eval_loss"), (int, float))]
        if losses:
            scored.append((losses[-1], step, ckpt))

    if scored:
        loss, step, best = min(scored, key=lambda x: (x[0], -x[1]))
        print(f"选择 {best}（eval_loss={loss:.6f}）", file=sys.stderr)
    else:
        step, best = ckpts[-1]
        print(f"没有 eval_loss，选择最后 checkpoint: {best}", file=sys.stderr)
    print(best.resolve())


if __name__ == "__main__":
    main()
