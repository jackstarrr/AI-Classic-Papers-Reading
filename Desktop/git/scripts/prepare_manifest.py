"""生成 manifest.csv（三列：image_path,label,split），后续所有脚本的唯一数据入口。

两种模式：
  from-dirs 数据集已按 root/<class>/<split>/ 组织（当前数据集布局，如
            /mnt/data_sda/xujie/BTCD_xujie/accident/train/...），直接汇总；
  resplit   数据集按类别分目录（root/<class>/*.jpg，无 split 子目录），
            本脚本做分层 8:1:1 划分（固定种子）。

只识别十个已知类别目录，root 下的其他目录/文件（如 BTCD/、*.py）自动忽略。

★ 重要：SFT 与论文 Table 2 的可比性依赖【完全相同的 test 集】。
  from-dirs 模式直接使用磁盘上已有划分，与历史实验天然一致。
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import ALL_CLASSES, IMG_EXTS


def scan_class_dir(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="数据集根目录，如 /mnt/data_sda/xujie/BTCD_xujie")
    ap.add_argument("--mode", choices=["from-dirs", "resplit"], default="from-dirs")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    if args.mode == "from-dirs":
        # 布局：root/<class>/<split>/图片
        for cls in ALL_CLASSES:
            for split in ("train", "val", "test"):
                d = root / cls / split
                if not d.is_dir():
                    print(f"[warn] 未找到目录: {d}，跳过")
                    continue
                imgs = scan_class_dir(d)
                for p in imgs:
                    rows.append((str(p), cls, split))
                print(f"{cls:10s} {split:5s}: {len(imgs)}")
    else:
        rng = random.Random(args.seed)
        for cls in ALL_CLASSES:
            d = root / cls
            if not d.is_dir():
                cands = [x for x in root.iterdir() if x.is_dir() and x.name.lower() == cls]
                if not cands:
                    print(f"[warn] 未找到类别目录: {cls}，跳过")
                    continue
                d = cands[0]
            imgs = scan_class_dir(d)
            rng.shuffle(imgs)
            n = len(imgs)
            n_train = int(n * args.train_ratio)
            n_val = int(n * args.val_ratio)
            for i, p in enumerate(imgs):
                split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
                rows.append((str(p), cls, split))
            print(f"{cls:10s} total={n}  train={n_train}  val={n_val}  "
                  f"test={n - n_train - n_val}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "split"])
        w.writerows(rows)
    print(f"\nmanifest 已写出: {out}（共 {len(rows)} 行）")
    print("★ 请抽查几行路径，确认与 CNN/VLM 历史实验用的是同一份划分。")


if __name__ == "__main__":
    main()
