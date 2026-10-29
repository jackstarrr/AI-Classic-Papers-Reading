"""修复 ms-swift 合并模型与旧版 Transformers 的 tokenizer 配置兼容性。

仅当 merged/tokenizer_config.json 的 extra_special_tokens 是 list 时，
将其替换为基础模型中的 dict 值；修改前备份，其他字段不变。
用法：python scripts/fix_merged_tokenizer.py BASE_MODEL_DIR MERGED_DIR [MERGED_DIR ...]
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_model")
    ap.add_argument("merged_dirs", nargs="+")
    args = ap.parse_args()

    base_path = Path(args.base_model) / "tokenizer_config.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_extra = base.get("extra_special_tokens", {})
    if not isinstance(base_extra, dict):
        raise TypeError(f"基础模型的 extra_special_tokens 不是 dict: {base_path}")

    for directory in args.merged_dirs:
        path = Path(directory) / "tokenizer_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        value = config.get("extra_special_tokens")
        print(f"{path}: extra_special_tokens 当前类型为 {type(value).__name__}")
        if not isinstance(value, list):
            continue

        backup = path.with_name(path.name + ".before_vllm_fix")
        if not backup.exists():
            shutil.copy2(path, backup)
        config["extra_special_tokens"] = base_extra
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
        print(f"  已恢复为基础模型的 dict 配置；原文件备份为 {backup}")


if __name__ == "__main__":
    main()
