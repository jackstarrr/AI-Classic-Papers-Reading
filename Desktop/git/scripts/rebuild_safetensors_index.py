"""从本地分片头部重建 model.safetensors.index.json。

只读取各分片的 safetensors 头部，不加载或修改权重；若索引已存在则不覆盖。
用法：python scripts/rebuild_safetensors_index.py /path/to/model_dir
"""
import argparse
import json
import struct
from pathlib import Path


def read_shard_header(path):
    """返回 {tensor_name: 字节数}；safetensors 前 8 字节记录 JSON 头长度。"""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"文件不是有效的 safetensors: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_len))

    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        if not (0 <= begin <= end):
            raise ValueError(f"{path} 中 {name} 的 data_offsets 无效")
        tensors[name] = end - begin
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="包含 model-xxxxx-of-xxxxx.safetensors 的目录")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    output = model_dir / "model.safetensors.index.json"
    if output.exists():
        print(f"索引已存在，未改动: {output}")
        return

    shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"没有找到分片权重: {model_dir}/model-*-of-*.safetensors")

    weight_map = {}
    total_size = 0
    for shard in shards:
        tensors = read_shard_header(shard)
        if not tensors:
            raise ValueError(f"分片中没有张量: {shard}")
        for name, nbytes in tensors.items():
            if name in weight_map:
                raise ValueError(f"张量 {name} 同时出现在 {weight_map[name]} 和 {shard.name}")
            weight_map[name] = shard.name
            total_size += nbytes
        print(f"读取 {shard.name}: {len(tensors)} 个张量")

    index = {"metadata": {"total_size": total_size},
             "weight_map": dict(sorted(weight_map.items()))}
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(f"已生成 {output}：{len(shards)} 个分片，{len(weight_map)} 个张量，"
          f"total_size={total_size}")


if __name__ == "__main__":
    main()
