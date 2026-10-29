"""公共工具：vLLM OpenAI 兼容接口调用、图片编码、verdict 解析、prompt 构建。"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import zlib
from pathlib import Path

import yaml
from openai import AsyncOpenAI

# 十个类别（九类有害 + normal），与论文口径一致
TOXIC_CLASSES = ["accident", "drug", "fire", "gambling", "horror",
                 "political", "porn", "robbery", "weapon"]
ALL_CLASSES = ["normal"] + TOXIC_CLASSES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".bmp": "image/bmp"}


def stable_seed(category: str, base: int = 20261023) -> int:
    """按类别名生成确定性种子，保证多次运行采样池一致（断点续跑/扩量安全）。"""
    return base + zlib.crc32(category.encode("utf-8"))


def load_template(path) -> str:
    """读取文本模板（如 rationalize 前缀）；以 '##' 开头的行为注释，不参与发送。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("##")).strip()


def load_prompts(path) -> dict:
    """读取按类别的 prompt 配置（yaml: category -> prompt 原文，论文冻结版本）。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {str(k).lower(): str(v).strip() for k, v in data.items()}


def encode_image(path) -> str:
    path = Path(path)
    mime = _MIME.get(path.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def make_client(api_base: str, timeout: float = 300.0) -> AsyncOpenAI:
    # vLLM 本地服务 api_key 任意非空字符串即可
    return AsyncOpenAI(base_url=api_base, api_key="EMPTY",
                       timeout=timeout, max_retries=0)


async def chat_with_image(client: AsyncOpenAI, model: str, prompt: str,
                          image_path, temperature: float = 0.02,
                          top_p: float = 0.9, max_tokens: int = 1024,
                          retries: int = 3) -> dict:
    """发送一条图文请求，带指数退避重试。成功返回 {text, latency_ms, usage}，最终失败抛异常。"""
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
        {"type": "text", "text": prompt},
    ]}]
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = await client.chat.completions.create(
                model=model, messages=msg, temperature=temperature,
                top_p=top_p, max_tokens=max_tokens)
            dt = (time.perf_counter() - t0) * 1000
            usage = getattr(resp, "usage", None)
            return {
                "text": resp.choices[0].message.content or "",
                "latency_ms": round(dt, 1),
                "usage": (usage.model_dump() if usage is not None else {}),
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(min(2 ** attempt * 2, 20))
    raise last_err


# 论文 prompt 的输出契约：分析后"最后一行只含 YES 或 NO"（无 Decision: 前缀）。
# 解析大小写敏感：句子里的普通单词 "no"/"No"（如 "no clear evidence"）不得误判，
# 宁可计入 parse_fail 重试也不要静默误解析。
_DECISION_RE = re.compile(r"Decision\s*:\s*\*{0,2}(YES|NO)\b")  # 兼容带 Decision: 的输出
_WORD_RE = re.compile(r"\b(YES|NO)\b")  # 全大写整词


def parse_verdict(text: str):
    """从模型输出解析 YES/NO；取全文中最后一个匹配（最终答案在末尾），失败返回 None。"""
    if not text:
        return None
    matches = _DECISION_RE.findall(text)
    if matches:
        return matches[-1]
    words = _WORD_RE.findall(text)
    if words:
        return words[-1]
    return None


def iter_jsonl(path):
    """逐行读 jsonl，容错坏行；文件不存在时返回空迭代。"""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
