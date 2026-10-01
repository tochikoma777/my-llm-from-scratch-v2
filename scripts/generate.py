#!/usr/bin/env python
"""文本生成入口。

用法：
    python scripts/generate.py --prompt "Every effort moves you"
    python scripts/generate.py --prompt "..." --temperature 0.8 --top-k 50 --max-new-tokens 100
    python scripts/generate.py --prompt "..." --checkpoint outputs/checkpoints/last.pt

默认未指定权重时加载 HF `gpt2`（需配到 gpt2-small.yaml 的尺寸）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from my_llm.config import GPTConfig
from my_llm.generate.sampling import generate
from my_llm.model.gpt import GPTModel


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命名空间对象。
    """
    parser = argparse.ArgumentParser(description="用 GPT-2 生成文本")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gpt2-small.yaml"), help="模型配置"
    )
    parser.add_argument("--prompt", type=str, default="Every effort moves you", help="提示词")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="本地 checkpoint；缺省用 HF 权重"
    )
    parser.add_argument("--max-new-tokens", type=int, default=50, help="新生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="<=0 走贪婪解码")
    parser.add_argument("--top-k", type=int, default=None, help="top-k 阈值")
    parser.add_argument("--seed", type=int, default=123, help="随机种子")
    return parser.parse_args()


def main() -> None:
    """执行生成。

    Returns:
        None。
    """
    args = parse_args()
    cfg = GPTConfig.from_yaml(args.config)
    model = GPTModel(cfg)
    _ = generate(
        model=model,
        idx=None,  # type: ignore[arg-type]  # TODO: 用 tokenizer 编码 args.prompt
        max_new_tokens=args.max_new_tokens,
        context_size=cfg.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
