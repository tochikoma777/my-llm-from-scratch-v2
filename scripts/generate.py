#!/usr/bin/env python
"""文本生成入口。

用法：
    python scripts/generate.py --prompt "Every effort moves you"
    python scripts/generate.py --prompt "..." --temperature 0.8 --top-k 50 --max-new-tokens 100
    python scripts/generate.py --prompt "..." --checkpoint outputs/checkpoints/last.pt

默认未指定权重时加载 HF `gpt2`（需配到 gpt2-small.yaml 的尺寸）。

注意：`--config` 决定**架构**，权重来自 HF 或 checkpoint，两者尺寸必须一致；
否则 `load_weights_from_hf` 会直接报形状不匹配（不会静默乱加载）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from my_llm.config import GPTConfig
from my_llm.generate.sampling import generate
from my_llm.model.gpt import GPTModel
from my_llm.tokenizer import build_tokenizer
from my_llm.weights.hf import load_hf_state_dict, load_weights_from_hf


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
    parser.add_argument("--top-p", type=float, default=None, help="nucleus 阈值；None 表示不启用")
    parser.add_argument("--seed", type=int, default=123, help="随机种子")
    return parser.parse_args()


def load_model(cfg: GPTConfig, checkpoint: Path | None) -> GPTModel:
    """按配置建模型并装载权重。

    Args:
        cfg: 架构配置。
        checkpoint: 本地 checkpoint；`None` 时拉 HF `gpt2` 权重。

    Returns:
        已装载权重的模型（eval 模式）。
    """
    model = GPTModel(cfg)
    if checkpoint is None:
        load_weights_from_hf(model, load_hf_state_dict("gpt2"))
    else:
        ckpt: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # save_checkpoint 存的是 {"step", "model", "optimizer", ...}；裸 state_dict 也兼容
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    """执行生成。

    Returns:
        None。
    """
    args = parse_args()
    cfg = GPTConfig.from_yaml(args.config)
    torch.manual_seed(args.seed)

    tokenizer = build_tokenizer()
    model = load_model(cfg, args.checkpoint)

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        context_size=cfg.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
