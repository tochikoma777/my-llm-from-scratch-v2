#!/usr/bin/env python
"""权重下载/转换入口。

用法：
    python scripts/download_weights.py --source hf --model gpt2
    python scripts/download_weights.py --source openai --model-size 124M --models-dir gpt2

为什么既能拉 HF 又能拉 OpenAI：两者可以做**交叉验证**——同一层的张量应当完全一致，
这是 `tests/test_parity_hf.py` 里最强的一条断言。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from my_llm.weights.hf import load_hf_state_dict
from my_llm.weights.openai_tf import download_and_load_gpt2


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命名空间对象。
    """
    parser = argparse.ArgumentParser(description="下载 GPT-2 权重")
    parser.add_argument("--source", choices=("hf", "openai"), default="hf", help="权重来源")
    parser.add_argument("--model", type=str, default="gpt2", help="HF 模型名（source=hf 时）")
    parser.add_argument(
        "--model-size",
        choices=("124M", "355M", "774M", "1558M"),
        default="124M",
        help="OpenAI 模型尺寸（source=openai 时）",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("gpt2"), help="OpenAI 权重落盘目录")
    return parser.parse_args()


def main() -> None:
    """按来源下载权重。

    Returns:
        None。
    """
    args = parse_args()
    if args.source == "hf":
        _state = load_hf_state_dict(args.model)
    else:
        _settings, _params = download_and_load_gpt2(args.model_size, args.models_dir)


if __name__ == "__main__":
    main()
