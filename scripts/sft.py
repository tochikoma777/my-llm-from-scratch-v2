#!/usr/bin/env python
"""Alpaca 指令微调入口。

用法：
    python scripts/sft.py --config configs/gpt2-small.yaml --data configs/sft-alpaca.yaml
    python scripts/sft.py --config configs/gpt2-tiny.yaml --test-mode   # 只跑少量样本

注意：SFT 的**数据侧**超参（划分比例、batch、lr、epochs）来自 `--data` 指向的 yaml，
模型架构来自 `--config`。两者分离，避免 v1 那样把架构配置就地 `update()` 覆盖
（`SFT:446` 的 in-place 修改问题）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from my_llm.config import GPTConfig
from my_llm.model.gpt import GPTModel


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命名空间对象。
    """
    parser = argparse.ArgumentParser(description="Alpaca 指令微调")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gpt2-small.yaml"), help="模型配置"
    )
    parser.add_argument(
        "--data", type=Path, default=Path("configs/sft-alpaca.yaml"), help="SFT 参数"
    )
    parser.add_argument("--json", type=Path, default=None, help="本地指令数据 json")
    parser.add_argument("--test-mode", action="store_true", help="只跑 10 条样本，快速验证")
    parser.add_argument("--epochs", type=int, default=2, help="微调轮数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--seed", type=int, default=123, help="随机种子")
    return parser.parse_args()


def main() -> None:
    """执行微调并保存产物。

    Returns:
        None。
    """
    args = parse_args()
    cfg = GPTConfig.from_yaml(args.config)
    _model = GPTModel(cfg)  # noqa: F841  # P1 接线后进 run_sft
    raise NotImplementedError  # TODO(P1): 装配 InstructionDataset/DataLoader 并调用 run_sft


if __name__ == "__main__":
    main()
