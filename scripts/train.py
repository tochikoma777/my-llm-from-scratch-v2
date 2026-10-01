#!/usr/bin/env python
"""预训练入口。

用法：
    python scripts/train.py --config configs/gpt2-tiny.yaml
    python scripts/train.py --config configs/gpt2-small.yaml --resume outputs/checkpoints/last.pt

配置**只能**来自 yaml（`GPTConfig.from_yaml`），不在脚本里写任何超参——
这是相对 v1 的关键改动（v1 把全套超参埋在 `module_train.py:530-546` 的 `__main__` 里）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from my_llm.config import GPTConfig
from my_llm.model.gpt import GPTModel
from my_llm.train.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命名空间对象。
    """
    parser = argparse.ArgumentParser(description="从头预训练 GPT-2")
    parser.add_argument("--config", type=Path, required=True, help="模型配置 yaml 路径")
    parser.add_argument(
        "--data", type=Path, default=Path("data/raw/the-verdict.txt"), help="语料路径"
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="批大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--seed", type=int, default=123, help="随机种子")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint 路径，续训用")
    return parser.parse_args()


def main() -> None:
    """执行预训练。

    Returns:
        None。

    Raises:
        FileNotFoundError: 配置文件或语料不存在。
    """
    args = parse_args()
    cfg = GPTConfig.from_yaml(args.config)
    model = GPTModel(cfg)
    trainer = Trainer(
        model=model,
        optimizer=None,  # type: ignore[arg-type]  # TODO: 由 setup_optimizer(cfg, args.lr) 提供
        cfg=TrainerConfig(num_epochs=args.epochs, seed=args.seed),
    )
    trainer.train(train_loader=None, val_loader=None, device="cpu", resume_from=args.resume)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
