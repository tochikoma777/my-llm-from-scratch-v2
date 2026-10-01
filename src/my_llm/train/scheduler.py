"""学习率调度（v2 新增）。

v1 全程用固定 lr 的 AdamW（`TRAIN:467-471`、`SFT:479-483`），仓库内 grep
`scheduler|warmup|cosine` 全部 0 命中 —— 这是"复现 GPT-2 级训练"最缺的一环。

提供 `get_cosine_schedule_with_warmup`：线性 warmup 后按余弦衰减到 `base_lr * min_lr_ratio`。
语义与 HF `transformers.get_cosine_schedule_with_warmup` 一致，便于 parity 对照。
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LRScheduler


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> LRScheduler:
    """构建 warmup + cosine 调度器。

    Args:
        optimizer: 目标优化器。
        num_warmup_steps: warmup 步数，期间 lr 从 0 线性升到初始值。
        num_training_steps: 总训练步数。
        num_cycles: 余弦周期数（0.5 表示半个周期衰减到 0）。
        min_lr_ratio: 衰减下限相对初始 lr 的比例，0 表示衰减到 0。

    Returns:
        `LambdaLR` 调度器，按 `optimizer.step()` 计数。

    Raises:
        ValueError: 步数配置非法。
    """
    raise NotImplementedError


def warmup_cosine_lr(
    step: int, base_lr: float, num_warmup_steps: int, num_training_steps: int
) -> float:
    """单个步的学习率系数，便于单测。

    Args:
        step: 当前全局步（从 0 开始）。
        base_lr: 基础学习率。
        num_warmup_steps: warmup 步数。
        num_training_steps: 总步数。

    Returns:
        该步的学习率绝对值。
    """
    raise NotImplementedError
