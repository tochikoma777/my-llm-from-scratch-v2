"""学习率调度（v2 新增）。

v1 全程用固定 lr 的 AdamW（`TRAIN:467-471`、`SFT:479-483`），仓库内 grep
`scheduler|warmup|cosine` 全部 0 命中 —— 这是"复现 GPT-2 级训练"最缺的一环。

提供 `get_cosine_schedule_with_warmup`：线性 warmup 后按余弦衰减到 `base_lr * min_lr_ratio`。
语义与 HF `transformers.get_cosine_schedule_with_warmup` 一致，便于 parity 对照。
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def _validate(num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float) -> None:
    """校验调度参数。

    Args:
        num_warmup_steps: warmup 步数。
        num_training_steps: 总步数。
        min_lr_ratio: 衰减下限比例。

    Raises:
        ValueError: 步数为负、总步数非正、warmup 占满全程，或 `min_lr_ratio` 不在 `[0, 1]`。
    """
    if num_warmup_steps < 0:
        msg = f"num_warmup_steps 不能为负，收到 {num_warmup_steps}"
        raise ValueError(msg)
    if num_training_steps <= 0:
        msg = f"num_training_steps 必须为正，收到 {num_training_steps}"
        raise ValueError(msg)
    if num_warmup_steps >= num_training_steps:
        msg = f"warmup 步数 {num_warmup_steps} 必须小于总步数 {num_training_steps}"
        raise ValueError(msg)
    if not 0.0 <= min_lr_ratio <= 1.0:
        msg = f"min_lr_ratio 必须在 [0, 1]，收到 {min_lr_ratio}"
        raise ValueError(msg)


def _lr_factor(
    step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> float:
    """第 `step` 步的学习率**倍率**（相对初始 lr）。

    Args:
        step: 当前全局步（从 0 开始）。
        num_warmup_steps: warmup 步数，期间倍率从 0 线性升到 1。
        num_training_steps: 总训练步数。
        num_cycles: 余弦周期数（0.5 = 半个周期，单调衰减到底）。
        min_lr_ratio: 衰减下限，0 表示衰减到 0。

    Returns:
        倍率；warmup 阶段落在 `[0, 1]`，之后落在 `[min_lr_ratio, 1]`。

    Raises:
        ValueError: 步数配置非法（见 `_validate`）。
    """
    _validate(num_warmup_steps, num_training_steps, min_lr_ratio)
    # step < num_warmup_steps 且 step >= 0 蕴含 num_warmup_steps > 0，不会除零
    if step < num_warmup_steps:
        return step / num_warmup_steps
    progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    progress = min(progress, 1.0)
    # 与 HF get_cosine_schedule_with_warmup 同一公式，便于 parity 对照
    cosine = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def warmup_cosine_lr(
    step: int,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> float:
    """单个步的学习率系数，便于单测。

    Args:
        step: 当前全局步（从 0 开始）。
        base_lr: 基础学习率。
        num_warmup_steps: warmup 步数。
        num_training_steps: 总步数。
        num_cycles: 余弦周期数。
        min_lr_ratio: 衰减下限相对初始 lr 的比例。

    Returns:
        该步的学习率绝对值。

    Raises:
        ValueError: 步数配置非法（见 `_validate`）。
    """
    return base_lr * _lr_factor(
        step, num_warmup_steps, num_training_steps, num_cycles, min_lr_ratio
    )


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
        `LambdaLR` 调度器，按 `scheduler.step()` 计数（`optimizer.step()` 之后调用）。

    Raises:
        ValueError: 步数配置非法（见 `_validate`）。
    """
    _validate(num_warmup_steps, num_training_steps, min_lr_ratio)

    def lr_lambda(step: int) -> float:
        return _lr_factor(step, num_warmup_steps, num_training_steps, num_cycles, min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)
