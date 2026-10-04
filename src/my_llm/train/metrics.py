"""评估指标（v2 新增）。

v1 只在训练日志里打印 train/val 交叉熵（`TRAIN:349`），仓库内 grep `perplexity` 为 0 命中。
语言模型的标准汇报指标是 perplexity = exp(cross_entropy)，这里补上，
避免在 notebook 里手工 `math.exp`。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from my_llm.train.losses import DeviceLike, calc_loss_loader


def loss_to_perplexity(loss: float | torch.Tensor) -> float:
    """把交叉熵损失换算为困惑度。

    Args:
        loss: 交叉熵损失（自然对数底）。

    Returns:
        `exp(loss)`；输入非有限值时返回 `inf`。
    """
    value = loss.item() if isinstance(loss, torch.Tensor) else loss
    if not math.isfinite(value):
        return float("inf")
    return math.exp(value)


def evaluate_perplexity(
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    device: DeviceLike,
    num_batches: int | None = None,
) -> float:
    """在 loader 上计算困惑度。

    Args:
        data_loader: 数据 loader。
        model: 语言模型。
        device: 计算设备。
        num_batches: 限制批次数；`None` 表示整轮。

    Returns:
        困惑度；空 loader 返回 `nan`（不转成 `inf`，好让"数据为空"和"模型发散"区分开）。
    """
    loss = calc_loss_loader(data_loader, model, device, num_batches)
    if math.isnan(loss):
        return float("nan")
    return loss_to_perplexity(loss)


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: DeviceLike,
    eval_iter: int,
) -> tuple[float, float]:
    """一次拿到 train / val 两个损失（v1 `TRAIN:179`）。

    Args:
        model: 语言模型。
        train_loader: 训练 loader。
        val_loader: 验证 loader。
        device: 计算设备。
        eval_iter: 每侧跑多少批（快速评估用）。

    Returns:
        `(train_loss, val_loss)`；任一侧 loader 为空时该侧为 `nan`。

    Note:
        相对 v1 的一处修正：v1 结尾无条件 `model.train()`（`TRAIN:215`），
        对"本来就在 eval 模式的调用方"是个副作用——评估完模型被悄悄切回训练模式，
        dropout 重新打开。这里改为**恢复进入时的模式**。
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    if was_training:
        model.train()
    return train_loss, val_loss
