"""交叉熵损失：批次级与 loader 级平均。

搬运来源：v1 `src/modules/module_train.py:87` `calc_loss_batch`、`:129` `calc_loss_loader`。

保留 v1 的两个约定：
1. 用 `F.cross_entropy` 的 `mean` 归约，因此 `-100`（ignore_index）能天然屏蔽 padding ——
   这是 `finetune/custom_collate_fn` 做损失屏蔽的前提；
2. `calc_loss_loader` 遇到空 loader 返回 `float("nan")` 而不是除零（`TRAIN:153-154`），
   并保持"由调用方控制 `model.eval()` / `no_grad()`"的契约。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DeviceLike = str | torch.device


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: DeviceLike,
) -> torch.Tensor:
    """计算单个批次的交叉熵损失（v1 `TRAIN:87`）。

    Args:
        input_batch: 形状 `(batch, seq_len)` 的 token ID。
        target_batch: 形状 `(batch, seq_len)` 的目标 ID，padding 位置为 `-100`。
        model: 语言模型，输出 `(batch, seq_len, vocab_size)` 的 logits。
        device: 计算设备。

    Returns:
        标量损失。
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    device: DeviceLike,
    num_batches: int | None = None,
) -> float:
    """计算 loader 上若干批次的平均损失（v1 `TRAIN:129`）。

    Args:
        data_loader: 验证/训练 loader。
        model: 语言模型。
        device: 计算设备。
        num_batches: 限制的批次数；`None` 表示跑完整轮。

    Returns:
        平均损失；空 loader 返回 `nan`。
    """
    if len(data_loader) == 0:
        return float("nan")
    # 上限取实际可用批次数，避免传入过大值时整轮跑完还不退出（v1 `TRAIN:158`）
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches
