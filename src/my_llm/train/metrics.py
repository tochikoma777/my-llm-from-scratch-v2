"""评估指标（v2 新增）。

v1 只在训练日志里打印 train/val 交叉熵（`TRAIN:349`），仓库内 grep `perplexity` 为 0 命中。
语言模型的标准汇报指标是 perplexity = exp(cross_entropy)，这里补上，
避免在 notebook 里手工 `math.exp`。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from my_llm.train.losses import DeviceLike


def loss_to_perplexity(loss: float | torch.Tensor) -> float:
    """把交叉熵损失换算为困惑度。

    Args:
        loss: 交叉熵损失（自然对数底）。

    Returns:
        `exp(loss)`；输入非有限值时返回 `inf`。
    """
    raise NotImplementedError


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
        困惑度；空 loader 返回 `nan`。
    """
    raise NotImplementedError
