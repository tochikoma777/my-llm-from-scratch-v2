"""可视化（搬运 v1 的 loss 曲线，新增注意力热力图）。

搬运来源：v1 `module_train.py:362` `plot_losses`（双 x 轴：epoch 与 tokens seen）。

相对 v1 的修正：
- v1 在 `plot_losses` 内部 `plt.show()`（`TRAIN:410`），再回到 `__main__` 里 `plt.savefig`
  （`TRAIN:559`），无头环境行为不可控且可能拿到空图；这里统一接受显式 `out_path` 并内部 `savefig`。
- v1 有一份重复实现（`SFT:282`），合并到这里。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch


def plot_losses(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    tokens_seen: Sequence[int] | None = None,
    out_path: str | Path = "outputs/loss.pdf",
    *,
    dpi: int = 300,
) -> Path:
    """绘制训练/验证损失曲线（v1 `TRAIN:362`）。

    Args:
        train_losses: 训练损失序列。
        val_losses: 验证损失序列。
        tokens_seen: 与 loss 同步的累计 token 数；给出时绘制顶部次坐标轴。
        out_path: 输出文件路径（父目录自动创建）。
        dpi: 输出分辨率。

    Returns:
        实际写入的路径。

    Raises:
        ValueError: `train_losses` 与 `val_losses` 长度不一致。
    """
    raise NotImplementedError


def plot_attention_heatmap(
    attn_weights: torch.Tensor,
    out_path: str | Path = "outputs/attention.pdf",
    *,
    layer: int = 0,
    head: int = 0,
    dpi: int = 200,
) -> Path:
    """绘制单层单头的注意力权重热力图（v2 新增，用于直观检查因果掩码）。

    Args:
        attn_weights: 注意力权重，形状 `(heads, tokens, tokens)` 或
            `(batch, heads, tokens, tokens)`。
        out_path: 输出路径。
        layer: 层序号，仅用于图标题。
        head: 头序号。
        dpi: 输出分辨率。

    Returns:
        实际写入的路径。
    """
    raise NotImplementedError


__all__ = ["plot_attention_heatmap", "plot_losses"]
