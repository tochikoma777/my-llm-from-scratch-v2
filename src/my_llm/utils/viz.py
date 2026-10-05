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

import matplotlib.pyplot as plt
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
        ValueError: `train_losses` 与 `val_losses` 长度不一致，或 `tokens_seen`
            给出时长度与它们不一致。
    """
    if len(train_losses) != len(val_losses):
        msg = f"train_losses({len(train_losses)}) 与 val_losses({len(val_losses)}) 长度必须一致"
        raise ValueError(msg)
    if tokens_seen is not None and len(tokens_seen) != len(train_losses):
        msg = f"tokens_seen({len(tokens_seen)}) 长度必须与 loss 序列({len(train_losses)})一致"
        raise ValueError(msg)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 主坐标轴：横轴是评估点（每轮一次评估时即 epoch）。v1 传的是 linspace 出来的
    # epochs 数组（`SFT:510`），这里没有 epochs 参数，直接按下标画，语义相同。
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, train_losses, label="Training loss", linewidth=2)
    ax1.plot(epochs, val_losses, linestyle="-.", label="Validation loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    if tokens_seen is not None:
        # v1 `TRAIN:400-402` 的写法：画一条透明曲线只为让次坐标轴刻度对齐
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha=0)
        ax2.set_xlabel("Tokens seen", fontsize=12)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_attention_heatmap(
    attn_weights: torch.Tensor,
    out_path: str | Path = "outputs/attention.pdf",
    *,
    layer: int = 0,
    head: int = 0,
    dpi: int = 200,
) -> Path:
    """绘制单层单头的注意力权重热力图（v2 新增，用于直观检查因果掩码）。

    只用 matplotlib 的 `imshow`：seaborn 仅存在于 `viz` extra，CI 走
    `pip install -e ".[dev]"` 没有它，引了会直接 ImportError。

    Args:
        attn_weights: 注意力权重，形状 `(heads, tokens, tokens)` 或
            `(batch, heads, tokens, tokens)`；4 维时取第 0 个样本。
        out_path: 输出路径。
        layer: 层序号，仅用于图标题。
        head: 头序号。
        dpi: 输出分辨率。

    Returns:
        实际写入的路径。

    Raises:
        ValueError: `attn_weights` 不是 3/4 维，或 `head` 越界。
    """
    weights = attn_weights.detach().cpu()
    if weights.dim() == 4:
        weights = weights[0]
    if weights.dim() != 3:
        msg = f"attn_weights 形状应为 (heads, q, k) 或 (batch, heads, q, k)，收到 {weights.shape}"
        raise ValueError(msg)
    if not 0 <= head < weights.shape[0]:
        msg = f"head 越界: {head}，共 {weights.shape[0]} 个头"
        raise ValueError(msg)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    matrix = weights[head].to(torch.float32).numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(f"layer {layer} / head {head}")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


__all__ = ["plot_attention_heatmap", "plot_losses"]
