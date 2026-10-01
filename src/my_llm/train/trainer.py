"""训练 Trainer 与 checkpoint 续训（v2 新增主体，loss 部分搬运 v1）。

对应 v1 `module_train.py:265` `train_model_simple` / `:179` `evaluate_model` / `:219`
`generate_and_print_sample`。相对 v1 补齐四项能力：

1. **checkpoint 续训**（v1 缺陷 §1.2-2）：只有训练结束后的一次 `torch.save(model.state_dict())`
   （`TRAIN:564`），不存 optimizer / epoch / 步数，无法中断续跑。这里
   `save_checkpoint/load_checkpoint` 把 model+optimizer+scheduler+step 一起落盘。
2. **梯度累积**（v1 缺陷 §1.2-4）：`grad_accum_steps` 参数。
3. **混合精度**（v1 缺陷 §1.2-4）：`torch.autocast` + `GradScaler`，CPU 上自动退化。
4. **学习率调度**（v1 缺陷 §1.2-3）：每步 `scheduler.step()`。

同时保留 v1 的观察手段：`eval_freq` 定期评估 + 每轮结束用固定 prompt 采样看生成质量
（`TRAIN:336-356`），这在调早期模型时比纯 loss 更有信息量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from my_llm.train.losses import DeviceLike


@dataclass
class TrainerConfig:
    """Trainer 的运行参数（与模型架构无关的那一部分）。

    Attributes:
        num_epochs: 训练轮数。
        eval_freq: 每多少个全局步评估一次（沿用 v1 `TRAIN:518` 的语义）。
        eval_iter: 每次评估跑多少批（v1 `TRAIN:519`）。
        grad_accum_steps: 梯度累积步数，1 表示禁用。
        grad_clip_norm: 梯度裁剪阈值，<=0 表示禁用。
        amp: 是否启用自动混合精度（仅 CUDA 生效）。
        start_context: 每轮结束后用于采样演示的 prompt。
        checkpoint_dir: checkpoint 落盘目录，默认 `outputs/checkpoints`。
        save_every: 每隔多少步保存一次；0 表示只在每轮结束保存。
        seed: 随机数种子（见 `utils.seed`）。
    """

    num_epochs: int = 1
    eval_freq: int = 50
    eval_iter: int = 1
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    amp: bool = False
    start_context: str = "Every effort moves you"
    checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))
    save_every: int = 0
    seed: int = 123


@dataclass
class TrainHistory:
    """训练过程记录。

    Attributes:
        train_losses: 每次评估的训练损失。
        val_losses: 每次评估的验证损失。
        perplexities: 与上述同步的困惑度。
        tokens_seen: 每次评估时累计见过的 token 数。
        global_steps: 每次评估对应的全局步。
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    perplexities: list[float] = field(default_factory=list)
    tokens_seen: list[int] = field(default_factory=list)
    global_steps: list[int] = field(default_factory=list)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    path: str | Path,
    **extra: Any,
) -> None:
    """保存完整训练状态（model + optimizer + scheduler + step）。

    Args:
        model: 模型。
        optimizer: 优化器。
        scheduler: 调度器，可为 None。
        step: 当前全局步（用于续训）。
        path: 目标文件路径；父目录会自动创建。
        **extra: 额外要写入 checkpoint 的可 pickle 对象（如 epoch、config）。
    """
    raise NotImplementedError


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    path: str | Path,
) -> int:
    """恢复训练状态。

    Args:
        model: 模型，就地加载权重。
        optimizer: 优化器，就地加载状态。
        scheduler: 调度器，为 None 时跳过其状态恢复。
        path: checkpoint 路径。

    Returns:
        checkpoint 里记录的全局步，用于继续执行 `range(resumed_step, total_steps)`。

    Raises:
        FileNotFoundError: checkpoint 不存在。
    """
    raise NotImplementedError


class Trainer:
    """训练循环（替代 v1 `train_model_simple`）。

    Attributes:
        model: 被训练的模型。
        optimizer: 优化器。
        scheduler: 学习率调度器，可为 None（固定 lr）。
        cfg: 运行参数。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: TrainerConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        """初始化 Trainer。

        Args:
            model: 被训练的模型。
            optimizer: 优化器。
            cfg: 运行参数。
            scheduler: 调度器。
        """
        raise NotImplementedError

    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        device: DeviceLike,
        epoch: int,
    ) -> TrainHistory:
        """跑完一个 epoch。

        Args:
            train_loader: 训练数据。
            val_loader: 验证数据。
            device: 计算设备。
            epoch: 当前 epoch 序号（0 起）。

        Returns:
            本轮的记录。
        """
        raise NotImplementedError

    def train(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        device: DeviceLike,
        resume_from: str | Path | None = None,
    ) -> TrainHistory:
        """完整训练主循环，支持从 checkpoint 续训。

        Args:
            train_loader: 训练数据。
            val_loader: 验证数据。
            device: 计算设备。
            resume_from: checkpoint 路径；给出则恢复权重/优化器/步数。

        Returns:
            全部轮次累计的记录。
        """
        raise NotImplementedError
