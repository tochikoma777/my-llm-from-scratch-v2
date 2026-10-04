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

**依赖存根的处理**（三个模块尚未落地，这里刻意不 import 它们）：
- `generate/sampling.py`（P0）→ 采样通过可选回调 `sample_fn` 注入，不注入就不采样；
- `utils/seed.py`（P2）→ 直接用 `torch.manual_seed`，等它落地后替换；
- `utils/logging.py`（P2）→ 用标准库 `logging`，不 `print`。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import grad_scaler
from torch.utils.data import DataLoader

from my_llm.train.losses import DeviceLike, calc_loss_batch
from my_llm.train.metrics import evaluate_model, loss_to_perplexity

logger = logging.getLogger(__name__)


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
    samples: list[str] = field(default_factory=list)

    def extend(self, other: TrainHistory) -> None:
        """把另一个 `TrainHistory` 的记录追加到本对象末尾。

        Args:
            other: 通常是单个 epoch 的记录。
        """
        self.train_losses.extend(other.train_losses)
        self.val_losses.extend(other.val_losses)
        self.perplexities.extend(other.perplexities)
        self.tokens_seen.extend(other.tokens_seen)
        self.global_steps.extend(other.global_steps)
        self.samples.extend(other.samples)


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    payload.update(extra)
    torch.save(payload, path)


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

    Note:
        `load_state_dict` 是**就地 `copy_`**，不会替换 `nn.Parameter` 对象，
        因此 weight tying（`out_head.weight is tok_emb.weight`）不会被打断（硬约束 1）。
        读盘用 `weights_only=True`：checkpoint 只含张量与内置容器，
        没必要为它开 pickle 反序列化任意对象的口子。
    """
    path = Path(path)
    if not path.is_file():
        msg = f"checkpoint 不存在: {path}"
        raise FileNotFoundError(msg)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["step"])


class Trainer:
    """训练循环（替代 v1 `train_model_simple`）。

    Attributes:
        model: 被训练的模型。
        optimizer: 优化器。
        scheduler: 学习率调度器，可为 None（固定 lr）。
        cfg: 运行参数。
        sample_fn: 可选的采样回调，签名 `(model, prompt) -> str`。
        global_step: 已处理的**微批次**数（与 v1 `TRAIN:322` 同义）。
        tokens_seen: 累计见过的 token 数。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: TrainerConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        sample_fn: Callable[[nn.Module, str], str] | None = None,
    ) -> None:
        """初始化 Trainer。

        Args:
            model: 被训练的模型。
            optimizer: 优化器。
            cfg: 运行参数。
            scheduler: 调度器。
            sample_fn: 每轮结束后用于采样演示文本的回调；为 `None` 则不采样。
                `generate/sampling.py` 落地后由调用方传入，这里刻意不 import 它。
        """
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.scheduler = scheduler
        self.sample_fn = sample_fn
        self.global_step = 0
        self.tokens_seen = 0

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

        Note:
            两个"步"的口径不同，别混：`global_step` 数的是**微批次**（沿用 v1，
            `eval_freq` 按它触发），而 `optimizer.step()` / `scheduler.step()` 每
            `grad_accum_steps` 个微批次才走一次（`global_step` 之外单独计数）。
        """
        history = TrainHistory()
        dev = torch.device(device)
        amp_enabled = self.cfg.amp and dev.type == "cuda"
        scaler = grad_scaler.GradScaler(dev.type, enabled=amp_enabled)
        accum = max(1, self.cfg.grad_accum_steps)
        num_batches = len(train_loader)

        self.model.train()
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            self.global_step += 1
            self.tokens_seen += input_batch.numel()

            with torch.autocast(device_type=dev.type, enabled=amp_enabled):
                loss = calc_loss_batch(input_batch, target_batch, self.model, dev)
            # 用 torch.autograd.backward 而不是 Tensor.backward：后者在 torch 的类型存根里
            # 没有注解，mypy --strict 会报 no-untyped-call；两者对标量损失等价。
            torch.autograd.backward(scaler.scale(loss / accum))

            is_boundary = (batch_idx + 1) % accum == 0 or (batch_idx + 1) == num_batches
            if is_boundary:
                if self.cfg.grad_clip_norm > 0:
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            if self.global_step % self.cfg.eval_freq == 0:
                self._record_eval(history, train_loader, val_loader, dev)
                if self.cfg.save_every > 0 and self.global_step % self.cfg.save_every == 0:
                    self._save_step_ckpt()

        if self.sample_fn is not None and self.cfg.start_context:
            history.samples.append(self._sample())
        return history

    def _record_eval(
        self,
        history: TrainHistory,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> None:
        """评估一次并把结果追加进 `history`（v1 `TRAIN:336-356`）。

        Args:
            history: 目标记录对象。
            train_loader: 训练 loader。
            val_loader: 验证 loader。
            device: 计算设备。
        """
        train_loss, val_loss = evaluate_model(
            self.model, train_loader, val_loader, device, self.cfg.eval_iter
        )
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.perplexities.append(loss_to_perplexity(val_loss))
        history.tokens_seen.append(self.tokens_seen)
        history.global_steps.append(self.global_step)
        logger.info(
            "step %d: train loss %.3f, val loss %.3f", self.global_step, train_loss, val_loss
        )

    def _save_step_ckpt(self) -> None:
        """按 `save_every` 落一次快照。"""
        path = self.cfg.checkpoint_dir / f"step-{self.global_step:06d}.pt"
        save_checkpoint(self.model, self.optimizer, self.scheduler, self.global_step, path)

    def _sample(self) -> str:
        """用 `sample_fn` 基于 `start_context` 采样一段文本。

        Returns:
            采样出的文本。
        """
        self.model.eval()
        with torch.no_grad():
            text = self.sample_fn(self.model, self.cfg.start_context)  # type: ignore[misc]  # 已在上层判空
        self.model.train()
        return text

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

        Note:
            种子只在 `train()` 入口设一次（`utils/seed.py` 属 P2，落地后改用它）。

            v1 用 `global_step = -1` 起步（`TRAIN:322`）来让"第 1 个微批次就评估一次"，
            代价是计数比实际批次数少 1、续训时容易算错。这里改成 `global_step` 就是
            **已处理的微批次数**，基线评估在训练前显式做一次，效果相同但计数自洽。
        """
        torch.manual_seed(self.cfg.seed)
        history = TrainHistory()
        if resume_from is not None:
            self.global_step = load_checkpoint(
                self.model, self.optimizer, self.scheduler, resume_from
            )
            logger.info("从 %s 续训，global_step=%d", resume_from, self.global_step)
        else:
            self.global_step = 0
            self._record_eval(history, train_loader, val_loader, torch.device(device))
        for epoch in range(self.cfg.num_epochs):
            logger.info("epoch %d/%d 开始", epoch + 1, self.cfg.num_epochs)
            epoch_history = self.train_epoch(train_loader, val_loader, device, epoch)
            history.extend(epoch_history)
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.global_step,
                self.cfg.checkpoint_dir / "last.pt",
                epoch=epoch,
            )
        return history
