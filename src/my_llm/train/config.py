"""训练运行配置层（与模型架构配置 `GPTConfig` 并列）。

为什么单独一份：v1 把 `lr/epochs/batch_size/weight_decay` 埋在 `module_train.py:541-546`
的 `OTHER_SETTINGS` 里，与架构配置 `GPT_CONFIG_124M` 混在同一个 `__main__`，既不能 import
也不能被 CLI 覆盖。v2 把它做成 `TrainConfig`，从 `configs/train-*.yaml` 读。

**形态说明**：yaml 顶层是扁平标量，加 `optimizer` / `scheduler` / `data` 三个**二级段**
（仍是一次 `yaml.safe_load` 解析完，不是 sft-*.yaml 那种 `model.config` 再指向别的文件的嵌套）。

对齐 v1 的默认值来源：
- `train_ratio=0.9`（v1 `TRAIN:477`：90% 训练 / 10% 验证，无测试集）
- `lr=5e-4` / `weight_decay=0.1` / `num_epochs=10` / `batch_size=2`（v1 `TRAIN:541-546`）
- `eval_freq=5` / `eval_iter=1`（v1 `TRAIN:518-519`）
- `max_length=256` / `stride=128`（v1 `DATA:122-123` 的 DataLoader 默认值）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OptimizerConfig:
    """AdamW 超参（v1 `TRAIN:467-471` 只有 lr + weight_decay）。

    Attributes:
        lr: 学习率。
        weight_decay: 权重衰减。
        betas: Adam 的一阶/二阶矩系数。
        eps: 数值稳定项。
    """

    lr: float = 5.0e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8


@dataclass
class SchedulerConfig:
    """warmup + cosine 调度参数（v1 完全没有，`train/scheduler.py` 是新增）。

    Attributes:
        warmup_steps: warmup 步数。
        min_lr_ratio: 衰减下限相对初始 lr 的比例，0 表示衰减到 0。
    """

    warmup_steps: int = 10
    min_lr_ratio: float = 0.0


@dataclass
class DataConfig:
    """语料切分与滑窗参数。

    Attributes:
        train_ratio: 训练集占比，其余为验证集（v1 `TRAIN:477`）。
        max_length: 滑窗长度。
        stride: 滑窗步长。
    """

    train_ratio: float = 0.9
    max_length: int = 256
    stride: int = 128


@dataclass
class TrainConfig:
    """一次预训练运行的全部超参（架构之外的部分）。

    Attributes:
        seed: 随机种子。
        num_epochs: 训练轮数。
        batch_size: 批大小。
        eval_freq: 每多少个微批次评估一次。
        eval_iter: 每次评估跑多少批。
        save_every: 每隔多少步存一次快照；0 表示只在每轮结束存。
        grad_accum_steps: 梯度累积步数。
        grad_clip: 梯度裁剪阈值，<=0 表示禁用。
        precision: `"fp32" | "fp16" | "bf16"`；只有 fp16 会启用 GradScaler。
        optimizer: AdamW 超参。
        scheduler: warmup + cosine 参数。
        data: 语料切分与滑窗参数。
    """

    seed: int = 123
    num_epochs: int = 10
    batch_size: int = 2
    eval_freq: int = 5
    eval_iter: int = 1
    save_every: int = 100
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    precision: str = "fp32"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        """校验取值合法性，尽早失败。

        Raises:
            ValueError: `precision` 非法、`train_ratio` 不在 (0, 1)，或批/步数为非正。
        """
        if self.precision not in ("fp32", "fp16", "bf16"):
            msg = f"precision 必须是 fp32/fp16/bf16 之一，收到 {self.precision}"
            raise ValueError(msg)
        if not 0.0 < self.data.train_ratio < 1.0:
            msg = f"data.train_ratio 必须在 (0, 1)，收到 {self.data.train_ratio}"
            raise ValueError(msg)
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            msg = (
                f"batch_size 与 grad_accum_steps 必须为正: "
                f"{self.batch_size} / {self.grad_accum_steps}"
            )
            raise ValueError(msg)

    def total_steps(self, batches_per_epoch: int) -> int:
        """估算总优化器步数（给 scheduler 用）。

        Args:
            batches_per_epoch: 每个 epoch 的微批次数。

        Returns:
            `ceil(batches_per_epoch / grad_accum_steps) * num_epochs`。
        """
        per_epoch = math.ceil(batches_per_epoch / self.grad_accum_steps)
        return per_epoch * self.num_epochs

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """从 yaml 构造配置（二级段会被转成对应 dataclass）。

        Args:
            path: yaml 路径。顶层键是本 dataclass 的字段名，其中
                `optimizer` / `scheduler` / `data` 三段再各自对应一个 dataclass。

        Returns:
            TrainConfig 实例。

        Raises:
            TypeError: yaml 含未知字段。
        """
        with Path(path).open(encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            msg = f"{path} 含未知字段: {sorted(unknown)}，合法字段: {sorted(known)}"
            raise TypeError(msg)

        sections = {"optimizer": OptimizerConfig, "scheduler": SchedulerConfig, "data": DataConfig}
        kwargs: dict[str, Any] = {}
        for name, value in raw.items():
            if name in sections:
                kwargs[name] = _build_section(sections[name], value, path, name)
            else:
                kwargs[name] = value
        return cls(**kwargs)


def _build_section(cls: type[Any], raw: Any, path: str | Path, name: str) -> Any:
    """把 yaml 里的二级段构造成对应 dataclass。

    Args:
        cls: 目标 dataclass 类型。
        raw: yaml 中该段的原始内容。
        path: yaml 路径（仅用于报错信息）。
        name: 段名（仅用于报错信息）。

    Returns:
        `cls` 的实例。

    Raises:
        TypeError: 该段不是映射，或含未知字段。
    """
    if not isinstance(raw, dict):
        msg = f"{path} 的 {name} 段必须是映射，实际是 {type(raw).__name__}"
        raise TypeError(msg)
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        msg = f"{path} 的 {name} 段含未知字段: {sorted(unknown)}，合法字段: {sorted(known)}"
        raise TypeError(msg)
    # betas 在 yaml 里是列表 [0.9, 0.95]，dataclass 里是 tuple
    if "betas" in raw and isinstance(raw["betas"], list):
        raw = {**raw, "betas": tuple(raw["betas"])}
    return cls(**raw)
