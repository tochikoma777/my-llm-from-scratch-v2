"""训练：loss 计算 / 指标 / 学习率调度 / Trainer。

相对 v1 的能力补齐（审计结论 §1.2）：

| 能力 | v1 | v2 |
|---|---|---|
| lr scheduler | 无（grep 0 命中） | `scheduler.py`：warmup + cosine |
| checkpoint 续训 | 只 `torch.save` 权重一次 | `trainer.py`： optimizer/scheduler/step 一并存取 |
| 梯度累积 / AMP | 无 | `trainer.py` |
| perplexity | 无（只有交叉熵） | `metrics.py` |
"""

from my_llm.train.losses import calc_loss_batch as calc_loss_batch
from my_llm.train.losses import calc_loss_loader as calc_loss_loader
from my_llm.train.metrics import loss_to_perplexity as loss_to_perplexity
from my_llm.train.scheduler import (
    get_cosine_schedule_with_warmup as get_cosine_schedule_with_warmup,
)
from my_llm.train.trainer import Trainer as Trainer
from my_llm.train.trainer import load_checkpoint as load_checkpoint
from my_llm.train.trainer import save_checkpoint as save_checkpoint

__all__ = [
    "Trainer",
    "calc_loss_batch",
    "calc_loss_loader",
    "get_cosine_schedule_with_warmup",
    "load_checkpoint",
    "loss_to_perplexity",
    "save_checkpoint",
]
