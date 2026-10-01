"""Alpaca 指令微调。

搬运来源：v1 `src/modules/module_fine_tuning.py`
（`InstructionDataset:59`、`custom_collate_fn:136`、`format_input:248`、主流程 `main:315`）。

**必须原样保留的语义**（写错最容易"训练能跑但效果崩"的两点）：

1. **prompt 模板**：`format_input` 里的三段式模板（`SFT:270-277`）必须逐字保持，
   训练与推理共用同一个字符串，否则模型看到的分布不一致。
2. **保留第一个 pad 作为预测目标**（`SFT:197-198`）：targets 里除第一个 pad 之外的
   padding 位置全部置 `-100`，这样模型既不在无意义 padding 上算损失，
   又能学到"生成到这里该停"——即学会输出 `<|endoftext|>`。

相对 v1 的修正：
- collate 的 `device` 不再通过 `partial` 提前绑定后再被改成 `"cpu"`（v1 `SFT:371-375` vs `:421`
  的 device 不一致 bug）；改为默认 `"cpu"`，由 Trainer 负责搬运。
- 解除 `--test_mode` 与 model/device 的耦合，测试模式只影响数据量 steps。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from my_llm.tokenizer.protocol import Tokenizer


class InstructionDataset(Dataset[list[int]]):
    """Alpaca 指令数据集（v1 `SFT:59`）。

    构造时一次性完成分词，避免训练循环里重复编码。

    Attributes:
        encoded_texts: 每条样本的 token ID 列表。
    """

    def __init__(self, data: Sequence[Mapping[str, str]], tokenizer: Tokenizer) -> None:
        """初始化并预分词。

        Args:
            data: 含 `instruction` / `input` / `output` 字段的字典序列。
            tokenizer: 分词器。
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """样本数量。

        Returns:
            样本数。
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> list[int]:
        """取第 index 条的 token ID 列表。

        Args:
            index: 样本下标。

        Returns:
            token ID 列表。
        """
        raise NotImplementedError


def format_input(entry: Mapping[str, str]) -> str:
    """把一条 Alpaca 样本格式化成标准 prompt（v1 `SFT:248`）。

    Args:
        entry: 含 `instruction` 与可选 `input` 的字典。

    Returns:
        模板化后的文本（不含 `### Response:` 部分）。
    """
    raise NotImplementedError


def custom_collate_fn(
    batch: Sequence[list[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """变长批次整理 + 损失屏蔽（v1 `SFT:136`）。

    Args:
        batch: 每条样本的 token ID 列表。
        pad_token_id: 填充 token，默认 `<|endoftext|>`。
        ignore_index: 参与交叉熵时被忽略的标签值。
        allowed_max_length: 超长截断阈值；`None` 表示不截断。

    Returns:
        `(inputs, targets)`，形状均为 `(batch, max_len - 1)`。
    """
    raise NotImplementedError


def run_sft(
    model: torch.nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device,
) -> None:
    """执行一轮完整的监督微调（v1 `SFT:491-550` 的训练部分）。

    Args:
        model: 待微调模型（通常已加载预训练权重）。
        train_loader: 训练数据。
        val_loader: 验证数据。
        num_epochs: 轮数（微调通常 2~3）。
        learning_rate: 学习率（比预训练小 1~2 个数量级）。
        weight_decay: 权重衰减。
        device: 计算设备。

    Returns:
        None；副作用是模型参数被更新。
    """
    raise NotImplementedError


def generate_responses(
    model: torch.nn.Module,
    test_data: Sequence[Mapping[str, str]],
    tokenizer: Tokenizer,
    device: str | torch.device,
    max_new_tokens: int = 256,
) -> list[dict[str, str]]:
    """为测试集逐条生成回复（v1 `SFT:518-538`）。

    Args:
        model: 微调后的模型。
        test_data: 测试数据。
        tokenizer: 分词器。
        device: 计算设备。
        max_new_tokens: 单条回复的最大生成长度。

    Returns:
        带 `model_response` 字段的样本列表。
    """
    raise NotImplementedError
