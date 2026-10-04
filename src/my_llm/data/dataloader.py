"""DataLoader 工厂（v1 `DATA:122`）。

保持 v1 的默认值语义，但把分词器改为必填的显式依赖，避免"每次调用都新建 encoder"。
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from my_llm.data.dataset import GPTDatasetV1
from my_llm.tokenizer.protocol import Tokenizer


def create_dataloader_v1(
    txt: str,
    tokenizer: Tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """创建语言建模 DataLoader。

    Args:
        txt: 原始文本。
        tokenizer: 分词器。
        batch_size: 批大小。
        max_length: 序列长度。
        stride: 滑窗步长。
        shuffle: 是否打乱。
        drop_last: 是否丢弃最后不完整批次。
        num_workers: 数据加载进程数。

    Returns:
        产出 `(inputs, targets)` 的 DataLoader，两个张量形状均为 `(batch_size, max_length)`。
    """
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
