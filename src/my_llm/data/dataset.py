"""数据：滑窗数据集与 DataLoader 构建。

搬运来源：v1 `src/modules/data_preprocess.py:21` `GPTDatasetV1`、`:122` `create_dataloader_v1`。

自回归样本对的构造规则（不变）：
    输入 = `token_ids[i : i + max_length]`
    目标 = `token_ids[i + 1 : i + max_length + 1]`（整体右移一位）

相对 v1 的调整：分词器由**外部注入**（依赖 `Tokenizer` 协议），而不是在函数内部
`tiktoken.get_encoding("gpt2")`，否则每次构造 DataLoader 都会重建编码器，也无法替换实现。
"""

from __future__ import annotations

from typing import Final

import torch
from torch.utils.data import Dataset

from my_llm.tokenizer.protocol import Tokenizer

# v1 `DATA:59` 显式允许 `<|endoftext|>`，否则它会被当成普通文本切成 BPE 片段。
# 提到模块常量是为了让调用方不必关心这个细节，同时避免每个样本都新建集合。
_DEFAULT_ALLOWED_SPECIAL: Final[frozenset[str]] = frozenset({"<|endoftext|>"})


class GPTDatasetV1(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """滑窗语言建模数据集（v1 `DATA:21`）。

    Attributes:
        input_ids: 输入序列列表，每个元素形状 `(max_length,)`。
        target_ids: 目标序列列表，每个元素形状 `(max_length,)`。
    """

    def __init__(self, txt: str, tokenizer: Tokenizer, max_length: int, stride: int) -> None:
        """切分文本为训练样本。

        Args:
            txt: 原始文本。
            tokenizer: 分词器（依赖注入）。
            max_length: 每个样本的序列长度。
            stride: 滑动步长；等于 `max_length` 时样本无重叠。

        Raises:
            ValueError: `max_length` 或 `stride` 非正数。

        Note:
            文本编码后若不足 `max_length + 1` 个 token，则**一个样本都生成不了**
            （v1 `DATA:63` 的 `range` 行为原样保留）。这是调用方需要保证的前提，
            不是本类的缺陷——`create_dataloader_v1` 拿到空数据集后，
            `calc_loss_loader` 会以 `nan` 显式暴露（见 `train/losses.py`）。
        """
        if max_length <= 0:
            msg = f"max_length 必须为正，收到 {max_length}"
            raise ValueError(msg)
        if stride <= 0:
            msg = f"stride 必须为正，收到 {stride}"
            raise ValueError(msg)

        token_ids = tokenizer.encode(txt, allowed_special=_DEFAULT_ALLOWED_SPECIAL)
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        for start in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[start : start + max_length]))
            self.target_ids.append(torch.tensor(token_ids[start + 1 : start + max_length + 1]))

    def __len__(self) -> int:
        """返回样本数量。

        Returns:
            样本总数。
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """取单个样本。

        Args:
            idx: 样本索引。

        Returns:
            `(input_tensor, target_tensor)`，均为形状 `(max_length,)` 的 int64 张量。
        """
        return self.input_ids[idx], self.target_ids[idx]
