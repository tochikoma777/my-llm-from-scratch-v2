"""滑窗数据集与 DataLoader 的切窗正确性。

这里刻意用**字符级 dummy 分词器**而不是 tiktoken：这两块只关心"整数序列怎么切窗"，
与真实 BPE 无关；用 dummy 还能精确控制 token 数（1 字符 = 1 token），
并且让快测完全离线（`test_tokenizer.py` 才需要真实词表）。
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence, Set

import pytest
import torch

from my_llm.data import GPTDatasetV1, create_dataloader_v1
from my_llm.tokenizer.protocol import Tokenizer


class CharTokenizer:
    """字符级分词器：`encode` 即逐字符 `ord()`，满足 `Tokenizer` 协议。"""

    @property
    def n_vocab(self) -> int:
        """词表大小（ASCII 范围足够测试用）。"""
        return 256

    def encode(self, text: str, *, allowed_special: Set[str] | None = None) -> list[int]:
        """把文本逐字符转成码点。

        Args:
            text: 输入文本。
            allowed_special: 忽略，字符级分词没有特殊 token。

        Returns:
            码点列表。
        """
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        """`encode` 的逆操作。

        Args:
            ids: 码点序列。

        Returns:
            还原出的文本。
        """
        return "".join(chr(int(i)) for i in ids)


TEXT = "abcdefghijklmnopqrstuvwxyz"  # 26 字符 → 26 个 token
MAX_LENGTH = 4
STRIDE = 2


@pytest.fixture
def tokenizer() -> Tokenizer:
    """字符级分词器。"""
    return CharTokenizer()


def _ord_ids(text: str) -> list[int]:
    """文本 → token ID 列表（与 `CharTokenizer` 同规则）。"""
    return [ord(c) for c in text]


def test_target_is_input_shifted_by_one(tokenizer: Tokenizer) -> None:
    """`target` 必须是 `input` 整体右移一位的结果。"""
    dataset = GPTDatasetV1(TEXT, tokenizer, max_length=MAX_LENGTH, stride=1)
    ids = _ord_ids(TEXT)
    expected_len = len(ids) - MAX_LENGTH
    msg = f"样本数应为 {expected_len}（v1 range 语义），实际 {len(dataset)}"
    assert len(dataset) == expected_len, msg

    for idx in (0, 1, len(dataset) - 1):
        inputs, targets = dataset[idx]
        assert torch.equal(inputs, torch.tensor(ids[idx : idx + MAX_LENGTH]))
        assert torch.equal(targets, torch.tensor(ids[idx + 1 : idx + MAX_LENGTH + 1]))
        # 错位关系：target 去掉最后一个 == input 去掉第一个
        assert torch.equal(targets[:-1], inputs[1:])


def test_stride_controls_overlap(tokenizer: Tokenizer) -> None:
    """`stride` 决定相邻样本起点的间隔（等于 `max_length` 时无重叠）。"""
    ids = _ord_ids(TEXT)
    dataset = GPTDatasetV1(TEXT, tokenizer, max_length=MAX_LENGTH, stride=STRIDE)
    for idx in range(len(dataset)):
        (inputs, _) = dataset[idx]
        start = idx * STRIDE
        assert torch.equal(inputs, torch.tensor(ids[start : start + MAX_LENGTH]))

    no_overlap = GPTDatasetV1(TEXT, tokenizer, max_length=MAX_LENGTH, stride=MAX_LENGTH)
    (first, _) = no_overlap[0]
    (second, _) = no_overlap[1]
    assert not torch.equal(first, second)


def test_text_shorter_than_max_length_is_empty(tokenizer: Tokenizer) -> None:
    """文本长度不足 `max_length` 时**不崩溃**，样本数为 0（v1 `DATA:63` 行为）。"""
    short = "abc"
    dataset = GPTDatasetV1(short, tokenizer, max_length=MAX_LENGTH, stride=1)
    msg = f"短文本应产出 0 个样本，实际 {len(dataset)}"
    assert len(dataset) == 0, msg


def test_non_positive_window_raises(tokenizer: Tokenizer) -> None:
    """`max_length` / `stride` 非正数时报错，而不是静默产出空数据集。"""
    with pytest.raises(ValueError, match="max_length"):
        GPTDatasetV1(TEXT, tokenizer, max_length=0, stride=1)
    with pytest.raises(ValueError, match="stride"):
        GPTDatasetV1(TEXT, tokenizer, max_length=MAX_LENGTH, stride=0)


def test_dataloader_batch_shape(tokenizer: Tokenizer) -> None:
    """DataLoader 产出的 batch 形状为 `(batch_size, max_length)`，dtype 为 int64。"""
    batch_size = 2
    loader = create_dataloader_v1(
        TEXT,
        tokenizer,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        shuffle=False,
        drop_last=True,
    )
    inputs, targets = next(iter(loader))
    assert inputs.shape == (batch_size, MAX_LENGTH)
    assert targets.shape == (batch_size, MAX_LENGTH)
    assert inputs.dtype is torch.int64
    assert targets.dtype is torch.int64


def test_dataloader_drops_last_partial_batch(tokenizer: Tokenizer) -> None:
    """`drop_last=True` 时不返回不足 `batch_size` 的尾部批次。"""
    dataset = GPTDatasetV1(TEXT, tokenizer, max_length=MAX_LENGTH, stride=STRIDE)
    expected_batches = len(dataset) // 3
    loader = create_dataloader_v1(
        TEXT,
        tokenizer,
        batch_size=3,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        shuffle=False,
        drop_last=True,
    )
    msg = f"批次数应为 {expected_batches}（丢弃尾部），实际 {len(loader)}"
    assert len(loader) == expected_batches, msg


def test_dataloader_defaults_are_v1_compatible() -> None:
    """默认值必须与 v1 一致，且**只**出现在函数签名里（不散落在实现内部）。"""
    sig = inspect.signature(create_dataloader_v1)
    empty = inspect.Parameter.empty
    defaults = {name: p.default for name, p in sig.parameters.items() if p.default is not empty}
    expected = {
        "batch_size": 4,
        "max_length": 256,
        "stride": 128,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 0,
    }
    msg = f"默认值与 v1 不一致: {defaults}"
    assert defaults == expected, msg
