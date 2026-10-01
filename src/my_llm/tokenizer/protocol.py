"""Tokenizer 抽象协议。

只声明 v2 真正用到的最小接口，刻意不把 tiktoken 的 API 全盘搬进来
（例如 `allowed_special` 的多种取值保持为集合形式，而不是 `set | str`）。

实现者：
- `my_llm.tokenizer.tiktoken_impl.TiktokenTokenizer`（默认）
- `my_llm.tokenizer.bpe`（P3，待实现）
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """分词器协议。

    约定：
    - `encode` 必须能处理任意 UTF-8 文本；未显式允许的特殊 token 一律按普通文本处理；
    - `decode` 是 `encode` 的逆操作（对合法 token ID 序列而言）；
    - `n_vocab` 必须与模型 `GPTConfig.vocab_size` 一致，否则 embedding 查表会越界。
    """

    @property
    def n_vocab(self) -> int:
        """词表大小（含特殊 token）。"""
        ...

    def encode(self, text: str, *, allowed_special: Set[str] | None = None) -> list[int]:
        """把文本编码为 token ID 列表。

        Args:
            text: 输入文本。
            allowed_special: 允许按特殊字串处理的 token 集合，例如 `{"<|endoftext|>"}`；
                为 `None` 时表示不允许任何特殊 token（按普通 BPE 处理）。

        Returns:
            token ID 列表。
        """
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """把 token ID 序列解码回文本。

        Args:
            ids: token ID 序列。

        Returns:
            解码后的文本。
        """
        ...
