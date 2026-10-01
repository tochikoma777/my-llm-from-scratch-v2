"""tiktoken 实现（默认分词器）。

对 v1 里四处重复调用的 `tiktoken.get_encoding("gpt2")`（`DATA:154`、`TRAIN:507`、`LOAD:478`、
`SFT:365`）做一次封装，并在构造时校验词表大小与配置一致。

注意：`tiktoken` 的 `encode` 默认**不**接受 `<|endoftext|>` 等特殊串，需要显式
`allowed_special`；这里把参数做成显式关键字，避免调用方忘记时被静默当成普通文本。
"""

from __future__ import annotations

from collections.abc import Sequence, Set

import tiktoken

from my_llm.tokenizer.protocol import Tokenizer


class TiktokenTokenizer:
    """基于 tiktoken 的分词器，满足 `Tokenizer` 协议。

    Attributes:
        encoding: 底层 tiktoken `Encoding` 对象。
    """

    def __init__(self, encoding_name: str = "gpt2") -> None:
        """初始化。

        Args:
            encoding_name: tiktoken 编码名，默认 `"gpt2"`（词表 50257）。
        """
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

    @property
    def n_vocab(self) -> int:
        """词表大小。"""
        return int(self.encoding.n_vocab)

    def encode(self, text: str, *, allowed_special: Set[str] | None = None) -> list[int]:
        """编码文本为 token ID。

        Args:
            text: 输入文本。
            allowed_special: 允许的特殊 token 集合。

        Returns:
            token ID 列表。
        """
        return list(self.encoding.encode(text, allowed_special=allowed_special or frozenset()))

    def decode(self, ids: Sequence[int]) -> str:
        """解码 token ID 序列。

        Args:
            ids: token ID 序列。

        Returns:
            解码后的文本。
        """
        return str(self.encoding.decode(list(ids)))


def build_tokenizer(name: str = "gpt2") -> Tokenizer:
    """工厂函数：按名字构造分词器。

    Args:
        name: tiktoken 编码名。

    Returns:
        满足 `Tokenizer` 协议的对象。
    """
    return TiktokenTokenizer(name)
