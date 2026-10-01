"""分词器：往返一致性与词表大小。

BPE 对 UTF-8 字节是无损的，所以 `decode(encode(s)) == s` 应该对所有合法输入成立；
这里刻意覆盖空串、超长文本、换行/制表符、HTML 片段与多语言混排。
"""

from __future__ import annotations

import pytest

from my_llm.tokenizer.protocol import Tokenizer
from my_llm.tokenizer.tiktoken_impl import TiktokenTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """gpt2 BPE 分词器（tiktoken 首次调用会下载/读取本地缓存）。"""
    return TiktokenTokenizer()


CASES: list[tuple[str, str]] = [
    ("ascii", "Every effort moves you toward the goal."),
    ("empty", ""),
    ("newline_tabs", "line1\nline2\t\tindented\r\nwindows-end"),
    ("special_chars", '<a href="#">&amp;</a> \\ backslash \x0c formfeed'),
    ("unicode", "你好，世界！こんにちは — emoji: 🚀✨"),
    ("long", ("the quick brown fox jumps over the lazy dog. " * 500).strip()),
]


def test_vocab_size_is_50257(tokenizer: Tokenizer) -> None:
    """词表必须是 50257，否则嵌入查表会越界（`GPTConfig.vocab_size` 同值）。"""
    size = tokenizer.n_vocab
    msg = f"词表大小应为 50257，实际: {size}"
    assert size == 50257, msg


@pytest.mark.parametrize(("name", "text"), CASES, ids=[c[0] for c in CASES])
def test_roundtrip(tokenizer: Tokenizer, name: str, text: str) -> None:
    """`decode(encode(s)) == s`，覆盖多种文本形态。"""
    ids = tokenizer.encode(text)
    restored = tokenizer.decode(ids)
    prefix = repr(restored[:80])
    msg = (
        f"[{name}] 往返不一致: 原文 {len(text)} 字符 → {len(ids)} ids → "
        f"{len(restored)} 字符, got {prefix}"
    )
    assert restored == text, msg
    print(f"[roundtrip/{name}] {len(text)} 字符 -> {len(ids)} ids")


def test_eot_token_roundtrip(tokenizer: Tokenizer) -> None:
    """显式允许 `<|endoftext|>` 时，它能被编码为单个 ID 50256。"""
    special = {"<|endoftext|>"}
    ids = tokenizer.encode("hello<|endoftext|>world", allowed_special=special)
    eot_id = TiktokenTokenizer().eot_token
    msg = f"未在 ids 中找到 eot token: {ids}"
    assert eot_id in ids, msg
    assert TiktokenTokenizer().decode([eot_id]) == "<|endoftext|>"
