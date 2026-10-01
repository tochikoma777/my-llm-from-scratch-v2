"""模型本体：归一化 / 注意力 / Transformer 块 / GPT。

搬运自 v1 `src/modules/language_module.py`，按职责拆成四个文件。
重新导出仅使用显式形式（mypy strict 的 `no_implicit_reexport`）。
"""

from my_llm.model.attention import MultiHeadAttention as MultiHeadAttention
from my_llm.model.block import FeedForward as FeedForward
from my_llm.model.block import TransformerBlock as TransformerBlock
from my_llm.model.gpt import GPTModel as GPTModel
from my_llm.model.norm import LayerNorm as LayerNorm
from my_llm.model.norm import NewGELU as NewGELU

__all__ = [
    "FeedForward",
    "GPTModel",
    "LayerNorm",
    "MultiHeadAttention",
    "NewGELU",
    "TransformerBlock",
]
