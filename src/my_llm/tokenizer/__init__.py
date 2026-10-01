"""分词器协议与默认实现。

v1 直接使用 `tiktoken.get_encoding("gpt2")`（`DATA:154`、`TRAIN:507`、`LOAD:478`、`SFT:365`），
没有抽象层。v2 抽出 `Tokenizer` 协议，好处：

- 模型/数据模块只依赖协议，未来接入自实现 BPE（`bpe.py`，P3）无需改调用方；
- parity 测试可以用同一个假分词器对比 HF 与本实现。
"""

from my_llm.tokenizer.protocol import Tokenizer as Tokenizer
from my_llm.tokenizer.tiktoken_impl import TiktokenTokenizer as TiktokenTokenizer

__all__ = ["TiktokenTokenizer", "Tokenizer"]
