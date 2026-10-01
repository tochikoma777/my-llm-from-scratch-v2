"""滑窗语言建模数据集（搬运 v1 `DATA:21`）。

v2 改动：不再在函数内部构造分词器，改为依赖注入，便于 parity 测试替换 DUMMY 实现。
"""

from my_llm.data.dataset import GPTDatasetV1 as GPTDatasetV1

__all__ = ["GPTDatasetV1"]
