"""生成：采样策略与 KV cache。"""

from my_llm.generate.sampling import generate as generate
from my_llm.generate.sampling import sample_next_token as sample_next_token

__all__ = ["generate", "sample_next_token"]
