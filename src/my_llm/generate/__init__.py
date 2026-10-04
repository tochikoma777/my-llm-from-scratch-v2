"""生成：采样策略与 KV cache。"""

from my_llm.generate.kv_cache import KVCache as KVCache
from my_llm.generate.kv_cache import generate_with_cache as generate_with_cache
from my_llm.generate.sampling import apply_temperature as apply_temperature
from my_llm.generate.sampling import apply_top_k as apply_top_k
from my_llm.generate.sampling import apply_top_p as apply_top_p
from my_llm.generate.sampling import generate as generate
from my_llm.generate.sampling import sample_next_token as sample_next_token

__all__ = [
    "KVCache",
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "generate",
    "generate_with_cache",
    "sample_next_token",
]
