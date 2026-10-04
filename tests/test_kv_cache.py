"""KV cache：与无 cache 路径的数值一致性 + 加速倍数。

标 `slow` 的原因：要跑 60 个 token 的真实生成才有稳定的耗时比，
快测里跑会拖慢日常提交（`pytest` 默认 `-m 'not slow'`）。

一致性断言只用**贪婪**：采样本身带随机性，且 `generate_with_cache` 的签名里没有 `top_p`，
用 temperature / top-k 对拍会得到"随机但合法"的差异，结论不干净。
"""

from __future__ import annotations

import time

import pytest
import torch

from my_llm.config import GPTConfig
from my_llm.generate.kv_cache import KVCache, generate_with_cache
from my_llm.generate.sampling import generate
from my_llm.model.gpt import GPTModel

pytestmark = pytest.mark.slow

NEW_TOKENS = 60


@pytest.fixture(scope="module")
def model(tiny_cfg: GPTConfig) -> GPTModel:
    """固定种子的 tiny 模型（eval 模式，关闭 dropout 才能保证两条路径一致）。"""
    torch.manual_seed(0)
    m = GPTModel(tiny_cfg)
    m.eval()
    return m


def test_cache_matches_plain_generation(model: GPTModel, tiny_cfg: GPTConfig) -> None:
    """同一 prompt、贪婪解码：有 cache 与无 cache 的输出必须逐 token 相等。"""
    torch.manual_seed(0)
    prompt = torch.randint(0, 1000, (2, 8), dtype=torch.long)

    plain = generate(model, prompt, NEW_TOKENS, tiny_cfg.context_length, temperature=0.0)
    cached = generate_with_cache(model, prompt.clone(), NEW_TOKENS, temperature=0.0)
    msg = f"两条路径不一致:\n无 cache {plain[0].tolist()}\n有 cache {cached[0].tolist()}"
    assert torch.equal(plain, cached), msg


def test_cache_restores_model_forward(model: GPTModel, tiny_cfg: GPTConfig) -> None:
    """旁路结束后必须把 `att.forward` 还原成类方法，不能污染后续调用。"""
    prompt = torch.randint(0, 1000, (1, 4), dtype=torch.long)
    generate_with_cache(model, prompt, 4, temperature=0.0)
    att = model.trf_blocks[0].att
    assert "forward" not in att.__dict__, "实例上不应残留 forward 属性"
    # 还原后再走一次无 cache 路径，结果应仍与另一条一致
    plain = generate(model, prompt, 4, tiny_cfg.context_length, temperature=0.0)
    cached = generate_with_cache(model, prompt.clone(), 4, temperature=0.0)
    assert torch.equal(plain, cached)


def test_cache_speedup_is_measured(model: GPTModel, tiny_cfg: GPTConfig) -> None:
    """记录加速倍数（只打印不断言阈值：不同机器/负载下倍数会浮动）。"""
    torch.manual_seed(0)
    prompt = torch.randint(0, 1000, (1, 16), dtype=torch.long)

    start = time.perf_counter()
    generate(model, prompt, NEW_TOKENS, tiny_cfg.context_length, temperature=0.0)
    plain_secs = time.perf_counter() - start

    start = time.perf_counter()
    generate_with_cache(model, prompt.clone(), NEW_TOKENS, temperature=0.0)
    cached_secs = time.perf_counter() - start

    speedup = plain_secs / cached_secs if cached_secs > 0 else float("inf")
    print(f"\n[KV cache] 无 cache {plain_secs:.3f}s / 有 cache {cached_secs:.3f}s = {speedup:.2f}x")
    assert cached_secs > 0


def test_kv_cache_update_and_reset() -> None:
    """`update` 返回完整 K/V 并推进 `seq_len`；越界与超长都抛 `ValueError`。"""
    cache = KVCache(n_layers=2, batch_size=1, max_seq_len=4, n_heads=2, head_dim=3)
    assert cache.seq_len == 0

    keys = torch.zeros(1, 2, 2, 3)
    values = torch.ones(1, 2, 2, 3)
    full_k, full_v = cache.update(0, keys, values)
    assert cache.seq_len == 2
    assert full_k.shape == (1, 2, 2, 3)
    assert torch.equal(full_v, values)

    full_k, _ = cache.update(0, keys, values)
    assert cache.seq_len == 4
    assert full_k.shape[2] == 4

    with pytest.raises(ValueError, match="缓存溢出"):
        cache.update(0, keys, values)
    with pytest.raises(ValueError, match="layer_idx 越界"):
        cache.update(2, keys, values)

    cache.reset()
    assert cache.seq_len == 0
    assert all(k.shape[2] == 0 for k, _ in cache.cache)
