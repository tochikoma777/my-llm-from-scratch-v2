"""KV cache（v2 新增）。

为什么必须有：v1 的 `generate` 每一步都把整段历史塞回模型重算
（`LOAD:398-402`、`GEN:55-63`），生成 T 个 token 的复杂度是 O(T²)。
`SFT:523-529` 用 `max_new_tokens=256` 遍历测试集，是整条流水线最慢的一段。

实现约束：
- **数值必须与无 cache 路径一致**（`tests/test_kv_cache.py` 的核心断言），
  因此只在注意力层复用 K/V，**不改变**任何计算顺序；
- cache 只对自回归解码（query 长度为 1）生效，训练/预填充阶段不启用。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KVCache:
    """逐层缓存注意力 K/V。

    Attributes:
        n_layers: 层数。
        max_seq_len: 单条序列的最大缓存长度（`<= GPTConfig.context_length`）。
        cache: 逐层的 `(keys, values)` 列表，已追加部分。
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """初始化空 cache。

        Args:
            n_layers: Transformer 层数。
            batch_size: 批次大小。
            max_seq_len: 最大缓存长度。
            n_heads: 注意力头数。
            head_dim: 每头维度。
            device: 设备。
            dtype: 数据类型。
        """
        raise NotImplementedError

    @property
    def seq_len(self) -> int:
        """当前已缓存的序列长度。

        Returns:
            已追加的 token 数。
        """
        raise NotImplementedError

    def update(
        self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """追加本步的 K/V 并返回完整缓存。

        Args:
            layer_idx: 层号。
            keys: 本步 key，形状 `(batch, n_heads, new_len, head_dim)`。
            values: 本步 value，形状同上。

        Returns:
            `(full_keys, full_values)`，形状 `(batch, n_heads, cache_len, head_dim)`。

        Raises:
            ValueError: `layer_idx` 越界，或超出 `max_seq_len`。
        """
        raise NotImplementedError

    def reset(self) -> None:
        """清空缓存（新序列开始时调用）。"""
        raise NotImplementedError


def generate_with_cache(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    eos_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """带 KV cache 的自回归生成，输出需与 `generate.generate` 一致。

    Args:
        model: 语言模型（需支持 `forward_with_cache` 钩子，见实现）。
        idx: 起始序列，形状 `(batch, seq_len)`。
        max_new_tokens: 新生成 token 上限。
        eos_id: 结束符 ID。
        temperature: 温度；<= 0 走贪婪。
        top_k: top-k 阈值。
        generator: 随机源。

    Returns:
        完整序列，形状 `(batch, seq_len + generated)`。
    """
    raise NotImplementedError
