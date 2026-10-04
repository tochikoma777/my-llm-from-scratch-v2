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

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from my_llm.generate.sampling import sample_next_token


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
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self._len = 0
        # 每层一对空张量，`update` 用 cat 追加（返回完整缓存即可，
        # 预分配 + 切片赋值更快但会把"已缓存长度"藏在切片里，可读性更差）
        shape = (batch_size, n_heads, 0, head_dim)
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in range(n_layers)
        ]

    @property
    def seq_len(self) -> int:
        """当前已缓存的序列长度。

        Returns:
            已追加的 token 数。
        """
        return self._len

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
        if not 0 <= layer_idx < self.n_layers:
            msg = f"layer_idx 越界: {layer_idx}，合法范围 [0, {self.n_layers})"
            raise ValueError(msg)
        new_len = keys.shape[2]
        if self._len + new_len > self.max_seq_len:
            msg = f"缓存溢出：已缓存 {self._len} + 新增 {new_len} > max_seq_len {self.max_seq_len}"
            raise ValueError(msg)
        old_keys, old_values = self.cache[layer_idx]
        full_keys = torch.cat((old_keys, keys), dim=2)
        full_values = torch.cat((old_values, values), dim=2)
        self.cache[layer_idx] = (full_keys, full_values)
        self._len = full_keys.shape[2]
        return full_keys, full_values

    def reset(self) -> None:
        """清空缓存（新序列开始时调用）。"""
        shape = (self.batch_size, self.n_heads, 0, self.head_dim)
        keys, values = self.cache[0]
        self.cache = [
            (
                torch.zeros(shape, device=keys.device, dtype=keys.dtype),
                torch.zeros(shape, device=values.device, dtype=values.dtype),
            )
            for _ in range(self.n_layers)
        ]
        self._len = 0


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
        model: 语言模型。**不需要**它支持任何 cache 接口：K/V 通过 `W_key` / `W_value`
            的 forward hook 捕获，解码时用实例级旁路替换每层 `attention.forward`，
            位置嵌入用 hook 换成"从当前绝对位置取窗口"。`model/` 下的一行代码都不改。
        idx: 起始序列，形状 `(batch, seq_len)`。
        max_new_tokens: 新生成 token 上限。
        eos_id: 结束符 ID。
        temperature: 温度；<= 0 走贪婪。
        top_k: top-k 阈值。
        generator: 随机源。

    Returns:
        完整序列，形状 `(batch, seq_len + generated)`。

    Raises:
        ValueError: 模型没有 `trf_blocks` / `pos_emb`（旁路实现靠这两个属性定位）。

    Note:
        与 `generate` 的差异只在 attention 的 matmul 形状（query 长度 1 对全部 key，
        而不是 T×T 取最后一行），其余算子逐字相同，因此数值一致。
    """
    blocks = getattr(model, "trf_blocks", None)
    pos_emb = getattr(model, "pos_emb", None)
    if blocks is None or pos_emb is None:
        msg = "generate_with_cache 需要模型具备 trf_blocks / pos_emb 属性"
        raise ValueError(msg)

    model.eval()
    device = idx.device
    context_size = int(pos_emb.weight.shape[0])
    # 与无 cache 路径对齐：超长 prompt 先截到 context_size，生成量也不越过 context_size
    if idx.shape[1] > context_size:
        idx = idx[:, -context_size:]
    max_new_tokens = max(0, min(max_new_tokens, context_size - idx.shape[1]))

    blocks_any: Any = blocks
    first_att: Any = blocks_any[0].att
    cache = KVCache(
        n_layers=len(blocks),
        batch_size=idx.shape[0],
        max_seq_len=idx.shape[1] + max_new_tokens,
        n_heads=int(first_att.num_heads),
        head_dim=int(first_att.head_dim),
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    with torch.no_grad():
        # ① 预填充：hook 拿到的是 W_key / W_value 的**真实输出**，不重算、不改数值路径
        pending: list[dict[str, torch.Tensor]] = [{} for _ in range(len(blocks))]
        hooks: list[RemovableHandle] = []
        for layer_idx, block in enumerate(blocks_any):
            att = block.att
            hooks.append(att.W_key.register_forward_hook(_capture(pending[layer_idx], "k", att)))
            hooks.append(att.W_value.register_forward_hook(_capture(pending[layer_idx], "v", att)))
            hooks.append(att.register_forward_hook(_commit(cache, layer_idx, pending[layer_idx])))
        logits = model(idx)
        for hook in hooks:
            hook.remove()

        next_token = sample_next_token(
            logits, temperature=temperature, top_k=top_k, generator=generator
        )
        idx = torch.cat((idx, next_token.to(idx.dtype)), dim=1)

        # ② 解码：每层 attention 换成"只算新 token + 复用 cache"的旁路实现；
        #    位置嵌入用 hook 换成"从当前绝对位置取窗口"，否则单 token 输入永远取 position 0
        offset = {"start": idx.shape[1] - 1}
        pos_hook = pos_emb.register_forward_hook(_window(pos_emb, offset))
        # 旁路只改实例属性（att.forward），恢复时 del 掉让类方法重新可见
        patched = [(block.att, block.att.forward) for block in blocks_any]
        for layer_idx, (att, _original) in enumerate(patched):
            att.forward = _make_cached_forward(att, cache, layer_idx)
        try:
            for _step in range(max_new_tokens - 1):
                offset["start"] = idx.shape[1] - 1
                step_logits = model(idx[:, -1:])
                next_token = sample_next_token(
                    step_logits, temperature=temperature, top_k=top_k, generator=generator
                )
                idx = torch.cat((idx, next_token.to(idx.dtype)), dim=1)
                if eos_id is not None and bool((idx[:, -1] == eos_id).all()):
                    break
        finally:
            pos_hook.remove()
            for att, _original in patched:
                del att.forward
    return idx


def _capture(sink: dict[str, torch.Tensor], key: str, att: nn.Module) -> Callable[..., None]:
    """构造一个把 `W_key` / `W_value` 输出按多头形状收进 `sink` 的 hook。

    Args:
        sink: 该层的暂存字典。
        key: `"k"` 或 `"v"`。
        att: 所属注意力模块，用于取 `num_heads` / `head_dim`。

    Returns:
        可直接注册到 `register_forward_hook` 的回调。
    """

    mod: Any = att

    def hook(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
        batch_size, num_tokens, _emb = output.shape
        heads = int(mod.num_heads)
        head_dim = int(mod.head_dim)
        sink[key] = output.view(batch_size, num_tokens, heads, head_dim).transpose(1, 2)

    return hook


def _commit(cache: KVCache, layer_idx: int, sink: dict[str, torch.Tensor]) -> Callable[..., None]:
    """构造一个在注意力算完后把 K/V 提交进 cache 的 hook。

    Args:
        cache: 目标 KV cache。
        layer_idx: 层号。
        sink: 与 `_capture` 共用的暂存字典。

    Returns:
        可直接注册到 `register_forward_hook` 的回调。
    """

    def hook(_module: nn.Module, _args: tuple[Any, ...], _output: torch.Tensor) -> None:
        cache.update(layer_idx, sink["k"], sink["v"])

    return hook


def _window(pos_emb: nn.Module, offset: dict[str, int]) -> Callable[..., torch.Tensor]:
    """构造一个把位置嵌入替换成"当前绝对位置那一段"的 hook。

    单 token 解码时 `GPTModel.forward` 传进来的是 `arange(1)`，直接查表会永远命中
    position 0；这里按 `offset["start"]` 取窗口，等价于无 cache 路径的第 p 行。

    Args:
        pos_emb: 位置嵌入模块（`nn.Embedding`）。
        offset: 可变字典，`"start"` 为当前绝对位置。

    Returns:
        可直接注册到 `register_forward_hook` 的回调。
    """

    def hook(module: nn.Module, args: tuple[Any, ...], _output: torch.Tensor) -> torch.Tensor:
        num_tokens = int(torch.as_tensor(args[0]).numel())
        start = offset["start"]
        weight = cast(torch.Tensor, module.weight)
        return weight[start : start + num_tokens]

    return hook


def _make_cached_forward(
    att: nn.Module, cache: KVCache, layer_idx: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """构造"只算新 token、复用 cache"的注意力旁路实现。

    算子顺序与 `model/attention.py:84` 的 `forward` 逐字一致，只是 query 长度为 1、
    K/V 来自 cache，且不需要因果掩码（所有缓存 key 的位置都不在 query 之后）。

    Args:
        att: 注意力模块。
        cache: KV cache。
        layer_idx: 层号。

    Returns:
        替换 `att.forward` 的函数（实例属性赋值不绑定 self，故只接收 `x`）。
    """

    mod: Any = att

    def cached_forward(x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _emb = x.shape
        heads = int(mod.num_heads)
        head_dim = int(mod.head_dim)
        keys_new = (
            cast(torch.Tensor, mod.W_key(x))
            .view(batch_size, num_tokens, heads, head_dim)
            .transpose(1, 2)
        )
        values_new = (
            cast(torch.Tensor, mod.W_value(x))
            .view(batch_size, num_tokens, heads, head_dim)
            .transpose(1, 2)
        )
        queries = (
            cast(torch.Tensor, mod.W_query(x))
            .view(batch_size, num_tokens, heads, head_dim)
            .transpose(1, 2)
        )
        keys, values = cache.update(layer_idx, keys_new, values_new)
        attn_scores = (queries @ keys.transpose(2, 3)) / (head_dim**0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = cast(torch.Tensor, mod.dropout(attn_weights))
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, int(mod.d_out))
        return cast(torch.Tensor, mod.out_proj(context_vec))

    return cached_forward
