"""HuggingFace 权重加载（v2 新增，parity 测试的取数路径）。

为什么需要它：v1 只支持 OpenAI TF 检查点，且 `tensorflow` 是硬依赖（`LOAD:24`）。
v2 以 HF 权重为主，TF 兼容层退化为可选（`weights/openai_tf.py`）。

键名对照（实测 transformers 5.12.0 的 `GPT2LMHeadModel.state_dict()`）：

    transformer.wte.weight            -> tok_emb.weight
    transformer.wpe.weight            -> pos_emb.weight
    transformer.h.{i}.ln_1.weight     -> trf_blocks[i].norm1.scale
    transformer.h.{i}.attn.c_attn.*   -> trf_blocks[i].att.W_{query,key,value}.*（切 q/k/v）
    transformer.h.{i}.attn.c_proj.*   -> trf_blocks[i].att.out_proj.*（转置）
    transformer.h.{i}.mlp.c_fc.*      -> trf_blocks[i].ff.fc1.*（转置）
    transformer.h.{i}.mlp.c_proj.*    -> trf_blocks[i].ff.fc2.*（转置）
    transformer.ln_f.*                -> final_norm.*
    lm_head.weight                    -> 不加载（已与 tok_emb 共享）

用法见 `scripts/download_weights.py`。
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from my_llm.config import GPTConfig
from my_llm.model.gpt import GPTModel


def load_hf_state_dict(
    model_name: str = "gpt2", *, revision: str = "main"
) -> dict[str, torch.Tensor]:
    """从 HF Hub 拉取 `GPT2LMHeadModel` 的 state_dict。

    Args:
        model_name: Hub 上的模型名，如 `"gpt2"` / `"gpt2-medium"`。
        revision: 副本分支/提交号。

    Returns:
        HF 原生键名的 state_dict。

    Raises:
        ValueError: 拉取结果不是 GPT2LMHeadModel。
    """
    raise NotImplementedError


def hf_key_to_ours(key: str) -> str:
    """单个 HF 键名 → 本仓库参数名（含 `.T` 需求由调用方处理）。

    Args:
        key: 例如 `"transformer.h.0.attn.c_attn.weight"`。

    Returns:
        本仓库侧的参数名；无法映射时返回空字符串（如 `lm_head.weight`）。
    """
    raise NotImplementedError


def load_weights_from_hf(gpt: GPTModel, hf_state: Mapping[str, torch.Tensor]) -> None:
    """把 HF state_dict 写入 `GPTModel`。

    会校验 `lm_head` 与 `wte` 在 HF 侧是否一致，并在写入后确认 tie 仍然成立。

    Args:
        gpt: 目标模型。
        hf_state: HF 原生键名的 state_dict。

    Raises:
        ValueError: 键缺失或形状不匹配。
    """
    raise NotImplementedError


def gpt_config_from_hf(model_name: str = "gpt2") -> GPTConfig:
    """读取 HF 模型的配置并转成 `GPTConfig`。

    Args:
        model_name: Hub 上的模型名。

    Returns:
        对齐该变体的配置（`qkv_bias` 恒为 True）。
    """
    raise NotImplementedError
