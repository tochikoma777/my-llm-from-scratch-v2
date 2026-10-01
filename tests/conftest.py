"""pytest 公共 fixture。

parity 相关约定（见 CODEBUDDY.md 硬约束 2）：
- 所有对比类测试跑 CPU + fp32，阈值 1e-5 不可放宽；
- 需要真实 GPT-2 权重的 fixture（如下一轮的 `hf_gpt2`）必须配 `@pytest.mark.slow` 的用例。

`DEVICE` / `DTYPE` 放在模块级常量而不是 fixture 里，是为了让测试能直接 import 使用，
也方便一眼看出这套测试不碰 CUDA。

parity fixture（`hf_gpt2` / `our_gpt2_loaded`）会真实下载 GPT-2 权重，只能被标了
`@pytest.mark.slow` 的用例使用；下载慢时走 `make test-full`（它注入
`HF_ENDPOINT=https://hf-mirror.com`）。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from my_llm.config import GPTConfig
from my_llm.model.gpt import GPTModel
from my_llm.weights.hf import load_hf_weights_into_gpt

DEVICE = "cpu"
DTYPE = torch.float32

# 相对仓库根的路径；pytest 从根运行，但这里用 __file__ 定位，避免换 CWD 就崩
REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_CONFIG_PATH = REPO_ROOT / "configs" / "gpt2-tiny.yaml"
SMALL_CONFIG_PATH = REPO_ROOT / "configs" / "gpt2-small.yaml"


@pytest.fixture(scope="session")
def tiny_cfg() -> GPTConfig:
    """tiny 配置（等价 `GPTConfig.gpt2_tiny()`，唯一来源仍是 yaml）。"""
    return GPTConfig.from_yaml(TINY_CONFIG_PATH)


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg: GPTConfig) -> GPTModel:
    """随机初始化的 tiny 模型：`eval()` + `no_grad`，不含真实权重。

    Args:
        tiny_cfg: `configs/gpt2-tiny.yaml` 读出的配置。

    Returns:
        参数已确定、`requires_grad` 仍为 True 但不做反向的 GPTModel。
    """
    torch.manual_seed(0)
    model = GPTModel(tiny_cfg)
    model.eval()
    model.to(device=DEVICE, dtype=DTYPE)
    return model


@pytest.fixture
def sample_ids() -> torch.Tensor:
    """固定的 token ID 批次 `(2, 16)`，值域 `[0, 10000)`。"""
    torch.manual_seed(0)
    return torch.randint(0, 10000, (2, 16), device=DEVICE)


@pytest.fixture(scope="session")
def hf_gpt2() -> Any:  # noqa: ANN401  # transformers 无类型存根，实际是 GPT2LMHeadModel
    """HF 官方 `gpt2`（124M），parity 的基准侧。

    显式指定 `attn_implementation="eager"`：默认的 sdpa 不返回注意力权重
    （`output_attentions=True` 时 `attentions` 是空元组），eager 才是纯 PyTorch 数学路径。

    Returns:
        eval 模式、CPU + fp32 的 `GPT2LMHeadModel`。
    """
    pytest.importorskip("transformers")
    from transformers import GPT2LMHeadModel  # noqa: PLC0415  # 惰性导入：可选依赖

    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    model.to(device=DEVICE, dtype=DTYPE)
    return model


@pytest.fixture(scope="session")
def our_gpt2_loaded(hf_gpt2: Any) -> GPTModel:  # noqa: ANN401
    """加载了同一份 HF 权重的本仓库 `GPTModel`。

    Args:
        hf_gpt2: 基准模型，取它的 state_dict 作为权重来源。

    Returns:
        eval 模式、CPU + fp32 的 `GPTModel`。

    Raises:
        AssertionError: 有 HF 键未能映射。
    """
    cfg = GPTConfig.from_yaml(SMALL_CONFIG_PATH)
    model = GPTModel(cfg)
    # 权重会被完整覆盖，manual_seed 只为让未覆盖部分（不存在，但保持可复现）稳定
    torch.manual_seed(0)
    unmatched = load_hf_weights_into_gpt(model, hf_gpt2.state_dict())
    unmatched_msg = f"有 HF 键未能映射: {sorted(unmatched)}"
    assert not unmatched, unmatched_msg
    model.eval()
    model.to(device=DEVICE, dtype=DTYPE)
    return model


@pytest.fixture(scope="session")
def parity_fp64(hf_gpt2: Any, our_gpt2_loaded: GPTModel) -> tuple[Any, GPTModel]:  # noqa: ANN401
    """fp64 副本 `(hf, ours)`，用于绝对 1e-5 断言。

    fp32 下 GPT-2 残差流的量级达到 3e3，相对误差 ~1e-7 换算成绝对误差就是 ~2e-4，
    高于 1e-5 阈值——这不是实现差异（fp64 对拍互差 ~1e-13）。因此绝对阈值放在 fp64 跑，
    fp32 按张量量级缩放后断言（见 `tests/test_parity_hf.py` 的 `_assert_parity`）。

    Returns:
        `(hf_gpt2 的 fp64 副本, our_gpt2_loaded 的 fp64 副本)`，均为 eval 模式。
    """
    hf64 = copy.deepcopy(hf_gpt2).double().eval()
    ours64 = copy.deepcopy(our_gpt2_loaded).double().eval()
    return hf64, ours64
