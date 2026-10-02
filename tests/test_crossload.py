"""双路径交叉验证：OpenAI TF 检查点 vs HuggingFace 权重。

同一个 `gpt2`（124M）可以走两条完全独立的路装进我们的 `GPTModel`：

- **TF 路**：`openai_tf.load_openai_tf_weights_into_gpt`，从 OpenAI 的 TensorFlow 检查点读，
  QKV 是"先按最后一维三分、再各自转置"（v1 的写法）；
- **HF 路**：`hf.load_hf_weights_into_gpt`，从 `GPT2LMHeadModel` 的 state_dict 读，
  QKV 是"先整体 `.T`、再三等分"。

两条路径的数值**必须一致**——任何一处转置方向写反、QKV 顺序写错、LayerNorm 的 g/b 接反，
都会在这里现形。这是对单看一侧最难发现的一类 bug 的防线。

跳过约定（`slow` + skip 而非 fail）：
- 没装 tensorflow → skip（TF 只是可选依赖，不是失败）；
- OpenAI 的 Azure 源下不动 → skip 并说明原因，不用 fail 掩盖网络问题。

注意分工：`tests/test_parity_hf.py` 是「我们 vs HF」的对拍，本文件是「TF 路 vs HF 路」的对拍，
两者互补、不重复。
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
from my_llm.weights.openai_tf import download_and_load_gpt2, load_openai_tf_weights_into_gpt

pytestmark = pytest.mark.slow

ATOL = 1e-5
DEVICE = "cpu"
DTYPE = torch.float32

REPO_ROOT = Path(__file__).resolve().parents[1]
SMALL_CONFIG_PATH = REPO_ROOT / "configs" / "gpt2-small.yaml"
# 500MB 的检查点落在 gitignore 覆盖的输出目录里，换会话也能复用
TF_MODELS_DIR = REPO_ROOT / "outputs" / "openai-tf"


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """两个张量的最大绝对差。"""
    return (a - b).abs().max().item()


def _assert_same(ours: torch.Tensor, theirs: torch.Tensor, label: str) -> None:
    """断言两侧数值一致，失败时带上实际 diff。

    Args:
        ours: TF 路径的结果。
        theirs: HF 路径的结果。
        label: 断言消息里用的名字，如 `"logits"`。

    Raises:
        AssertionError: diff 超过阈值。
    """
    diff = _max_diff(ours, theirs)
    msg = f"{label}: max|diff|={diff:.3e}（阈值 {ATOL:.0e}）; shape={tuple(ours.shape)}"
    assert diff < ATOL, msg
    print(f"[crossload] {msg}")


def _new_model(cfg: GPTConfig) -> GPTModel:
    """构造一个会被权重完整覆盖的空白模型。

    Args:
        cfg: 模型配置。

    Returns:
        eval 模式、CPU + fp32 的 `GPTModel`。
    """
    torch.manual_seed(0)
    model = GPTModel(cfg)
    model.eval()
    return model.to(device=DEVICE, dtype=DTYPE)


def _tf_params_from_hf_state(hf_state: dict[str, torch.Tensor], n_layers: int) -> dict[str, Any]:
    """把 HF state_dict 摆成 OpenAI TF 检查点的嵌套参数形态。

    用途：在不装 tensorflow、不下 500MB 的情况下，单独验证 TF 路加载器的自身正确性
    （QKV 切分顺序、各处 `.T`、LayerNorm 的 g/b 映射）。
    TF 与 HF 的 Conv1D 都按 `(in, out)` 存权重，所以 c_attn / c_proj / c_fc 可以直接用同一份数据。

    Args:
        hf_state: HF 原生键名的 state_dict。
        n_layers: 层数。

    Returns:
        `load_openai_tf_weights_into_gpt` 能直接吃的嵌套字典。

    Raises:
        KeyError: HF state_dict 里缺少预期的键。
    """

    def _npy(key: str) -> Any:
        tensor = hf_state[key]
        if tensor is None:
            msg = f"HF state_dict 缺少 {key}"
            raise KeyError(msg)
        return tensor.detach().numpy()

    blocks: list[dict[str, Any]] = []
    for index in range(n_layers):
        prefix = f"transformer.h.{index}."
        blocks.append(
            {
                "attn": {
                    "c_attn": {
                        "w": _npy(prefix + "attn.c_attn.weight"),
                        "b": _npy(prefix + "attn.c_attn.bias"),
                    },
                    "c_proj": {
                        "w": _npy(prefix + "attn.c_proj.weight"),
                        "b": _npy(prefix + "attn.c_proj.bias"),
                    },
                },
                "mlp": {
                    "c_fc": {
                        "w": _npy(prefix + "mlp.c_fc.weight"),
                        "b": _npy(prefix + "mlp.c_fc.bias"),
                    },
                    "c_proj": {
                        "w": _npy(prefix + "mlp.c_proj.weight"),
                        "b": _npy(prefix + "mlp.c_proj.bias"),
                    },
                },
                "ln_1": {"g": _npy(prefix + "ln_1.weight"), "b": _npy(prefix + "ln_1.bias")},
                "ln_2": {"g": _npy(prefix + "ln_2.weight"), "b": _npy(prefix + "ln_2.bias")},
            }
        )
    return {
        "wte": _npy("transformer.wte.weight"),
        "wpe": _npy("transformer.wpe.weight"),
        "blocks": blocks,
        "g": _npy("transformer.ln_f.weight"),
        "b": _npy("transformer.ln_f.bias"),
    }


@pytest.fixture(scope="session")
def tf_params() -> dict[str, Any]:
    """真实的 OpenAI TF 检查点参数。

    Returns:
        `load_gpt2_params_from_tf_ckpt` 风格的嵌套字典。

    Raises:
        Skipped: 未装 tensorflow，或下载失败（网络问题不是 bug，不该 fail）。
    """
    pytest.importorskip("tensorflow", reason="未安装 tensorflow（TF 路径是可选依赖）")
    try:
        _settings, params = download_and_load_gpt2("124M", TF_MODELS_DIR)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"无法获取 OpenAI TF 检查点：{type(exc).__name__}: {exc}")
    return params


def test_openai_tf_loader_matches_hf_loader(
    hf_gpt2: Any,  # noqa: ANN401  # transformers 无类型存根
    our_gpt2_loaded: GPTModel,
    sample_ids: torch.Tensor,
) -> None:
    """TF 路加载器本身 vs HF 路加载器：同一份 HF 权重，两条 loader 必须得出同一个模型。

    这条用例不需要 tensorflow、不需要下载 OpenAI 权重（HF 权重已缓存在 fixture 里），
    因此即使本机没装 TF 也能跑。它覆盖的是**映射逻辑**（QKV 顺序 / 转置方向 / g-b 映射）。

    它覆盖不到的那一环——TF 变量的真实命名（是不是真的叫 `model/h0/attn/c_attn/w`）——
    由下面 `test_tf_and_hf_agree` 用真实检查点兜住。

    Args:
        hf_gpt2: HF 官方模型，仅用于取 state_dict。
        our_gpt2_loaded: 已装好 HF 权重的本仓库模型（基准侧）。
        sample_ids: 固定 token 批次。
    """
    cfg = GPTConfig.from_yaml(SMALL_CONFIG_PATH)
    tf_params = _tf_params_from_hf_state(dict(hf_gpt2.state_dict()), cfg.n_layers)

    tf_model = _new_model(cfg)
    unmatched = load_openai_tf_weights_into_gpt(tf_model, tf_params)
    unmatched_msg = f"有 TF 变量未能映射到本仓库参数: {unmatched}"
    assert not unmatched, unmatched_msg
    tf_model.eval().to(device=DEVICE, dtype=DTYPE)

    tie_msg = "TF 路径加载后 out_head.weight 必须与 tok_emb.weight 仍是同一个 nn.Parameter"
    assert tf_model.out_head.weight is tf_model.tok_emb.weight, tie_msg

    with torch.no_grad():
        ours = tf_model(sample_ids)
        theirs = our_gpt2_loaded(sample_ids)
    _assert_same(ours, theirs, "logits（TF 加载器 vs HF 加载器，fp32）")

    ours64 = copy.deepcopy(tf_model).double().eval()
    theirs64 = copy.deepcopy(our_gpt2_loaded).double().eval()
    with torch.no_grad():
        ours_64 = ours64(sample_ids)
        theirs_64 = theirs64(sample_ids)
    _assert_same(ours_64, theirs_64, "logits（TF 加载器 vs HF 加载器，fp64）")


def test_tf_and_hf_agree(
    tf_params: dict[str, Any],
    hf_gpt2: Any,  # noqa: ANN401  # transformers 无类型存根
    sample_ids: torch.Tensor,
) -> None:
    """真实 OpenAI TF 检查点 vs HF 权重：两条独立路径的 logits 必须相等。

    这一个用例同时证明三件事：TF 映射正确、HF 映射正确、两者数值等价。

    Args:
        tf_params: 真实 TF 检查点读出的嵌套参数。
        hf_gpt2: HF 官方模型，取它的 state_dict 作为另一条路径的权重来源。
        sample_ids: 固定 token 批次。
    """
    cfg = GPTConfig.from_yaml(SMALL_CONFIG_PATH)

    hf_model = _new_model(cfg)
    unmatched_hf = load_hf_weights_into_gpt(hf_model, hf_gpt2.state_dict())
    unmatched_hf_msg = f"有 HF 键未能映射到本仓库参数: {sorted(unmatched_hf)}"
    assert not unmatched_hf, unmatched_hf_msg

    tf_model = _new_model(cfg)
    unmatched_tf = load_openai_tf_weights_into_gpt(tf_model, tf_params)
    unmatched_tf_msg = f"有 TF 变量未能映射到本仓库参数: {unmatched_tf}"
    assert not unmatched_tf, unmatched_tf_msg

    tie_msg = "TF 路径加载后 out_head.weight 必须与 tok_emb.weight 仍是同一个 nn.Parameter"
    assert tf_model.out_head.weight is tf_model.tok_emb.weight, tie_msg

    with torch.no_grad():
        ours = tf_model(sample_ids)
        theirs = hf_model(sample_ids)
    _assert_same(ours, theirs, "logits（OpenAI TF vs HuggingFace，fp32）")

    ours64 = copy.deepcopy(tf_model).double().eval()
    theirs64 = copy.deepcopy(hf_model).double().eval()
    with torch.no_grad():
        ours_64 = ours64(sample_ids)
        theirs_64 = theirs64(sample_ids)
    _assert_same(ours_64, theirs_64, "logits（OpenAI TF vs HuggingFace，fp64）")
