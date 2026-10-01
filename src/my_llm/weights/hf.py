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

两条实现约定（改动前先读）：

1. **转置**：HF 的 `Conv1D` 存 `(in, out)`，本仓库的 `nn.Linear` 存 `(out, in)`，
   因此 `c_attn / c_proj / c_fc / mlp.c_proj` 的 **weight** 都要 `.T`；bias 与
   `wte / wpe / LayerNorm` 参数不转置。
2. **QKV 顺序**：`c_attn` 的合并张量按 **q → k → v** 切分。顺序写反不会报错，
   只会让 parity diff 变成 1e0 量级，所以切分逻辑只有这一处，别在别处重写。
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch

from my_llm.config import GPTConfig
from my_llm.model.gpt import GPTModel

# transformers 只是 parity 用的可选依赖，缺失时给可执行提示而不是裸 ImportError
_TRANSFORMERS_HINT = (
    "需要 transformers（仅 parity 测试用）：pip install -e '.[dev]'；"
    "下载慢时设置 HF_ENDPOINT=https://hf-mirror.com（见 Makefile）"
)

# tie 已由 GPTModel.__init__ 保证，加载时必须跳过：给 out_head 赋值会重新打断 tie
_SKIPPED_HF_KEYS = frozenset({"lm_head.weight"})

_EMB_MAP = {"wte": "tok_emb.weight", "wpe": "pos_emb.weight"}
_FINAL_MAP = {"weight": "final_norm.scale", "bias": "final_norm.shift"}

_BLOCK_SUFFIX_MAP = {
    "ln_1.weight": "norm1.scale",
    "ln_1.bias": "norm1.shift",
    "ln_2.weight": "norm2.scale",
    "ln_2.bias": "norm2.shift",
    "attn.c_proj.weight": "att.out_proj.weight",
    "attn.c_proj.bias": "att.out_proj.bias",
    "mlp.c_fc.weight": "ff.fc1.weight",
    "mlp.c_fc.bias": "ff.fc1.bias",
    "mlp.c_proj.weight": "ff.fc2.weight",
    "mlp.c_proj.bias": "ff.fc2.bias",
}

# c_attn 一个 HF 键对应我们三个参数，由 load_hf_weights_into_gpt 单独处理
_QKV_SUFFIX_MAP = {"attn.c_attn.weight": "weight", "attn.c_attn.bias": "bias"}
_QKV_TARGETS = ("att.W_query", "att.W_key", "att.W_value")

# 需要 .T 的键（都是 nn.Linear 的 weight）；bias / 嵌入 / LayerNorm 不在其中
_TRANSPOSED_SUFFIXES = ("att.out_proj.weight", "ff.fc1.weight", "ff.fc2.weight")

_HF_KEY_RE = re.compile(
    r"^transformer\."
    r"(?:(?P<emb>wte|wpe)\.weight"
    r"|h\.(?P<block>\d+)\.(?P<rest>.+)"
    r"|ln_f\.(?P<final>weight|bias))$"
)


def _require_transformers() -> None:
    """确认 `transformers` 可用，否则抛出带安装/镜像提示的 ImportError。

    Raises:
        ImportError: 未安装 transformers。
    """
    try:
        import transformers  # noqa: F401  # 仅探测可用性
    except ImportError as exc:
        raise ImportError(_TRANSFORMERS_HINT) from exc


def _split_qkv(merged: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 `c_attn` 的合并张量按 **q → k → v** 切成三段。

    采用做法 A：先整体转置，再按 dim=0 三等分。

    - weight：HF 存 `(emb, 3*emb)`，`.T` 后 `(3*emb, emb)`，三等分得三个 `(emb, emb)`；
    - bias：HF 存 `(3*emb,)`，一维无需转置，同样按 dim=0 三等分
      （一维张量上用 `.T` 已被 PyTorch 标记为 deprecated，会刷警告，故按维数分支）。

    Args:
        merged: HF 的 `attn.c_attn.weight` 或 `.bias`。

    Returns:
        `(q, k, v)`，顺序与 HF `split(split_size, dim=2)` 一致。
    """
    flat = merged.T if merged.dim() == 2 else merged
    chunks = flat.chunk(3, dim=0)
    return chunks[0], chunks[1], chunks[2]


def _copy_into(
    params: Mapping[str, torch.Tensor],
    our_key: str,
    src: torch.Tensor,
    hf_key: str,
) -> None:
    """形状校验后把单个张量拷进目标参数。

    Args:
        params: `dict(model.named_parameters())` 的结果。
        our_key: 本仓库侧参数名。
        src: 已按需转置的源张量。
        hf_key: 原始 HF 键名，仅用于报错信息。

    Raises:
        KeyError: 目标参数不存在（说明映射表写错了）。
        ValueError: 形状不匹配。
    """
    dst = params.get(our_key)
    if dst is None:
        msg = f"本仓库没有参数 {our_key}（来自 HF 键 {hf_key}）"
        raise KeyError(msg)
    if tuple(dst.shape) != tuple(src.shape):
        msg = f"形状不匹配：HF {hf_key} {tuple(src.shape)} vs 本仓库 {our_key} {tuple(dst.shape)}"
        raise ValueError(msg)
    with torch.no_grad():
        dst.copy_(src)


def load_hf_state_dict(
    model_name: str = "gpt2", *, revision: str = "main"
) -> dict[str, torch.Tensor]:
    """从 HF Hub 拉取 `GPT2LMHeadModel` 的 state_dict。

    Args:
        model_name: Hub 上的模型名，如 `"gpt2"` / `"gpt2-medium"`。
        revision: 副本分支/提交号。

    Returns:
        HF 原生键名的 state_dict（已从计算图 detach）。

    Raises:
        ImportError: 未安装 transformers。
    """
    _require_transformers()
    from transformers import GPT2LMHeadModel  # noqa: PLC0415  # 惰性导入：可选依赖

    model = GPT2LMHeadModel.from_pretrained(model_name, revision=revision)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def hf_key_to_ours(key: str) -> str:
    """单个 HF 键名 → 本仓库参数名（是否要 `.T` 由 `_TRANSPOSED_SUFFIXES` 决定）。

    Args:
        key: 例如 `"transformer.h.0.attn.c_attn.weight"`。

    Returns:
        本仓库侧的参数名。返回空串表示**该函数不直接映射**，有两种情况：
        `lm_head.weight`（已 tie，跳过）与 `attn.c_attn.*`（一个键对应 q/k/v 三个参数，
        由 `load_hf_weights_into_gpt` 处理）。
    """
    matched = _HF_KEY_RE.match(key)
    if matched is None:
        return ""

    emb = matched["emb"]
    if emb:
        return _EMB_MAP[emb]

    final = matched["final"]
    if final:
        return _FINAL_MAP[final]

    rest = matched["rest"]
    if rest in _QKV_SUFFIX_MAP:
        return ""

    suffix = _BLOCK_SUFFIX_MAP.get(rest)
    if suffix is None:
        return ""
    return f"trf_blocks.{matched['block']}.{suffix}"


def load_hf_weights_into_gpt(gpt: GPTModel, hf_state_dict: Mapping[str, torch.Tensor]) -> list[str]:
    """把 HF state_dict 逐个参数写入 `GPTModel`。

    `out_head.weight` **不会**被赋值：它与 `tok_emb.weight` 是同一个 `nn.Parameter`，
    HF 侧的 `lm_head.weight` 直接跳过（硬约束 1）。

    Args:
        gpt: 目标模型。
        hf_state_dict: HF 原生键名的 state_dict。

    Returns:
        未能映射到本仓库参数的 HF 键列表（正常情况下应为空）。

    Raises:
        KeyError: 映射表指向了不存在的参数。
        ValueError: 形状不匹配。
    """
    params: dict[str, torch.Tensor] = dict(gpt.named_parameters())
    unmatched: list[str] = []

    for hf_key, src in hf_state_dict.items():
        if hf_key in _SKIPPED_HF_KEYS:
            continue

        matched = _HF_KEY_RE.match(hf_key)
        if matched is not None and matched["rest"] in _QKV_SUFFIX_MAP:
            block = matched["block"]
            kind = _QKV_SUFFIX_MAP[matched["rest"]]
            for target, piece in zip(_QKV_TARGETS, _split_qkv(src), strict=True):
                _copy_into(params, f"trf_blocks.{block}.{target}.{kind}", piece, hf_key)
            continue

        our_key = hf_key_to_ours(hf_key)
        if our_key == "":
            unmatched.append(hf_key)
            continue

        tensor = src.T if our_key.endswith(_TRANSPOSED_SUFFIXES) else src
        _copy_into(params, our_key, tensor, hf_key)

    return unmatched


def load_weights_from_hf(gpt: GPTModel, hf_state: Mapping[str, torch.Tensor]) -> None:
    """严格版加载：不允许残留未匹配的 HF 键，并在结束后确认 tie 仍成立。

    Args:
        gpt: 目标模型。
        hf_state: HF 原生键名的 state_dict。

    Raises:
        ValueError: HF 侧 `lm_head` 与 `wte` 不一致（不是 tied 的 GPT-2），
            或存在未匹配的键，或形状不匹配。
        AssertionError: 加载后 weight tying 被破坏（不该发生）。
    """
    wte = hf_state.get("transformer.wte.weight")
    lm_head = hf_state.get("lm_head.weight")
    if wte is not None and lm_head is not None and not torch.equal(wte, lm_head):
        msg = "HF 侧 lm_head.weight 与 transformer.wte.weight 不一致：该检查点不是 tied 的 GPT-2"
        raise ValueError(msg)

    unmatched = load_hf_weights_into_gpt(gpt, hf_state)
    if unmatched:
        msg = f"以下 HF 键未能映射到本仓库参数: {sorted(unmatched)}"
        raise ValueError(msg)

    tie_msg = "加载后 out_head.weight 必须与 tok_emb.weight 仍是同一个 nn.Parameter"
    assert gpt.out_head.weight is gpt.tok_emb.weight, tie_msg


def gpt_config_from_hf(model_name: str = "gpt2") -> GPTConfig:
    """读取 HF 模型的配置并转成 `GPTConfig`。

    Args:
        model_name: Hub 上的模型名。

    Returns:
        对齐该变体的配置（`qkv_bias` 恒为 True）。

    Raises:
        ImportError: 未安装 transformers。
    """
    _require_transformers()
    from transformers import GPT2Config  # noqa: PLC0415  # 惰性导入：可选依赖

    hf_cfg = GPT2Config.from_pretrained(model_name)
    return GPTConfig(
        vocab_size=int(hf_cfg.vocab_size),
        context_length=int(hf_cfg.n_positions),
        emb_dim=int(hf_cfg.n_embd),
        n_layers=int(hf_cfg.n_layer),
        n_heads=int(hf_cfg.n_head),
        drop_rate=float(hf_cfg.embd_pdrop),
        qkv_bias=True,
    )
