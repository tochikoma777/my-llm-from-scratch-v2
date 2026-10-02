"""OpenAI GPT-2 TensorFlow 检查点 → PyTorch 权重映射（v1 资产移植）。

搬运来源：v1 `src/modules/module_load_param.py`（`download_and_load_gpt2:53`、
`download_file:125`、`load_gpt2_params_from_tf_ckpt:168`、`assign:235`、`load_weights_into_gpt:256`）。

> ⚠️ **警告（硬约束 1）：本模块任何地方都不得给 `out_head` 赋值。**
> weight tying 由 `GPTModel.__init__` 保证（`self.out_head.weight = self.tok_emb.weight`，
> 同一个 `nn.Parameter` 对象）。任何形式的赋值（含 copy / assign / deepcopy）都会把 tie 重新打断，
> 导致微调时两个头漂移。v1 正是在 `LOAD:361` 栽在这里。

完整键映射见 `docs/00-现状盘点.md` 第三部分。相对 v1 的三处**刻意**改动：

1. **不再给 `out_head.weight` 赋值**。v1 `LOAD:361` 把 `wte` 拷进去，等于加了一份没被 tie 的副本；
   v2 直接跳过，`tok_emb.weight` 就是输出头权重。
2. **tensorflow 改为惰性导入**。它是本模块唯一的可选依赖（`LOAD:24` 在 v1 里是硬 import，
   导致 `--test_mode` 也要装 TF），这里放进函数体内部 import。
3. **不再替换 `nn.Parameter` 对象**（`assign` 现在返回张量而非 `nn.Parameter`）。v1 用
   `gpt.tok_emb.weight = assign(...)` 整体替换，即使不碰 `out_head` 也会让两者指向不同对象；
   v2 一律 `copy_` 进现有参数，tie 因此不会被"间接"打断。

保留的语义（不得改动）：
- QKV 按 **q, k, v** 顺序切分（这里用 v1 的写法：先在最后一维三分再各自转置；
  `hf.py` 用另一种写法：先整体 `.T` 再三等分。两条路径数值必须完全一致，
  由 `tests/test_crossload.py` 交叉验证）；
- 所有 Conv1D 形态的权重（TF 存 `(in, out)`）必须 `.T` 后才对得上 `nn.Linear` 的 `(out, in)`；
- LayerNorm 的 `g/b` 映射到 `scale/shift`。

本仓库侧的参数名（与 `hf.py` 完全一致的唯一真名，不要再猜）：

    tok_emb.weight / pos_emb.weight / final_norm.scale / final_norm.shift
    trf_blocks.{i}.att.W_query|W_key|W_value|out_proj（各含 weight 与 bias）
    trf_blocks.{i}.ff.fc1|fc2（各含 weight 与 bias）
    trf_blocks.{i}.norm1|norm2（各含 scale 与 shift）
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from my_llm.model.gpt import GPTModel

# 官方权重所在目录；HF 镜像相关环境变量见 Makefile
DEFAULT_BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"
ALLOWED_SIZES = ("124M", "355M", "774M", "1558M")
CHECKPOINT_FILENAMES = (
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
)

# OpenAI 的 Azure 源在国内经常超时，因此默认就带超时与重试（v1 的 `download_file:125` 两者皆无）
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
_BLOCK_SIZE = 1 << 16  # 64KB，比 v1 的 1KB 少两个数量级的 Python 层循环

_TF_HINT = (
    "需要 tensorflow（仅 OpenAI TF 检查点这条路用）：pip install tensorflow-cpu；"
    "若不想装 TF，改用 HF 路径 `load_weights_from_hf` / `scripts/download_weights.py --source hf`"
)

# TF 变量名 → 本仓库参数后缀。值为 `(目标后缀, 是否需要 .T)`
_TF_ROOT_MAP = {
    "wte": ("tok_emb.weight", False),
    "wpe": ("pos_emb.weight", False),
    "g": ("final_norm.scale", False),
    "b": ("final_norm.shift", False),
}
_TF_BLOCK_MAP = {
    "attn.c_proj.w": ("att.out_proj.weight", True),
    "attn.c_proj.b": ("att.out_proj.bias", False),
    "mlp.c_fc.w": ("ff.fc1.weight", True),
    "mlp.c_fc.b": ("ff.fc1.bias", False),
    "mlp.c_proj.w": ("ff.fc2.weight", True),
    "mlp.c_proj.b": ("ff.fc2.bias", False),
    "ln_1.g": ("norm1.scale", False),
    "ln_1.b": ("norm1.shift", False),
    "ln_2.g": ("norm2.scale", False),
    "ln_2.b": ("norm2.shift", False),
}
# c_attn 一个 TF 变量对应我们三个参数
_TF_QKV_MAP = {"w": "weight", "b": "bias"}
_TF_QKV_TARGETS = ("att.W_query", "att.W_key", "att.W_value")


def _require_tensorflow() -> Any:  # noqa: ANN401  # tensorflow 无类型存根
    """确认 `tensorflow` 可用，否则抛出带安装/替代方案提示的 ImportError。

    Returns:
        已导入的 tensorflow 模块（`Any`，因为它没有类型存根）。

    Raises:
        ImportError: 未安装 tensorflow。
    """
    try:
        import tensorflow as tf  # noqa: PLC0415  # 惰性导入：可选依赖
    except ImportError as exc:
        raise ImportError(_TF_HINT) from exc
    return tf


def download_file(
    url: str,
    destination: str | Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> None:
    """带进度提示的单文件下载（v1 `LOAD:125`）。

    幂等：目标文件已存在且大小与远端一致时直接跳过，避免重复下载 500MB。

    Args:
        url: 远端 URL。
        destination: 本地保存路径。
        timeout: 单次请求超时秒数。
        retries: 最大尝试次数（每次重试前线性退避）。

    Raises:
        RuntimeError: 重试耗尽仍失败。
    """
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: BaseException | None = None

    for attempt in range(1, retries + 1):
        try:
            _download_once(url, dest, timeout)
            return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            print(f"⚠️ 下载失败（第 {attempt}/{retries} 次）{dest.name}: {exc}")
            if attempt < retries:
                time.sleep(DEFAULT_TIMEOUT * 0.1 * attempt)

    msg = (
        f"无法下载 {url}（已重试 {retries} 次，最后错误: {last_error}）。"
        f"OpenAI 的 Azure 源在国内经常超时/中断；办法有三：重跑一次让它续下、"
        f"手动下载后放到 {dest}（文件名保持一致即可被识别为已存在）、"
        f"或改用 HF 路径 `scripts/download_weights.py --source hf`。"
    )
    raise RuntimeError(msg)


def _download_once(url: str, dest: Path, timeout: float) -> None:
    """单次下载尝试，`urllib` 相关异常交给 `download_file` 重试。

    Args:
        url: 远端 URL。
        dest: 本地保存路径；先写 `.part` 再 rename，避免中断后留下伪装成完整的半截文件。
        timeout: 超时秒数。

    Raises:
        OSError: 网络或写盘失败（`URLError` / `TimeoutError` 都是它的子类）。
    """
    # 固定的 https 官方源，无用户可控地址：ruff S310 对此处的告警不适用
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        remote_size = int(response.headers.get("Content-Length", 0) or 0)
        if remote_size and dest.exists() and dest.stat().st_size == remote_size:
            print(f"已存在且大小一致，跳过下载: {dest.name}")
            return

        print(f"开始下载 {dest.name}" + (f"（{remote_size / 1e6:.1f} MB）" if remote_size else ""))
        part_path = dest.with_name(dest.name + ".part")
        downloaded = 0
        next_mark = 0.25
        with part_path.open("wb") as file:
            while True:
                chunk = response.read(_BLOCK_SIZE)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if remote_size:
                    ratio = downloaded / remote_size
                    while next_mark <= ratio:
                        print(f"  {dest.name}: {next_mark:.0%}")
                        next_mark += 0.25
        part_path.replace(dest)
        print(f"下载完成: {dest.name}（{downloaded / 1e6:.1f} MB）")


def download_and_load_gpt2(
    model_size: str, models_dir: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """下载并解析指定尺寸的 GPT-2 权重（v1 `LOAD:53`）。

    Args:
        model_size: `"124M" | "355M" | "774M" | "1558M"`。
        models_dir: 本地目录，实际落在 `<models_dir>/<model_size>/`。

    Returns:
        `(settings, params)`：`settings` 来自 `hparams.json`，`params` 是嵌套字典形式的权重。

    Raises:
        ValueError: `model_size` 不在 `ALLOWED_SIZES` 内，或目录下找不到 TF 检查点。
        ImportError: 未安装 tensorflow。
        RuntimeError: 下载失败。
    """
    if model_size not in ALLOWED_SIZES:
        msg = f"不支持的模型大小: {model_size}。可选: {ALLOWED_SIZES}"
        raise ValueError(msg)
    tf = _require_tensorflow()

    model_dir = Path(models_dir) / model_size
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in CHECKPOINT_FILENAMES:
        download_file(f"{DEFAULT_BASE_URL}/{model_size}/{filename}", model_dir / filename)

    settings: dict[str, Any] = json.loads((model_dir / "hparams.json").read_text(encoding="utf-8"))
    ckpt_path = tf.train.latest_checkpoint(str(model_dir))
    if ckpt_path is None:
        msg = f"{model_dir} 下找不到 TF 检查点（model.ckpt.*），请重新下载或检查目录"
        raise ValueError(msg)
    params = load_gpt2_params_from_tf_ckpt(ckpt_path, settings)
    print(f"成功加载 GPT-2 {model_size} 权重")
    return settings, params


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """把 TF 检查点读成嵌套字典（v1 `LOAD:168`）。

    变量名形如 `model/h0/attn/c_attn/w`，去掉 `model/` 后按 `/` 拆层，
    `h<n>` 对应 `params["blocks"][n]`。

    Args:
        ckpt_path: TF 检查点前缀路径。
        settings: 超参字典，需要 `n_layer`。

    Returns:
        嵌套字典：`{"wpe", "wte", "blocks": [{attn, mlp, ln_1, ln_2}], "g", "b"}`。

    Raises:
        ImportError: 未安装 tensorflow。
    """
    tf = _require_tensorflow()
    params: dict[str, Any] = {"blocks": [{} for _ in range(int(settings["n_layer"]))]}

    for name, _ in tf.train.list_variables(ckpt_path):
        # squeeze 去掉 TF 里的单例维度（如 bias 存成 (768, 1) -> (768,)）
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        parts = name.split("/")[1:]
        target: dict[str, Any] = params
        if parts[0].startswith("h"):
            layer_number = int(parts[0][1:])
            target = params["blocks"][layer_number]
        for key in parts[1:-1]:
            target = target.setdefault(key, {})
        target[parts[-1]] = variable_array

    return params


def assign(left: torch.Tensor, right: npt.NDArray[np.float32]) -> torch.Tensor:
    """形状校验后把 numpy 数组转成可直接 `copy_` 的张量。

    与 v1 `LOAD:235` 的差别：v1 返回 `nn.Parameter` 供调用方整体替换属性，那样会把
    weight tying 间接打断（见模块 docstring 第 3 点）；这里只返回张量。

    Args:
        left: 目标参数，**只用于取形状与 dtype/device**。
        right: 源 numpy 数组。

    Returns:
        与 `left` 同 dtype/device 的张量。

    Raises:
        ValueError: 形状不一致。
    """
    if tuple(left.shape) != tuple(right.shape):
        msg = f"形状不匹配。目标: {tuple(left.shape)}, 源: {tuple(right.shape)}"
        raise ValueError(msg)
    return torch.as_tensor(np.ascontiguousarray(right), dtype=left.dtype, device=left.device)


def _flatten_tf_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """把嵌套参数字典摊平成 `{点分路径: 数组}`，便于统计哪些变量没被消费。

    Args:
        params: `load_gpt2_params_from_tf_ckpt` 的输出。

    Returns:
        扁平字典，键如 `"blocks.0.attn.c_attn.w"`、`"wte"`、`"g"`。
    """
    flat: dict[str, Any] = {}

    def _walk(node: Mapping[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, Mapping):
                _walk(value, path)
            elif isinstance(value, list):  # 只有 blocks 是列表，逐层编号
                for index, item in enumerate(value):
                    _walk(item, f"{path}.{index}")
            else:
                flat[path] = value

    _walk(params, "")
    return flat


def _take_source(flat: Mapping[str, Any], consumed: set[str], src_key: str) -> Any:  # noqa: ANN401
    """取出一个 TF 变量并标记为已消费。

    Args:
        flat: 扁平化后的参数字典。
        consumed: 已消费键集合，会被就地修改。
        src_key: 扁平路径，如 `"blocks.0.attn.c_attn.w"`。

    Returns:
        对应的 numpy 数组。

    Raises:
        KeyError: 检查点里没有该变量（说明结构与预期不符）。
    """
    if src_key not in flat:
        msg = f"TF 参数字典里没有 {src_key}：checkpoint 结构不符合预期（模型层数是否匹配？）"
        raise KeyError(msg)
    consumed.add(src_key)
    return flat[src_key]


def _write_param(
    model_params: Mapping[str, torch.Tensor],
    target_key: str,
    source: Any,  # noqa: ANN401
    src_key: str,
    *,
    transpose: bool,
) -> None:
    """形状校验后就地写入单个参数（不替换 `nn.Parameter` 对象，保住 weight tying）。

    Args:
        model_params: `dict(model.named_parameters())` 的结果。
        target_key: 本仓库侧参数名。
        source: 源 numpy 数组。
        src_key: 原始 TF 扁平路径，仅用于报错信息。
        transpose: 是否需要先 `.T`（TF 的 Conv1D 形态权重需要）。

    Raises:
        KeyError: 目标参数不存在（说明映射表写错了）。
        ValueError: 形状不匹配。
    """
    dst = model_params.get(target_key)
    if dst is None:
        msg = f"本仓库没有参数 {target_key}（来自 TF 变量 {src_key}）"
        raise KeyError(msg)
    array = source.T if transpose else source
    with torch.no_grad():
        dst.copy_(assign(dst, array))


def load_openai_tf_weights_into_gpt(gpt: GPTModel, params: Mapping[str, Any]) -> list[str]:
    """把 TF 参数字典写入 `GPTModel`。

    `out_head.weight` **不会**被触碰：它与 `tok_emb.weight` 是同一个 `nn.Parameter`，
    写好 `tok_emb` 就等于写好了输出头（硬约束 1）。

    Args:
        gpt: 目标模型。
        params: `load_gpt2_params_from_tf_ckpt` 的输出（嵌套字典）。

    Returns:
        未能映射到本仓库参数的 TF 变量列表（正常情况下应为空）。

    Raises:
        KeyError: 预期的 TF 变量缺失，或映射表指向了不存在的参数。
        ValueError: 形状不匹配。
    """
    flat = _flatten_tf_params(params)
    model_params: dict[str, torch.Tensor] = dict(gpt.named_parameters())
    consumed: set[str] = set()

    for src_key, (target_key, transpose) in _TF_ROOT_MAP.items():
        _write_param(
            model_params,
            target_key,
            _take_source(flat, consumed, src_key),
            src_key,
            transpose=transpose,
        )

    for index in range(int(gpt.cfg.n_layers)):
        for src_suffix, (target_suffix, transpose) in _TF_BLOCK_MAP.items():
            src_key = f"blocks.{index}.{src_suffix}"
            _write_param(
                model_params,
                f"trf_blocks.{index}.{target_suffix}",
                _take_source(flat, consumed, src_key),
                src_key,
                transpose=transpose,
            )

        # c_attn 合成了 Q/K/V，按 q → k → v 在最后一维三分后各自转置（v1 `LOAD:283` 的写法）
        for tf_suffix, kind in _TF_QKV_MAP.items():
            src_key = f"blocks.{index}.attn.c_attn.{tf_suffix}"
            merged = _take_source(flat, consumed, src_key)
            q_w, k_w, v_w = np.split(merged, 3, axis=-1)
            for target_suffix, piece in zip(_TF_QKV_TARGETS, (q_w, k_w, v_w), strict=True):
                _write_param(
                    model_params,
                    f"trf_blocks.{index}.{target_suffix}.{kind}",
                    piece.T,
                    src_key,
                    transpose=False,
                )

    return sorted(set(flat) - consumed)


def load_weights_into_gpt(gpt: GPTModel, params: Mapping[str, Any]) -> None:
    """严格版 TF 加载：不允许残留未消费的变量，并在结束后确认 tie 仍成立。

    Args:
        gpt: 目标模型。
        params: `load_gpt2_params_from_tf_ckpt` 的输出。

    Raises:
        ValueError: 存在未消费的 TF 变量。
        AssertionError: 加载后 weight tying 被破坏（不该发生）。
    """
    unmatched = load_openai_tf_weights_into_gpt(gpt, params)
    if unmatched:
        msg = f"以下 TF 变量未能映射到本仓库参数: {unmatched}"
        raise ValueError(msg)

    tie_msg = "加载后 out_head.weight 必须与 tok_emb.weight 仍是同一个 nn.Parameter"
    assert gpt.out_head.weight is gpt.tok_emb.weight, tie_msg
