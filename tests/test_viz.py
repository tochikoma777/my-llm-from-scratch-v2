"""可视化：只断言"文件写出来了 + 父目录被创建"，不断言图像内容。

输出一律走 `tmp_path`，不写进 `outputs/`（产物目录已 gitignore，但测试不该依赖它）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境必须显式选后端，否则 CI 上可能拿到空图

import pytest  # noqa: E402  # 必须在 matplotlib.use 之后才导入 pyplot 相关模块
import torch  # noqa: E402

from my_llm.utils.viz import plot_attention_heatmap, plot_losses  # noqa: E402


def test_plot_losses_creates_parent_dir_and_returns_path(tmp_path: Path) -> None:
    """`plot_losses` 自动创建父目录，并返回实际写入的路径。

    Args:
        tmp_path: pytest 提供的临时目录。
    """
    out = tmp_path / "nested" / "loss.pdf"
    written = plot_losses([2.0, 1.5, 1.2], [2.1, 1.6, 1.3], out_path=out)
    assert written == out
    assert written.is_file()


def test_plot_losses_accepts_tokens_seen(tmp_path: Path) -> None:
    """给出 `tokens_seen` 时走双 x 轴分支，同样要落盘。

    Args:
        tmp_path: pytest 提供的临时目录。
    """
    out = tmp_path / "loss-tokens.pdf"
    written = plot_losses([2.0, 1.5], [2.1, 1.6], tokens_seen=[100, 200], out_path=out)
    assert written.is_file()


def test_plot_losses_rejects_length_mismatch(tmp_path: Path) -> None:
    """两个 loss 序列长度不一致时报错（画出来也是错位的，不如早失败）。

    Args:
        tmp_path: pytest 提供的临时目录。
    """
    with pytest.raises(ValueError, match="长度"):
        plot_losses([2.0, 1.5], [2.1], out_path=tmp_path / "x.pdf")


def test_plot_attention_heatmap_handles_both_shapes(tmp_path: Path) -> None:
    """3 维 `(heads, q, k)` 与 4 维 `(batch, heads, q, k)` 都要能画。

    Args:
        tmp_path: pytest 提供的临时目录。
    """
    weights = torch.rand(4, 6, 6)
    three_dim = plot_attention_heatmap(weights, out_path=tmp_path / "att3.pdf")
    four_dim = plot_attention_heatmap(weights.unsqueeze(0), out_path=tmp_path / "att4.pdf")
    assert three_dim.is_file()
    assert four_dim.is_file()


def test_plot_attention_heatmap_rejects_bad_shape(tmp_path: Path) -> None:
    """形状不是注意力矩阵时报错，而不是画出一堆无意义的色块。

    Args:
        tmp_path: pytest 提供的临时目录。
    """
    with pytest.raises(ValueError, match="形状"):
        plot_attention_heatmap(torch.rand(6, 6), out_path=tmp_path / "bad.pdf")
