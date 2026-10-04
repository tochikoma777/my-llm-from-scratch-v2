"""困惑度换算与 train/val 评估。

`evaluate_model` 会切换 `model.train()` / `eval()`，所以这里**不用** conftest 的 session 级
`tiny_model` fixture（改了它的模式会污染后续用例），而是模块内自建一份模型。
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from my_llm.config import GPTConfig
from my_llm.model.gpt import GPTModel
from my_llm.train import calc_loss_loader, evaluate_model, evaluate_perplexity, loss_to_perplexity

DEVICE = "cpu"


@pytest.fixture(scope="module")
def model(tiny_cfg: GPTConfig) -> GPTModel:
    """模块内自建的 tiny 模型（eval 模式），避免污染 session 级 fixture。"""
    torch.manual_seed(0)
    m = GPTModel(tiny_cfg)
    m.eval()
    return m


@pytest.fixture
def loader(sample_ids: torch.Tensor) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """两批 `(2, 16)` 的 token 批次，target 为 input 左移一位。"""
    targets = torch.roll(sample_ids, shifts=-1, dims=1)
    dataset = TensorDataset(sample_ids, targets)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def empty_loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """样本数为 0 的 loader，用于验证 NaN 防御。"""
    zeros = torch.zeros(0, 16, dtype=torch.long)
    return DataLoader(TensorDataset(zeros, zeros), batch_size=2)


@pytest.mark.parametrize("loss", [0.0, 0.5, 1.0, 2.5, 7.0])
def test_loss_to_perplexity_matches_exp(loss: float) -> None:
    """perplexity 就是 `math.exp(loss)`。"""
    result = loss_to_perplexity(loss)
    msg = f"loss={loss} 的 perplexity 应为 {math.exp(loss)}，实际 {result}"
    assert result == pytest.approx(math.exp(loss)), msg


def test_loss_to_perplexity_known_values() -> None:
    """已知 loss → 已知 perplexity（ln2 → 2，ln10 → 10）。"""
    assert loss_to_perplexity(math.log(2.0)) == pytest.approx(2.0)
    assert loss_to_perplexity(math.log(10.0)) == pytest.approx(10.0)
    assert loss_to_perplexity(0.0) == pytest.approx(1.0)


def test_loss_to_perplexity_accepts_tensor() -> None:
    """张量输入与浮点输入结果一致。"""
    value = math.log(3.0)
    expected = loss_to_perplexity(value)
    result = loss_to_perplexity(torch.tensor(value))
    msg = f"张量输入应与浮点一致: {result} vs {expected}"
    assert result == pytest.approx(expected), msg


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_loss_to_perplexity_non_finite(bad: float) -> None:
    """非有限 loss 收敛到 `inf`，不让 `math.exp` 抛出或静默返回 nan。"""
    msg = f"非有限 loss {bad} 应返回 inf"
    assert loss_to_perplexity(bad) == float("inf"), msg


def test_evaluate_perplexity_matches_exp_of_loader_loss(
    model: GPTModel, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """`evaluate_perplexity` = `exp(calc_loss_loader)`。"""
    loss = calc_loss_loader(loader, model, DEVICE)
    ppl = evaluate_perplexity(loader, model, DEVICE)
    msg = f"perplexity {ppl} 应等于 exp({loss}) = {math.exp(loss)}"
    assert ppl == pytest.approx(math.exp(loss)), msg


def test_evaluate_perplexity_empty_loader_returns_nan(
    model: GPTModel, empty_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """空 loader 返回 `nan`（而不是 `inf`），保留 v1 `TRAIN:153-154` 的防御。"""
    result = evaluate_perplexity(empty_loader, model, DEVICE)
    msg = f"空 loader 应返回 nan，实际 {result}"
    assert math.isnan(result), msg


def test_evaluate_model_returns_train_and_val_loss(
    model: GPTModel, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """`evaluate_model` 一次返回 `(train_loss, val_loss)`，与分别计算一致。"""
    train_loss, val_loss = evaluate_model(model, loader, loader, DEVICE, eval_iter=2)
    expected = calc_loss_loader(loader, model, DEVICE, num_batches=2)
    assert train_loss == pytest.approx(expected)
    assert val_loss == pytest.approx(expected)


def test_evaluate_model_restores_eval_mode(
    model: GPTModel, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """进入时是 eval 模式，评估完必须仍是 eval（v1 `TRAIN:215` 无条件切回 train 的修正）。"""
    model.eval()
    evaluate_model(model, loader, loader, DEVICE, eval_iter=1)
    msg = "评估不应把 eval 模式的模型切成 train 模式"
    assert model.training is False, msg


def test_evaluate_model_restores_train_mode(
    model: GPTModel, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """进入时是 train 模式，评估完恢复 train。"""
    model.train()
    evaluate_model(model, loader, loader, DEVICE, eval_iter=1)
    assert model.training is True
    model.eval()
