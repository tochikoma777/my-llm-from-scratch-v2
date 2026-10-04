"""Trainer：checkpoint 往返、梯度累积、评估节奏、续训与采样回调。

会真实更新权重，所以**不用** conftest 的 session 级 `tiny_model`（会污染其他用例），
每个用例用 `tiny_cfg` 现造一个模型。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from my_llm.config import GPTConfig
from my_llm.model.gpt import GPTModel
from my_llm.train import get_cosine_schedule_with_warmup, load_checkpoint, save_checkpoint
from my_llm.train.trainer import Trainer, TrainerConfig, TrainHistory

DEVICE = "cpu"
BATCH_SIZE = 2
NUM_BATCHES = 4
SEQ_LEN = 16

BatchLoader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


@pytest.fixture
def model(tiny_cfg: GPTConfig) -> GPTModel:
    """新建的 tiny 模型（训练会改动权重，故不用 session fixture）。"""
    torch.manual_seed(0)
    return GPTModel(tiny_cfg)


@pytest.fixture
def loader() -> BatchLoader:
    """`NUM_BATCHES` 批 `(BATCH_SIZE, SEQ_LEN)` 的 token 批次。"""
    torch.manual_seed(0)
    ids = torch.randint(0, 10000, (NUM_BATCHES * BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    targets = torch.roll(ids, shifts=-1, dims=1)
    return DataLoader(TensorDataset(ids, targets), batch_size=BATCH_SIZE)


def _sgd(model: GPTModel, lr: float = 0.1) -> torch.optim.SGD:
    """构造一个 SGD 优化器。"""
    return torch.optim.SGD(model.parameters(), lr=lr)


def test_checkpoint_roundtrip_restores_state(
    model: GPTModel, tiny_cfg: GPTConfig, tmp_path: Path
) -> None:
    """`save_checkpoint` → `load_checkpoint`：step / 权重 / optimizer / scheduler 全回来。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 2, 10)
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, scheduler, 7, path, epoch=1)
    assert path.is_file()

    fresh = GPTModel(tiny_cfg)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_sched = get_cosine_schedule_with_warmup(fresh_opt, 2, 10)
    step = load_checkpoint(fresh, fresh_opt, fresh_sched, path)

    assert step == 7
    expected_lr = optimizer.param_groups[0]["lr"]
    assert fresh_opt.param_groups[0]["lr"] == pytest.approx(expected_lr)
    pairs = zip(model.named_parameters(), fresh.named_parameters(), strict=True)
    for (name_a, param_a), (name_b, param_b) in pairs:
        assert name_a == name_b
        assert torch.equal(param_a, param_b), name_a


def test_load_checkpoint_keeps_weight_tying(
    model: GPTModel, tiny_cfg: GPTConfig, tmp_path: Path
) -> None:
    """`load_state_dict` 必须就地 `copy_`，不能打断 tie（硬约束 1）。"""
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, _sgd(model), None, 0, path)

    fresh = GPTModel(tiny_cfg)
    load_checkpoint(fresh, _sgd(fresh), None, path)
    tie_msg = "checkpoint 恢复后 out_head.weight 与 tok_emb.weight 必须是同一个 nn.Parameter"
    assert fresh.out_head.weight is fresh.tok_emb.weight, tie_msg


def test_load_checkpoint_missing_file_raises(model: GPTModel, tmp_path: Path) -> None:
    """checkpoint 不存在时明确报错，而不是静默从头训练。"""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(model, _sgd(model), None, tmp_path / "nope.pt")


def test_train_epoch_updates_parameters(model: GPTModel, loader: BatchLoader) -> None:
    """跑完一轮后权重确实变了，且 `eval_freq=1` 时每步都有记录。"""
    before = model.tok_emb.weight.detach().clone()
    cfg = TrainerConfig(eval_freq=1, eval_iter=1)
    trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=1e-2), cfg)
    history = trainer.train_epoch(loader, loader, DEVICE, epoch=0)

    assert not torch.equal(before, model.tok_emb.weight)
    assert history.global_steps == [1, 2, 3, 4]
    assert len(history.train_losses) == NUM_BATCHES
    assert len(history.val_losses) == NUM_BATCHES
    assert len(history.perplexities) == NUM_BATCHES
    assert trainer.tokens_seen == NUM_BATCHES * BATCH_SIZE * SEQ_LEN
    tie_msg = "训练后 weight tying 必须仍然成立"
    assert model.out_head.weight is model.tok_emb.weight, tie_msg


def test_grad_accumulation_steps_less_often(model: GPTModel, loader: BatchLoader) -> None:
    """`grad_accum_steps=2` 时 optimizer/scheduler 只走 `ceil(批次 / 2)` 次。"""
    optimizer = _sgd(model)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 1, 100)
    cfg = TrainerConfig(eval_freq=10_000, grad_accum_steps=2)
    trainer = Trainer(model, optimizer, cfg, scheduler=scheduler)

    trainer.train_epoch(loader, loader, DEVICE, epoch=0)
    msg = f"scheduler 应只 step {NUM_BATCHES // 2} 次，实际 {scheduler.last_epoch}"
    assert scheduler.last_epoch == NUM_BATCHES // 2, msg


def test_train_evaluates_baseline_at_step_zero(
    model: GPTModel, loader: BatchLoader, tmp_path: Path
) -> None:
    """v1 `TRAIN:322` 的 `-1` 起步语义：第 1 个微批次就评估一次基线（step 0）。"""
    cfg = TrainerConfig(num_epochs=1, eval_freq=5, eval_iter=1, checkpoint_dir=tmp_path)
    trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=1e-3), cfg)
    history = trainer.train(loader, loader, DEVICE)
    msg = f"4 个批次 + eval_freq=5 应只在 step 0 评估一次，实际 {history.global_steps}"
    assert history.global_steps == [0], msg


def test_train_resume_continues_global_step(
    model: GPTModel, tiny_cfg: GPTConfig, loader: BatchLoader, tmp_path: Path
) -> None:
    """续训时 `global_step` 接着上次的步数走，不从 0 重来。"""
    cfg = TrainerConfig(num_epochs=1, eval_freq=10_000, checkpoint_dir=tmp_path)
    first = Trainer(model, _sgd(model), cfg)
    first.train(loader, loader, DEVICE)
    ckpt = tmp_path / "last.pt"
    assert ckpt.is_file(), "每轮结束应写出 last.pt"

    fresh = GPTModel(tiny_cfg)
    second = Trainer(fresh, _sgd(fresh), cfg)
    second.train(loader, loader, DEVICE, resume_from=ckpt)
    expected = 2 * NUM_BATCHES
    msg = f"续训后 global_step 应为 {expected}，实际 {second.global_step}"
    assert second.global_step == expected, msg


def test_sample_fn_records_per_epoch(model: GPTModel, loader: BatchLoader, tmp_path: Path) -> None:
    """给了 `sample_fn` 就每轮采样一次；没给就一条都不采。"""
    cfg = TrainerConfig(num_epochs=2, eval_freq=10_000, checkpoint_dir=tmp_path)
    sampled = Trainer(model, _sgd(model, lr=0.0), cfg, sample_fn=lambda m, prompt: f"<{prompt}>")
    assert sampled.train(loader, loader, DEVICE).samples == ["<Every effort moves you>"] * 2

    quiet_cfg = TrainerConfig(num_epochs=1, eval_freq=10_000, checkpoint_dir=tmp_path)
    quiet = Trainer(model, _sgd(model, lr=0.0), quiet_cfg)
    assert quiet.train(loader, loader, DEVICE).samples == []


def test_history_extend_concatenates() -> None:
    """`TrainHistory.extend` 逐字段追加（多轮累计用）。"""
    first = TrainHistory(
        train_losses=[1.0], val_losses=[2.0], perplexities=[3.0], tokens_seen=[4], global_steps=[0]
    )
    second = TrainHistory(
        train_losses=[1.5], val_losses=[2.5], perplexities=[4.0], tokens_seen=[8], global_steps=[1]
    )
    first.extend(second)
    assert first.train_losses == [1.0, 1.5]
    assert first.val_losses == [2.0, 2.5]
    assert first.perplexities == [3.0, 4.0]
    assert first.tokens_seen == [4, 8]
    assert first.global_steps == [0, 1]
