#!/usr/bin/env python
"""预训练入口。

用法：
    python scripts/train.py --config configs/gpt2-tiny.yaml
    python scripts/train.py --config configs/gpt2-small.yaml --resume outputs/checkpoints/last.pt
    python scripts/train.py --config configs/gpt2-tiny.yaml --train-config configs/train-demo.yaml

配置**只能**来自 yaml（`GPTConfig.from_yaml` + `TrainConfig.from_yaml`），不在脚本里写任何超参
——这是相对 v1 的关键改动（v1 把全套超参埋在 `module_train.py:530-546` 的 `__main__` 里）。

`--epochs` / `--batch-size` / `--lr` / `--seed` 是**可选覆盖**：默认 `None`，
`None` 时回落到 `--train-config` 指定的 yaml。yaml 兜底 + CLI 覆盖，不是"改源码才能调参"。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from my_llm.config import GPTConfig
from my_llm.data import create_dataloader_v1
from my_llm.model.gpt import GPTModel
from my_llm.tokenizer import build_tokenizer
from my_llm.train import TrainConfig, Trainer, TrainerConfig, get_cosine_schedule_with_warmup
from my_llm.train.trainer import TrainHistory


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命名空间对象。
    """
    parser = argparse.ArgumentParser(description="从头预训练 GPT-2")
    parser.add_argument("--config", type=Path, required=True, help="模型架构 yaml 路径")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train-default.yaml"),
        help="训练超参 yaml 路径（optimizer / scheduler / data 三段都在这里）",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/raw/the-verdict.txt"), help="语料路径"
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 yaml 的 num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 yaml 的 batch_size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 yaml 的 optimizer.lr")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 yaml 的 seed")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint 路径，续训用")
    return parser.parse_args()


def split_corpus(txt: str, train_ratio: float) -> tuple[str, str]:
    """按字符比例切分语料为训练 / 验证两段。

    v1 是先编码再按 token 切（`TRAIN:477`）；这里在**字符层**切，因为 `GPTDatasetV1`
    接收的是原始文本、编码在它内部完成。比例语义一致（前 `train_ratio` 训练，其余验证），
    切点最多相差一个 token 的边界。

    Args:
        txt: 原始语料。
        train_ratio: 训练集占比，取值 (0, 1)。

    Returns:
        `(train_txt, val_txt)`。
    """
    cut = int(len(txt) * train_ratio)
    return txt[:cut], txt[cut:]


def main() -> None:
    """执行预训练。

    Returns:
        None。

    Raises:
        FileNotFoundError: 配置文件或语料不存在。
    """
    args = parse_args()
    if not args.data.is_file():
        msg = f"语料不存在: {args.data}"
        raise FileNotFoundError(msg)

    cfg = GPTConfig.from_yaml(args.config)
    tc = TrainConfig.from_yaml(args.train_config)

    # CLI 可选覆盖：给了就用，没给就回落 yaml
    num_epochs = args.epochs if args.epochs is not None else tc.num_epochs
    batch_size = args.batch_size if args.batch_size is not None else tc.batch_size
    lr = args.lr if args.lr is not None else tc.optimizer.lr
    seed = args.seed if args.seed is not None else tc.seed

    # Trainer 只把 batch 搬到 device（`calc_loss_batch` 里 `.to(device)`），模型由调用方搬
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(cfg).to(device)

    txt = args.data.read_text(encoding="utf-8")
    train_txt, val_txt = split_corpus(txt, tc.data.train_ratio)
    tokenizer = build_tokenizer()
    train_loader = create_dataloader_v1(
        train_txt,
        tokenizer,
        batch_size=batch_size,
        max_length=tc.data.max_length,
        stride=tc.data.stride,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader_v1(
        val_txt,
        tokenizer,
        batch_size=batch_size,
        max_length=tc.data.max_length,
        stride=tc.data.stride,
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=tc.optimizer.weight_decay,
        betas=tc.optimizer.betas,
        eps=tc.optimizer.eps,
    )
    warmup = min(tc.scheduler.warmup_steps, max(1, tc.total_steps(len(train_loader)) - 1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=tc.total_steps(len(train_loader)),
        min_lr_ratio=tc.scheduler.min_lr_ratio,
    )
    trainer_cfg = TrainerConfig(
        num_epochs=num_epochs,
        eval_freq=tc.eval_freq,
        eval_iter=tc.eval_iter,
        grad_accum_steps=tc.grad_accum_steps,
        grad_clip_norm=tc.grad_clip,
        precision=tc.precision,
        save_every=tc.save_every,
        seed=seed,
    )
    trainer = Trainer(model, optimizer, trainer_cfg, scheduler=scheduler)
    history: TrainHistory = trainer.train(train_loader, val_loader, device, resume_from=args.resume)

    if history.train_losses:
        print(f"train loss {history.train_losses[-1]:.3f} | val loss {history.val_losses[-1]:.3f}")
        print(f"perplexity {history.perplexities[-1]:.3f} | steps {history.global_steps[-1]}")
    else:
        print("本轮没有触发评估（增大 num_epochs 或调小 eval_freq）")


if __name__ == "__main__":
    main()
