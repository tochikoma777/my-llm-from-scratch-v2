#!/usr/bin/env python
"""Alpaca 指令微调入口。

用法：
    python scripts/sft.py --json data/instruction-sample.json --test-mode
    python scripts/sft.py --config configs/gpt2-tiny.yaml --sft-config configs/sft-alpaca.yaml

配置分两份，**架构**来自 `--config`（缺省时用 `--sft-config` 里 `model.config` 指向的
yaml），**数据侧 / 训练侧**超参来自 `--sft-config`。两者分离，避免 v1 那样把架构配置
就地 `update()` 覆盖（`SFT:446` 的 in-place 修改问题）。

**不联网**：本脚本不会下载指令数据。数据只能来自 `--json` 或 yaml 的 `data.local_json`，
两者都没有就直接报错退出（而不是偷偷去 GitHub 拉），这样离线环境行为可预期。

`--epochs` / `--lr` / `--seed` 是**可选覆盖**：默认 `None`，`None` 时回落
`--sft-config` 的 `train:` 段（与 `scripts/train.py` 同款：yaml 兜底 + CLI 覆盖）。
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from my_llm.config import GPTConfig
from my_llm.finetune.sft import (
    InstructionDataset,
    custom_collate_fn,
    generate_responses,
    run_sft,
)
from my_llm.model.gpt import GPTModel
from my_llm.tokenizer import build_tokenizer
from my_llm.utils.logging import configure_logging, get_logger
from my_llm.utils.seed import seed_worker, set_seed

logger = get_logger("scripts.sft")

# --test-mode 时每个划分取多少条（v1 `SFT:354-356` 用的是 10）
TEST_MODE_SAMPLES = 10
_PRECISION_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命名空间对象。
    """
    parser = argparse.ArgumentParser(description="Alpaca 指令微调")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="模型架构 yaml；缺省时用 --sft-config 里 model.config 的路径",
    )
    parser.add_argument(
        "--sft-config",
        type=Path,
        default=Path("configs/sft-alpaca.yaml"),
        help="SFT 运行参数 yaml（嵌套形态：model / data / train / generation / output）",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="本地指令数据 json；覆盖 yaml 的 data.local_json"
    )
    parser.add_argument("--test-mode", action="store_true", help="每个划分只取 10 条，快速验证")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 yaml 的 train.num_epochs")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 yaml 的 train.learning_rate")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 yaml 的 train.seed")
    return parser.parse_args()


def split_data(
    data: list[dict[str, str]], train_ratio: float, test_ratio: float
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """按 v1 `SFT:344-349` 的比例切分训练 / 测试 / 验证。

    Args:
        data: 完整指令数据。
        train_ratio: 训练集占比，其余按 `test_ratio` 给测试集、再剩下的给验证集。
        test_ratio: 测试集占比。

    Returns:
        `(train, test, val)`。
    """
    train_portion = int(len(data) * train_ratio)
    test_portion = int(len(data) * test_ratio)
    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]
    return train_data, test_data, val_data


def main() -> None:
    """执行微调并保存产物。

    Returns:
        None。

    Raises:
        FileNotFoundError: 配置文件或指令数据不存在。
        ValueError: 既没有 `--json` 也没有 yaml 的 `data.local_json`（离线策略：不联网下载）。
    """
    args = parse_args()
    configure_logging()

    raw: dict[str, Any] = yaml.safe_load(args.sft_config.read_text(encoding="utf-8"))
    model_section: dict[str, Any] = raw["model"]
    data_section: dict[str, Any] = raw["data"]
    train_section: dict[str, Any] = raw["train"]
    gen_section: dict[str, Any] = raw["generation"]
    out_section: dict[str, Any] = raw["output"]

    # 嵌套形态：model.config 再指向一份架构 yaml，不能直接喂 GPTConfig.from_yaml
    arch_path = args.config if args.config is not None else Path(model_section["config"])
    cfg = GPTConfig.from_yaml(arch_path)

    num_epochs = args.epochs if args.epochs is not None else int(train_section["num_epochs"])
    lr = args.lr if args.lr is not None else float(train_section["learning_rate"])
    seed = args.seed if args.seed is not None else int(train_section["seed"])

    local_json = data_section.get("local_json")
    json_path = args.json if args.json is not None else (Path(local_json) if local_json else None)
    if json_path is None:
        msg = (
            "离线策略：SFT 不联网下载数据，请用 --json 指定本地 json，"
            "或在 yaml 的 data.local_json 写路径"
        )
        raise ValueError(msg)
    if not json_path.is_file():
        msg = f"指令数据不存在: {json_path}"
        raise FileNotFoundError(msg)

    data: list[dict[str, str]] = json.loads(json_path.read_text(encoding="utf-8"))
    train_data, test_data, val_data = split_data(
        data, float(data_section["train_ratio"]), float(data_section["test_ratio"])
    )
    if args.test_mode:
        # 只影响数据量，不与 model / device 耦合（v1 `SFT:407-421` 把两者绑在一起）
        train_data = train_data[:TEST_MODE_SAMPLES]
        test_data = test_data[:TEST_MODE_SAMPLES]
        val_data = val_data[:TEST_MODE_SAMPLES]
    logger.info(
        "数据: 共 %d 条 -> 训练 %d / 测试 %d / 验证 %d",
        len(data),
        len(train_data),
        len(test_data),
        len(val_data),
    )

    set_seed(seed)
    tokenizer = build_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(cfg).to(device)
    dtype = _PRECISION_DTYPE.get(str(model_section.get("precision", "fp32")))
    if dtype is not None:
        model.to(dtype=dtype)
    # 参数量按实际架构算：--config 覆盖时 yaml 里的 size 标注可能已经不符（只作记录）
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "模型: %s（%d 参数，%s，%s）", arch_path, num_params, dtype or torch.float32, device
    )

    collate = partial(
        custom_collate_fn, allowed_max_length=int(train_section["allowed_max_length"])
    )
    num_workers = int(train_section["num_workers"])
    batch_size = int(train_section["batch_size"])
    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    run_sft(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=float(train_section["weight_decay"]),
        device=device,
    )

    # 贪婪解码（temperature=0）保证评测可复现
    responses = generate_responses(
        model=model,
        test_data=test_data,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=int(gen_section["max_new_tokens"]),
    )
    responses_path = Path(out_section["responses_json"])
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    responses_path.write_text(json.dumps(responses, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info("回复已保存: %s（%d 条）", responses_path, len(responses))

    model_path = Path(out_section["model_pth"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info("模型已保存: %s", model_path)


if __name__ == "__main__":
    main()
