"""Alpaca 指令微调。

搬运来源：v1 `src/modules/module_fine_tuning.py`
（`InstructionDataset:59`、`custom_collate_fn:136`、`format_input:248`、主流程 `main:315`）。

**必须原样保留的语义**（写错最容易"训练能跑但效果崩"的两点）：

1. **prompt 模板**：`format_input` 里的三段式模板（`SFT:270-277`）必须逐字保持，
   训练与推理共用同一个字符串，否则模型看到的分布不一致。
2. **保留第一个 pad 作为预测目标**（`SFT:197-198`）：targets 里除第一个 pad 之外的
   padding 位置全部置 `-100`，这样模型既不在无意义 padding 上算损失，
   又能学到"生成到这里该停"——即学会输出 `<|endoftext|>`。

相对 v1 的修正：
- collate 的 `device` 不再通过 `partial` 提前绑定后再被改成 `"cpu"`（v1 `SFT:371-375` vs `:421`
  的 device 不一致 bug）；改为默认 `"cpu"`，由 Trainer 负责搬运。
- 解除 `--test_mode` 与 model/device 的耦合，测试模式只影响数据量 steps。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from my_llm.generate.sampling import generate
from my_llm.model.gpt import GPTModel
from my_llm.tokenizer.protocol import Tokenizer
from my_llm.train.losses import calc_loss_batch, calc_loss_loader
from my_llm.utils.logging import get_logger

logger = get_logger("finetune.sft")

# 训练前 / 每轮结束评估时，最多跑多少个批次（v1 `SFT:468-469` 用的是 5）
_EVAL_BATCHES = 5


class InstructionDataset(Dataset[list[int]]):
    """Alpaca 指令数据集（v1 `SFT:59`）。

    构造时一次性完成分词，避免训练循环里重复编码。

    Attributes:
        encoded_texts: 每条样本的 token ID 列表。
    """

    def __init__(self, data: Sequence[Mapping[str, str]], tokenizer: Tokenizer) -> None:
        """初始化并预分词。

        Args:
            data: 含 `instruction` / `input` / `output` 字段的字典序列。
            tokenizer: 分词器。
        """
        self.data = data
        self.encoded_texts: list[list[int]] = []

        for entry in data:
            # 与推理时用同一个 format_input（语义 1），再拼上回复部分一起编码
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(instruction_plus_input + response_text))

        lengths = [len(seq) for seq in self.encoded_texts]
        logger.info(
            "数据集: %d 条，平均 %.1f tokens，最长 %d，最短 %d",
            len(self.encoded_texts),
            sum(lengths) / len(lengths) if lengths else 0.0,
            max(lengths, default=0),
            min(lengths, default=0),
        )

    def __len__(self) -> int:
        """样本数量。

        Returns:
            样本数。
        """
        return len(self.encoded_texts)

    def __getitem__(self, index: int) -> list[int]:
        """取第 index 条的 token ID 列表。

        Args:
            index: 样本下标。

        Returns:
            token ID 列表。
        """
        return self.encoded_texts[index]


def format_input(entry: Mapping[str, str]) -> str:
    """把一条 Alpaca 样本格式化成标准 prompt（v1 `SFT:248`）。

    Args:
        entry: 含 `instruction` 与可选 `input` 的字典。

    Returns:
        模板化后的文本（不含 `### Response:` 部分）。
    """
    # 模板（v1 `SFT:270-277`）逐字保持：训练与推理共用同一个字符串，改一个字就会训推不一致
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def custom_collate_fn(
    batch: Sequence[list[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """变长批次整理 + 损失屏蔽（v1 `SFT:136`）。

    Args:
        batch: 每条样本的 token ID 列表。
        pad_token_id: 填充 token，默认 `<|endoftext|>`。
        ignore_index: 参与交叉熵时被忽略的标签值。
        allowed_max_length: 超长截断阈值；`None` 表示不截断。

    Returns:
        `(inputs, targets)`，形状均为 `(batch, max_len - 1)`。

    Note:
        **保留第一个 pad 作为预测目标**（语义 2，`SFT:197-198`）：targets 里除第一个
        pad 之外的 padding 全置 `ignore_index`。少这一个例外，模型就学不到"该停了"。

        `ignore_index` 默认 `-100` 不能改：`train/losses.py` 的 `F.cross_entropy` 用
        `mean` 归约，`-100` 会被天然屏蔽，两边必须同一套约定。

        张量恒定留在 CPU（语义 3）：v1 用 `partial` 预绑定 device，又在别处改成 `"cpu"`
        （`SFT:371-375` vs `:421`），搬运统一交给 Trainer / 调用方。
    """
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst: list[torch.Tensor] = []
    targets_lst: list[torch.Tensor] = []

    for item in batch:
        # +1 是为了给结尾补一个 <|endoftext|>；再补齐到批内最大长度
        new_item = list(item) + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        # 只有"多于一个 padding"时才有得屏蔽：第一个 pad 留着当预测目标
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst), torch.stack(targets_lst)


def run_sft(
    model: torch.nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device,
) -> None:
    """执行一轮完整的监督微调（v1 `SFT:491-550` 的训练部分）。

    Args:
        model: 待微调模型（通常已加载预训练权重）。
        train_loader: 训练数据。
        val_loader: 验证数据。
        num_epochs: 轮数（微调通常 2~3）。
        learning_rate: 学习率（比预训练小 1~2 个数量级）。
        weight_decay: 权重衰减。
        device: 计算设备。

    Returns:
        None；副作用是模型参数被更新。

    Note:
        与 `train/trainer.py:Trainer` 同款约定：**只把 batch 搬到 device，模型由调用方搬**
        （`calc_loss_batch` 内部 `.to(device)`），这里不代劳 `.to()`。
    """
    dev = torch.device(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 微调前的基线（v1 `SFT:466-472`）：比只看训练曲线更容易发现"加载错了权重"
    model.eval()
    with torch.no_grad():
        base_train = calc_loss_loader(train_loader, model, dev, num_batches=_EVAL_BATCHES)
        base_val = calc_loss_loader(val_loader, model, dev, num_batches=_EVAL_BATCHES)
    logger.info("微调前: train loss %.3f | val loss %.3f", base_train, base_val)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for input_batch, target_batch in train_loader:
            loss = calc_loss_batch(input_batch, target_batch, model, dev)
            optimizer.zero_grad(set_to_none=True)
            # 用 torch.autograd.backward 而非 Tensor.backward：后者在 torch 类型存根里
            # 没有注解，mypy --strict 会报 no-untyped-call（与 Trainer 保持同款写法）
            torch.autograd.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        model.eval()
        with torch.no_grad():
            val_loss = calc_loss_loader(val_loader, model, dev)
        train_loss = total_loss / num_batches if num_batches else float("nan")
        logger.info(
            "epoch %d/%d: train loss %.3f | val loss %.3f",
            epoch + 1,
            num_epochs,
            train_loss,
            val_loss,
        )


def generate_responses(
    model: torch.nn.Module,
    test_data: Sequence[Mapping[str, str]],
    tokenizer: Tokenizer,
    device: str | torch.device,
    max_new_tokens: int = 256,
) -> list[dict[str, str]]:
    """为测试集逐条生成回复（v1 `SFT:518-538`）。

    Args:
        model: 微调后的模型。
        test_data: 测试数据。
        tokenizer: 分词器。
        device: 计算设备。
        max_new_tokens: 单条回复的最大生成长度。

    Returns:
        带 `model_response` 字段的样本列表。

    Note:
        走**贪婪解码**（`temperature=0`），评测要可复现；需要采样时直接用
        `generate/sampling.py:generate`。

        `context_size` 取 `model.cfg.context_length`（`GPTModel` 把配置挂在 `.cfg` 上，
        `model/gpt.py:52`）；模型不是 `GPTModel` 时取它自己的属性会报 AttributeError。
    """
    dev = torch.device(device)
    # 需要 context_size 做生成时的上下文截断，而它只存在于 GPTModel 的配置快照里
    if not isinstance(model, GPTModel):
        got = type(model).__name__
        msg = f"generate_responses 需要 GPTModel（读 cfg.context_length），收到 {got}"
        raise TypeError(msg)
    context_size = model.cfg.context_length
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    model.eval()

    results: list[dict[str, str]] = []
    for entry in test_data:
        prompt = format_input(entry)
        prompt_ids = tokenizer.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
        out = generate(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            temperature=0.0,
            eos_id=eos_id,
        )
        # 只解码新生成的部分，再去掉模型自己生成的 "### Response:" 前缀
        # （v1 `SFT:535` 是按字符切 `generated_text[len(input_text):]`，对不上就整段错位）
        new_text = tokenizer.decode(out[0][len(prompt_ids) :].tolist())
        results.append({**entry, "model_response": new_text.replace("### Response:", "").strip()})
    return results
