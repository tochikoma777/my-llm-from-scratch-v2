"""Alpaca 指令微调：模板、collate 的屏蔽语义、数据集形状。

全部用**造的小数据 + 字符级分词器**，不联网、不下载权重。

核心是 `custom_collate_fn` 的"第一个 pad 保留为预测目标"这条语义（v1 `SFT:197-198`）：
丢掉它模型就学不到 `<|endoftext|>`，生成会停不下来，而且很容易被误判成超参问题。
所以这里断言到**具体位置的具体值**，不只看形状。
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence, Set

import torch
import torch.nn.functional as functional

from my_llm.finetune.sft import InstructionDataset, custom_collate_fn, format_input
from my_llm.tokenizer.protocol import Tokenizer

# v1 `SFT:270-277` 的三段式模板，逐字固定
PREFIX = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)
PAD_ID = 50256
IGNORE_INDEX = -100

SAMPLES: list[dict[str, str]] = [
    {"instruction": "Name a fruit.", "input": "", "output": "Apple"},
    {"instruction": "Translate to French.", "input": "Hello", "output": "Bonjour"},
    {"instruction": "Sum 1 and 2.", "input": "", "output": "3"},
]


class CharTokenizer:
    """字符级分词器：`encode` 即逐字符 `ord()`，满足 `Tokenizer` 协议。"""

    @property
    def n_vocab(self) -> int:
        """词表大小（ASCII 范围足够测试用）。"""
        return 256

    def encode(self, text: str, *, allowed_special: Set[str] | None = None) -> list[int]:
        """把文本逐字符转成码点。

        Args:
            text: 输入文本。
            allowed_special: 忽略，字符级分词没有特殊 token。

        Returns:
            码点列表。
        """
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        """`encode` 的逆操作。

        Args:
            ids: 码点序列。

        Returns:
            还原出的文本。
        """
        return "".join(chr(int(i)) for i in ids)


def test_format_input_with_input_matches_alpaca_template() -> None:
    """有 `input` 时是"指令段 + 输入段"，逐字等于 v1 的模板。"""
    got = format_input(SAMPLES[1])
    expected = f"{PREFIX}\n\n### Instruction:\nTranslate to French.\n\n### Input:\nHello"
    msg = f"模板与 Alpaca 标准不一致:\n{got!r}"
    assert got == expected, msg


def test_format_input_without_input_omits_section() -> None:
    """`input` 为空时**整段省略**（不是留空标题），与 v1 `SFT:277` 一致。"""
    got = format_input(SAMPLES[0])
    expected = f"{PREFIX}\n\n### Instruction:\nName a fruit."
    msg = f"空 input 应省略 ### Input 段:\n{got!r}"
    assert got == expected, msg
    assert "### Input" not in got


def test_format_input_excludes_response_section() -> None:
    """`format_input` 不含 `### Response:` —— 训练与推理共用同一个字符串。"""
    for sample in SAMPLES:
        msg = f"format_input 不应包含 Response 段: {sample['instruction']}"
        assert "### Response" not in format_input(sample), msg


def test_collate_keeps_first_pad_and_masks_the_rest() -> None:
    """核心语义：targets 里**第一个 pad 保留**，其余 padding 置 `ignore_index`。

    批次 `[[1,2,3], [4,5]]`（pad=50256）：批内最大长度 4。
    - 第 0 条补齐到 4 后只有 1 个 pad → 保留，targets=[2,3,50256]
    - 第 1 条有 2 个 pad → 第一个保留、第二个置 -100，targets=[5,50256,-100]
    """
    inputs, targets = custom_collate_fn([[1, 2, 3], [4, 5]], pad_token_id=PAD_ID)

    assert inputs.tolist() == [[1, 2, 3], [4, 5, PAD_ID]]
    assert targets[0].tolist() == [2, 3, PAD_ID]
    assert targets[1].tolist() == [5, PAD_ID, IGNORE_INDEX]

    # 断言到具体位置：第 1 条的第 1 位是"保留的 pad"，第 2 位才是被屏蔽的
    msg = f"第 1 条的第 1 个 pad 必须保留为预测目标，实际 {int(targets[1][1])}"
    assert int(targets[1][1]) == PAD_ID, msg
    msg = f"第 1 条的第 2 个 pad 必须被屏蔽，实际 {int(targets[1][2])}"
    assert int(targets[1][2]) == IGNORE_INDEX, msg


def test_collate_shapes_are_max_len_minus_one() -> None:
    """`inputs` / `targets` 形状均为 `(batch, max_len - 1)`。"""
    batch = [[1, 2, 3], [4, 5], [6]]
    inputs, targets = custom_collate_fn(batch, pad_token_id=PAD_ID)
    msg = f"形状应为 (3, 3)，实际 {tuple(inputs.shape)} / {tuple(targets.shape)}"
    assert inputs.shape == (3, 3), msg
    assert targets.shape == (3, 3), msg


def test_collate_without_padding_masks_nothing() -> None:
    """序列等长（无 padding）时不应误屏蔽：结尾的 `<|endoftext|>` 仍是预测目标。"""
    inputs, targets = custom_collate_fn([[1, 2], [3, 4]], pad_token_id=PAD_ID)
    assert inputs.tolist() == [[1, 2], [3, 4]]
    assert targets.tolist() == [[2, PAD_ID], [4, PAD_ID]]
    msg = f"无 padding 时不应出现 {IGNORE_INDEX}"
    assert IGNORE_INDEX not in targets.tolist(), msg


def test_collate_allowed_max_length_truncates() -> None:
    """`allowed_max_length` 截断 inputs 与 targets 到同一长度。"""
    inputs, targets = custom_collate_fn(
        [[1, 2, 3, 4, 5]], pad_token_id=PAD_ID, allowed_max_length=3
    )
    msg = f"截断后应为 (1, 3)，实际 {tuple(inputs.shape)}"
    assert inputs.shape == (1, 3), msg
    assert targets.shape == (1, 3), msg
    assert inputs.tolist() == [[1, 2, 3]]


def test_collate_ignore_index_default_is_minus_100() -> None:
    """默认值必须是 `-100`：`train/losses.py` 的 `F.cross_entropy` 靠它屏蔽 padding。"""
    sig = inspect.signature(custom_collate_fn)
    default = sig.parameters["ignore_index"].default
    msg = f"ignore_index 默认应为 {IGNORE_INDEX}，实际 {default}"
    assert default == IGNORE_INDEX, msg


def test_ignore_index_matches_cross_entropy_semantics() -> None:
    """`-100` 在 `functional.cross_entropy(mean)` 下确实被排除（两边同一套约定的证据）。

    logits 全 0 ⇒ 每个位置损失都是 `log(vocab_size)`；被屏蔽的位置不参与平均，
    因此均值仍等于 `log(vocab_size)`——若 `ignore_index` 与 losses.py 不一致，
    均值会变成 `log(vocab_size) * (有效位置数 / 总位置数)`。
    """
    vocab_size = 5
    logits = torch.zeros(2, 3, vocab_size)
    targets = torch.tensor([[1, 2, IGNORE_INDEX], [3, IGNORE_INDEX, IGNORE_INDEX]])
    loss = functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    expected = math.log(vocab_size)
    msg = f"被屏蔽位置不应参与平均: {loss.item()} vs {expected}"
    assert math.isclose(loss.item(), expected, rel_tol=1e-6), msg


def test_instruction_dataset_shapes_and_content() -> None:
    """`__len__` 为样本数，`__getitem__` 是"prompt + 回复"的完整 token 序列。"""
    tokenizer: Tokenizer = CharTokenizer()
    dataset = InstructionDataset(SAMPLES, tokenizer)

    msg = f"样本数应为 {len(SAMPLES)}，实际 {len(dataset)}"
    assert len(dataset) == len(SAMPLES), msg

    for i, sample in enumerate(SAMPLES):
        ids = dataset[i]
        assert isinstance(ids, list)
        expected = format_input(sample) + f"\n\n### Response:\n{sample['output']}"
        msg = f"第 {i} 条应为 format_input + Response 的编码结果"
        assert ids == [ord(c) for c in expected], msg


def test_instruction_dataset_pre_tokenizes_once() -> None:
    """构造时一次性分词：`encoded_texts` 长度与样本数一致且非空。"""
    tokenizer: Tokenizer = CharTokenizer()
    dataset = InstructionDataset(SAMPLES, tokenizer)
    msg = "encoded_texts 应在构造时填好"
    assert len(dataset.encoded_texts) == len(SAMPLES), msg
    assert all(len(seq) > 0 for seq in dataset.encoded_texts)
