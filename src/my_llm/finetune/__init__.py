"""指令微调（Alpaca SFT），搬运自 v1 `module_fine_tuning.py`。"""

from my_llm.finetune.sft import InstructionDataset as InstructionDataset
from my_llm.finetune.sft import custom_collate_fn as custom_collate_fn
from my_llm.finetune.sft import format_input as format_input

__all__ = ["InstructionDataset", "custom_collate_fn", "format_input"]
