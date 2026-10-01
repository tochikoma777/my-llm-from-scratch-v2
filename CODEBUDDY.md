# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Repository state

`v2-work/` is a freshly initialized git repo (no commits, no files) intended to host the **v2** rewrite.
All currently existing code lives in the sibling directory `../v1-reference/` — a from-scratch GPT
implementation (`my-LLM-from-scratch`, by tochikoma777) that v2 is derived from. Until v2 files exist,
treat `../v1-reference/` as the source of truth for behavior, and read it before changing anything.

There is no `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md`
anywhere in either tree. The only prose documentation is `../v1-reference/README.md` (Chinese).

## Environment

Python >= 3.8, PyTorch >= 2.0. There is **no** `requirements.txt`, `pyproject.toml`, `setup.py`, or
`setup.cfg` in v1-reference — dependencies are documented only as prose in its README:

```bash
pip install torch tiktoken matplotlib tqdm tensorflow numpy requests
```

`tensorflow` is a hard import-time dependency of the weight-loading module
(`v1-reference/src/modules/module_load_param.py:24`) and is required even for code paths that download
nothing (e.g. `--test_mode`), because it is used to read the OpenAI TF checkpoints.

## Commands

Scripts must be run **from inside the directory that contains them**, because every import is flat
(`from language_module import GPTModel`), not package-relative:

```bash
cd ../v1-reference/src/modules

python module_train.py                    # pretrain a small GPT on the-verdict.txt
python module_load_param.py               # download OpenAI GPT-2 weights + generate text
python module_fine_tuning.py              # instruction fine-tune (Alpaca format)
python module_fine_tuning.py --test_mode  # tiny model, fast smoke run; the only CLI flag in the repo
```

Running `python src/modules/module_train.py` from the repo root makes imports resolve (the script dir
is prepended to `sys.path`) but breaks path-dependent I/O, since data files and outputs are all
CWD-relative — it re-downloads the dataset into the repo root. `python -m src.modules.module_train`
fails: the modules are not importable as a package.

Generated artifacts land in the CWD and are not gitignored: `loss.pdf`, `model.pth`,
`loss-plot-standalone.pdf`, `instruction-data-with-response-standalone.json`, `<Model>-sft-standalone.pth`,
and the `gpt2/` weight download directory.

**Tests, lint, and CI: none exist.** `tests/` contains only an empty `__init__.py`. There is no pytest
config, `Makefile`, `tox.ini`, `.pre-commit-config.yaml`, or linter/formatter config (no ruff, flake8,
black, mypy). There is no single-test command to run.

## Architecture

Six modules in `v1-reference/src/modules/`, all importing each other by bare filename:

- **`language_module.py`** — the model. `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`,
  `TransformerBlock` (Pre-LN + residuals), `GPTModel`. Consumes a config dict with keys
  `vocab_size, emb_dim, context_length, n_heads, n_layers, drop_rate, qkv_bias`. Note that
  `GPT_CONFIG_124M` is *not* defined here — it is defined redundantly in three places with conflicting
  `context_length` values (256 in `module_train.py`, 1024 in `archives/gpt_module.py`, 256 in
  `archives/pretrain_module.py`).
- **`data_preprocess.py`** — `create_dataloader_v1` (tiktoken GPT-2 BPE, sliding-window input/target
  pairs; defaults `batch_size=4, max_length=256, stride=128`).
- **`module_train.py`** — the training core, and the module everything else leans on:
  `calc_loss_batch`, `calc_loss_loader`, `evaluate_model`, `generate_and_print_sample`,
  `train_model_simple`, `plot_losses`, `main`. Downloads `the-verdict.txt` on demand.
- **`module_load_param.py`** — downloads OpenAI GPT-2 TF checkpoints into `gpt2/<size>/`, then maps them
  into `GPTModel` (`download_and_load_gpt2`, `load_gpt2_params_from_tf_ckpt`, `assign`,
  `load_weights_into_gpt` — handles QKV splitting, transposes, `g`/`b` → `scale`/`shift`, and ties
  `out_head` to `wte`). Also holds `generate` (temperature + top-k sampling).
- **`module_fine_tuning.py`** — instruction tuning. Alpaca prompt built by `format_input`
  (`### Instruction:` / optional `### Input:` / `### Response:`), `InstructionDataset`, and
  `custom_collate_fn`, which pads with `50256` and masks with `ignore_index=-100` so loss is only
  computed on the response (deliberately keeping the *first* pad token as a target so the model learns
  to stop). Uses `calc_loss_loader` and `train_model_simple` from `module_train.py`.
- **`generate_text_simple.py`** — greedy decoding helper, library only.

`archives/` holds superseded early drafts (no `__init__.py`, so run them from inside `archives/`). Much
of it duplicates `src/modules/` — model classes, `assign`/`load_weights_into_gpt`, `generate`,
`plot_losses`, and the tokenizer helpers — and roughly half of each file is commented-out dead code kept
as a manual "test" record.

## Conventions and gotchas

- **Chinese throughout**: comments, docstrings, `print()` output, plot labels, and argparse help text are
  all Chinese. Filenames and identifiers are ASCII. Match the surrounding language when editing.
- **CWD-relative paths everywhere**: `the-verdict.txt`, `instruction-data.json`, `gpt2/`, and all output
  filenames. Nothing uses `pathlib` or paths relative to `__file__`.
- **Config dicts are mutated in place**: `BASE_CONFIG.update(model_configs[CHOOSE_MODEL])` in both
  `module_load_param.py` and `module_fine_tuning.py`. `archives/load_pretrained_gpt_module.py` copies
  first; the `src/` versions do not.
- **`generate()` assumes batch size 1** (`idx_next.item()`).
- **v1 README is partly wrong**: it documents `src/models/` (the directory is `src/modules/`), links a
  `LICENSE` that does not exist, and shows `python module_train.py` from the repo root, which cannot
  work. Do not trust it over the code.
- `archives/gpt_download.py` is vendored Apache-2.0 code from Raschka's *LLMs-from-scratch*, which is the
  upstream reference for the whole project.
