.PHONY: install test test-full check lint fmt demo clean

export HF_ENDPOINT ?= https://hf-mirror.com

install:
	pip install -e ".[dev,viz]"
	pre-commit install

test:
	pytest -q

test-full:
	pytest -q -m slow -v        # parity 测试，会下载 GPT-2 权重

lint:
	ruff check src tests scripts
	mypy src

fmt:
	ruff format src tests scripts

check: lint test

demo:
	python scripts/train.py --config configs/gpt2-tiny.yaml --train-config configs/train-demo.yaml
	python scripts/generate.py --config configs/gpt2-tiny.yaml \
		--checkpoint outputs/checkpoints/last.pt --prompt "Every effort moves you"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov
	find . -name __pycache__ -type d -exec rm -rf {} +
