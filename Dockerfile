FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

RUN python -m pip install --upgrade pip

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install -e ".[dev,viz]"

COPY . .

CMD ["python", "scripts/generate.py", "--prompt", "Every effort moves you"]
