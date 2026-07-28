.PHONY: install env test lint typecheck check build-sample search-sample eval-sample

install: env
	uv sync

env:
	python3 scripts/sync_env.py

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

check: lint typecheck test

build-sample:
	uv run cybernaut-mini build \
		--input data/sample/documents.jsonl \
		--index artifacts/sample \
		--config configs/tiny.yaml

search-sample:
	uv run cybernaut-mini search \
		--index artifacts/sample \
		--mode hybrid \
		--question "What links gut bacteria to immune response?"

eval-sample:
	uv run cybernaut-mini eval \
		--index artifacts/sample \
		--judgments data/sample/judgments.jsonl
