.PHONY: install install-prod env test lint typecheck check \
	build-sample search-sample eval-sample \
	ingest ingest-prod build-prod eval-pipeline viz pipelines

install: env
	uv sync

# Corpus acquisition from the Hub plus real sentence-transformers embeddings.
install-prod: env
	uv sync --extra hf --extra st

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

# ------------------------------------------------------------------ #
# Kedro pipelines                                                     #
# ------------------------------------------------------------------ #

pipelines:
	uv run kedro registry list

# Acquisition only: raw -> intermediate -> primary. Reads the local source
# declared in conf/base/catalog.yml (data/01_raw/corpus_source.jsonl).
ingest:
	uv run kedro run --pipeline corpus_ingest

# Acquisition from the pinned Hugging Face dataset. Needs `make install-prod`
# and, for gated repos, HF_TOKEN.
ingest-prod:
	uv run kedro run --pipeline corpus_ingest --env prod

# Full production shard: acquire, normalise, select, embed, shard, write.
# Aborts immediately unless embedding.revision is pinned in conf/prod.
build-prod:
	uv run kedro run --pipeline production --env prod

# Evaluation through the catalog; writes data/08_reporting/eval_report.json.
eval-pipeline:
	uv run kedro run --pipeline evaluation --params \
		index_path=artifacts/sample,judgments_path=data/sample/judgments.jsonl,embedding.provider=hash,embedding.dim=256,offline=true

viz:
	uv run kedro viz
