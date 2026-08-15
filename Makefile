.PHONY: install install-mac install-prod env accel test lint typecheck check \
	build-fixture search-fixture eval-fixture \
	ingest ingest-prod build-prod eval-pipeline viz pipelines \
	notebooks lab entire doctor

install: env
	uv sync

# ------------------------------------------------------------------ #
# Native macOS — the default development environment                  #
# ------------------------------------------------------------------ #

# Apple silicon: Accelerate-linked numpy, torch with the MPS backend, and thread
# counts tuned to performance cores. The devcontainer is still supported (see
# `install`) but is native arm64 *Linux*, which reaches none of those.
install-mac: env
	scripts/setup_mac.sh

# What this machine actually offers: BLAS, SIMD, torch backend, resolved device.
# Run it after install-mac, and any time a build looks slower than it should.
accel:
	uv run python -c "from cybernaut_mini.accel import describe; print(describe().render())"

# Corpus acquisition from the Hub plus real sentence-transformers embeddings.
install-prod: env
	uv sync --extra hf --extra st

env:
	python3 scripts/sync_env.py

# Install the Entire CLI and (re)wire its git + Claude Code hooks. Run after a
# release bumps the hook format, or whenever commits print the "[entire] ... not
# installed" warning. Also runs on devcontainer create.
entire:
	scripts/entire_setup.sh

# Verify the container: node/claude/omc/uv/.venv plus kedro, notebooks, catalog
# and Entire.
doctor:
	scripts/devcontainer_doctor.sh

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

check: lint typecheck test

build-fixture:
	uv run cybernaut-mini build \
		--input data/01_raw/fixtures/documents.jsonl \
		--index artifacts/fixture \
		--config configs/tiny.yaml

search-fixture:
	uv run cybernaut-mini search \
		--index artifacts/fixture \
		--mode hybrid \
		--question "What links gut bacteria to immune response?"

eval-fixture:
	uv run cybernaut-mini eval \
		--index artifacts/fixture \
		--judgments data/01_raw/fixtures/judgments.jsonl

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
		index_path=artifacts/fixture,judgments_path=data/01_raw/fixtures/judgments.jsonl,embedding.provider=hash,embedding.dim=256,offline=true

viz:
	uv run kedro viz

# ------------------------------------------------------------------ #
# Notebooks                                                           #
# ------------------------------------------------------------------ #

# Execute every notebook headlessly against the real catalog. Also run as part of
# `make check`, so a notebook cannot silently drift off the pipeline.
notebooks:
	uv run pytest tests/test_notebooks.py -q

# JupyterLab with catalog/context/session/pipelines pre-injected (port 8888).
lab:
	scripts/lab.sh
