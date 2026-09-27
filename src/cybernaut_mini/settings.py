"""Kedro project settings: the Omega config loader and the reproducibility hook.

Defaults are used except where noted below.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — the settings wire the
    run-level guards that make ``(corpus, config, seed) -> index`` hold for the build
    stages the post describes. Local copy:
    ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - ``OmegaConfigLoader`` is required, not cosmetic: ``conf/base/catalog.yml`` uses
      Omega's ``${runtime_params:...}`` resolver to let the Typer CLI forward
      ``--input`` / ``--index`` as Kedro runtime params.
    - ``base_env="base"`` and ``default_run_env="local"`` keep the Kedro convention,
      so ``kedro run`` needs no ``--env`` while ``--env prod`` switches the catalog.
    - ``ReproducibilityHooks`` is always registered. It is the only hook, and it is
      what fails a ``prod`` run early when ``embedding.revision`` is not pinned.

Alternatives considered:
    - Kedro's default ``ConfigLoader``: rejected because it cannot resolve the
      ``${runtime_params:name,default}`` references the catalog depends on.
    - Enforcing the pinned-revision rule inside the embedding provider: rejected
      because it fires after ingestion and tokenisation have already run, wasting the
      expensive part of the pipeline; a hook at context creation fails in about a
      second.
    - A CI-only check on ``conf/prod/parameters.yml``: rejected as the sole mechanism
      because ``--params`` can override the revision at run time, which no static file
      check can see.
"""

from kedro.config import OmegaConfigLoader

from cybernaut_mini.hooks import ReproducibilityHooks

CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}

HOOKS = (ReproducibilityHooks(),)
