"""Kedro lifecycle hooks.

Guards that belong to the *run*, not to any single node, live here so they fire
once per session regardless of which pipeline is executed.

Blog ref: https://nosible.com/blog/the-road-to-cybernaut-1 — enforces the build-time
    contract that an index is a rebuildable artifact, the property the whole
    "(corpus, config, seed) -> index" story in the build guide depends on. Local
    copy: ``data/00_reference/the-road-to-cybernaut-1.md``.

Assumptions:
    - Only the ``prod`` environment is strict. Local iteration with unpinned weights
      is normal and useful; shipping an index built that way is not.
    - ``after_context_created`` is early enough. It runs before any node executes,
      so an unpinned production build fails in about a second rather than after an
      hour of embedding.
    - A run without ``embedding`` params does not embed, so it has nothing to pin.

Alternatives considered:
    - Validating inside ``embed_documents``: closer to where the weights are used,
      and impossible to bypass. Rejected because it fails *after* ingestion and text
      processing have already run, wasting the expensive part of the pipeline.
    - A CI check on ``conf/prod/parameters.yml``: catches the same mistake before
      merge. Rejected as the sole mechanism because ``--params`` can override the
      revision at run time, which no static file check can see.
    - Failing on unpinned weights everywhere: simplest rule, no environment
      special-casing. Rejected because it makes the offline sample flow — the first
      thing a new reader runs — require a Hub lookup to get started.
"""

from __future__ import annotations

import logging
from typing import Any

from kedro.framework.hooks import hook_impl

logger = logging.getLogger(__name__)

#: Environments whose builds must be reproducible from (corpus, config, seed).
_STRICT_ENVS = frozenset({"prod"})


class ReproducibilityHooks:
    """Fail a production run early when its output could not be rebuilt later.

    A shard index is only meaningful as a durable artifact if the exact weights that
    produced its vectors can be fetched again. Checking at context creation means an
    unpinned production build fails in a second, rather than after an hour of
    embedding.
    """

    @hook_impl
    def after_context_created(self, context: Any) -> None:
        from cybernaut_mini.config import EmbeddingConfig

        embedding_params = context.params.get("embedding")
        if not isinstance(embedding_params, dict):
            return

        embedding = EmbeddingConfig.model_validate(embedding_params)
        if embedding.is_pinned():
            return

        message = (
            f"embedding model {embedding.model_name!r} is not pinned "
            f"(embedding.revision is {embedding.revision!r}); the resulting index "
            f"cannot be rebuilt byte-for-byte"
        )
        if context.env in _STRICT_ENVS:
            from cybernaut_mini.config import ConfigError

            raise ConfigError(
                f"{message}. Set embedding.revision to a commit SHA in "
                f"conf/{context.env}/parameters.yml."
            )
        logger.warning("%s. Acceptable locally; pin it before a production build.", message)
