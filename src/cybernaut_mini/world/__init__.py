"""The WORLD pillar: an event store plus the indices computed over it.

Where the SEARCH pillar answers questions over documents, the WORLD pillar turns
de-duplicated news *events* into point-in-time risk and uncertainty series. One
module per moving part:

- :mod:`~cybernaut_mini.world.events` — the event store (one record per real event),
- :mod:`~cybernaut_mini.world.vectors` — the frozen-embedder truncate-and-renorm policy,
- :mod:`~cybernaut_mini.world.ner` / :mod:`~cybernaut_mini.world.tickers` /
  :mod:`~cybernaut_mini.world.countries` / :mod:`~cybernaut_mini.world.topics` —
  the tag layers,
- :mod:`~cybernaut_mini.world.anchors` / :mod:`~cybernaut_mini.world.smoother` —
  the shared anchor-signal engine,
- :mod:`~cybernaut_mini.world.timeseries` — dated-series arithmetic,
- :mod:`~cybernaut_mini.world.indices` — the GPR and TPU/EPU index families,
- :mod:`~cybernaut_mini.world.validate` — the benchmark validation harness.

Blog ref: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world,
    https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty,
    https://nosible.com/blog/turning-news-into-a-risk-on-risk-off-equity-signal,
    https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities.
    Local copies under ``docs/blog-archive/``; see ``README.md`` in this package
    for the dataflow diagram.

Assumptions: submodules are imported lazily by callers (``from cybernaut_mini.world
import anchors``) rather than re-exported here, because the tag layers import each
other through the package and an eager re-export list would fix an import order
for no API gain.

Alternatives rejected: a flat ``world.py`` module (the pillar has ~10 independent
concerns and the posts document them separately); eager re-exports like
``cybernaut_mini.entities`` (that package is a single loop with one public surface,
this one is a toolbox).
"""

from __future__ import annotations
