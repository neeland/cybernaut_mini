"""The WORLD index families computed over the event store.

- :mod:`~cybernaut_mini.world.indices.gpr` — the geopolitical-risk family:
  global daily attention share, per-country and bilateral-pair monthly indices
  over the shared ``B(m)`` denominator, the trade-coercion OR-patch, and the
  relevance-weighted Oil-GPR AND-gate.
- :mod:`~cybernaut_mini.world.indices.policy` — the policy-uncertainty family:
  daily NOSIBLE-TPU, monthly US-over-US NOSIBLE-EPU, category sub-indices,
  lever ablations, the signed net-polarity series, and the explain-month
  diagnostic.

Blog ref: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
    and https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty.
    Local copies under ``docs/blog-archive/``; the package ``README.md`` holds the
    dataflow diagram.

Assumptions: each family is one module of pure functions over
``Sequence[WorldEvent]`` plus precomputed anchor scores — no state, no I/O — so
every index is replayable byte-for-byte from a frozen event table.

Alternatives rejected: a shared ``Index`` base class (the two families share only
:mod:`~cybernaut_mini.world.timeseries`, which is already the shared layer).
"""

from __future__ import annotations
