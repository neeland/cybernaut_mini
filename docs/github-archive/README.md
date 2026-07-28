---
source: https://github.com/NosibleAI
fetched: 2026-07-28
archived-by: GitHub REST API (educational archive)
---

# NOSIBLE GitHub Archive

Notes on every **public** repository under the [`NosibleAI`](https://github.com/NosibleAI)
GitHub organisation (org id `129729013`), captured 2026-07-28 for educational study.

These are *derived notes* — API surfaces, defaults, and design decisions extracted from
public source — not verbatim copies. All source is © Nosible Inc. under the licences noted
per repo. `cybernaut_mini` is an unaffiliated clean-room educational replica.

## Why this matters

The [blog archive](../blog-archive/) tells you *what NOSIBLE built*. The GitHub org tells
you the **contract**: the exact parameter names, defaults, bounds, and vocabularies the
production API actually exposes. Where the blog says "we probe shards," `nosible-py` says
`n_probes: int = 30` with `min 5, max 50`. That turns prose into a spec — see
[`../REVERSE_ENGINEERING_GUIDE.md`](../REVERSE_ENGINEERING_GUIDE.md) §1.

## Repositories

| Repo | Language | Licence | Stars | What it is | Notes |
|---|---|---|---|---|---|
| [`nosible-py`](nosible-py.md) | Python | MIT | 12 | Official client for the NOSIBLE Search API v2 | The richest disclosure — full filter vocabulary + `hybrid-3` defaults |
| [`nosible-mcp`](nosible-mcp.md) | Python | *(none stated)* | 2 | MCP server wrapping `nosible-py` | Ships the **verbatim query-expansion prompt** |
| [`smol-html`](smol-html.md) | Python | MIT | 10 | HTML cleaner/minifier | The ingest-side preprocessor for the crawl corpus |
| `nosible-email` | — | Unlicense | 0 | Email signature assets | No technical content; not documented here |

Created dates: `nosible-py` 2025-07-01, `nosible-mcp` 2025-08-28, `smol-html` 2025-09-10,
`nosible-email` 2026-04-13.

## The disclosed stack, end to end

Reading the three technical repos together recovers the outer shell of the pipeline in
[`../REVERSE_ENGINEERING_GUIDE.md`](../REVERSE_ENGINEERING_GUIDE.md) §0:

```
  crawl ──► smol-html ──────────► [ event layer — NOT public, the moat ]
            clean + minify + brotli          │
                                             ▼
                              nosible-py ──► Search API v2 (hybrid-3, 8 stages)
                                   │              │
                                   │              └─► search(agent="cybernaut-1")
                                   ▼
                              nosible-mcp ──► agent tools: fast-search / scrape-url / topic-trend
```

What is **still absent** from all public code: shard construction, the reranker stack,
event de-duplication, `total_netlocs` breadth, and IPTC/GICS tagging. Those are computed
server-side; the client only *selects* against them via filter arguments. This confirms the
guide's §8 conclusion — the event layer is the moat, and approximating it remains the only
original engineering required.

## Related

- [`../huggingface-archive/`](../huggingface-archive/) — the published models and datasets
- [`../blog-archive/`](../blog-archive/) — the 12 public technical posts
- [`../REVERSE_ENGINEERING_GUIDE.md`](../REVERSE_ENGINEERING_GUIDE.md) — the clean-room build guide
