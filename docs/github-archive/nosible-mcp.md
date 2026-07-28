---
title: "nosible-mcp — MCP server for the NOSIBLE Search API"
source: https://github.com/NosibleAI/nosible-mcp
endpoint: https://nosible-mcp.onrender.com/mcp/
licence: none stated
fetched: 2026-07-28
archived-by: GitHub REST API (educational archive)
---

# `nosible-mcp`

> Derived notes, not a verbatim copy. Source © Nosible Inc.
> Read alongside [`nosible-py.md`](nosible-py.md) and
> [`../REVERSE_ENGINEERING_GUIDE.md`](../REVERSE_ENGINEERING_GUIDE.md) §1–§2.

A thin `FastMCP` server exposing three NOSIBLE tools over streamable HTTP. Small repo,
but it contains one thing found nowhere else: **the exact query-expansion prompt**, with
its guidelines and sampling parameters.

Note: this server is already configured in this workspace as the `nosible` MCP server
(`mcp__nosible__fast-search`, `scrape-url`, `topic-trend`).

## Architecture

```
FastAPI (uvicorn, 0.0.0.0:10000)
├── CORSMiddleware            — wide open; README notes "tighten for production"
├── PerUserKeyMiddleware      — X-Nosible-Api-Key header → ContextVar, per-request
├── GET /healthz              — {"ok": true}
└── /mcp/  → FastMCP("nosible-mcp", stateless_http=True)
             ├── fast-search
             ├── scrape-url
             └── topic-trend
```

Three design choices worth copying:

1. **Per-request credentials via `ContextVar`.** The API key rides in the
   `X-Nosible-Api-Key` header, is `set()` into a `ContextVar` for the duration of one
   request, and is `reset()` in a `finally`. Process-wide env vars are never touched, so
   one server instance safely multiplexes many users' keys.
2. **`stateless_http=True`** and lazy `from nosible import Nosible` *inside* each tool
   body — "keeps server startup instant," and makes the server horizontally scalable with
   no session affinity.
3. **Errors are returned, not raised**: every tool wraps its call in
   `try/except` and returns `{"error": str(e)}`. An agent gets a readable failure it can
   reason about instead of a transport-level exception.

`dns_rebinding_protection` is explicitly disabled via `TransportSecuritySettings`.

## The query-expansion prompt [disclosed]

When `fast-search` is called **without** `expansions`, the server asks the *calling client's*
LLM to generate them via MCP sampling (`ctx.sample`) — NOSIBLE never pays for this token
spend. Sampling parameters:

```python
temperature=0.6, max_tokens=800,
system_prompt="You generate search query expansions. Output strict JSON only.",
model_preferences={"speedPriority": 0.6, "costPriority": 0.2, "intelligencePriority": 0.7},
```

The user prompt, abridged to its operative content:

> **TASK DESCRIPTION** — Given a search question you must generate a list of 10 similar
> questions that have the same exact semantic meaning but are contextually and lexically
> different to improve search recall.
>
> **RESPONSE FORMAT** — a list of ten strings, each a grammatically correct question.
>
> **EXPANSION GUIDELINES**
> 1. **Use specific named entities** — mention specific people, locations, organizations,
>    products, places.
> 2. **Expansions must be highly targeted** — semantically unambiguous, **between ten and
>    fifteen words**.
> 3. **Expansions must improve recall** — leverage semantic and contextual expansion.
>    *Semantic:* swap "climate change" with "global warming" or "environmental change".
>    *Contextual:* swap "diabetes treatment" with "insulin therapy" or "blood sugar management".

This is the most directly reusable artifact in the whole GitHub org. Three constraints are
doing real work, and all three follow from the retrieval design in
[`nosible-py.md`](nosible-py.md):

- **Named entities are mandatory.** Entity mentions are what let the shard router pick
  the right 30 of ~250,000 shards. A vague expansion probes the wrong shards and returns
  nothing.
- **10–15 words.** Long enough to be semantically unambiguous, matching the blog's claim
  that hybrid retrieval wins on *long high-intent AI queries*.
- **Same meaning, different lexicon.** Expansions exist to diversify the *lexical* half of
  hybrid retrieval, which is why the guidelines are about vocabulary swaps rather than
  broadening scope.

Post-processing: strip, drop any expansion equal to the original (case-insensitive),
dedupe, **cap at 10**. On sampling or JSON-parse failure it degrades silently to
`expansions = []` and searches anyway.

> **Bug worth noting** — the prompt specifies a bare JSON *array*, but the parser reads
> `data.get("expansions", [])`, i.e. an *object* with an `expansions` key. Unless the
> client model happens to emit the wrapped form, this parse yields `[]` and the
> `except` path is never even reached. Do not copy this pattern without fixing the
> contract mismatch.

→ `cybernaut_mini` mirror: `src/cybernaut_mini/expansion.py`,
`src/cybernaut_mini/providers/query_generator.py`

## Tools

### `fast-search`

Signature mirrors `Nosible.fast_search` with **client-level-only options dropped**:
no `sql_filter`, no `include_companies` / `exclude_companies`, no
`include_docs` / `exclude_docs`, no `autogenerate_expansions` (the server decides).
Everything else — `n_probes` (default 30, min 5, max 50), `n_contextify` (128–1024),
`algorithm="hybrid-3"`, the date pairs, `certain`, netloc lists, and the full
GICS/IAB/geography/brand-safety facet set — is exposed verbatim. Returns
`ResultSet.to_dict()`.

The docstrings here carry **tighter bounds than the Python client's** and are the best
public source for the parameter contract; they are transcribed in
[`nosible-py.md`](nosible-py.md).

### `scrape-url`

`(html="", recrawl=False, render=False, url=None)` → `WebPageData.to_dict()`.
`render=True` allows JS rendering; `html=...` processes supplied markup instead of
fetching. → pairs with [`smol-html.md`](smol-html.md).

### `topic-trend`

`(query, start_date=None, end_date=None)` → `{"2005-01-31": ..., "2020-12-31": ...}`.
The MCP surface **omits** the client's `sql_filter`, so faceted trends are not reachable
from an agent.

## Client configuration

VS Code (`.vscode/mcp.json`) and Cursor use the HTTP transport directly:

```json
{
  "servers": {
    "nosible-mcp": {
      "type": "http",
      "url": "https://nosible-mcp.onrender.com/mcp/",
      "headers": { "X-Nosible-Api-Key": "YOUR_NOSIBLE_API_KEY_HERE" }
    }
  }
}
```

Claude Desktop bridges through `npx mcp-remote` with `--header X-Nosible-Api-Key:${NOSIBLE_API_KEY}`.

## Takeaways for `cybernaut_mini`

1. **The expansion prompt is the highest-value disclosure** — port its three guidelines
   into `expansion.py` and evaluate against them.
2. **Push expansion cost to the caller.** Sampling through the client's LLM is why the
   free tier can afford autogenerated expansions.
3. **The MCP surface is a deliberate subset.** Agents get retrieval knobs and facets;
   they do not get raw SQL or document-hash filters. Useful precedent when scoping
   `agent/actions.py` — a smaller action space is a feature.
