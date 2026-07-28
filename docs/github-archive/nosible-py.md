---
title: "nosible-py — the official NOSIBLE Search API client"
source: https://github.com/NosibleAI/nosible-py
package: https://pypi.org/project/nosible/
docs: https://nosible-py.readthedocs.io/
api-spec: https://www.nosible.ai/search/v2/docs/
licence: MIT
fetched: 2026-07-28
archived-by: GitHub REST API (educational archive)
---

# `nosible-py`

> Derived notes, not a verbatim copy. Source © Nosible Inc., MIT licensed.
> Read alongside [`../REVERSE_ENGINEERING_GUIDE.md`](../REVERSE_ENGINEERING_GUIDE.md) §1–§2.

**The single most useful public artifact for this project.** The blog posts describe the
Hybrid-3 pipeline in prose; this client publishes its *parameter contract* — every knob,
its default, and its bound. Everything below is **[disclosed]** unless marked otherwise.

- Python 3.9+
- Deps: `polars`, `duckdb`, `openai`, `tantivy`, `pyrate-limiter`, `tenacity`,
  `cryptography`, `pyarrow`, `pandas`
- Auth: `NOSIBLE_API_KEY` env var (keys have the form `nos_sk_...`; requires package
  ≥ 0.3.12), optional `LLM_API_KEY` for query expansion
- Transport: HTTP against `https://www.nosible.ai/search/v2/...`, key sent in an
  `api-key` header, `Accept-Encoding: gzip`

## Public surface

```python
from nosible import (
    Nosible,                      # client
    Search, SearchSet,            # query construction
    Result, ResultSet,            # results
    Snippet, SnippetSet,          # extracted passages
    WebPageData,                  # scrape output
)
```

`Nosible` methods:

| Method | Rate-limit bucket | Purpose |
|---|---|---|
| `search(prompt, agent="cybernaut-1")` | `fast` | **The agentic entry point.** One prompt, no knobs. |
| `fast_search(...)` | `fast` | Single interactive query, ≤ 100 results |
| `fast_searches(questions=[...], ...)` | `fast` | Concurrent batch of interactive queries |
| `bulk_search(...)` | `bulk` (API: `slow`) | Offline/large pulls, default `n_results=1000` |
| `answer(query, ...)` | — | RAG: `fast_search` → LLM answer over retrieved docs |
| `scrape_url(url, html, recrawl, render)` | `scrape-url` (API: `visit`) | Fetch + parse one page → `WebPageData` |
| `topic_trend(query, start_date, end_date, sql_filter)` | — | News-volume time series for a query |
| `close()` | — | Shut down the thread pool |

Rate limits are **not hard-coded** — the client calls `_get_limits()` at construction and
builds sliding-window `pyrate-limiter` buckets from whatever the server returns for the
`fast` / `slow` / `visit` query types. So limits are per-key and server-authoritative.

## Cybernaut-1 is a one-line API

```python
with Nosible(nosible_api_key="nos_sk_...") as client:
    results = client.search(
        prompt="Find me interesting technical blogs about Monte Carlo Tree Search."
    )
```

`search()` POSTs `{"prompt": ..., "agent": "cybernaut-1"}` to `/search/v2/search` and
returns a `ResultSet`. That is the entire client-side implementation.

The README describes the agent as having *"unrestricted access to everything in NOSIBLE
including every shard, algorithm, selector, reranker, and signal. It knows what these
things are and can tune them on the fly."*

**Read this against the MCTS blog post.** The agent's action space is exactly the
`fast_search` parameter set below — it is a policy that emits `Search` objects. That is a
strong confirmation of the guide's §2 model: *the agent searches over query
reformulations and retrieval settings, not over some separate latent space.* The `agent`
parameter being a string implies more agents are expected. **[inferred]** — only
`"cybernaut-1"` is documented.

→ `cybernaut_mini` mirror: `src/cybernaut_mini/agent/{policy,search,actions,node,state}.py`

## `fast_search` — the full parameter contract

Bounds are taken from the MCP server's docstrings, which are stricter and more explicit
than the client's.

### Retrieval

| Parameter | Default | Bounds | Meaning |
|---|---|---|---|
| `question` | — | — | The query string |
| `expansions` | `None` | **max 10** | Related queries that boost recall |
| `autogenerate_expansions` | `False` | — | Have the client generate them via LLM |
| `sql_filter` | `None` | — | Raw SQL predicate over document metadata |
| `n_results` | `100` | max 100 (`fast`); floored to ≥ 10 internally | Result count |
| `n_probes` | `30` | **min 5, max 50** | **Shards probed.** More = better recall, slower |
| `n_contextify` | `128` | **min 128, max 1024** | Context window returned per result |
| `algorithm` | `"hybrid-3"` | — | Retrieval algorithm |
| `min_similarity` | `0` (`None` → 0) | — | Similarity floor |
| `instruction` | `None` | — | Instruction paired with the query |

**`n_probes` bounds are the headline number.** The blog claims ~250,000 shards; the API
probes **5–50 of them**, default 30 — i.e. ~0.012% of the index per query. The shard router
must therefore be extremely precise, and recall is bought by *expansions* (up to 10 queries
× 30 probes) rather than by scanning more shards. `min_similarity` defaulting to `0` means
filtering is expected to happen through metadata, not the similarity floor.

→ `cybernaut_mini` mirror: `src/cybernaut_mini/{routing,sharding,retrieval,rrf,expansion}.py`

### Lexical and temporal filters

| Parameter | Notes |
|---|---|
| `must_include` / `must_exclude` | `list[str]`; hard lexical gates over result text |
| `publish_start` / `publish_end` | ISO date; when the document was **published** |
| `visited_start` / `visited_end` | ISO date; when **NOSIBLE crawled** it |
| `certain` | `bool` — only docs whose date is known with certainty |

Two independent time axes (publish vs. visited) plus a `certain` flag is a
**point-in-time-correctness** design: it is what lets the quant-signal posts run
backtests without look-ahead bias. Date confidence is modelled as a first-class,
queryable property. See guide §6 and §8.

### Source and document filters

| Parameter | Cap | Notes |
|---|---|---|
| `include_netlocs` / `exclude_netlocs` | 50 | Domain allow/deny lists |
| `include_companies` / `exclude_companies` | 50 | **Google Knowledge Graph IDs** of public companies |
| `include_docs` / `exclude_docs` | 50 | URL hashes |

Companies are keyed by **Google KG ID, not ticker or name**. That means entity resolution
happens at index time against a canonical external KG — direct evidence for the
self-organizing entity tagging and point-in-time KG posts (guide §4, §7). `exclude_docs`
by URL hash is exactly what an MCTS agent needs to avoid re-expanding documents it has
already visited.

### Classification facets

Every result carries these, and every one is also a filter:

| Axis | Values |
|---|---|
| `brand_safety` | `safe` \| `sensitive` \| `unsafe` |
| `language` | ISO 639-1 |
| `continent` / `region` / `country` | e.g. `"Europe"`, `"Southern Africa"`, geographic |
| `sector` / `industry_group` / `industry` / `sub_industry` | **GICS**, 4 levels |
| `iab_tier_1` … `iab_tier_4` | **IAB content taxonomy**, 4 levels |

Two independent 4-level taxonomies (GICS for economic subject, IAB for content type) plus
3-level geography and brand safety. This is the **faceted-search self-organization** post
made concrete — and it is the part of the event layer that is *visible* through the API but
not *reproducible* from it. See guide §7 and §8.

## `Result` schema

```
url, title, description, netloc, published, visited, author,
content, best_chunk, language, similarity, url_hash,
brand_safety, continent, region, country,
sector, industry_group, industry, sub_industry
```

`best_chunk` alongside `content` confirms chunk-level retrieval with the winning passage
surfaced separately — the natural unit for an agent's evidence and for the
frozen-embedding feature work in guide §5.

## `scrape_url` → `WebPageData`

Returns far more than text:

```
full_text, languages (dict of code → probability), metadata, page,
request, snippets (SnippetSet), statistics, structured, url_tree, companies
```

Notable: `languages` is a **distribution**, not a single label (multilingual documents are
expected — guide §1 stage 1); `structured` holds schema.org/OpenGraph extractions;
`companies` means **entity extraction runs at scrape time**, not query time; `url_tree`
gives breadcrumb/navigation structure.

## `topic_trend`

```python
nos.topic_trend("Christmas Shopping", start_date="2005-01-01", end_date="2020-12-31")
# {'2005-01-31': ..., ..., '2020-12-31': ...}
```

Month-end keys, back to **2005**. This is the primitive under every quant-signal post
(GPR, risk-on/risk-off, TPU/EPU — guide §6): those signals are attention-share series, and
this endpoint returns the numerator. `sql_filter` is what turns a generic trend into a
country- or sector-conditioned one.

→ `cybernaut_mini` mirror: the breadth-weighted attention-share template in guide §6.

## `answer` — the RAG default

```python
def answer(self, query, n_results=100, min_similarity=0.65,
           model="google/gemini-2.0-flash-001", show_context=True) -> str
```

Note `min_similarity=0.65` here versus `0` in `fast_search`: **generation gets a much
stricter floor than retrieval.** Retrieve broadly, ground narrowly.

## Client-level defaults

`Nosible.__init__` accepts *every* filter above as a session-wide default, plus:

| Parameter | Default |
|---|---|
| `expansions_model` | `"openai/gpt-4o"` |
| `sentiment_model` | `"openai/gpt-4o"` |
| `openai_base_url` | OpenRouter's endpoint |
| `timeout`, `retries`, `concurrency` | HTTP/threading controls |

Retries use `tenacity` with exponential backoff (`multiplier=1, min=1, max=20`) on
`httpx.RequestError` only. Concurrency is a `ThreadPoolExecutor`.

**OpenRouter as the default LLM base URL** matches the blog's `gemma-3n-e4b-it` translation
stage — NOSIBLE routes auxiliary LLM calls through OpenRouter rather than binding to one
vendor. → `cybernaut_mini` mirror: `src/cybernaut_mini/providers/`.

## What this repo does *not* disclose

- How shards are built, or how the router picks which `n_probes` to hit
- Anything about the reranker stack
- Event de-duplication, `total_netlocs` breadth, or event dating
- How GICS/IAB/geography labels are assigned
- The Cybernaut-1 policy itself — it is entirely server-side

All of these remain **[inferred]** in the build guide. Confirms guide §8.
