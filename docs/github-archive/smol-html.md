---
title: "smol-html — HTML cleaner/minifier for web-scale storage"
source: https://github.com/NosibleAI/smol-html
package: https://pypi.org/project/smol-html/
licence: MIT
fetched: 2026-07-28
archived-by: GitHub REST API (educational archive)
---

# `smol-html`

> Derived notes, not a verbatim copy. Source © Nosible Inc., MIT licensed.
> Relevant to [`../REVERSE_ENGINEERING_GUIDE.md`](../REVERSE_ENGINEERING_GUIDE.md) §8
> (the ingest side of the event layer).

*"Small, dependable HTML cleaner/minifier with sensible defaults."*

The only public component of NOSIBLE's **ingest** path. Everything else in the org sits at
query time; this is what happens to a page between crawl and index.

## Motivation [disclosed]

From the README:

> Nosible is a search engine, which means we need to store and process a very large number
> of webpages. To make this tractable, we strip out visual chrome and other non-essential
> components that don't matter for downstream tasks (indexing, ranking, retrieval, and LLM
> pipelines) while preserving the important content and structure.

Note the framing: **HTML is kept, not discarded.** The output is still HTML — structure
(headings, tables, lists, links) survives, only chrome is removed. That matters downstream:
chunking, `best_chunk` selection, and table extraction all depend on structure the cleaner
preserves. Compare with plain-text extraction, which throws it away.

## Stack

Built on `minify-html` (Rust minifier), `lxml[html-clean]`, `BeautifulSoup4`, and Google's
`brotli`. Python 3.9+. Optional extras: `readability-lxml`
(`pip install "smol-html[readability]"`) and `selectolax`
(`smol-html[selectolax]`); `[all]` for both.

## Usage

```python
from smol_html import SmolHtmlCleaner

cleaner = SmolHtmlCleaner()          # all ctor args are keyword-only and optional
cleaned = cleaner.make_smol(raw_html=html)
```

Compressed output — clean, Brotli, then URL-safe Base64 **by default**:

```python
compressed = cleaner.make_smol_bytes(raw_html=html, compression_level=11)
# base64_encode=True by default → decode before decompressing
decompressed = brotli.decompress(base64.urlsafe_b64decode(compressed)).decode("utf-8")

# or skip Base64 entirely
raw = cleaner.make_smol_bytes(raw_html=html, compression_level=11, base64_encode=False)
brotli.decompress(raw)
```

`make_smol_bytes` defaults: `compression_level=4`, `base64_encode=True`. Level 4 is the
throughput/ratio compromise for bulk ingest; level 11 is for archival. Base64-by-default
costs ~33% size but makes the blob safe to put in a JSON column or text field — a strong
hint that cleaned pages live in a **columnar/text store**, consistent with the
`polars`/`duckdb`/`pyarrow` dependencies in [`nosible-py`](nosible-py.md).

## Design worth stealing

### `attr_stop_words` — heuristic chrome detection

A token set matched against `id` / `class` / `role` / `item_type` **on small elements
only**; matches are deleted as likely non-content. Defaults are "common UI/navigation
tokens," extensible in place:

```python
cleaner.attr_stop_words.add("advert")
```

The size gate is the clever part: a large `<div class="menu">` is probably mislabelled
content and is kept, while a small one is chrome. Cheap, no ML, no per-page cost.

### Three escalating tiers of aggression

| Tier | Flag | Cost | Use |
|---|---|---|---|
| Default | — | baseline | Structure-preserving clean |
| Pre-pass | `prepass_selectolax=True` | very low | Kill known tags/classes/ids *before* full parse |
| Isolation | `content_isolator="readability"` | higher | Extract the primary article first |

The `selectolax` pre-pass (`prepass_kill_tags` / `prepass_kill_classes` /
`prepass_kill_ids`) drops cookie banners and sidebars before lxml ever builds a tree — a
per-site optimisation applied when you already know the offenders. Readability isolation is
the opposite trade: general, but lossier and slower.

### Point-in-time hygiene

- `strip_tracking_query` — removes `utm_*`, `gclid`, `fbclid` from `<a href>`, so URLs
  canonicalise and **dedupe correctly**. Directly relevant to the event layer's
  one-record-per-event requirement (guide §8).
- `strip_tracking_pixels` — drops 1×1 and CSS-hidden images before lxml cleaning.
- `table_normalize` — converts `<br>` inside `td`/`th` to spaces so tables linearise
  cleanly for RAG and summarisation.

All three default to `False`.

### `report_stats`

With `report_stats=True`, every `make_smol` call records into `cleaner.last_stats`:

```python
{"bytes_before": ..., "bytes_after": ..., "pct_delta": ...,
 "node_count": ..., "wall_time_ms": ...}
```

Per-page instrumentation as a first-class option — the ingest pipeline measures its own
compression ratio continuously rather than sampling offline.

## Defaults reference

Cleaning defaults (lxml `Cleaner` options): `javascript=True`, `comments=True`,
`style=True`, `processing_instructions=True`, `embedded=True`, `frames=True`,
`forms=True`, `annoying_tags=True`, `remove_unknown_tags=True`, `safe_attrs_only=True`
(all removing), against `meta=False`, `page_structure=False`, `scripts=False`,
`links=True` (sanitise, don't strip).

`scripts=False` while `javascript=True` is deliberate: JS behaviour and event handlers go,
but `<script>` elements carrying **JSON-LD structured data** survive. That is what feeds
the `structured` field of `WebPageData` in [`nosible-py`](nosible-py.md), and it is an easy
detail to get wrong.

`non_text_to_keep` defaults to media/meta/table/`<br>` tags — empty elements that still
carry meaning are whitelisted rather than swept up by generic empty-node removal.

Other defaults: `minify=True`, `minify_kwargs={}`, `aggressive_strip=False`,
`remove_header_lists=True`, `remove_footer_lists=True`, `kill_tags=None`,
`image_inline_threshold=None` (reserved; stored but not yet acted on).

## Relevance to `cybernaut_mini`

The repo's sample corpus is pre-cleaned JSONL, so `smol-html` is not a runtime dependency.
It matters as **evidence about the corpus** feeding `src/cybernaut_mini/ingest.py` and
`text.py`:

- Documents entering the index are cleaned HTML with structure intact, not raw pages and
  not flat text.
- Navigation, footers, forms, and tracking artifacts are gone before embedding — so
  embedding quality is measured on *content*, and a replica that embeds raw crawled HTML
  will underperform for reasons unrelated to the model.
- URLs are canonicalised at ingest, which is a precondition for de-duplication.

If you ever extend `ingest.py` to accept live HTML, `pip install smol-html` and call
`make_smol` — it is MIT, public, and exactly what production uses.
