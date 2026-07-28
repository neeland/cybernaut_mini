---
title: "NOSIBLE datasets — shared schema and annotation pipeline"
source: https://huggingface.co/NOSIBLE
licence: ODC-By v1.0 (attribution required)
fetched: 2026-07-28
archived-by: Hugging Face Hub API (educational archive)
---

# NOSIBLE datasets — the shared spec

> Derived notes, not verbatim copies. Data © Nosible Inc. under **ODC-By v1.0**.
> Read alongside [`../../REVERSE_ENGINEERING_GUIDE.md`](../../REVERSE_ENGINEERING_GUIDE.md) §3.

Three datasets, released 2025-11-26 (last modified 2025-12-01), all `v1.0.0`. They share a
schema, a size, a split, a licence, a source, and an annotation pipeline — only the label
column differs. Document the common parts once here.

| Dataset | Labels | Notes |
|---|---|---|
| [`financial-sentiment`](financial-sentiment.md) | `positive` / `neutral` / `negative` | 3-class |
| [`forward-looking`](forward-looking.md) | `forward` / `not forward` | binary, tense |
| [`prediction`](prediction.md) | `prediction` / `not-prediction` | binary, modality |

## Schema

| Field | Type | Description |
|---|---|---|
| `text` | string | Text chunk from a search result |
| `label` | string | Class label (see per-dataset cards) |
| `netloc` | string | Source domain |
| `url` | string | Document URL |

One `train` split, **100,000 rows**, no validation or test split published. Parquet;
loadable via `datasets`, `pandas`, `polars`, `mlcroissant`. Language `en`;
`size_categories: 100K<n<1M`.

Two things follow from this schema. **`netloc` is a first-class column**, which makes
source-stratified splits possible — critical here, because random splitting would leak
near-duplicate syndicated wire copy across the boundary and inflate accuracy. And the unit
is a **chunk, not a document**, matching the `best_chunk` field and the 128–1024-token
`n_contextify` window in
[`../../github-archive/nosible-py.md`](../../github-archive/nosible-py.md). Labels apply to
retrievable passages, not articles.

The absence of published eval splits is why the external
[Financial PhraseBank](financial-sentiment.md) benchmark carries the weight in the model
cards.

## Provenance

Sourced from **NOSIBLE Search Feeds** over public web content, using finance-related
queries. The `forward-looking` card describes the rows as *"cleaned, deduplicated, and
labeled"* — that cleaning is [`smol-html`](../../github-archive/smol-html.md), and the
deduplication is the event layer (guide §8).

So the corpus is **not** a random web sample: it is the output of NOSIBLE's own retrieval
stack against finance queries. Rows are already high-precision, on-topic, deduplicated
passages. A replica that labels raw crawled text is solving a harder problem than these
datasets represent.

## The annotation pipeline [disclosed]

Identical across all three cards, and it *is* the ensemble-and-distil method of guide §3:

1. **Hand-label ~200 samples** for prompt tuning — human effort spent on the *prompt*,
   not the corpus.
2. **Label 100k rows with eight LLM annotators** — Grok 4, Gemini 2.5, GPT variants
   (incl. GPT-5), Llama 4, Qwen3.
3. **Train linear ensemble models over text embeddings** — the `financial-sentiment` and
   `prediction` cards specify **six embedding models**. Cheap, deterministic, no
   per-sample LLM cost.
4. **Iteratively relabel disagreements with an oracle** — **GPT-5.1** adjudicates rows
   where the committee and the ensemble diverge.
5. **Drop underperforming annotators and repeat until convergence.**

Step 3 is the load-bearing one. The linear probe over frozen embeddings is not there to be
accurate; it is there to be *cheap and consistent*, so that committee-vs-probe disagreement
becomes a **signal for where labels are wrong**. Expensive oracle calls go only to those
rows. That is active learning with a free acquisition function, and it is the same
frozen-embedding geometry as guide §5.

Step 5 — dropping annotators — means the committee is **not a fixed vote**. Models that
persistently disagree with the converged consensus are removed rather than down-weighted.

Two consequences for anyone training on this data:

- **Labels are model-derived, not human.** They encode the committee's consensus,
  including its shared blind spots. The 2-epoch / NEFTune-alpha-5 configuration in the
  model cards is a direct response — see
  [`../models/financial-sentiment.md`](../models/financial-sentiment.md).
- **No inter-annotator agreement statistics, per-model accuracies, class balance, or
  date range are published.** Convergence is asserted, not shown.

## Licence and attribution

**Open Data Commons Attribution License (ODC-By) v1.0** — more permissive than the
Apache-2.0 model weights in one sense (commercial use fine, derivatives fine) but it
**requires attribution** in any public use or derived database. If you train on these
rows and publish, you must credit the source.

Contributors named on all three cards: **Matthew Dicks, Simon van Dyk, Stuart Reid**, and
NOSIBLE Inc.

## For `cybernaut_mini`

Direct mapping onto the repo:

| Pipeline step | Module |
|---|---|
| 8-LLM committee | `src/cybernaut_mini/providers/judge.py` |
| Linear probe over frozen embeddings | `src/cybernaut_mini/providers/embeddings.py` |
| Disagreement detection + oracle loop | `src/cybernaut_mini/evals.py` |
| Corpus loading / splits | `src/cybernaut_mini/datasets.py` |
| Distil target | `src/cybernaut_mini/models.py` |

Two recommendations. **Split by `netloc`, not by row** — the column exists for a reason.
And **hold out an external benchmark**: with no published eval split and model-derived
labels, in-distribution accuracy mostly measures agreement with the committee, not
correctness.
