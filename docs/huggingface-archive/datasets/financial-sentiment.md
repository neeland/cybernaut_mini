---
title: "NOSIBLE/financial-sentiment dataset"
source: https://huggingface.co/datasets/NOSIBLE/financial-sentiment
licence: ODC-By v1.0 (attribution required)
version: v1.0.0
fetched: 2026-07-28
archived-by: Hugging Face Hub API (educational archive)
---

# `NOSIBLE/financial-sentiment`

> Derived notes, not a verbatim copy. Data © Nosible Inc. under **ODC-By v1.0**.
> Shared schema, provenance, and annotation pipeline: [`README.md`](README.md).

100,000 curated news samples labelled for **financial sentiment** — the training corpus
behind [`financial-sentiment-v1.1-base`](../models/financial-sentiment.md).

| | |
|---|---|
| Rows | 100,000 (single `train` split) |
| Labels | `positive` / `neutral` / `negative` |
| Schema | `text`, `label`, `netloc`, `url` |
| Language | `en` |
| Licence | ODC-By v1.0 |
| Released | 2025-11-26 (modified 2025-12-01) |
| Downloads / likes | 73 / 2 |
| Tags | `financial-sentiment`, `finance`, `sentiment-analysis`, `nlp` |

The most-liked and most-downloaded of the three, and the only 3-class one.

## The label definition is the interesting part

Entries are categorised *"based on potential financial impact"* — **not** authorial tone,
and not the sentiment of the writer. This is the definition that separates a usable
financial classifier from a repurposed product-review model:

- *"The company announced sweeping layoffs"* — grim in tone, frequently **positive** by
  cost impact.
- *"Regulators approved the merger unanimously"* — neutral prose, **positive** impact.
- *"Record revenue, but guidance was cut"* — the impact label follows the forward-looking
  clause, not the headline.

Because that judgment is not lexical, it is exactly the kind of task where an 8-model LLM
committee beats keyword or lexicon methods, and where FinBERT — trained on a narrower
notion of sentiment — is beatable. See
[`../models/financial-sentiment.md`](../models/financial-sentiment.md) for the benchmark
numbers.

**Not published:** class balance. With three classes over financial news, `neutral` almost
certainly dominates; check the distribution before choosing a loss or reporting accuracy.

## Annotation

Per the card: a multi-stage LLM pipeline using **eight** language models (Grok, Gemini,
GPT variants, Llama, Qwen), then iterative refinement with **ensemble linear models** over
text embeddings — the card specifies **six embedding models** — and oracle validation. Full
method and its implications in [`README.md`](README.md).

## Loading

```python
from datasets import load_dataset

ds = load_dataset("NOSIBLE/financial-sentiment", split="train")
# features: text (string), label (string), netloc (string), url (string)
```

Split by `netloc` rather than randomly — syndicated wire copy appears across domains and
random splits leak it.

## Contributors

Matthew Dicks, Simon van Dyk, Stuart Reid; NOSIBLE Inc.

## Related

- [`../models/financial-sentiment.md`](../models/financial-sentiment.md) — the distilled model
- [`../../blog-archive/ensemble-and-distil.md`](../../blog-archive/ensemble-and-distil.md)
- [`../../blog-archive/news-sentiment-showdown-who-checks-vibes-best.md`](../../blog-archive/news-sentiment-showdown-who-checks-vibes-best.md)
- [`../../REVERSE_ENGINEERING_GUIDE.md`](../../REVERSE_ENGINEERING_GUIDE.md) §3
