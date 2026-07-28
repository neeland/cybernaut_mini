---
title: "NOSIBLE/forward-looking dataset"
source: https://huggingface.co/datasets/NOSIBLE/forward-looking
licence: ODC-By v1.0 (attribution required)
version: v1.0.0
fetched: 2026-07-28
archived-by: Hugging Face Hub API (educational archive)
---

# `NOSIBLE/forward-looking`

> Derived notes, not a verbatim copy. Data © Nosible Inc. under **ODC-By v1.0**.
> Shared schema, provenance, and annotation pipeline: [`README.md`](README.md).

100,000 *"cleaned, deduplicated, and forward-looking labeled news"* — the training corpus
behind [`forward-looking-v1.1-base`](../models/forward-looking.md).

| | |
|---|---|
| Rows | 100,000 (single `train` split) |
| Labels | `forward` / `not forward` |
| Schema | `text`, `label`, `netloc`, `url` |
| Language | `en` |
| Licence | ODC-By v1.0 |
| Released | 2025-11-26 (modified 2025-12-01) |
| Downloads / likes | 26 / 3 |
| Tags | `tense-prediction`, `finance`, `nlp` |

Label definitions, verbatim in substance: **`forward`** — the statement contains
forward-looking content; **`not forward`** — it does not. Note the label string uses a
**space**, where [`prediction`](prediction.md) uses a hyphen (`not-prediction`).

The card is the only one of the three to state explicitly that rows are *cleaned and
deduplicated* — corroborating [`smol-html`](../../github-archive/smol-html.md) on the
ingest side and the event layer's one-record-per-event dedup (guide §8).

## Annotation

The most explicitly enumerated of the three cards:

1. Hand-label ~200 samples for prompt tuning
2. Apply **8 LLM annotators** — Grok, Gemini, **GPT-5**, Llama, Qwen
3. Train **linear ensemble models over text embeddings**
4. Iteratively relabel disagreements with an **oracle LLM (GPT-5.1)**
5. Ship the converged set as training data for `forward-looking-v1.1-base`

Full method and its consequences in [`README.md`](README.md).

## Where the labels get hard

Tense is cleaner to annotate than sentiment, which is part of why this is a good first
target for a replica distil loop. But the residual ambiguity is structural, not lexical:

- **Mixed chunks.** A passage that recaps results *and* gives an outlook has no single
  correct label. With a 128–1024-token retrieval window
  ([`nosible-py`](../../github-archive/nosible-py.md)), mixed chunks are common, so
  chunking strategy moves the numbers as much as the model does.
- **Reported forecasts.** *"Last year we predicted a downturn"* is grammatically past but
  about a forecast — `not forward` here, `prediction` in the sibling dataset. This is the
  exact cell where the two labels dissociate.
- **Conditionals and hedges.** *"Should conditions improve…"* — where the committee landed
  is not documented.

**Not published:** class balance, inter-annotator agreement, or per-annotator accuracy.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("NOSIBLE/forward-looking", split="train")
# features: text (string), label (string), netloc (string), url (string)
```

Split by `netloc`, not randomly.

## Contributors

Matthew Dicks, Simon van Dyk, Stuart Reid; NOSIBLE Inc.

## Related

- [`../models/forward-looking.md`](../models/forward-looking.md) — the distilled model
- [`prediction.md`](prediction.md) — the sibling modality dataset
- [`../../REVERSE_ENGINEERING_GUIDE.md`](../../REVERSE_ENGINEERING_GUIDE.md) §3
