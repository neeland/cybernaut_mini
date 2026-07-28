---
title: "NOSIBLE/prediction dataset"
source: https://huggingface.co/datasets/NOSIBLE/prediction
licence: ODC-By v1.0 (attribution required)
version: v1.0.0
fetched: 2026-07-28
archived-by: Hugging Face Hub API (educational archive)
---

# `NOSIBLE/prediction`

> Derived notes, not a verbatim copy. Data © Nosible Inc. under **ODC-By v1.0**.
> Shared schema, provenance, and annotation pipeline: [`README.md`](README.md).

100,000 rows labelled for **predictive modality** — whether a passage contains a forecast.
Corpus behind [`prediction-v1.1-base`](../models/prediction.md).

| | |
|---|---|
| Rows | 100,000 (single `train` split) |
| Labels | `prediction` / `not-prediction` |
| Schema | `text`, `label`, `netloc`, `url` |
| Language | `en` |
| Licence | ODC-By v1.0 |
| Released | 2025-11-26 (modified 2025-12-01) |
| Downloads / likes | 23 / 2 |
| Tags | `tense-prediction`, `finance`, `nlp` |

Label definitions: **`prediction`** — the text includes a prediction, estimate, or
forecast; **`not-prediction`** — it does not. The hyphen matters:
[`forward-looking`](forward-looking.md) uses a space (`not forward`). Build decoding
grammars per checkpoint, not from memory.

> **Name mismatch.** [`prediction-v1.1-base`](../models/prediction.md) declares its
> training data as `NOSIBLE/predictive`, which is not a public repo. This dataset is
> presumably the same corpus under its final name.

Least downloaded of the three, and the only one whose model never received a v1.2.

## Annotation

The most specific model list of the three cards:

1. Hand-label ~200 samples for prompt tuning
2. Label 100k rows with **eight LLMs** — **Grok 4, Gemini 2.5, GPT variants, Llama 4,
   Qwen3**
3. Train **linear ensemble models** over text embeddings from **six embedding models**
4. Iteratively relabel via disagreement detection with **GPT-5.1 as oracle**
5. **Drop underperforming models until convergence**

Step 5 is stated most plainly here: the committee is pruned, not merely re-weighted. Full
discussion in [`README.md`](README.md).

## Modality vs. tense

This dataset only earns its existence where it *disagrees* with
[`forward-looking`](forward-looking.md):

| Statement | forward | prediction |
|---|---|---|
| "We will open 40 stores next year." | yes | **no** — a commitment |
| "In hindsight, we had forecast a downturn." | no | **yes** — reported forecast |

A joint model would collapse those two cells. Keeping them separate is what lets a
downstream signal treat issuer guidance (an intention the company controls) differently
from an analyst estimate (one they do not). See
[`../models/prediction.md`](../models/prediction.md).

Hardest cases: hedged guidance (*"margins should improve"*) sitting between commitment and
forecast, and implicit forecasts carried by valuation or target-price language. Neither is
documented.

**Not published:** class balance, agreement statistics, per-annotator accuracy, date range.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("NOSIBLE/prediction", split="train")
# features: text (string), label (string), netloc (string), url (string)
```

Split by `netloc`, not randomly.

## Contributors

Matthew Dicks, Simon van Dyk, Stuart Reid; NOSIBLE Inc.

## Related

- [`../models/prediction.md`](../models/prediction.md) — the distilled model
- [`forward-looking.md`](forward-looking.md) — the sibling tense dataset
- [`../../REVERSE_ENGINEERING_GUIDE.md`](../../REVERSE_ENGINEERING_GUIDE.md) §3
