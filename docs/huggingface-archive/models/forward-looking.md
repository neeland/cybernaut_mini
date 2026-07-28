---
title: "NOSIBLE forward-looking — v1.1 & v1.2 base"
source:
  - https://huggingface.co/NOSIBLE/forward-looking-v1.1-base
  - https://huggingface.co/NOSIBLE/forward-looking-v1.2-base
base-model: Qwen/Qwen3-0.6B-Base
licence: apache-2.0
fetched: 2026-07-28
archived-by: Hugging Face Hub API (educational archive)
---

# `forward-looking-v1.x-base`

> Derived notes, not a verbatim copy. Weights © NOSIBLE, Apache-2.0.
> Read alongside [`financial-sentiment.md`](financial-sentiment.md) — same recipe, same
> hyperparameters, different label.

Binary tense classifier: does a statement contain **forward-looking** content?

| | v1.1 | v1.2 |
|---|---|---|
| Released | 2025-11-25 | 2026-04-28 |
| Downloads | 724 | 2.8K |
| Params | 596M | 596M |
| Base | `Qwen/Qwen3-0.6B-Base` | `Qwen/Qwen3-0.6B-Base` |
| Language | English | not stated |
| Training data | [`NOSIBLE/forward-looking`](../datasets/forward-looking.md) | not stated |
| Live inference | `featherless-ai` | weights only |

Labels: **`forward`** / **`not forward`**. Tagged `finance`, `nlp`; the dataset carries
`tense-prediction`, pairing it with [`prediction`](prediction.md).

Cited prior work: `arxiv:2505.09388`, `arxiv:2510.14276` — the same two as the
sentiment model, minus the FinBERT reference (no binary-tense baseline to beat).

## Same recipe, different head

The v1.2 card is thin: no benchmark table, no separate training section. That is
informative in itself — this is the *same pipeline* as
[`financial-sentiment`](financial-sentiment.md), re-run against a different label column
of an identically-shaped 100k dataset. Treat the sentiment card's configuration as the
shared spec:

- LR 2e-5, cosine schedule with 0.03 warmup, batch 64, **2 epochs**
- AdamW fused, bfloat16, NEFTune alpha 5, weight decay 0.1, max seq len 2048
- System prompt on every example; user + system masked with `labels=-100`, loss on the
  assistant token only

The three usage requirements carry over and are not optional:

1. `enable_thinking=False`
2. Use the model's exact system prompt verbatim — it is a task key, not a description
3. Constrain generation to the label set via grammar/regex or guided decoding

Deployment likewise: SGLang with an OpenAI-compatible API, chosen for guided decoding.

```bash
python3 -m sglang.launch_server --model-path NOSIBLE/forward-looking-v1.2-base
```

> **Verify the label strings before use.** The dataset card gives `forward` /
> `not forward` (space, not hyphen), while [`prediction`](prediction.md) uses
> `not-prediction` (hyphen). The two families are inconsistent, and a regex built on the
> wrong form silently fails. Read the tokenizer config / card of the exact checkpoint you
> load.

## Why a tense classifier earns its own model

Tense is the cheapest available proxy for **information value** in financial text.
"Revenue fell 12%" is already priced; "we expect margins to recover" is not. A retrieval
system that cannot separate the two returns yesterday's news for a question about
tomorrow.

Combined with the other two heads it forms the decomposition described in
[`../README.md`](../README.md):

| | backward-looking | forward-looking |
|---|---|---|
| **not a prediction** | reported results | stated plans, guidance |
| **is a prediction** | *(rare — retrospective forecasts)* | analyst forecasts, outlooks |

The empty-ish cell is the useful part: `forward-looking` and `prediction` are correlated
but **not redundant**. "We will open 40 stores next year" is forward-looking but is a
commitment, not a forecast. Separating intent from estimate is what lets a downstream
signal weight company guidance differently from third-party predictions.

Note the download inversion: v1.2 (2.8K) far exceeds v1.1 (724), the reverse of the
sentiment pair. This head only became widely used after the v1.2 release.

## Limitations

Not enumerated on the v1.2 card; the family-level constraints from
[`financial-sentiment.md`](financial-sentiment.md) apply — 0.6B parameters, 2048-token
context requiring chunking, financial domain only, no aspect/entity binding, and unverified
per-language quality. Binary-tense labels are also inherently softer than sentiment at
sentence boundaries: a chunk mixing a results recap with an outlook has no clean answer,
and chunking strategy will move the numbers.

## For `cybernaut_mini`

A useful **second target** for the distil loop in `providers/judge.py` / `models.py`. Two
practical advantages over the sentiment head: binary labels make agreement statistics much
easier to read during committee refinement, and tense is far less culturally loaded, so
LLM annotators agree more often. Start the ensemble-and-distil implementation here, then
port the converged loop to the three-class problem.
