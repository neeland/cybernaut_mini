---
title: "NOSIBLE prediction-v1.1-base"
source: https://huggingface.co/NOSIBLE/prediction-v1.1-base
base-model: Qwen/Qwen3-0.6B-Base
licence: apache-2.0
fetched: 2026-07-28
archived-by: Hugging Face Hub API (educational archive)
---

# `prediction-v1.1-base`

> Derived notes, not a verbatim copy. Weights © NOSIBLE, Apache-2.0.
> Read alongside [`financial-sentiment.md`](financial-sentiment.md) — same recipe,
> different label.

Binary modality classifier: does a statement contain **a prediction, estimate, or
forecast**?

| | v1.1 |
|---|---|
| Released | 2025-11-25 |
| Downloads | 365 |
| Params | 596M |
| Base | `Qwen/Qwen3-0.6B-Base` |
| Language | English |
| Training data | [`NOSIBLE/prediction`](../datasets/prediction.md) |
| Live inference | `featherless-ai` |

Labels: **`prediction`** / **`not-prediction`** (hyphenated — unlike
[`forward-looking`](forward-looking.md), which uses a space; verify per checkpoint before
writing a decoding grammar).

Cited prior work: `arxiv:2505.09388`, `arxiv:2510.14276`.

> **Note the model/dataset name mismatch.** The model's card metadata points at
> `dataset:NOSIBLE/predictive`, but the published dataset is
> [`NOSIBLE/prediction`](../datasets/prediction.md). The `predictive` repo is not public.
> Assume the published dataset is the same corpus under its final name.

## No v1.2

The only head in the family without a v1.2 refresh — `financial-sentiment` and
`forward-looking` both shipped one on 2026-04-28. It is also the least downloaded (365 vs.
4.5K and 724 for the other v1.1 models). Whether that reflects lower demand or an
unpublished successor, the public artifact set stops at v1.1 here. If you are building on
this family, this is the head most likely to be stale.

## Recipe

The card publishes no separate training configuration; the family spec from
[`financial-sentiment.md`](financial-sentiment.md) applies — LR 2e-5, cosine with 0.03
warmup, batch 64, 2 epochs, AdamW fused, bfloat16, NEFTune alpha 5, weight decay 0.1,
2048-token max length, loss masked to the assistant label token only.

The three usage requirements are mandatory here as elsewhere:

1. `enable_thinking=False`
2. Use the checkpoint's exact system prompt verbatim
3. Constrain generation to the label set via grammar/regex or guided decoding

```bash
python3 -m sglang.launch_server --model-path NOSIBLE/prediction-v1.1-base
```

## Modality is not tense

The distinction from [`forward-looking`](forward-looking.md) is the point of having both
models, and it is easy to conflate:

| Statement | forward | prediction |
|---|---|---|
| "Revenue fell 12% last quarter." | no | no |
| "We will open 40 stores next year." | **yes** | no — a commitment |
| "Analysts expect margins to recover." | **yes** | **yes** |
| "In hindsight, we had forecast a downturn." | no | **yes** — reported forecast |

`forward-looking` asks *when*; `prediction` asks *with what epistemic status*. The two
off-diagonal cells are the ones a single model would collapse. Company guidance
("we will") is an intention the issuer controls; an analyst forecast ("expect") is an
estimate they do not. Any signal that weights insider guidance differently from
third-party speculation needs both axes.

The `tense-prediction` tag shared with `forward-looking` marks them as one family, but
they are trained as independent binary heads over separate 100k datasets — not as a joint
multi-label model.

## Limitations

Not enumerated on the card. Family-level constraints from
[`financial-sentiment.md`](financial-sentiment.md) apply: 0.6B parameters, 2048-token
context requiring chunking, financial domain only, English-primary with unverified
multilingual behaviour, and **no entity binding** — the model tells you a prediction is
present, never *whose* it is or *about what*. Attribution is the event layer's job
(guide §8).

Hedged language is the obvious hard case: "margins should improve" sits between guidance
and forecast, and where the annotator committee landed on such phrasing is not documented.

## For `cybernaut_mini`

Least useful of the three as a distillation target — the smallest download base, no v1.2,
and a dataset-name inconsistency. Its value is **compositional**: running all three heads
over the same chunk yields a 3-tuple (valence, tense, modality) that is a far richer
retrieval feature than any one label, and it is exactly the kind of cheap deterministic
enrichment the frozen-embedding work in guide §5 is built around. If you implement one
distil loop in `providers/judge.py`, do sentiment; if you implement the *feature* side in
`evals.py`, use all three.
