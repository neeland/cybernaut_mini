---
source: https://huggingface.co/NOSIBLE
fetched: 2026-07-28
archived-by: Hugging Face Hub API (educational archive)
---

# NOSIBLE Hugging Face Archive

Notes on every public repository under the [`NOSIBLE`](https://huggingface.co/NOSIBLE)
Hugging Face organisation, captured 2026-07-28 for educational study.

These are *derived notes*, not verbatim copies. Models are Apache-2.0; datasets are
**ODC-By v1.0** (attribution required). `cybernaut_mini` is an unaffiliated clean-room
educational replica.

## Why this matters

Unlike the [blog archive](../blog-archive/) (prose) and the
[GitHub archive](../github-archive/) (API contract), this is the one place where NOSIBLE
released **the artifacts themselves** — weights you can run and 300,000 labelled rows you
can train on. Section §3 of
[`../REVERSE_ENGINEERING_GUIDE.md`](../REVERSE_ENGINEERING_GUIDE.md) ("Ensemble & distil →
tiny fine-tuned classifiers") is the one part of the stack that does not need
reverse-engineering at all: **the output is downloadable.**

## Contents

### Models — 5 repos, all Apache-2.0

All are 596M-parameter `qwen3` causal LMs fine-tuned from `Qwen/Qwen3-0.6B-Base`, run as
constrained single-token classifiers.

| Model | Task | Downloads | Card |
|---|---|---|---|
| `financial-sentiment-v1.2-base` | positive / neutral / negative | 2.9K | [notes](models/financial-sentiment.md) |
| `financial-sentiment-v1.1-base` | positive / neutral / negative | 4.5K | [notes](models/financial-sentiment.md) |
| `forward-looking-v1.2-base` | forward / not forward | 2.8K | [notes](models/forward-looking.md) |
| `forward-looking-v1.1-base` | forward / not forward | 724 | [notes](models/forward-looking.md) |
| `prediction-v1.1-base` | prediction / not-prediction | 365 | [notes](models/prediction.md) |

v1.1 released 2025-11-25, v1.2 on 2026-04-28. There is **no `prediction-v1.2`**.
v1.1 models are served live by `featherless-ai`; v1.2 are weights-only.

### Datasets — 3 repos, all ODC-By v1.0, 100,000 rows each

| Dataset | Labels | Downloads | Card |
|---|---|---|---|
| `financial-sentiment` | positive / neutral / negative | 73 | [notes](datasets/financial-sentiment.md) |
| `forward-looking` | forward / not forward | 26 | [notes](datasets/forward-looking.md) |
| `prediction` | prediction / not-prediction | 23 | [notes](datasets/prediction.md) |

All share one schema — `text`, `label`, `netloc`, `url` — one `train` split, and one
annotation pipeline. See [`datasets/README.md`](datasets/README.md) for the shared method.

Contributors named across all three: **Matthew Dicks, Simon van Dyk, Stuart Reid**.

## The three-classifier set is not arbitrary

Read together, the tasks decompose a financial statement along orthogonal axes:

| Axis | Question | Model |
|---|---|---|
| **Valence** | Is the financial impact good or bad? | `financial-sentiment` |
| **Tense** | Is this about the future? | `forward-looking` |
| **Modality** | Is this a forecast, or a report? | `prediction` |

"Revenue fell 12%" is negative, backward-looking, not a prediction. "We expect margins to
recover" is positive, forward-looking, *and* a prediction. Crossing the three gives you
guidance-vs-results, forecast-vs-fact, and hope-vs-history — the distinctions the
quant-signal posts need but never name. Both `forward-looking` and `prediction` carry the
`tense-prediction` tag, confirming they are treated as one family.

**Deliberate gap:** the entity/aspect axis is missing. The v1.2 card states the model
"does not perform aspect-based sentiment analysis." *Which company* a sentiment attaches to
is resolved elsewhere — via Google KG IDs at index time, per
[`../github-archive/nosible-py.md`](../github-archive/nosible-py.md). The classifiers score
text; the event layer binds it to entities. Confirms guide §8.

## The pattern to copy

Every model here is the terminal step of the same loop, and it is the cheapest
high-leverage technique NOSIBLE has published:

```
hand-label ~200  →  8-LLM committee labels 100k  →  linear ensemble over
embeddings finds disagreements  →  GPT-5.1 oracle adjudicates  →  drop weak
annotators, repeat until convergence  →  fine-tune a 0.6B model on the result
```

The result matches or beats frontier models on the narrow task at a fraction of the
cost — the blog's "fast enough to matter" claim, and guide §3's core recipe.

→ `cybernaut_mini` mirrors: `src/cybernaut_mini/providers/judge.py` (the committee),
`src/cybernaut_mini/evals.py`, `src/cybernaut_mini/datasets.py`,
`src/cybernaut_mini/models.py`

## Related

- [`../github-archive/`](../github-archive/) — the client, MCP server, and HTML cleaner
- [`../blog-archive/ensemble-and-distil.md`](../blog-archive/ensemble-and-distil.md) — the method post
- [`../blog-archive/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction.md`](../blog-archive/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction.md)
- [`../REVERSE_ENGINEERING_GUIDE.md`](../REVERSE_ENGINEERING_GUIDE.md) §3, §5
