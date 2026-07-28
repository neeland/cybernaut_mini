---
title: "NOSIBLE financial-sentiment — v1.1 & v1.2 base"
source:
  - https://huggingface.co/NOSIBLE/financial-sentiment-v1.1-base
  - https://huggingface.co/NOSIBLE/financial-sentiment-v1.2-base
base-model: Qwen/Qwen3-0.6B-Base
licence: apache-2.0
fetched: 2026-07-28
archived-by: Hugging Face Hub API (educational archive)
---

# `financial-sentiment-v1.x-base`

> Derived notes, not a verbatim copy. Weights © NOSIBLE, Apache-2.0.
> Read alongside [`../../REVERSE_ENGINEERING_GUIDE.md`](../../REVERSE_ENGINEERING_GUIDE.md) §3.

The flagship of the distil family and the most-downloaded NOSIBLE artifact. Classifies a
text snippet as **positive / neutral / negative** by *likely financial impact* — not by
authorial tone.

| | v1.1 | v1.2 |
|---|---|---|
| Released | 2025-11-25 | 2026-04-28 |
| Downloads | 4.5K | 2.9K |
| Params | 596M | 596M |
| Base | `Qwen/Qwen3-0.6B-Base` | `Qwen/Qwen3-0.6B-Base` |
| Languages | English | **94** (English + 93) |
| Training data | [`NOSIBLE/financial-sentiment`](../datasets/financial-sentiment.md) | not stated |
| Live inference | `featherless-ai` | weights only |

## Usage requirements — all three are mandatory

The cards are unusually emphatic. Getting any of these wrong degrades output badly:

1. **Disable thinking.** `enable_thinking=False`. It is a `qwen3` model with reasoning
   capability that was fine-tuned to emit exactly one label token; leaving reasoning on
   puts it off-distribution.
2. **Use the exact system prompt**, verbatim:
   > `Classify the financial sentiment as positive, neutral, or negative.`

   This string was present on *every* training example, so it functions as a task key, not
   a description. Paraphrasing it is out-of-distribution.
3. **Constrain output** to the label set, via grammar or regex
   `(positive|neutral|negative)`, or guided decoding.

Recommended deployment is SGLang with an OpenAI-compatible API:

```bash
python3 -m sglang.launch_server --model-path NOSIBLE/financial-sentiment-v1.2-base
```

SGLang is the recommendation *because* it supports guided decoding — requirement 3 is a
deployment concern, not a client one.

## v1.2 — multilingual is the whole story

| Metric | v1.1 | v1.2 | Δ |
|---|---|---|---|
| English accuracy | 87.70% | 87.97% | **+0.27pp** |
| Multilingual accuracy | 76.69% | 83.16% | **+6.47pp** |
| Currency / geo feeds | 67.30% | 76.17% | **+8.87pp** |

English is flat; multilingual and the newly added currency/G10-geography feeds carry the
release. The English number was already near the ceiling of what a 0.6B model extracts from
this label set, so the work went into coverage instead. The currency/geo jump also shows
those feeds were the weakest slice at v1.1 (67.3%) — v1.2 added them to training.

This mirrors the multilingual-first design of the retrieval stack (guide §1 stage 1:
fasttext language detection + 140-language translation). A monolingual classifier would be
the bottleneck in a 94-language pipeline.

## v1.1 benchmarks

- **NOSIBLE validation set** (1,000 held-out samples): beats FinBERT and leading
  general-purpose LLMs on accuracy, and is the cheapest per token — the card's cost
  analysis assumes a conservative **100:1 input:output token ratio**, fair given
  single-token outputs.
- **Financial PhraseBank**: beats FinBERT's reported **86%**. PhraseBank was **excluded
  from training and reserved entirely for evaluation** — a genuinely clean external
  benchmark, and the reason to trust the headline number.

Cited prior work: `arxiv:1908.10063` (FinBERT), `arxiv:2505.09388`, `arxiv:2510.14276`.

## Training configuration [disclosed]

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Scheduler | cosine, 0.03 warmup ratio |
| Batch size | 64 |
| Epochs | **2** |
| Optimizer | AdamW (Torch fused) |
| Precision | bfloat16 |
| NEFTune noise alpha | **5** |
| Weight decay | 0.1 |
| Max sequence length | 2048 |

**Preprocessing:** the system prompt was included on every example; user input and system
instruction were masked with `labels=-100` so **loss is computed only on the assistant
response** — i.e. only on the single label token.

Two choices carry most of the weight. **Two epochs** on 100k LLM-labelled rows is
deliberate under-training: the labels are committee-derived and noisy, so more epochs would
fit the annotators' errors. **NEFTune alpha=5** injects embedding noise for regularisation,
which matters precisely because the label set is tiny and the model is small enough to
memorise. Together they say: treat distillation labels as a soft signal.

## Limitations [disclosed]

- **Financial contexts only** — not a general sentiment model.
- **0.6B params** limits nuanced reasoning.
- **No aspect-based sentiment** — will not tell you *which* entity the sentiment attaches
  to. That binding happens in the event layer (guide §8).
- **2048-token context** — longer documents must be chunked.
- Per-language quality varies; lower-resource languages show larger gaps (v1.2).
- v1.1 is English-primary; non-English performance unverified.

> **Disclaimer (from the card):** outputs are not financial advice or investment
> recommendations, and financial decisions should never rely solely on model outputs
> without professional consultation.

## Citation

```bibtex
@misc{nosible2025financialsentiment,
  author = {NOSIBLE},
  title = {Financial Sentiment v1.1 Base},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {https://huggingface.co/NOSIBLE/financial-sentiment-v1.1-base}
}
```

## For `cybernaut_mini`

The repo's judge/eval path (`providers/judge.py`, `evals.py`, `models.py`) implements the
committee side of guide §3. This model is the **reference distillation target**: same
task, same label set, published hyperparameters, and an external benchmark
(PhraseBank) to check against. Chunk to 2048 tokens, pin the system prompt exactly, and
constrain decoding — then the only variable left is your label quality.
