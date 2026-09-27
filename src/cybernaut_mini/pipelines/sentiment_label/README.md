# `sentiment_label` — label a story set with the whole pool, then report

Blog ref: [news-sentiment-showdown-who-checks-vibes-best](https://nosible.com/blog/news-sentiment-showdown-who-checks-vibes-best) —
the post's method as a static DAG: curate a representative dataset, label it
with every model under one shared few-shot prompt, publish the agreement
matrices, the timing table, and the 10M-story cost extrapolation. Local copy:
[`docs/blog-archive/news-sentiment-showdown-who-checks-vibes-best.md`](../../../../docs/blog-archive/news-sentiment-showdown-who-checks-vibes-best.md).

## Flow

```mermaid
flowchart TB
    R[("sentiment_rows<br/>cached NOSIBLE/financial-sentiment JSONL<br/>(opt-in download, data/01_raw)")]
    R --> A["prepare_sentiment_stories"]
    R --> G["extract_sentiment_gold<br/>positive/neutral/negative → +1/0/−1"]
    A --> S[("sentiment_stories")]
    S --> L["label_sentiment_stories<br/>labelers.build_pool → one column per labeler"]
    L --> M[("sentiment_label_matrix")]
    M --> B["benchmark_sentiment_labels<br/>agreement sorted by reference · kappa · Random row"]
    G --> B
    S --> T["time_sentiment_pool"]
    B --> O1[("sentiment_agreement_report")]
    T --> O2[("sentiment_timing_report")]
    C["estimate_sentiment_costs"] --> O3[("sentiment_cost_report")]
```

## Not registered yet

The integration pass registers this pipeline and adds the catalog entries and
parameter blocks listed in `pipeline.py`'s docstring; `configs/sentiment/labelers.yaml`
is the reference parameter block. Until then it runs through its node functions
directly (that is what the offline tests do).

## Offline behaviour

With the default parameters the pool is TextBlob + VADER + Random — zero
downloads, zero network. Flair / FinBERT / LLM columns are opt-in via the
parameter block and require their respective installs, downloads or API keys.
