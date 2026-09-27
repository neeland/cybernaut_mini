# `sentiment_distil` — ensemble the labelers, distil into OLS students

Blog ref: [ensemble-and-distil](https://nosible.com/blog/ensemble-and-distil) —
steps three and four of "Curate → Label → Ensemble → Distil → Scale": greedy
iterative-addition ensemble selection with a per-step trace, 1,000-run
bootstrap stability, a teacher that is the unweighted row-sum of the winning
members (±1 band), and `LinearRegression` students on frozen sentence
embeddings graded against the teacher. Local copy:
[`docs/blog-archive/ensemble-and-distil.md`](../../../../docs/blog-archive/ensemble-and-distil.md).

## Flow

```mermaid
flowchart TB
    M[("sentiment_label_matrix<br/>from sentiment_label")]
    G[("sentiment_gold<br/>(or proxy_column at scale)")]
    S[("sentiment_stories")]
    M --> E["select_sentiment_ensemble<br/>greedy forward selection, per-step trace"]
    G --> E
    M --> B["bootstrap_sentiment_ensemble<br/>n runs on 75% subsamples, winner tally"]
    G --> B
    E --> T[("sentiment_ensemble_trace")]
    B --> O1[("sentiment_bootstrap_report")]
    S --> D["distil_sentiment_students<br/>teacher = row-sum of members<br/>OLS on frozen embeddings<br/>test_size=0.25, random_state=42"]
    M --> D
    T --> D
    D --> O2[("sentiment_distil_results<br/>Parameters / Runtime / Dimensions / Accuracy<br/>NaN-injected labeler baselines")]
```

## Not registered yet

The integration pass registers this pipeline (after `sentiment_label`) and adds
the three reporting catalog entries named in `pipeline.py`'s docstring.

## Offline behaviour

The default distil path embeds with the repo's deterministic hash provider —
the whole DAG runs with zero network. Setting `survey_checkpoints` to the
CPU-friendly list in `sentiment.distil.DEFAULT_SURVEY_CHECKPOINTS` reproduces
the post's sentence-transformer survey (each checkpoint is a download).
