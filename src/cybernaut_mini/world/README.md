# `world` — the WORLD pillar: event store, tag layers, anchor engine, indices

Blog refs: [rebuilding-the-geopolitical-risk-index-from-nosible-world](https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world),
[an-embedding-based-approach-to-trade-and-economic-policy-uncertainty](https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty),
[turning-news-into-a-risk-on-risk-off-equity-signal](https://nosible.com/blog/turning-news-into-a-risk-on-risk-off-equity-signal),
[point-in-time-knowledge-graphs-over-named-entities](https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities).
Local copies under [`docs/blog-archive/`](../../../docs/blog-archive/).

Where the SEARCH pillar answers questions over documents, the WORLD pillar turns
de-duplicated news **events** into point-in-time risk and uncertainty series. One
real event is one record, no matter how many outlets repeat it; every field is
point-in-time safe (nothing encodes future information — the materiality
normalizer uses an *expanding* max, the denominators are *trailing* means, the
smoother's trigger statistics are strictly-before).

## Dataflow

```mermaid
flowchart TB
    D[("dated corpus<br/>(Document rows + frozen embeddings)")]
    D --> DD["dedup.cluster_documents<br/>(WS2: EventCluster — apex, coverage-peak date, total_netlocs)"]
    DD --> EV["events.build_events<br/>one WorldEvent per cluster<br/>centroid embedding, expanding-max materiality"]
    EV --> NER["ner.tag_events<br/>entities{TYPE:{surface:count}}<br/>suffix + surname folding"]
    NER --> TK["tickers.tag_events<br/>exact alias match → tickers[{name,ticker_eodhd}]"]
    NER --> CO["countries.tag_events<br/>strict whole-string resolver → event.country, ent_gpe"]
    EV --> TP["topics.tag_events<br/>embedding-nearest IPTC row → iptc_level_1..3"]
    subgraph engine ["anchor engine (shared by every index)"]
        Y[("configs/anchors/*.yaml<br/>verbatim sentences: 17 stress, trade 3+2,<br/>EPU 60-sentence appendix, 3 oil, 1 trade-coercion")]
        Y --> EMB["vectors.FrozenEmbedder<br/>Matryoshka truncate + L2 renorm"]
        EMB --> SC["anchors.score_events<br/>max-cos relevance + floor,<br/>best-pair tanh polarity, w_unc"]
    end
    EV --> SC
    TP --> G["topics.geopolitical(e)<br/>exact ontology-code filter"]
    G --> GPR["indices/gpr.py<br/>daily share, country & pair / B(m),<br/>trade OR-patch, Oil-GPR AND-gate"]
    SC --> GPR
    CO --> GPR
    SC --> POL["indices/policy.py<br/>TPU daily, EPU US-over-US monthly,<br/>categories, net polarity, explain-month"]
    POL --> SM["smoother.asymmetric_ewma<br/>3d fast on >1.5σ upticks, 30d slow"]
    GPR --> V["validate.py<br/>rebase, Pearson levels+changes,<br/>episode ranks ≥ 60th pct, spike checklist"]
    POL --> V
    R[("data/01_raw/reference/*.csv<br/>opt-in published benchmarks")] --> V
```

## Point-in-time rules

| Field / series | Rule that keeps it causal |
| --- | --- |
| `materiality_score` | expanding max in date order, never corpus-wide |
| `B(m)` denominator | trailing 12-month mean, global, `min_periods=6` |
| daily detrend | trailing-365d mean of zero-filled daily breadth |
| smoother trigger | mean/std of changes strictly *before* the day (no look-ahead) |
| embedder | frozen, revision-pinned, never fine-tuned on the corpus |

## Thresholds are data

Published floors (0.35 trade/EPU, 0.40 trade-coercion, 0.30 stress/oil, 0.25
category, temperature 0.1) were tuned on `text-embedding-3-large` cosines and live
in the YAML configs. Open-model users sweep them (`anchors.sweep_relevance_floor`)
against the four polarity-separation stats (`anchors.polarity_separation`): the
posts' targets are off-topic mean w_unc ≈ 0.44, on-topic ≈ 0.63, 60% > 0.6,
21% < 0.4.
