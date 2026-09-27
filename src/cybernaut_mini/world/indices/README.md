# `world.indices` — the GPR and TPU/EPU index families

Blog refs: [rebuilding-the-geopolitical-risk-index-from-nosible-world](https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world),
[an-embedding-based-approach-to-trade-and-economic-policy-uncertainty](https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty).
Local copies under [`docs/blog-archive/`](../../../../docs/blog-archive/).

Every index is a group-by over the event table: a numerator of breadth
(`total_netlocs`, optionally tilted by `w_unc` or oil relevance), a
corpus-growth-stripping trailing denominator, and a mask built from the topic
filter and/or the anchor engine.

```mermaid
flowchart LR
    EV[("event table<br/>date, breadth, embedding,<br/>iptc_level_1..3, attribution")]
    subgraph gpr ["gpr.py"]
        GEO["geopolitical(e)<br/>exact code filter"] --> GD["gpr_daily<br/>share of daily attention<br/>raw or /trailing-365d"]
        GEO --> ORP["trade_patched_mask<br/>geopolitical(e) OR trade(e) ≥ 0.40"]
        ORP --> GC["gpr_country_monthly / gpr_pairs_monthly<br/>explode attribution → Σ breadth / B(m)"]
        GEO --> OIL["oil_gpr<br/>AND-gate: geopolitical ∧ oil-relevance ≥ 0.30<br/>Σ relevance·breadth / B(m)"]
        B["monthly_breadth_denominator<br/>B(m): trailing 12-month GLOBAL mean"] --> GC
        B --> OIL
    end
    subgraph policy ["policy.py"]
        SC["AnchorScores<br/>relevance, polarity, w_unc"] --> TPU["tpu_daily<br/>Σ breadth·w_unc / trailing daily breadth"]
        SC --> EPU["epu_monthly<br/>US numerator / trailing US denominator<br/>+ category AND-gate at 0.25"]
        SC --> NP["net_polarity<br/>Σ breadth·polarity / Σ breadth"]
        SC --> EX["explain_month<br/>top-20 contributions +<br/>top-20 just below the floor"]
    end
    EV --> gpr
    EV --> policy
```

Design invariants the tests pin:

- `B(m)` is **global**: `per_country_denominator` exists only to demonstrate the
  self-cancellation the post warns about.
- The trade OR-patch leaves conflict-country series **bit-identical** — only
  events that were previously excluded can enter.
- Oil is an **AND**-gate (geopolitical ∧ relevance), weighted by
  `relevance × breadth`, never plain breadth.
- EPU is US-over-US on **both** lines, "the way the published index scales US
  articles by US articles".
- Lever ablations (the 9-vs-10-lever COVID gap) are a parameter upstream
  (`EmbeddedAnchorSet.embed(..., levers=...)`), not logic here.
