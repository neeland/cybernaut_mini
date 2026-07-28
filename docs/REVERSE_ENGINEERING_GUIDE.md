# Reverse-Engineering NOSIBLE: A Clean-Room Build Guide

**Purpose.** This is an educational field guide to every piece of technology NOSIBLE
has disclosed on its public blog. For each system it states *what was disclosed*, the
*exact formulas / hyperparameters* where given, an open-tool *build recipe* at
laptop scale, and the *deliberate gaps* you must approximate.

**Ground rules.**

- **Clean-room / educational only.** Everything here is derived from the 12 public
  posts archived in [`blog-archive/`](blog-archive/). No private data, credentials, or
  non-public interfaces were used. `cybernaut_mini` is unaffiliated with NOSIBLE and
  claims no benchmark parity.
- Every recipe step is tagged **[disclosed]** (stated in a post) or **[inferred]**
  (a gap the guide fills, with the assumption named). When a post prints an exact
  formula, prompt, or hyperparameter, it is quoted verbatim so the replica is faithful.
- The one component NOSIBLE never reveals — the upstream **event layer**
  (de-duplication into one-record-per-event, `total_netlocs` breadth, topic/country
  tagging, event dating) — is the moat behind almost every downstream number. Your main
  original engineering is approximating that layer; everything after it is disclosed math.

---

## 0. The NOSIBLE stack at a glance

Three product surfaces sit on one substrate.

```
                          ┌─────────────────────────────────────────┐
   Cybernaut-1 (agentic)  │  LLM-guided MCTS over "Hybrid-3"          │  §2
                          ├─────────────────────────────────────────┤
   Hybrid-3 search        │  8-stage multilingual retrieval pipeline │  §1
                          │  over a federation of 250,000 shards      │
                          ├─────────────────────────────────────────┤
   Enrichment / signals   │  NER + KG (§4), frozen-embedding features │  §5,§6,§7
                          │  (§5), distilled tiny classifiers (§3),   │
                          │  quant indices (§6)                       │
                          ├─────────────────────────────────────────┤
   Event layer (the moat) │  dedup → 1 record/event; breadth; IPTC    │  §8
                          │  topics; NER country; oai_vector; dating  │
                          └─────────────────────────────────────────┘
```

**Naming.** "Shards" (Road-to-Cybernaut post) and "collections" (faceted-search post)
are the same construct: ~250,000 semantically+lexically coherent, near-uniform mini
search engines learned from ~250M embeddings/documents. "Hybrid-3" is the 8-stage
pipeline; "Cybernaut-1" is the RL/MCTS agent on top of it.

**Recurring primitives worth memorizing** — they appear across unrelated posts:

- **RRF (Reciprocal Rank Fusion)** — the universal ensembling glue:
  `score(d) = Σ_i 1/(k + rank_i(d))`, `k=60` conventional (constant not disclosed).
- **Breadth-weighted attention share** — every quant signal is
  `Σ breadth·weight over matching events / (total breadth or its trailing 12-mo mean)`.
- **Frozen-embedding geometry** — cosine + tanh gate + power-mean / signed contrast,
  deliberately deterministic and backtest-safe (no LLM per document).
- **Ensemble → distil** — label with an LLM committee, then distill into a linear head
  or a 0.6B model that runs for ~$0.

---

## 1. Hybrid-3: the eight-stage retrieval pipeline

**Source:** `the-road-to-cybernaut-1.md`. **cybernaut_mini mapping:** this is the part
the repo already replicates — see `routing.py`, `retrieval.py`, `rrf.py`, `sharding.py`,
`indexing.py`, `expansion.py`, `providers/embeddings.py`.

### What was disclosed

A query flows through eight stages; each stage's tech stack is named in the post.

1. **Language detection & translation** — `fasttext-langdetect` (170+ langs);
   `gemma-3n-e4b-it` via OpenRouter for translation (140 langs).
2. **Multilingual tokenization** — a wrapper over ~12 NLP libs (spaCy, NLTK, BlingFire,
   pySBD, pyStemmer, jieba, mecab, wtpsplit, …): sentence-split, normalize, stem,
   stop-word removal. Claim: *well-calibrated lexical beats well-calibrated semantic on
   long high-intent AI queries; hybrid beats both.*
3. **Search-intent prediction** — extract "intents" = proximal token n-grams with a high
   **harmonic mean of per-token TF-IDF**, excluding function-word POS (pronouns,
   determiners, conjunctions, adverbs). Stack: NumPy, spaCy.
4. **Instruction tuning & embedding** — embed query + expansions with
   **`intfloat/multilingual-e5-large-instruct`** using an LLM-evolved instruction
   template. Disclosed template: *"Given a question, please retrieve any relevant
   {Language} Headlines, Leads, Passages, and Source URLs that focus on the same named
   entities as the question, and provide substantive answers to the question."* Claim:
   good instructions are worth **1–5%** precision/recall for free.
5. **Shard selection** — four ranking factors fused by RRF:
   (a) **vanilla dense** cosine(query, shard-summary embedding) over **HNSW**;
   (b) **Bayesian dense** — *proprietary, patent-pending, withheld*;
   (c) **vanilla sparse** TF-IDF keyword similarity;
   (d) **entity sparse** similarity. Stack: NumPy/SciPy/Numba/SimSIMD/USearch.
6. **Shard reranking** — RRF over: **bloom-filter** reranker (downrank shards whose
   phrase bloom filter never saw the query intents), **compression** reranker (the shard
   whose trained **Zstandard** dictionary compresses the query best wins), **neural**
   reranker (**`BAAI/bge-reranker-v2-m3`** over shard titles), **Personalized PageRank**
   over a shard kNN graph. LLM reranking works but "latency too high." Stack: Zstandard, rBloom.
7. **Shard-based query expansion** — each shard's co-occurrence **synonym graph**
   probabilistically expands terms; coherence disambiguates ("gene" only genetic here).
   Cap additions so originals aren't drowned. Stack: NumPy.
8. **Map-reduce retrieval** — broadcast to the top **10–30** shards; each runs
   SQL filter → BM25 lexical → semantic → fuse → full-text (Aho-Corasick) for intents;
   group by doc id; build user-length snippets; return ~100 results. Stack: Polars,
   SimSIMD, Numba, ahocorasick-rs.

### Concrete numbers

250,000 shards; ~1B webpages; last shard-training run = 250M embeddings; example shard
11,343: 134,523 docs / 8.4M words / vocab 64,992 / 2,523 keywords. Final routing = 10–30
shards, ~100 results.

### Build recipe (laptop scale)

1. **[disclosed]** Corpus → passages as JSONL `{id, text, metadata}`. **[inferred]**
   a few thousand–tens of thousands of docs.
2. **[disclosed]** Cluster into K coherent, near-uniform shards. **[inferred]** embed with
   `multilingual-e5-large-instruct` (or `all-MiniLM-L6-v2` for speed) and run
   **balanced/constrained KMeans** (`k-means-constrained`, or `MiniBatchKMeans` +
   post-balancing — which is what `sharding.py` does). *The proprietary segmentation model
   is unavailable; constrained clustering is the stand-in.*
3. **[disclosed]** Per shard build: BM25 (`rank_bm25`), a vector index (FAISS or
   `usearch`/HNSW), vocabulary, top-TF-IDF keywords, spaCy NER entities, a co-occurrence
   (PMI) synonym graph, a phrase **bloom filter** (`rbloom`), and a **Zstandard** dictionary
   (`zstandard.train_dictionary`).
4. **[disclosed]** Generate/collect an LLM summary per shard and embed it as the shard's
   dense descriptor. **[inferred]** fall back to the mean doc embedding if no LLM.
5. **[disclosed]** Stages 1–4 at query time: langdetect → tokenize (spaCy + snowball) →
   intents (harmonic-mean TF-IDF n-grams minus function words) → E5-instruct embed.
   **[inferred]** English-only build may skip translation.
6. **[disclosed]** Stage 5: fuse factors (a),(c),(d) with RRF; **omit (b)** (undisclosed).
   `score(d)=Σ 1/(60+rank_i(d))`. **[inferred]** `k=60` is the standard RRF constant.
7. **[disclosed]** Stage 6: bloom check + Zstd compression-ratio + `bge-reranker-v2-m3`
   cross-encoder over shard titles + optional Personalized PageRank (`networkx`); fuse with RRF.
8. **[disclosed]** Stage 7: expand via shard synonym graphs, capped.
9. **[disclosed]** Stage 8: broadcast to top 10–30 shards; per shard BM25 + vector, RRF,
   Aho-Corasick (`pyahocorasick`) intent match; group by doc id; snippet; return ~100.
10. **[inferred]** Evaluate with gold judgments (nDCG/recall@k) — mirrors per-shard "Evals"
    (the repo's `evals.py` / `data/sample/judgments.jsonl`).

### Gaps

The shard-learning algorithm; the patented Bayesian dense selector; exact RRF weights/`k`;
harmonic-TF-IDF formula and POS list; dynamic 10–30 shard count; the SQL filter schema;
instruction-evolution algorithm; hardware.

---

## 2. Cybernaut-1: LLM-guided MCTS over search

**Source:** `introducing-cybernaut-1-agentic-search-with-mcts.md`. **cybernaut_mini
mapping:** `agent/search.py`, `agent/node.py`, `agent/policy.py`, `agent/actions.py`,
`agent/state.py`.

### What was disclosed

Cybernaut-1 layers **LLM-guided Monte Carlo Tree Search** on top of Hybrid-3 to build a
search *plan* iteratively — moving from **wide-and-shallow** to **narrow-and-deep**,
explicitly balancing **exploration, exploitation, and inference cost**. Unlike black-box
agents that see only top results, it has full access to shards/selectors/rerankers/signals
and tunes them on the fly. Coverage cited: ~20M new webpages/day. Delivered via V2 Search
API and the `nosible` Python package. **No reward function, UCT constant, action space, or
depth is disclosed.**

### Build recipe

1. **[disclosed]** Use the §1 Hybrid-3 pipeline as the *environment* the search acts on.
2. **[inferred]** **State** = current search config (query text, expansion terms, active
   shards, filters, lexical/semantic weight). **Actions** = mutate it (rewrite/expand/narrow
   query, add/drop shards, retune weights, change filters). Assumption from "tune on the fly."
3. **[inferred]** Standard MCTS four phases with **UCT**:
   `argmax_a Q(s,a)/N(s,a) + c·√(ln N(s)/N(s,a))`. Constant `c` and depth undisclosed.
4. **[disclosed→inferred]** **LLM-guided expansion**: prompt an LLM with current results +
   policy to propose N candidate refinements → child nodes.
5. **[inferred]** **Reward** = relevance (LLM-as-judge or nDCG on retrieved quality) minus a
   cost penalty: `reward = relevance − λ·(num_LLM_calls)` (they explicitly balance cost).
6. **[disclosed]** Anneal exploration: broad/shallow early → focused/deep late (decay `c`,
   grow rollout depth as the tree matures).
7. **[inferred]** Terminate on a node/LLM-call budget; return the best leaf's result set.

### Gaps

Reward/value function, UCT variant + constant, rollout policy, tree width/depth, node budget,
the exact tunable action space, and how the LLM guides selection.

---

## 3. Ensemble & distil → tiny fine-tuned classifiers

**Sources:** `news-sentiment-showdown-who-checks-vibes-best.md` (benchmark),
`ensemble-and-distil.md` (pattern), `fast-enough-to-matter-...-tiny-transformers-...md`
(production). This is a three-post arc: *benchmark labelers → distill an ensemble into a
linear head → replace with a fine-tuned 0.6B model refined by active learning.*

### 3a. The benchmark (what to label with)

- Datasets carved from the index by (year × category × industry) top-N-by-coverage, with
  quality filters (verbatim SQL): `source_language_rank>=0.50`, `source_variance_rank>=0.50`,
  `accepted=true`, `apex_story=true`, `source_is_banned=false`, `media_coverage>=3`,
  `title_num_words 5..25`, `description_num_words 10..50`, `ORDER BY media_coverage DESC
  LIMIT 10` per cell. Tiers: small 10,368 / medium 22,422 / large 43,324.
- **250 hand-labeled** stories = ground truth. Input = headline + description.
- **Identical 9-shot prompt** for all LLMs ("You are a sentiment classification AI",
  reply POS/NEU/NEG, *"when unsure you MUST reply NEU"*, 3+3+3 examples). Full text in post.
- Results vs humans: **Text-Unicorn (PaLM-2) 84%**, GPT-4 74%, GPT-4-Turbo ≈66%,
  **FinBERT ≈69%** (beats GPT-3.5), FinBERT-Tone 53% (regression), VADER-0.10 56%.
  Cost: GPT-4 $0.00983/story; GPT-4-Turbo $0.00329 (2.98× cheaper). **VADER 339× faster than
  FinBERT.** 10M stories with GPT-4 ≈ **$92,000**.

### 3b. Ensemble → distil (the pattern)

Five steps: **Curate → Label → Ensemble → Distil → Scale.**

- **Ensemble (teacher)** by greedy *iterative addition*: start best, add the most additive
  model until none helps. Disclosed trace: `Unicorn(83.6%) +GPT-3.5(+2.8) +GPT-4(+2.0)
  +Sigma(+1.6) +Bison(+0.4) → 90.4%`. Teacher score = **raw sum** of member labels ∈
  {−1,0,1}; discretize `≥1→+1, ≤−1→−1, else 0`. Stability: best ensemble in **51%** of
  1,000 bootstraps.
- **Distil (students)**: encode `"{Headline}. {Description}"` with 35 sentence-transformers;
  fit `sklearn LinearRegression(embeddings → teacher sum)`; **75/25 split, random_state=42**;
  threshold at ±1; score 3-class agreement vs teacher.
- **Punchline**: OLS on **`sentence-t5-large`** = **80.40%**, and `e5-large-v2` = 80.21% —
  both **beat GPT-4 (79.90%)** at ~$0 marginal, ~100× faster. GPT-4 $92k/10M & 57 compute-days
  → student $0 & <1 day. Whole analysis = 198 lines of Python.

### 3c. Production: active learning + Qwen3-0.6B

- **100,000** real-world texts labeled by an **8-LLM majority vote** (Grok-4-Fast ×2,
  Gemini-2.5-Flash, GPT-5-Nano, GPT-4.1-Mini, gpt-oss-120b, Llama-4-Maverick, Qwen3-32B).
- **Baseline** = `SGDClassifier(loss='hinge', penalty='l2', alpha=1e-5)` per embedding
  (6 embedding models), 80/20 split. 86.65% train / 85.93% val.
- **Active-learning relabel loop**: find rows where *all* linear models agree but *disagree*
  with the majority vote → ask oracle **GPT-5.1** (`{"correct":bool,"label":...,"reason":...}`)
  → relabel if oracle sides with baselines → drop the worst linear model → retrain → repeat
  (~18 iterations). Gain: **+3.5% val → 89.48%**.
- **Fine-tune `Qwen/Qwen3-0.6B`** as *next-token classification* (Qwen3Guard style, no
  classifier head): chat template, **label-mask −100 on prompt tokens**, single-token label +
  `<|im_end|>`, **left padding**, bf16. Hyperparameters (verbatim): `lr=2e-5`, cosine,
  `warmup_ratio=0.03`, 5 epochs, batch 16 × grad-accum 4, `weight_decay=0.1`,
  `neftune_noise_alpha=5`, `optim=adamw_torch_fused`, `group_by_length=True`,
  best-model-on-`eval_val_loss`. **Metric pitfall**: exclude EOS from accuracy (mask −100 AND
  the eos id) or numbers look "suspiciously good."
- **Result**: **87.34%** real-world / **86.4%** Financial PhraseBank, beats FinBERT on both,
  matches GPT-5.1 at orders-of-magnitude less cost. **Serving**: SGLang on L40S, regex-constrained
  single-token decode (`max_tokens=1`, `regex="(positive|neutral|negative)"`, thinking off),
  logprobs → confidence via `math.exp(logprob)`.
- **Open-sourced** on HF: datasets `NOSIBLE/financial-sentiment` (100k),
  `NOSIBLE/forward-looking`, `NOSIBLE/prediction`; models `NOSIBLE/financial-sentiment-v1.1-base`
  et al. — **you can download these directly for the replica.**

### Build recipe (laptop scale)

1. **[disclosed]** Start from `NOSIBLE/financial-sentiment` (HF) or label a few-thousand-row
   corpus with 3–5 cheap API LLMs + the verbatim decision-tree prompt; majority vote.
2. **[disclosed]** Validate the prompt against ~200 hand labels; refine from disagreements.
3. **[disclosed]** Baseline: `SGDClassifier(hinge, l2, alpha=1e-5)` on 2–6 sentence-transformer
   embeddings (`e5-small-v2`, `bge-large`, `all-mpnet-base-v2`), 80/20. Confirms signal.
4. **[disclosed]** Active-learning relabel loop with a strong oracle model, drop-worst pruning,
   to convergence.
5. **[disclosed]** Either stop at the OLS/SGD student (Post 2 — already GPT-4-competitive) **or**
   fine-tune `Qwen/Qwen3-0.6B` with the disclosed script (label masking, left pad, EOS-excluded
   metric). **[inferred]** local `transformers`/ONNX inference is an acceptable stand-in for
   SGLang/L40S.
6. **[disclosed]** Evaluate on a 20% holdout + `takala/financial_phrasebank`; compare to
   `ProsusAI/finbert`.
7. **[disclosed]** Repeat for forward-looking / prediction using the other two disclosed prompts.

### Gaps

Image-only per-model accuracy tables; why Qwen3-0.6B beat ModernBERT/DeBERTa (no numbers);
Search-Feed sampling/negatives-oversampling; per-iteration relabel counts; 8-model tie-breaking;
fine-tune hardware/wall-clock.

---

## 4. Point-in-time knowledge graphs over NER entities

**Source:** `point-in-time-knowledge-graphs-over-named-entities.md`.

### What was disclosed

Nodes = NER entities (ORG/PERSON/GPE/LOC/PRODUCT); edges = statistical **co-mention lift**
within a **single year's** news; every edge date-bucketed so a graph "as of 2018" knows only
2018 — eliminating **lookahead bias** in backtests.

**Exact formula:** `lift = (co / target_events) / (entity_events / total_events)`.
Worked example (verbatim): NVIDIA–Blackwell 2024 `co=69, target_events=2651, entity_events=75,
total_events=2249039 → lift ≈ 780`. A *normalized* lift is used for ranking (formula withheld).
Validation: **24/24** sampled edges in-year across 955 docs. Reference appendix code:
`co_mentions_by_year(events, target_ticker, entity_type)` → `defaultdict(Counter)` keyed by
`event["date"][:4]`, counting each entity **once per event** (presence, not mention count).

### Build recipe

1. **[inferred]** News corpus with dates; optionally cluster near-dup articles into "events"
   first (embedding + agglomerative) to approximate one-record-per-event.
2. **[disclosed]** spaCy NER (`en_core_web_trf`) → per-doc `{type:{name:count}}` — yields exactly
   ORG/PERSON/GPE/LOC/PRODUCT.
3. **[inferred]** Resolve the target company to a ticker via a small alias list.
4. **[disclosed]** Canonicalize surface forms (case/suffix collapse; per-year surname→full-name
   folding).
5. **[disclosed]** Bucket by `date[:4]`; count co-mentions per (company, year, entity) as
   *presence per event* (port `co_mentions_by_year` verbatim).
6. **[disclosed]** Compute `target_events`, `entity_events`, `total_events` per year; score
   `lift`; keep edges well above 1.0.
7. **[inferred]** Normalize with `log(lift)` and a support floor (`co≥3`) since the real
   normalized-lift formula is withheld.
8. **[disclosed+inferred]** Build per-year graphs in `networkx` (ego graph + entity–entity edges
   by the same lift test); force-directed layout, node color = NER type, size ∝ strength.
9. **[disclosed]** Evolution matrix (entity × year) in matplotlib: dot size = strength, dot color
   = YoY change in the entity's coverage share.

### Gaps

Normalized-lift formula; NER model identity; full entity-resolution algorithm; edge thresholds;
entity–entity edge criterion; event dedup (the moat); embedding model / ontology facets.

---

## 5. Frozen-embedding classification ("contrastive geometry")

**Source:** `the-contrastive-geometry-of-risk.md`. Training-free, LLM-free, **backtest-safe**
classification from frozen sentence embeddings and elementary geometry. Two tricks.

### Trick A — multiclass relevance score

1. **[disclosed]** Author **20 reference sentences per bucket**, near mirror-images (same frame,
   only the class phrase varies), so topic/style cancel. Demo buckets: local/national/global.
2. **[disclosed]** Embed with a frozen model — post uses OpenAI **`text-embedding-3-large`,
   3,072-d, Sept-2021 cutoff** (chosen so it *cannot* know post-2021 outcomes). Two unrelated
   same-register sentences score ≈ **0.3** (the background floor to kill).
3. **[disclosed]** **tanh gate**, band **0.25 → 0.50** (below→0, above→1).
4. **[disclosed]** **cubic power mean (p=3)** per bucket: `(mean(gated³))^(1/3)`; argmax. A
   **dead-zone floor** returns "none" for off-topic (value withheld). Worked scores: local 0.315,
   national 0.417, global 0.739 → global.

### Trick B — binary signed contrast

1. **[disclosed]** **20 mirror pairs**, identical except the flipped attribute
   (single-company ↔ whole-market), topics varied *across* pairs so averaging cancels topic.
2. **[disclosed]** **Common-mode neutralization**: subtract the mean of all reference vectors,
   renormalize. Shared coordinate drops **+0.71 → +0.005**.
3. **[disclosed]** Score = `mean cos(x', set_A') − mean cos(x', set_B')`; sign = class. Running
   example: **−0.068** = systemic.
4. **[disclosed]** Fuse the two independent reads onto a 2-D plane (scale × scope).

**Accuracy**: geometry 16/18 geography, 15/18 scope, 13/18 both — a **dead heat** with
`gemini-2.5-flash` @ temp 0 (identical 16/15/13). Five stated wins over LLM labeling:
deterministic, defensible (re-derivable by hand), cheap/fast, no training data, **no
foreknowledge bias**.

### Build recipe

1. **[disclosed]** Reuse the printed reference sets, or author your own by the rules.
2. **[disclosed/inferred]** Frozen encoder — `text-embedding-3-large`, or a local
   sentence-transformer (`all-mpnet-base-v2`, `bge-large-en-v1.5`); **retune the gate band** since
   the 0.3 floor is model-specific.
3. **[disclosed]** L2-normalize; cosine vs references.
4. **[disclosed+inferred]** tanh gate on [0.25, 0.50] — assume `0.5·(1+tanh(k·(cos−0.375)))`,
   `k≈8–12`.
5. **[disclosed]** cubic power mean, argmax, dead-zone "none".
6. **[disclosed]** Binary axis: `mu=mean(all refs)`; neutralize `v'=normalize(v−mu)`; signed
   mean-cosine contrast (neutralize the input with the same `mu`).
7. **[disclosed]** Verify on ~18 hand-labeled sentences spanning the grid, incl. hard cases;
   optionally benchmark vs an LLM at temp 0.

### Gaps

Exact tanh parameterization; dead-zone value; whether the input is neutralized (implied);
projection method in figures; production axes/thresholds; ablations (why 20, why p=3).

---

## 6. Quant signals: the one template, five knobs

**Sources:** `rebuilding-the-geopolitical-risk-index-from-nosible-world.md`,
`turning-news-into-a-risk-on-risk-off-equity-signal.md`,
`an-embedding-based-approach-to-trade-and-economic-policy-uncertainty.md`.

Every signal is the **same template**: `Σ breadth·weight over matching events /
(total breadth or its trailing-12-month mean)`. Knobs: **(1) concept** (ontology filter or K
anchor phrases), **(2) admission** (`max-cosine ≥ floor`, or ontology AND/OR embedding),
**(3) weight** (`breadth`, `relevance·breadth`, or `breadth·w_unc`), **(4) denominator**
(same-day share or detrended trailing-12-mo share), **(5) post-processing** (rebase-to-mean, or
the full trading stack).

### 6a. Geopolitical Risk Index (GPR)

- **[disclosed]** Geopolitical filter on IPTC **codes** (never label words):
  `iptc_level_1=="conflict, war and peace" OR iptc_level_2=="international relations" OR
  iptc_level_3 ∈ {war crime, genocide, terrorism, nuclear policy}`.
- **[disclosed]** `NOSIBLE-GPR(t) = Σ breadth(e) over geopolitical events / Σ breadth(e) over
  all events`; detrended variant divides by the **trailing 12-month average** of total breadth.
- **[disclosed]** Country cut: `attribution(e) = {main country} ∪ {NER-resolved countries}`;
  group by (country, month) / **global** `B(m)`. Multi-country attribution lifts US corr
  **0.45→0.83**. Correlations: Iran 0.99, Israel/Ukraine 0.96, Russia 0.94; 76 countries >0.60.
- **[disclosed]** Pair cut via self-join into unordered pairs. Iran–USA 0.97, Russia–Ukraine
  0.95; **China–USA fails at 0.28** (tariffs live in the economy branch).
- **[disclosed]** *Trade-coercion fix*: admit if `geopolitical OR cosine(event, trade_anchor)
  ≥ 0.40`; China–USA **0.28→0.75** with only **8,079** events added. *Oil-GPR*: 3 supply anchors,
  keep `geopolitical AND relevance≥0.30`, weight `relevance·breadth`; corr 0.95–0.97. All anchor
  phrases printed verbatim in the post.
- Overall global corr vs AI-GPR: **0.90 levels / 0.79 changes** (raw). 13.2M events, 2019+.

### 6b. Risk-on/risk-off equity overlay

- **[disclosed]** **17 stress anchors** (all printed verbatim), embedded once; truncate event +
  anchor vectors to the **first 1024 of 3072 Matryoshka dims**, **L2-renorm**.
- **[disclosed]** `relevance(e)=max cos over 17 anchors`, gate `≥0.30`;
  `intensity(t)=Σ relevance·breadth (qualifying) / Σ breadth (all)`.
- **[disclosed]** Calibrate: `s=intensity.rolling(7).mean()`; robust
  `z=(s−median)/(1.4826·MAD)` over trailing **252** days; `z=EWMA(z, span=7)`.
- **[disclosed]** **Lag 2 trading days**, execute next bar. State machine: exit to `BIL.US` when
  `z>1.75`; re-enter when out ≥5 days AND `z<0.25`; cost 1bp × |Δw|.
- **[disclosed]** Selected on **2010–2013 only** (grid: exit_z {1.75..3.0}, enter_z {0..0.75},
  min_hold {3,5,10,15}), objective **Sortino improvement over B&H**, 252-day embargo, **plateau
  pick** (median of top-20% cells), then frozen; tested untouched **2015-01-02→2026-06-01**.
- **[disclosed]** Results (test): S&P Sharpe 0.64→0.89, max DD −34%→−18%, return +269%→+254%;
  identical frozen rule transfers to Nasdaq & Russell 2000. Prices: EODHD adjusted closes.

### 6c. Trade / Economic Policy Uncertainty (TPU / EPU)

- **[disclosed]** TPU anchors = **3 topic + 2 axis** sentences (all verbatim);
  `relevant(e)=max cos(event, 3 topics) ≥ 0.35`;
  `polarity(e)=tanh((uncertain_sim − certain_sim)/0.1) ∈ [−1,1]`;
  `w_unc=(1+polarity)/2`; `TPU(t)=Σ breadth·w_unc / trailing-12-mo mean of daily breadth`.
- **[disclosed]** The **signed** axis (resolved↔uncertain) is something the keyword indices lack.
  0.35 cutoff "not load-bearing" (0.30→0.40 moves corr ~0.02); chosen by a sweep vs the published
  index. Scale: **14.9M events scored in ~12 min on a laptop.**
- **[disclosed]** Results: TPU vs published **0.87 levels / 0.82 changes** monthly (the two
  official indices agree 0.96/0.70). EPU = **10 policy "levers"**, 60 verbatim sentences (matched
  uncertain/certain pairs), US-scoped numerator+denominator, category gate ≥0.25; US EPU corr
  **0.77 levels**. Two design lessons: a single generic anchor **fails** (concrete decision types
  needed); resolution must be written as the topic's concrete act, so use **matched U/C pairs**.

### Build recipe (shared)

1. **[inferred]** Corpus: GDELT / open news archive with dates; dedupe by embedding cosine >~0.9
   in a date window; **breadth = distinct source domains** (proxy for `total_netlocs`).
2. **[disclosed]** Embed events + anchors; for 6b/6c replicate `text-embedding-3-large` @1024-d +
   L2-renorm, or substitute a sentence-transformer and **re-sweep the floor** against the
   published index (the post itself set thresholds by such a sweep).
3. **[disclosed]** Apply the concept filter/anchors, admission rule, weight, and denominator per
   §6a/6b/6c. Copy anchor phrases verbatim from the archive files.
4. **[disclosed]** Validation: Pearson on levels + first differences vs the published series —
   GPR at matteoiacoviello.com/gpr.htm, TPU at /tpu.htm, EPU + categories at
   policyuncertainty.com, tariff rate via BEA NIPA on DBnomics. Rebase each series to its own mean.
5. **[disclosed]** For 6b only: the full causal trading stack + train/embargo/plateau/freeze
   protocol above; free price substitute = yfinance `^GSPC ^IXIC IWM BIL ^VIX`.

### Gaps

Event dedup + breadth + dating + topic/country tagging (the moat); the threshold sweeps; why
`tanh(gap/0.1)`; exact daily→monthly + "trailing 12-month" mechanics; Sharpe/Sortino conventions.
No code or raw index values released.

---

## 7. Vector-search substrate & self-organizing tagging

**Sources:** `using-vector-search-to-see-signals-in-company-news.md` (LSH substrate),
`can-faceted-search-at-web-scale-self-organize.md` (entity tagging).

### 7a. LSH-indexed weekly signals

- **[disclosed]** Corpus (Jan 2024): 55M+ embedded snippets; SBERT (checkpoint unnamed);
  **Random-Hyperplanes LSH** — `sign(X·H)` bits → integer hashes → Hamming distance; "300M+
  vectors/s", search over 55M in **<0.20 s**, ~10× a flat FAISS index. Filter-then-search: SQL
  metadata filter → candidate rows → LSH → exact cosine on top N×M.
- **[disclosed]** `get_signal` (verbatim mechanics): weekly Monday buckets; keep sims ≥ **70% of
  max code**; percentile weighting **+8/+4/+2/+1** at p80/p60/p40/p20; divide by
  `weekly_count·8`. **Triplet signal**: for baseline/positive/negative,
  `z=(SMA4 − SMA26)/STD26`; final `=(pos_z − base_z) − (neg_z − base_z)` — works around embeddings
  that can't separate "beat" vs "missed" and around earnings seasonality. Five production triplets
  listed verbatim.
- **[disclosed]** Company disambiguation: **Aho-Corasick** (`ahocorasick_rs`) high-conviction
  seeds → probability model on domain + co-mentioned GPEs → ensemble. AI-news filter = per-domain
  embedding-dispersion × volume → **domain blacklist**. Dedup = shard by time×metadata, elect an
  **apex story** (naive full dedup ≈ 11M s ≈ 4 months → minutes sharded).

### 7b. Self-organizing entity facets

- **[disclosed]** Two taggers per collection: a **fast Aho-Corasick** tagger (real-time, built
  from resolver output) + a **slow neural NER** sampling **X%** of incoming docs. Slow tagger
  proposes entities → accumulate → threshold → **Resolver Agent** (tools: Search, Wikipedia,
  LinkedIn, Market APIs) builds a canonical record (rich JSON schema disclosed: kgid/wiki_qid/cik/
  isin/cusip/lei/figi/ticker/GICS/aliases/subsidiaries…) → **distill** to a flat
  unigram/bigram/trigram pattern list → push to the collection → **retag at next flush**.
- Distillation principle: "the intelligence is in the tokens — distill it into strings + regex."

### Build recipe

1. **[disclosed]** SBERT snippets (`all-MiniLM-L6-v2` / `all-mpnet-base-v2`; checkpoint unnamed).
2. **[disclosed]** **Random-Hyperplanes LSH**: `k` Gaussian planes, codes `(X@H.T>0)` packed to
   ints, query by Hamming (`bitwise_xor`+`bit_count`) or FAISS `IndexLSH`. **[inferred]** 64–256 bits.
3. **[disclosed]** Metadata in DuckDB/Polars/SQLite; **filter then search**.
4. **[disclosed]** Port `get_signal` verbatim (Monday buckets, 70%-of-max floor, 8/4/2/1
   percentile weights, `/weekly_count·8`) + the triplet z-score transform.
5. **[disclosed]** Entity tagging: `pyahocorasick` fast tagger + spaCy/HF NER slow tagger sampling
   X%; SQLite proposal DB + threshold; a tool-calling **Resolver Agent** (Wikidata/ticker APIs,
   cache hard by normalized name); distill to n-gram patterns; retag on flush.
6. **[disclosed]** Facet search: query → facets → pre-filter → hybrid retrieve.

### Gaps

SBERT checkpoint + dims + chunking; LSH bits/tables/multi-probe + the 300M/s engineering;
dedup thresholds + apex rule; AI-news metric/thresholds; disambiguation model form; the
collection-segmentation model; sampling rate X + suggestion threshold; resolver LLM/prompt.

---

## 8. The event layer (the moat) — approximating it

Never disclosed in any post, yet upstream of almost every number. To replicate:

1. **De-dup to one-record-per-event** — **[inferred]** embed headlines/snippets; cluster
   near-duplicates within a sliding date window (cosine > ~0.9, or agglomerative/`DBSCAN`); each
   cluster = one event; pick an apex (earliest or most-covered).
2. **Breadth** — **[inferred]** `total_netlocs` ≈ count of **distinct source domains** in the
   cluster.
3. **Topic tags (IPTC)** — **[inferred]** zero-shot classify into IPTC buckets (embed IPTC label
   descriptions, nearest), or replace the ontology filter with an embedding anchor for the concept.
4. **Country tag** — **[inferred]** NER-majority GPE, or source-domain country.
5. **Event date** — **[inferred]** coverage-peak day of the cluster (post uses this; algorithm
   withheld).
6. **Event embedding** — **[disclosed]** `text-embedding-3-large` @1024-d + L2-renorm for exact
   replication of §6; a multilingual sentence-transformer otherwise (re-sweep thresholds).

Free data sources for the whole guide: **GDELT** (events + source domains), Hugging Face
`NOSIBLE/*` datasets (labels), `takala/financial_phrasebank`, published index files
(matteoiacoviello.com, policyuncertainty.com), EODHD/yfinance prices, DBnomics macro.

---

## 9. Suggested build order for a full educational replica

1. **§8 event layer** on a GDELT slice — the foundation everything else reads.
2. **§6c TPU** — smallest, most self-contained (5 anchors, ~12-min compute, published benchmark).
   Fast confidence that the substrate works.
3. **§6b overlay** — adds the causal calibration + train/embargo/freeze discipline.
4. **§5 frozen-embedding features** + **§3 distilled classifiers** — event enrichment
   (both have open HF artifacts to check against).
5. **§4 point-in-time KG** — NER + yearly lift over the same events.
6. **§1 Hybrid-3** — the retrieval pipeline (largest; the repo already scaffolds it).
7. **§2 Cybernaut-1 MCTS** — the agent on top, once §1 is a working environment.

**One-line summary of the architectural signature to preserve:** *date-partition everything,
freeze the encoder, weight by breadth, normalize by attention, and make every score re-derivable
from counts, cosines, means, and subtractions.*

---

*Sources: the 12 posts in [`blog-archive/`](blog-archive/). Reverse-engineering notes were
produced clean-room from public text for educational purposes; all content © NOSIBLE.
`cybernaut_mini` is unaffiliated with NOSIBLE and claims no benchmark parity.*
