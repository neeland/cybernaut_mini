# cybernaut-mini

An educational, local-first replica of the retrieval architecture described in
the NOSIBLE Cybernaut posts:

- [The Road to Cybernaut-1](https://nosible.com/blog/the-road-to-cybernaut-1)
- [Introducing Cybernaut-1: Agentic Search with MCTS](https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts)

**This project is unaffiliated with NOSIBLE and makes no claim of benchmark
parity. It is a clean-room, educational implementation written for learning
purposes only.**

---

## Architecture

```mermaid
graph LR
    A[documents.jsonl] -->|ingest| B[Shard & Embed]
    B -->|MiniBatchKMeans| C[Index Artifacts]
    C -->|LoadedIndex| D[Router]
    D -->|top-N shards| E[Retrieve]
    E -->|lexical + dense + RRF| F[Stage 1: Explore]
    F -->|UCT select| G[Stage 2: Refine]
    G -->|UCT select| H[Stage 3: Exploit]
    H -->|winner replay| I[SearchHit list + AgentTrace]
```

The build stage is a **Kedro pipeline** (`index_build`). Routing, retrieval, and
agent search are plain library calls — dynamically branching with a shared budget
counter, which makes them unsuitable for a static DAG. A separate Kedro
`evaluation` pipeline wraps `evals.evaluate` for reproducible offline benchmarks.

`kedro viz` renders both pipeline DAGs when the optional `[viz]` extra is
installed.

---

## Source-described vs replica choices

| Feature | NOSIBLE Cybernaut-1 | This replica |
|---|---|---|
| Index scale | 250 k shards | 8–12 shards (tiny.yaml: 8) |
| Build pipeline | 8-stage pipeline | Kedro `index_build` (6 nodes) |
| Embedding | Shard-specific LLMs | Hash embedder (offline) or e5-small |
| Similarity | Bayesian dense similarity | Exact cosine, float32 |
| Search tree | LLM-guided MCTS | Staged beam search with UCT budget allocation (honestly not full MCTS) |
| Judge / generator | LLM-based | Heuristic (token coverage, Jaccard) — 0 LLM calls |
| Reward max | Unspecified | 0.95 by design (weights 0.45+0.20+0.20+0.10−0.05); 1.0 is unreachable |
| Retrieval budget | Unspecified | 5+9+4 = 18 calls max |
| Stage schedule | Unspecified | Explore 5, Refine 9, Exploit 4 |
| Entities | Production NER | Regex fallback (spaCy optional via `[nlp]`) |
| Pipelines | Unknown internal | Kedro `index_build` + `evaluation` |

---

## Offline quick start

```bash
# Install (no GPU, no downloads)
uv sync

# Generate sample corpus (63 docs, 12 graded judgments)
uv run python scripts/generate_sample_corpus.py

# Build the sample index (hash embedder, 8 shards, ~0.3 s)
uv run cybernaut-mini build \
  --input data/sample/documents.jsonl \
  --index artifacts/sample \
  --config configs/tiny.yaml \
  --offline

# Inspect shards
uv run cybernaut-mini inspect-shards --index artifacts/sample

# Hybrid search
uv run cybernaut-mini search \
  --index artifacts/sample \
  --question "CRISPR screening immune pathways T-cell" \
  --mode hybrid \
  --offline

# Agent search (three-stage beam search, <=18 retrieval calls, 0 LLM calls)
uv run cybernaut-mini search \
  --index artifacts/sample \
  --question "transformer scaling laws language model" \
  --mode agent \
  --offline \
  --config configs/tiny.yaml \
  --trace-out run.json

# Evaluate all four modes
uv run cybernaut-mini eval \
  --index artifacts/sample \
  --judgments data/sample/judgments.jsonl \
  --config configs/tiny.yaml \
  --offline
```

### Kedro equivalents

```bash
# Build via Kedro (same artifacts as the CLI build above)
uv run kedro run --pipeline index_build \
  --params "input_path=data/sample/documents.jsonl,index_path=artifacts/sample,seed=42,offline=true"

# Evaluate via Kedro
uv run kedro run --pipeline evaluation \
  --params "index_path=artifacts/sample,judgments_path=data/sample/judgments.jsonl,offline=true"
```

---

## Artifact formats

```
artifacts/<index>/
  _VALID               # written last; loaders reject an index without this marker
  index_meta.json      # embedding_model, embedding_dim, n_shards, n_documents, seed
  documents.jsonl      # canonical-JSON, one Document per line (sorted keys)
  embeddings.npy       # float32 [n_docs x dim]
  row_map.json         # {doc_id -> row index}
  tokens.jsonl         # {id, tokens} per doc
  shards/
    shard_000.json     # ShardManifest: centroid, keywords, entities, term_graph
    ...

data/sample/
  documents.jsonl      # 63 synthetic docs across 6 topics, seed 42
  judgments.jsonl      # 12 graded queries (grades 1-2), referencing real doc IDs
```

All JSON artifacts are written through `canonical_dumps` (sorted keys, compact
separators, floats rounded to 8 decimal places) so two builds with the same seed
and provider are byte-for-byte identical within the same sklearn version.

---

## Annotated trace example

Question: `"transformer scaling laws language model"`
Config: `configs/tiny.yaml` (hash embedder, 8 shards, seed 42)

```json
{
  "question": "transformer scaling laws language model",
  "seed": 42,
  "retrieval_calls": 6,
  "llm_calls": 0,
  "stop_reason": null,
  "final_query": "transformer scaling laws language model graph spaces",
  "final_shard_ids": [2, 5, 0],
  "selected_path": [0, 3, 5, 6],

  "nodes_trimmed": [
    {
      "node_id": 1, "stage": "explore",
      "query": "transformer scaling laws language model",
      "shard_ids": [2, 5, 0, 1, 7, 4, 3, 6],
      "reward": 0.4726,
      "reward_components": {
        "relevance": 0.36, "coverage": 1.0,
        "dense": 0.3244, "lexical": 0.4666, "redundancy": 0.0192
      },
      "judge_reason": "heuristic: rel=0.36 cov=1.00 red=0.02 over 5 hits"
    },
    {
      "node_id": 3, "stage": "explore",
      "query": "transformer scaling laws language model graph",
      "reward": 0.4851,
      "judge_reason": "heuristic: rel=0.40 cov=1.00 red=0.02 over 5 hits"
    },
    {
      "node_id": 5, "stage": "refine",
      "query": "transformer scaling laws language model graph spaces",
      "shard_ids": [2, 5, 0, 1, 3],
      "reward": 0.5342,
      "judge_reason": "heuristic: rel=0.46 cov=1.00 red=0.02 over 5 hits"
    },
    {
      "node_id": 6, "stage": "exploit",
      "query": "transformer scaling laws language model graph spaces",
      "shard_ids": [2, 5, 0],
      "reward": 0.5281,
      "judge_reason": "heuristic: rel=0.46 cov=1.00 red=0.02 over 5 hits"
    }
  ]
}
```

**Trace annotations:**

- **Node 0 (root)**: original question, no retrieval.
- **Explore (nodes 1-3)**: heuristic generator emits up to 5 deduplicated
  candidate queries. Each executed once (1 retrieval call each). Reward is a
  weighted sum of heuristic judge scores plus min-max-normalised dense/BM25
  means. Top 3 by mean value survive.
- **Refine (nodes 4-5)**: UCT selects which parent to expand next (unvisited
  first, then UCT score with `c=1.2`, ties broken by action type then payload).
  Node 5 adds "spaces" as a term-graph expansion, raising reward from 0.479 to
  0.534.
- **Exploit (node 6)**: winner gets 2 children, narrowed to 3 shards, 40
  hits/shard. Node 6 is selected as terminal winner (highest mean value across
  all nodes).
- **Reward formula**: `0.45 x relevance + 0.20 x coverage + 0.20 x dense_norm
  + 0.10 x lexical_norm - 0.05 x redundancy`. Weights intentionally sum to 0.95;
  a perfect score of 1.0 is unreachable by design.
- **Budget**: 6 retrieval calls total, well within the 18-call limit (5+9+4).
- **UCT**: staged beam search, not full MCTS. The "MCTS" label in the NOSIBLE
  post inspired the design, but the implementation is a staged beam with UCT
  ordering in the Refine phase only.

---

## Evaluation results (sample corpus, offline, seed 42)

Run: `uv run cybernaut-mini eval --index artifacts/sample --judgments data/sample/judgments.jsonl --config configs/tiny.yaml --offline`

```
Mode           Recall@5    Recall@10       MRR@10      nDCG@10    Ret calls    LLM calls   Wall (s) (informational)
-------------------------------------------------------------------------------------------------------------------
lexical          0.5972       0.6944       0.9583       0.7935          1.0          0.0                       0.01
dense            0.5833       0.6111       1.0000       0.7902          1.0          0.0                       0.01
hybrid           0.6111       0.6528       0.8194       0.7019          1.0          0.0                       0.01
agent            0.6528       0.6944       0.8194       0.7140          6.3          0.0                       0.21
```

**When agent mode underperforms baseline hybrid:** On this small synthetic corpus
(63 docs, 12 queries) the agent mode does not consistently beat hybrid. In this
run, agent achieves higher Recall@5 and Recall@10 but lower MRR@10 and nDCG@10
than the simple lexical baseline. This is expected: the three-stage search is
designed to add value when the retrieval space is large and a good initial query
is hard to formulate. On 63 documents, a single BM25 or cosine call often
retrieves the top result on the first try, and 6.3 extra retrieval calls per
query add overhead without proportional quality gain. On a real-scale index
(250 k+ shards), routing and refinement would provide substantially more benefit.

---

## Replica-specific design choices

| Decision | Value | Rationale |
|---|---|---|
| Reward weights | 0.45/0.20/0.20/0.10/-0.05 | Sum to 0.95; perfect score unreachable by design |
| Retrieval-call budget | 5+9+4 = 18 | Stage schedule from spec resolution #1 |
| Hash embedder dim | 256 (tiny: 256, tests: 64) | `hashlib.blake2b` token+char-trigram buckets, L2-norm |
| spaCy | Optional `[nlp]` extra | Regex fallback tested by default; `en_core_web_sm` auto-detected |
| Byte-for-byte determinism | Same seed + provider + sklearn version -> identical | `canonical_dumps` + float32 npy |
| tiny.yaml shards | 8 | ~7-8 docs/shard for 63-doc corpus |
| default.yaml shards | 12 | Default per spec |

---

## Kedro vs library boundary

| Layer | Technology | Reason |
|---|---|---|
| `index_build` | Kedro pipeline | Pure map-reduce DAG; reproducible, replayable |
| `evaluation` | Kedro pipeline | Pure map-reduce DAG; offline benchmarks |
| Routing / retrieval | Python library | Per-request; input is a question string, not a dataset |
| Agent search | Python library | Shared budget counter, UCT ordering, cannot be a static DAG |

Search is not a Kedro pipeline because retrieval branches dynamically at runtime
(UCT selects the next node to expand based on prior results), and the budget
counter is shared across all three stages in a single request. Kedro's data
catalog model assumes a static, pre-known DAG — which the agent's tree search
violates. `kedro viz` renders only the `index_build` and `evaluation` pipelines.

---

## Limitations and ethical-use notes

- **Educational only.** Not a production search system. Results on real corpora
  depend heavily on embedding quality.
- **Hash embedder is the default.** Fast, offline, and deterministic, but poor
  recall on paraphrase-heavy queries. It is the baseline so the project runs on a
  bare `uv sync` with no downloads. For better quality, opt into real embeddings
  explicitly — `--config configs/default.yaml` (CLI) or `--env prod` (Kedro) —
  after installing the extra with `uv sync --extra st` (or `make install-prod`).
  That path needs internet on first run.
- **Small corpus.** The 63-doc synthetic corpus is for testing; metric numbers
  above should not be extrapolated to real workloads.
- **sklearn version drift.** KMeans output depends on the installed sklearn
  version. Byte-identical builds are guaranteed within the same environment only.
- **rank-bm25.** Unmaintained but stable for educational use.
- **No benchmark claims.** This replica has not been evaluated against any
  published NOSIBLE benchmark.
- **Unaffiliated.** Not affiliated with, endorsed by, or connected to NOSIBLE
  in any way.
- **License.** MIT. See `LICENSE`.
