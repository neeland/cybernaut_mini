#!/usr/bin/env bash
#
# A terminal slide deck on the technicalities of cybernaut-mini.
#
#   scripts/preso.sh              interactive: space/n next, p back, q quit
#   scripts/preso.sh --all        print every slide and exit
#   scripts/preso.sh --all | less -R
#   scripts/preso.sh --no-color   plain text
#   scripts/preso.sh --start 9    open on a given slide
#
# Bash and coreutils only, on purpose: the deck has to run before `uv sync` does.

set -uo pipefail

WIDTH=78
TOTAL=18

# ------------------------------------------------------------------------- #
# Colour                                                                      #
# ------------------------------------------------------------------------- #

USE_COLOR=1
[[ -t 1 ]] || USE_COLOR=0
[[ -n "${NO_COLOR:-}" ]] && USE_COLOR=0

set_palette() {
  if ((USE_COLOR)); then
    RESET=$'\033[0m'; BOLD=$'\033[1m'
    BCYAN=$'\033[96m'; YELLOW=$'\033[33m'; MAGENTA=$'\033[95m'; GREY=$'\033[90m'
  else
    RESET=''; BOLD=''; BCYAN=''; YELLOW=''; MAGENTA=''; GREY=''
  fi
}

# ------------------------------------------------------------------------- #
# Inline markup: `code`, *emphasis*, {accent}                                  #
# ------------------------------------------------------------------------- #

apply_markup() {
  sed -E \
    -e "s/\`([^\`]*)\`/${BCYAN}\1${RESET}/g" \
    -e "s/\*([^*]*)\*/${BOLD}\1${RESET}/g" \
    -e "s/\{([^}]*)\}/${YELLOW}\1${RESET}/g"
}

repeat() { # repeat <count> <char>
  local n=$1 c=$2 out=''
  while ((n-- > 0)); do out+="$c"; done
  printf '%s' "$out"
}

# ------------------------------------------------------------------------- #
# Slides                                                                      #
# ------------------------------------------------------------------------- #

SLIDE_KICKERS=(
  ""   "01" "02" "03" "04" "05" "06" "07" "08"
  "09" "10" "11" "12" "13" "14" "15" "16" ""
)

SLIDE_TITLES=(
  "cybernaut-mini"
  "The repo at a glance"
  "Module map"
  "Two paths, deliberately different shapes"
  "Hybrid-3 · the query pipeline"
  "One regex, one real bug"
  "Routing · which shards even get asked"
  "Three shard artifacts the post lists"
  "Retrieval · lexical + dense + RRF"
  "Agent search · staged beam, not full MCTS"
  "Storage · the 20 GB problem"
  "Four rejections worth reading"
  "Determinism is a hard requirement"
  "Degrade, never raise"
  "Replica vs. the real thing"
  "Driving it"
  "Where it stands"
  "fin"
)

slide_body() {
  case "$1" in
  0) cat <<'EOF'

  A local-first, clean-room replica of NOSIBLE's *Cybernaut-1*:
  sharded hybrid search plus three-stage agentic retrieval,
  reconstructed from two public blog posts.

  {Unaffiliated with NOSIBLE. No benchmark-parity claim.}
  {Educational implementation only.}

  ─────────────────────────────────────────────────────

  Source of truth        `docs/blog-archive/the-road-to-cybernaut-1.md`
                         `…/introducing-cybernaut-1-agentic-search-with-mcts.md`

  Rule of the repo       every module cites the paragraph it replicates,
                         states its assumptions, and names what it rejected

EOF
    ;;
  1) cat <<'EOF'

    Python              3.12 only  (`>=3.12,<3.13`, pinned hard)
    Package manager     uv + hatchling, `src/` layout
    CLI                 Typer — `cybernaut-mini`
    Orchestration       Kedro 0.19 — 4 registered pipelines
    Validation          pydantic v2 models everywhere on the boundary

    Source              *16,920* lines across 6 packages
    Tests               *8,436* lines · *570* test functions · 30 files
    History             44 commits

    Gates               `ruff` (E,F,W,I,UP,B,SIM,RUF) · `mypy --strict`
                        · `pytest` · notebooks executed headlessly

    ─────────────────────────────────────────────────────────────────

    The test suite is half the size of the source. That ratio is the
    design: the blog gives worked examples, and the tests *are* the
    spec — a token list from the post fails the build if it drifts.

EOF
    ;;
  2) cat <<'EOF'

  `src/cybernaut_mini/`

    indexing.py       897   build + `LoadedIndex`, the artifact contract
    storage.py        774   mmap / offset / LRU primitives   {new}
    retrieval.py      588   BM25 + dense + RRF + snippets
    shard_artifacts.py 543  bloom · zstd dict · vocabulary    {new}
    cli.py            421   build / inspect-shards / search / eval
    routing.py        250   shard scoring and rerank
    models.py         260   pydantic schemas
    rrf.py                  weighted reciprocal rank fusion
    sharding.py             MiniBatchKMeans clustering

    `agent/`          628   search.py + node · policy · state · actions
    `query/`         2.4k   s1_language · s2_tokenize · s3_intents ·
                            s4_instruct                       {new}
    `providers/`            embeddings · judge · query_generator
    `pipelines/`            corpus_ingest · index_build · evaluation

  {new} = present in the working tree, not yet committed

EOF
    ;;
  3) cat <<'EOF'

  *Build path* — static, so it is a Kedro DAG

    documents.jsonl → normalise → select → embed → shard → write
                                                    `index_build`, 6 nodes

  *Query path* — dynamically branching with a shared budget counter,
  so it is a plain library behind Typer. A static DAG cannot express
  "expand only if the budget survived the last stage".

    question → S1..S4 → route → retrieve → explore → refine → exploit

  ─────────────────────────────────────────────────────────────────

  Registered pipelines:

    `corpus_ingest`   raw → intermediate → primary
    `index_build`     primary → shard artifacts        {__default__}
    `evaluation`      offline benchmark → `eval_report.json`
    `production`      corpus_ingest + index_build

EOF
    ;;
  4) cat <<'EOF'

  The post describes eight stages before retrieval. Four are built:

  *S1 language*      fastText `lid.176` detection → optional translation
                     Translation sits behind a `Protocol`; the offline
                     default is `NullTranslator`. Detection never raises —
                     an unreadable question still gets a search.

  *S2 tokenize*      NFKC → sentence split → casefold → segment →
                     stopwords → stem. pySBD · jieba · MeCab · PyStemmer
                     behind one class, one method, every language.

  *S3 intents*       intent extraction, function-word handling, scoring

  *S4 instruct*      E5-style instruction templates + writer

  ─────────────────────────────────────────────────────────────────

  Worked example from the post, reproduced exactly as a test:

    "What lessons from bacteria and yeast actually translate into
     safer gene-editing medicines?"
  → `('lesson','bacteria','yeast','actual','translat','safer',`
    `'gene','edit','medicin')`

EOF
    ;;
  5) cat <<'EOF'

  The original `text.py` tokenizer matched `[a-z0-9]+`.

  On a Russian or Japanese query that does not segment *badly* —
  it segments to *nothing*. Zero tokens. The lexical index then
  returns zero hits while every health check stays green.

  S2 uses `[^\W_]+` — every Unicode letter and digit, minus the
  underscore, splitting on hyphen and apostrophe.

  ─────────────────────────────────────────────────────────────────

  Splitting on the hyphen is also what makes the post's example
  come out right: its expected tokens contain `gene` and `edit`
  separately, so "gene-editing" *must* be two words.

  And stems, not lemmas — `translat` is not a lemma. A better
  lemmatiser would *fail* the worked example. The lexical index only
  needs query and document to collide on the same string.

EOF
    ;;
  6) cat <<'EOF'

  NOSIBLE is "a federation of 250,000 smaller search engines called
  shards". A query touches a handful, never all of them.

  `route()` scores every shard on up to five signals, fuses, reranks:

    dense              centroid cosine
    sparse             lexical overlap
    entity             only when the query has entities  ({None} otherwise)
    rerank_intent      bloom filter — has this shard seen the intent?
    rerank_compression zstd dictionary gain

                         ↓  weighted RRF

    shard ids best-first + `RoutingSignals` carrying every
    intermediate score, so a routing decision is inspectable
    rather than merely correct.

EOF
    ;;
  7) cat <<'EOF'

  *Bloom filter*   of important phrases. "If a selected shard hasn't
                   seen any of the user's search intents, that shard
                   should get downranked."

  *Zstd dict*      trained per shard. "The shard that can compress the
                   input question best is more likely to contain the
                   most similar content." Read as *gain vs. no dict* —
                   a raw length is dominated by question length.

  *Vocabulary*     unique terms, capped at 65,536 (the post quotes
                   64,992 for shard 11,343). The cap drops the rarest
                   terms — a news shard's tail is hapax legomena.

  ─────────────────────────────────────────────────────────────────

  {The trap:} rBloom hashes with `hash()`, which CPython salts per
  process. rBloom guards its own `save_bytes` against the builtin —
  but a caller's `lambda o: hash(o)` slips straight past the guard
  and silently produces different bits every run.

  So the hash is not a parameter. BLAKE2b-128 is pinned, recorded in
  the payload, and a payload written under another hash is refused.

EOF
    ;;
  8) cat <<'EOF'

    BM25Okapi  ─┐
                ├─→  weighted RRF  ─→  fused hits  ─→  snippets
    cosine     ─┘

    RRF(item) = Σ over rankers r of  w_r / (k + rank_r(item))

    `k = 60`   dense 1.0 · lexical 1.0 · entity 0.5   (tiny.yaml)

  Ties break by ascending item id, so ordering is deterministic
  rather than dependent on dict insertion order.

  `FusedItem` keeps per-ranker `contributions`, `ranks`, and
  `ranker_scores` — you can ask *why* a hit ranked where it did.

  Recent change: hybrid results are fused across *all* shards at once
  rather than per-shard-then-merge, which is what the map-reduce
  formulation in the post actually implies.

EOF
    ;;
  9) cat <<'EOF'

                   candidates  shards  hits/shard  survivors
    *Explore*             5      12          3          3
    *Refine*        UCT-ordered budget allocation
    *Exploit*       replay the winner

    exploration_constant `1.2`   ·   max_retrieval_calls `18`
    judge `heuristic`            ·   query_generator `heuristic`

  Every node is one query executed across its selected shard set,
  drawing on a *shared* budget counter. When the budget is gone,
  expansion stops mid-stage and the trace records that it did.

  ─────────────────────────────────────────────────────────────────

  {Honesty note, from the README itself:}
  "Staged beam search with UCT budget allocation (honestly not full
  MCTS)." There is no rollout and no backpropagation to a root value.

  Judge and generator are heuristic — token coverage and Jaccard.
  *Zero LLM calls* on the default path. Reward maxes at 0.95 by
  construction (0.45+0.20+0.20+0.10−0.05); 1.0 is unreachable.

EOF
    ;;
  10) cat <<'EOF'

  Target scale from the post's own ratio: *1,000,000 documents over
  1,000 shards*. Loaded eagerly, that is:

    every `Document` as a pydantic model
    every token list into one dict
    a `BM25Okapi` per shard at load time

    ≈ *20 GB of heap*  on a box with *5 GB* free.
    The index cannot be opened at all.

  `storage.py` supplies the three primitives that replace eager load:

    1. random access into JSONL by record id (`.npy` offset sidecar)
    2. memory-mapped embeddings, symmetric per-matrix int8
    3. a bounded LRU over the per-shard objects a query touches

  {Not a rewrite — the layer the loader is meant to sit on.}
  Existing JSONL artifacts stay byte-for-byte unchanged; the sidecar
  can be deleted and rebuilt at any time.

EOF
    ;;
  11) cat <<'EOF'

  *SQLite* for the document store
    Right call for a mutable corpus. Rejected: it swaps a plain-text,
    diffable, canonically-ordered artifact for an opaque binary whose
    page layout is not byte-stable across libsqlite versions — losing
    the byte-identical rebuild for a feature the build never needs.

  *Offsets as JSON*
    1M int64 offsets = 8 MB as `.npy`, or 20 MB of text that parses
    into ~30 MB of heap on every open. The `.npy` sidecar is mmapped;
    opening costs kilobytes.

  *One `.npz` bundle* for the sidecar
    One file instead of five — but `np.savez` writes a ZIP whose
    entries carry wall-clock time, so identical input yields
    different bytes every run.

  *Product quantisation* instead of scalar int8
    Far better bytes-per-vector, and what a billion-scale index would
    really use. Out of scope here.

EOF
    ;;
  12) cat <<'EOF'

  Same input → same bytes, across processes and machines.

    `seed: 42`               everywhere it can be threaded
    BLAKE2b-128              pinned; never a caller's `hash()`
    zstd `threads=0`         COVER search becomes a fixed computation
    zstd `dict_id=0`         id derived from content, not randomised
    sample order preserved   zstd training depends on it; callers pass
                             shard text in stored order (by id)
    tie-break by id          in RRF and in every ranked output
    no `.npz`                ZIP timestamps break byte-stability
    LRU is not thread-safe   {on purpose} — a lock would make hit/miss
                             counters depend on interleaving

  ─────────────────────────────────────────────────────────────────

  `tests/test_reproducibility.py` is the enforcement, not a comment.

EOF
    ;;
  13) cat <<'EOF'

  The query path is written so that no stage can be the thing that
  kills a search:

    unknown language tag      →  `"und"`, segment without morphology
    fastText failure          →  `UNKNOWN_LANGUAGE`, confidence 0.0
    blank question            →  `LanguageResult(UNKNOWN, 0.0)`
    no Snowball stemmer       →  identity function
    no stopword list          →  keep everything
    no `cjk` / `nlp` extra    →  regex fallback, jieba stays core
                                 (pure Python) while MeCab is optional
    no API key                →  `NullTranslator`

  Credentials are read at *call* time, never import time — importing
  the query path must work on a machine with no keys at all.

  `ANTHROPIC_API_KEY` is deliberately *not* consulted: Anthropic's API
  is not OpenAI-shaped at that base URL, so accepting it would buy a
  401 at the far end instead of the actionable error raised locally.

EOF
    ;;
  14) cat <<'EOF'

                        NOSIBLE Cybernaut-1     this replica
    ─────────────────────────────────────────────────────────────
    Index scale         250k shards             8–12 (tiny: 8)
    Build               8-stage pipeline        Kedro, 6 nodes
    Embedding           shard-specific LLMs     hash / model2vec / e5
    Similarity          Bayesian dense          exact cosine, float32
    Search tree         LLM-guided MCTS         staged beam + UCT
    Judge / generator   LLM                     heuristic, 0 LLM calls
    Entities            production NER          regex (spaCy optional)
    Retrieval budget    unspecified             5+9+4 = 18 calls
    Reward max          unspecified             0.95 by construction

  ─────────────────────────────────────────────────────────────────

  Default embedder moved to *model2vec* so shards cluster on meaning
  rather than on hash collisions — the hash embedder remains the
  offline, no-download path.

EOF
    ;;
  15) cat <<'EOF'

  `make install`        uv sync                    (no GPU, no downloads)
  `make check`          ruff + mypy --strict + pytest
  `make doctor`         verify the container end to end

  `cybernaut-mini build      --config configs/tiny.yaml --offline`
      8 shards, hash embedder; fixture slice committed at data/01_raw/fixtures/

  `cybernaut-mini inspect-shards --index artifacts/fixture`
  `cybernaut-mini search  --mode hybrid|agent --trace-out run.json`
  `cybernaut-mini eval    --judgments data/01_raw/fixtures/judgments.jsonl`

  `make ingest` · `make build-prod` · `make eval-pipeline` · `make viz`
      the same work through Kedro; `build-prod` *aborts* unless
      `embedding.revision` is pinned in `conf/prod`

  `make notebooks`      every notebook executed against the real
                        catalog, so one cannot silently drift

EOF
    ;;
  16) cat <<'EOF'

  *Committed and working*
    build · shard · route · retrieve · fuse · agent search · eval
    three Kedro pipelines, notebooks under test, model2vec default

  *In the working tree, uncommitted*
    `query/` S1–S4        Hybrid-3 stages 1–4 of 8
    `storage.py`          mmap / offsets / int8 / LRU
    `shard_artifacts.py`  bloom · zstd dict · vocabulary
    six new test modules  `test_s1..s4` · `test_storage` ·
                          `test_shard_artifacts`

  *Not yet wired*
    the loader still loads eagerly — `storage.py` is the layer it
    is meant to sit on, and does not yet sit on
    Hybrid-3 stages 5–8
    corpus scale: stages 5–7 are meaningless below a few hundred
    shards, and the box has 7 GB RAM / 8 cores / 386 GB disk

EOF
    ;;
  17) cat <<'EOF'


        The interesting part of this repo is not that it retrieves.

        It is that every module says *which paragraph it replicates*,
        *what it assumed where the post was silent*, and *what it
        rejected and why* — and the tests hold it to all three.


        ─────────────────────────────────────────────────────

        `README.md`                        the honest comparison table
        `docs/REVERSE_ENGINEERING_GUIDE.md`
        `docs/blog-archive/`               the two source posts
        `.omc/wiki/`                       18 pages of session notes

EOF
    ;;
  esac
}

# ------------------------------------------------------------------------- #
# Rendering                                                                   #
# ------------------------------------------------------------------------- #

render_slide() {
  local idx=$1
  local kicker="${SLIDE_KICKERS[idx]}" title="${SLIDE_TITLES[idx]}"
  local right="$((idx + 1))/${TOTAL}"

  printf '\n'
  printf '%scybernaut-mini%*s%s\n' "$GREY" "$((WIDTH - 14))" "$right" "$RESET"
  printf '%s%s%s\n\n' "$GREY" "$(repeat "$WIDTH" '━')" "$RESET"

  if [[ -n "$kicker" ]]; then
    printf '  %s%s%s  %s%s%s%s\n\n' \
      "$MAGENTA" "$BOLD" "$kicker" "$RESET" "$BOLD$BCYAN" "$title" "$RESET"
  else
    printf '  %s%s%s\n\n' "$BOLD$BCYAN" "$title" "$RESET"
  fi

  slide_body "$idx" | apply_markup
}

render_footer() {
  local idx=$1 hint filled
  if ((idx == 0)); then
    hint="space/n next · p back · g first · G last · q quit"
  elif ((idx == TOTAL - 1)); then
    hint="p back · g first · q quit"
  else
    hint="space/n next · p back · q quit"
  fi
  filled=$(((idx + 1) * 40 / TOTAL))

  printf '\n%s%s%s\n' "$GREY" "$(repeat "$WIDTH" '─')" "$RESET"
  printf '  %s%s%s%s%s  %s%s%s\n' \
    "$BCYAN" "$(repeat "$filled" '█')" "$RESET" \
    "$GREY" "$(repeat $((40 - filled)) '░')" \
    "$GREY" "$hint" "$RESET"
}

clear_screen() {
  if ((USE_COLOR)); then printf '\033[2J\033[H'; else printf '\n\n\n'; fi
}

# ------------------------------------------------------------------------- #
# Input                                                                       #
# ------------------------------------------------------------------------- #

read_key() { # echoes a normalised key name
  local key rest
  IFS= read -rsn1 key || { printf 'q'; return; }
  if [[ $key == $'\033' ]]; then
    IFS= read -rsn2 -t 0.05 rest
    case "$rest" in
    '[C' | '[B') printf 'n' ;;
    '[D' | '[A') printf 'p' ;;
    *) printf '\033' ;;
    esac
    return
  fi
  [[ -z $key ]] && key='n' # bare Enter
  printf '%s' "$key"
}

# ------------------------------------------------------------------------- #
# Main                                                                        #
# ------------------------------------------------------------------------- #

ALL=0
START=1
while (($#)); do
  case "$1" in
  --all | -a) ALL=1 ;;
  --no-color) USE_COLOR=0 ;;
  --start | -s)
    START="${2:-1}"
    shift
    ;;
  -h | --help)
    sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    printf 'preso.sh: unknown option %s\n' "$1" >&2
    exit 2
    ;;
  esac
  shift
done

set_palette

cols=$(tput cols 2>/dev/null || printf 80)
((cols - 2 < WIDTH)) && WIDTH=$((cols - 2))
((WIDTH < 60)) && WIDTH=60

if ((ALL)) || [[ ! -t 0 ]]; then
  for ((i = 0; i < TOTAL; i++)); do
    render_slide "$i"
    printf '\n'
  done
  exit 0
fi

idx=$((START - 1))
((idx < 0)) && idx=0
((idx > TOTAL - 1)) && idx=$((TOTAL - 1))

while :; do
  clear_screen
  render_slide "$idx"
  render_footer "$idx"

  key=$(read_key)
  case "$key" in
  q | $'\003' | $'\004') break ;;
  n | ' ' | j | l | '')
    ((idx == TOTAL - 1)) && break
    ((idx++))
    ;;
  p | b | k | h | $'\177') ((idx > 0)) && ((idx--)) ;;
  g) idx=0 ;;
  G) idx=$((TOTAL - 1)) ;;
  [1-9])
    idx=$((key - 1))
    ((idx > TOTAL - 1)) && idx=$((TOTAL - 1))
    ;;
  esac
done

clear_screen
printf '  %s%scybernaut-mini%s%s  ·  end of deck%s\n\n' \
  "$BOLD" "$BCYAN" "$RESET" "$GREY" "$RESET"
