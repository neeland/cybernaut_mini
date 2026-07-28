# NOSIBLE Blog Archive

Local markdown archive of NOSIBLE's public technical blog, fetched 2026-07-28 via
the Exa `/contents` API for **educational study only**. All content is © NOSIBLE;
these files are unmodified article text with a provenance front-matter header.

This project (`cybernaut_mini`) is an unaffiliated, clean-room educational replica.
Nothing here is scraped private data — every post is publicly linked from
<https://nosible.com/blog> and its Ghost mirror.

## Posts

| File | Title | Layer |
|---|---|---|
| `the-road-to-cybernaut-1.md` | The Road to Cybernaut-1: Rebuilding Search for AI | Search core (8-stage pipeline, 250k shards) |
| `introducing-cybernaut-1-agentic-search-with-mcts.md` | Introducing Cybernaut-1: Agentic Search using MCTS | Agentic layer (LLM-guided MCTS) |
| `can-faceted-search-at-web-scale-self-organize.md` | Can Faceted Search at Web-Scale Self Organize? | Self-organizing entity tagging |
| `point-in-time-knowledge-graphs-over-named-entities.md` | Point-in-Time Knowledge Graphs with NOSIBLE World | Knowledge graphs (lift over NER) |
| `using-vector-search-to-see-signals-in-company-news.md` | Using Vector Search to See Signals in Company News | Vector substrate (LSH + weekly signals) |
| `the-contrastive-geometry-of-risk.md` | Two Tricks for Clean Features from Sentence Embeddings | Frozen-embedding classification |
| `ensemble-and-distil.md` | Distilling LLM Ensembles into Better Sentiment Models | Ensemble → distil pattern |
| `fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction.md` | Matching GPT-5.1 at Financial Sentiment with Active Learning and Qwen3 | Tiny fine-tuned classifiers |
| `news-sentiment-showdown-who-checks-vibes-best.md` | News Sentiment Showdown: Who Checks Vibes Best? | Sentiment model benchmark |
| `rebuilding-the-geopolitical-risk-index-from-nosible-world.md` | Rebuilding the Geopolitical Risk Index with NOSIBLE | Quant signal (GPR replica) |
| `turning-news-into-a-risk-on-risk-off-equity-signal.md` | Turning News into a Risk-On/Risk-Off Equity Signal | Quant signal (market stress overlay) |
| `an-embedding-based-approach-to-trade-and-economic-policy-uncertainty.md` | Trade & Economic Policy Uncertainty from News Embeddings | Quant signal (TPU/EPU replica) |

## Where to go next

See [`../REVERSE_ENGINEERING_GUIDE.md`](../REVERSE_ENGINEERING_GUIDE.md) — a clean-room
build guide that distills every technique disclosed above into open-tool recipes,
marks each step `[disclosed]` vs `[inferred]`, and lists the deliberate gaps.
