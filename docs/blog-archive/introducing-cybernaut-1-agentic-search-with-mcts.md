---
title: "Introducing Cybernaut-1: Agentic Search using MCTS"
source: https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts
published: 2025-08-26T00:00:00.000Z
fetched: 2026-07-28
archived-by: exa /contents (educational archive)
---

# Introducing Cybernaut-1: Agentic Search using MCTS

> Archived from <https://nosible.com/blog/introducing-cybernaut-1-agentic-search-with-mcts> for educational study. All content © NOSIBLE.

Introducing Cybernaut-1: Agentic Search using MCTS | NOSIBLE

Loading

ASK NOSIBLE

Introducing Cybernaut-1: Agentic Search using MCTS | NOSIBLE
[Skip to content](#main)
NEWAn Embedding-Based Approach to Trade and Economic Policy Uncertainty.[Read the post→](https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty)
[![NOSIBLE](https://nosible.com/logos/nosible-logo.svg)](https://nosible.com/#top)
[](https://nosible.com/search)[ENTER WORLD](https://nosible.world/world)[START TRIAL](https://nosible.com/start-trial)
Loading
ASK NOSIBLE
[cybernaut-1](https://nosible.com/blog/tag/cybernaut-1)
# Introducing Cybernaut-1: Agentic Search using MCTS
[Stuart Reid](https://www.linkedin.com/in/stuartgordonreid/)
2025-08-26
2 min read
Copy as Markdown
![Cybernaut-1 illustration representing agentic search with Monte Carlo Tree Search](https://nosible.com/blog/illustrations/cyber.png)
Today I am proud to announce the release of Cybernaut-1. Cybernaut-1 combines our powerful hybrid-3 search algorithm with LLM-guided [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) to deliver world class search results on difficult queries. Cybernaut-1 is available via our [V2 Search API](https://docs.nosible.com) and our [Python Package](https://github.com/NosibleAI/nosible-py).
```python
from nosible import Nosible

with Nosible(nosible_api_key="YOUR API KEY HERE") as nos:
  print(nos.search(prompt="Find technical blogs about Monte Carlo Tree Search"))
```
## We trust Cybernaut-1 with our signals
Cybernaut-1 is what we call a high-trust agentic search algorithm. What that means is that Cybernaut-1 has *direct access* to the internal logic in NOSIBLE. Our recent blog – ["The Road to Cybernaut-1: Rebuilding Search for AI"](https://nosible.com/blog/the-road-to-cybernaut-1) – goes into a lot of detail about what that internal logic encompasses.
![Cybernaut-1 concept diagram showing an agentic search system with access to NOSIBLE retrieval logic](https://nosible.com/images/2025/08/nosible-cybernaut-concept-2.png)Cybernaut-1 concept diagram showing an agentic search system with access to NOSIBLE retrieval logic
We trust Cybernaut-1 with ALL the signals from NOSIBLE
## Cybernaut-1 uses them to self-improve
Cybernaut-1 uses LLM-guided [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) to iteratively construct a high-quality search that aligns with your given prompt. It balances exploration, exploitation, and inference cost by slowly moving from wide and shallow searches to narrow and deep searches. This approach is illustrated below:
![Cybernaut-1 search algorithm diagram showing LLM-guided Monte Carlo Tree Search from broad to focused queries](https://nosible.com/images/2025/08/nosible-cybernaut-algorithm.png)Cybernaut-1 search algorithm diagram showing LLM-guided Monte Carlo Tree Search from broad to focused queries
Cybernaut-1 uses those signals to iterative and improve.
## So that you always get the best results
Next week, we will be open-sourcing a comprehensive set of evaluations showing how Cybernaut-1 as well as Hybrid-3, the algorithm it uses, consistently match or outperforms leading search engines, even as we continue expanding our web coverage (currently growing at \~20 million webpages per day).
In the meantime, we’d love for you to try it out. We are offering **$10,000 in Cybernaut-1 credits** to the first 20 AI startups that sign up early.
[Building AI and looking for better context? Let's chat!](https://calendly.com/operations-nosible/)
P.S. Looking for more of a technical deep dive? Go check out our blog: [*"The Road to Cybernaut-1: Rebuilding Search for AI*](https://nosible.com/blog/the-road-to-cybernaut-1)*".* Or, alternatively, check out:
1. [**Our Official Docs**](https://nosible-py.readthedocs.io/en/latest/) - how to use the Python package.
2. [**Our Python Package**](https://github.com/NosibleAI/nosible-py) - simply pip install nosible.
3. [**Our API documentation**](https://docs.nosible.com) - For non-Python users.
[All Research](https://nosible.com/blog)
Related Research
![Railway-track illustration representing the road to Cybernaut-1](https://nosible.com/blog/illustrations/track.png)
[cybernaut-1](https://nosible.com/blog/tag/cybernaut-1)[Technical](https://nosible.com/blog/tag/technical)
### [The Road to Cybernaut-1: Rebuilding Search for AI](https://nosible.com/blog/the-road-to-cybernaut-1)
2025-08-2017 min read
![NOSIBLE World knowledge graph showing entity connections over a decade](https://nosible.com/images/2026/07/kg-hero-decade.png)
[Research](https://nosible.com/blog/tag/research)[Knowledge Graphs](https://nosible.com/blog/tag/knowledge-graphs)[Nosible World](https://nosible.com/blog/tag/nosible-world)
### [Point-in-Time Knowledge Graphs over Named Entities with NOSIBLE World](https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities)
2026-07-168 min read
![Signed contrast number line separating systemic and idiosyncratic risk](https://nosible.com/images/2026/06/walkthrough_step9_number_line.png)
[Research](https://nosible.com/blog/tag/research)[Technical](https://nosible.com/blog/tag/technical)[Classification](https://nosible.com/blog/tag/classification)[Nosible World](https://nosible.com/blog/tag/nosible-world)
### [Two Tricks for Turning Sentence Embeddings into Clean Features](https://nosible.com/blog/the-contrastive-geometry-of-risk)
2026-06-1814 min read
