---
title: "Rebuilding the Geopolitical Risk Index with NOSIBLE"
source: https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world
published: 2026-06-06T00:00:00.000Z
fetched: 2026-07-28
archived-by: exa /contents (educational archive)
---

# Rebuilding the Geopolitical Risk Index with NOSIBLE

> Archived from <https://nosible.com/blog/rebuilding-the-geopolitical-risk-index-from-nosible-world> for educational study. All content © NOSIBLE.

Rebuilding the Geopolitical Risk Index with NOSIBLE | NOSIBLE

Loading

ASK NOSIBLE

Rebuilding the Geopolitical Risk Index with NOSIBLE | NOSIBLE
[Skip to content](#main)
NEWAn Embedding-Based Approach to Trade and Economic Policy Uncertainty.[Read the post→](https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty)
[![NOSIBLE](https://nosible.com/logos/nosible-logo.svg)](https://nosible.com/#top)
[](https://nosible.com/search)[ENTER WORLD](https://nosible.world/world)[START TRIAL](https://nosible.com/start-trial)
Loading
ASK NOSIBLE
[Research](https://nosible.com/blog/tag/research)[Geopolitical Risk](https://nosible.com/blog/tag/geopolitical-risk)[Nosible World](https://nosible.com/blog/tag/nosible-world)
# We Rebuilt the Geopolitical Risk Index with Nosible World
[Matthew Dicks](https://www.linkedin.com/in/matthewdicks98/)
2026-06-06
15 min read
Copy as Markdown
![NOSIBLE geopolitical risk signal compared with published geopolitical risk indices](https://nosible.com/images/2026/06/nosible-gpr-vs-published.png)
Markets move on geopolitics. A war, a coup, or a blockade moves oil, equities, and currencies within hours.
But risk models cannot read the news. They take numbers, not headlines. So "the world feels dangerous right now" never reaches the model.
We fixed that. We turned the Nosible World database into a single geopolitical risk signal, and it matches the benchmark that Federal Reserve economists built. It also matches the newer, harder version of that benchmark: the same risk scores, broken out for every country, every pair of countries, and for oil supply. We get there from data we already hold, in every language, without running a language model over every article.
## The problem: risk you can feel but cannot measure
Every investor and policymaker knows geopolitics moves markets. The hard part is that geopolitical risk is a feeling, not a figure. There is a number for inflation, a number for unemployment, and a number for market volatility. There was never a number for how dangerous the world looks this month. Without one, the single force everyone agrees matters could not enter a forecast, a risk model, or a backtest.
In 2018, two Federal Reserve economists, Dario Caldara and Matteo Iacoviello, [built that number](https://www.federalreserve.gov/econres/ifdp/files/ifdp1222.pdf). Their [Geopolitical Risk index](https://www.matteoiacoviello.com/gpr.htm) reads the daily newspaper record, measures how much of it is about wars, threats, and terrorism, and turns that into one series you can chart, compare, and test. For the first time, geopolitical risk became a variable you could put in a model.
It turned out to matter. In the authors' own work, a rise in the index comes before lower business investment and hiring, weaker stock returns, and capital moving out of emerging markets toward safer ground. That is why it spread beyond academia and into the institutions that price risk for a living:
* The **Federal Reserve** built the index and maintains it.
* The **IMF** uses it in its [Global Financial Stability Report](https://www.imf.org/-/media/files/publications/gfsr/2025/april/english/ch2.pdf) to study how geopolitical shocks move stock prices, bond yields, and bank exposures, and in its [World Economic Outlook](https://www.imf.org/-/media/files/publications/weo/2026/april/english/ch1.pdf), where the country-level index tracks regional risk inside the global forecast.
* The **European Central Bank and the European Systemic Risk Board** build it into financial-stability stress scenarios, bank-lending risk models, and growth-at-risk frameworks, in their [January 2026 fragmentation report](https://www.ecb.europa.eu/pub/pdf/other/ecb.report202601_financialstabilityrisks.en.pdf).
The use cases are just as broad. The index has been used to forecast investment and growth, to design stress scenarios, to price equity and credit risk, to identify oil supply shocks, and to track capital flight from emerging markets.
So the measure is valuable and trusted. The question we asked is whether the way it is built leaves room to do better.
## What we are matching
There are two published versions, both from the economists who created this field, and we test against both.
**The original Geopolitical Risk index** ([Caldara and Iacoviello, 2018](https://www.federalreserve.gov/econres/ifdp/files/ifdp1222.pdf)) counts geopolitical risk words across ten major newspapers, as a share of all articles published. It is the long-running benchmark, and it reaches back over a century.
**The AI-GPR index** ([Iacoviello and Tong, 2026](https://www.matteoiacoviello.com/research_files/AI_GPR_PAPER.pdf)) is the modern successor. Instead of counting words, it asks a language model to read each article and score how much real geopolitical risk it carries, from zero to one. That removes most of the miscounting that word-matching suffers from, and it lets the authors add cuts the original never had: a score for each country, for each pair of countries, and for oil supply.
The AI-GPR index is the harder target, and it is the one we hold ourselves to. It is the most accurate version, and it already publishes the country, pair, and oil breakdowns we want to reproduce. Matching those, from the data we already hold, is the bar we set.
## The solution: one number from the world's news
The Nosible World database already does the hard part. Every event is tagged with its topic, its main country, the geopolitical entities it names (extracted by named-entity recognition), and how many separate publishers covered it. One real event is one record, no matter how many outlets repeat it.
The database covers 2010 to today. This study uses the window from 2019 onward, where the published indices overlap, across 13.2 million de-duplicated events.
That makes the index simple to build. Every event is tagged with one topic from an ontology over the events: the IPTC Media Topics ontology, a structured three-level hierarchy that classifies what each event is about. Its top level has buckets such as `conflict, war and peace`, `economy, business and finance`, and `health`, each splitting into finer levels below. We treat an event as geopolitical with an exact filter on three ontology fields carried by every event (`iptc\_level\_1`, `iptc\_level\_2`, `iptc\_level\_3`):
```
geopolitical(e) is TRUE when ANY of these hold:
    iptc_level_1 == "conflict, war and peace"
    iptc_level_2 == "international relations"
    iptc_level_3 in { "war crime", "genocide", "terrorism", "nuclear policy" }
```
The level-1 bucket is the dense spine (armed conflict, terrorism, coups, civil unrest); the level-2 and level-3 additions bring in cross-border politics and a few high-precision leaves that sit outside that bucket. We match the ontology codes exactly, never the words in the labels, so "weather warning" or "tug-of-war" never leak in.
Write `breadth(e)` for the number of distinct publishers that covered an event `e`, which is the `total\_netlocs` field carried on every event. The index, each day, is the share of total publisher attention spent on geopolitical events:
```
                  sum of breadth(e) over geopolitical events on day t
NOSIBLE-GPR(t) = -----------------------------------------------------
                       sum of breadth(e) over all events on day t
```
Coverage breadth is the weight: a story carried by 200 outlets counts as one event weighted by 200, never as 200 separate events. For the charts we rescale each series to average 100 over 2020 to 2024, which makes them comparable and leaves every correlation unchanged. With Nosible World this is one group-by over the event table: filter to the conflict topics, sum `total\_netlocs` for the numerator, and divide by the same sum over all events, per day.
## It matches the benchmark
We put our index next to the published series, on the same scale. We show two versions of our signal. Both are the same share of publisher attention spent on geopolitical events, and differ only in what they divide by. The first divides by total attention that same day. The second divides by a trailing 12-month average of total attention.
The second version exists because our news corpus grew significantly from 2019 to 2026, and it broadened across topics as it grew. So the share spent on any one theme drifts down over the years even when the world is no calmer, which quietly flattens the most recent period. The benchmark runs on a small, stable set of newspapers and has no such drift. Dividing by a trailing 12-month baseline removes ours, so 2026 is measured on the same footing as 2019. This detrended version is the one we carry through the rest of this post, for every country, every pair, and oil.
![Nosible geopolitical risk signal, raw daily share and 12-month detrended, against the published Federal Reserve and AI versions, 2019 to 2026](https://nosible.com/images/2026/06/nosible-gpr-vs-published.png)Nosible geopolitical risk signal, raw daily share and 12-month detrended, against the published Federal Reserve and AI versions, 2019 to 2026
Every event that should appear, appears: Soleimani in 2020, the invasion of Ukraine in 2022, Israel and Gaza in 2023, Iran and Israel in 2024, and the US-Iran war in 2026.
Both versions track the AI-GPR index closely: the raw daily share at 0.90 on the levels and 0.79 on the stricter month-to-month changes, and the detrended version at 0.89 and 0.75. They are deliberately close, because the detrend changes the slope of the baseline, not which events are geopolitical. The chart shows where they part: the detrended line holds the 2026 surge up near the benchmark, while the raw share sags as the corpus swells. The two published versions agree with each other at about the same level, and we reach it on different data, with no keyword list.
**What you get:** a live geopolitical risk series, updated automatically, with no analyst in the loop.
## Risk for every country
You can also measure each country on its own, not just the world total.
Each event already carries one main country: the country it is mainly about. But most geopolitical events involve more than one country, and attributing an event only to its main country misses the rest. An event whose main country is Lebanon is often just as much about Israel. An event whose main country is the United States may be about sanctions on Iran.
So we use every country the event names, not just its main one. Each event already lists its geopolitical entities (countries, cities, regions) in the `ent\_gpe` field, extracted by named-entity recognition. We map those entity strings to countries and attribute the event to all of them, which means handling their many forms:
* Names and abbreviations (United States, U.S., USA).
* Nationalities (American, Russian, Israeli).
* Other languages and scripts (中国, Россия, ישראל).
We match each entity whole-string and case-insensitive against a table of country names, official aliases, nationalities of four letters or more, and native-language names. A partial match never counts, so `Indiana` never resolves to India, and genuinely ambiguous strings such as a bare `Georgia` (the US state) are dropped.
This is the difference between a weak score and a strong one. The United States is named in roughly a third of all geopolitical news, so attributing each event to its single main country alone misses most of where the US actually appears. Counting it everywhere it is named lifts its score from 0.45 to 0.83.
Per country, the index is the same publisher-attention share, attributed to every country an event names:
```
attribution(e) = { the event's main country } + { countries resolved from its named entities }

sum of breadth(e) over geopolitical events in month m
                                   where country c is in attribution(e)
GPR(country c, month m) = -----------------------------------------------------------
                                                B(m)
```
`B(m)` is one global denominator shared by every country: the trailing 12-month average of total monthly breadth across all events. We divide by this global figure, not by a country's own coverage, because in a crisis a country's own coverage spikes too and would cancel the signal; the trailing average also stops the corpus growing over time from masking real spikes. To build it: explode each event to its attribution set, group by country and month, and divide by `B(m)`.
![Per-country signal, same-day and 12-month detrended, against the published country indices for Russia, Israel, Ukraine, Iran, the USA, and India](https://nosible.com/images/2026/06/nosible-gpr-by-country.png)Per-country signal, same-day and 12-month detrended, against the published country indices for Russia, Israel, Ukraine, Iran, the USA, and India
The results hold across the major actors: Iran 0.99, Israel 0.96, Ukraine 0.96, Russia 0.94. Seventy-six countries score above 0.60. The AI-GPR index needs an extra language-model pass over every article to do this. We get it from tags the data already carries.
**What you get:** a risk monitor for any country, built the same way as the global one.
## Risk for every country pair
Because each event already carries every country named in it, we also know which two appear together. That gives a risk score for any pair of countries.
![Bilateral signal, same-day and 12-month detrended, for major country pairs against the published bilateral series](https://nosible.com/images/2026/06/nosible-gpr-bilateral.png)Bilateral signal, same-day and 12-month detrended, for major country pairs against the published bilateral series
From the same attribution sets, take every unordered country pair an event names:
```
                             sum of breadth(e) over geopolitical events in month m
                                    that name BOTH country a and country b
GPR(pair a-b, month m) = -----------------------------------------------------------
                                                B(m)
```
Same `B(m)` as the country index. To build it: self-join each event's attribution set into unordered pairs, then group by pair and month.
The major conflict pairs come through clearly: Iran and the USA at 0.97, Russia and Ukraine at 0.95, India and Pakistan at 0.95. Country-pair risk is the newest part of the AI-GPR index. We reproduce it as a by-product of the country work.
One pair does not match: China and the USA, at 0.28. The reason is precise, and it comes back to that one topic per event. A tariff story gets the topic "international trade," which sits in the economy branch of the ontology, not the conflict branch our filter reads, so the filter never sees it. We cannot just add the trade topic either, because it is mostly routine commerce and would drown the signal. This is a real limit, and it has a clean fix. We close it in the next section, and the score climbs from 0.28 to 0.75.
**What you get:** a tension tracker for any two countries, useful for trade, supply chains, and exposure.
## Closing the trade-war gap
The China-USA gap has a clean fix, and it does not touch the topic filter. The filter misses tariffs because each event carries only one topic, but every event also carries an embedding, a numeric summary of its meaning. We can read that directly.
We wrote one phrase for trade coercion: a government imposing tariffs, duties, or export controls on another country, and the threats of retaliation that follow. We compute the cosine similarity between the phrase's embedding and every event's embedding to get a trade-coercion score, and let an event into the index if it is geopolitical *or* it clears that score. Nothing else changed: the same publisher-breadth weight, the same country attribution, the same denominator.
```
trade(e) = cosine similarity between event e's embedding and this trade-coercion anchor phrase:
   "Tariffs, trade wars and economic coercion between countries: a government imposing import
    tariffs, retaliatory duties, export controls or other trade restrictions on another country,
    and the diplomatic tensions and threats of retaliation these trigger"

sum of breadth(e) over month m, country c in attribution(e),
                                   where geopolitical(e) OR trade(e) >= 0.40
GPR(country c, month m) = ------------------------------------------------------------------
                                                B(m)
```
The only change from the country index is the `OR trade(e) >= 0.40`: an event now counts if it is geopolitical or it clears the trade-coercion score. Everything else is identical, which is why the conflict countries do not move.
![China and the China-USA pair, before and after folding in trade coercion. AI-GPR in amber, the Nosible baseline in grey, the Nosible version with tariffs in green](https://nosible.com/images/2026/06/nosible-tariff-recovery.png)China and the China-USA pair, before and after folding in trade coercion. AI-GPR in amber, the Nosible baseline in grey, the Nosible version with tariffs in green
It closes the gap. China's country score rises from 0.53 to 0.78, and the China-USA pair rises from 0.28 to 0.75. It takes only 8,079 added events across the entire corpus, because one real trade-war event, weighted by how many outlets cover it, carries the spike. The conflict-driven countries do not move: Ukraine, Russia, Israel, and Iran stay where they were. The April 2025 tariff spike, missing before, now appears.
**What you get:** the same method, pointed at a different gap, with no new model to train.
## Risk to oil supply
Oil is the headline application in the AI-GPR paper, so we follow their method closely and compare to it directly.
Oil needs care because the direction is not obvious. Broad geopolitical risk usually *lowers* the oil price, because fear cuts demand. Risk inside oil-producing regions *raises* it, because it threatens supply. A useful signal has to isolate the supply side.
The paper does this in two steps. It keeps the geopolitical articles, filters them to the ones that mention oil, then asks a language model whether each one describes an oil supply disruption, and in which region.
Our topic filter cannot do this on its own. Each event gets a single topic, so a Gulf war is tagged conflict, never energy. That one label can tell us a story is geopolitical, or that it is about oil, but never both at once.
So we use the same second signal that closed the trade-war gap: the event embedding. We score each event against three oil-supply-risk phrases the same way, and keep the events that are both geopolitical and above the relevance floor. The embedding does the work of the paper's keyword filter and its disruption model in one step, with no extra model call.
```
relevance(e) = highest cosine similarity between event e's embedding and these three
               oil-supply-risk anchor phrases:
   "Armed conflict and war in or around major oil-producing regions"
   "Crude oil supply: OPEC production quotas, output levels and disruptions to oil production or exports"
   "Sanctions, embargoes and export restrictions on major oil-exporting countries such as Iran, Russia and Venezuela"

sum of relevance(e) x breadth(e) over events in month m that are
                   geopolitical AND have relevance(e) >= 0.30
Oil-GPR(m) = ----------------------------------------------------------------------
                                       B(m)
```
To build it: score every event's stored embedding against the anchor phrases, keep the geopolitical events above the relevance floor, weight each by relevance times breadth, and divide by `B(m)`. The per-region and per-country versions attribute each surviving event to its producer regions or countries first, exactly like the country index.
The paper's main oil chart plots its index against the real oil price. Here is ours, with the published academic version on the same axis.
![Nosible Oil-GPR, 12-month detrend in green and same-day in blue, with the published academic Oil-GPR in amber and the WTI oil price in grey. After Iacoviello and Tong, 2026, Figure 4](https://nosible.com/images/2026/06/nosible-oil-gpr-vs-wti.png)Nosible Oil-GPR, 12-month detrend in green and same-day in blue, with the published academic Oil-GPR in amber and the WTI oil price in grey. After Iacoviello and Tong, 2026, Figure 4
The index lines sit almost on top of each other. Our oil signal reproduces the published academic version closely under both denominators: the same-day share at 0.95, and the 12-month detrended version, the one we carry throughout, at 0.97.
The paper also breaks the oil signal down by producer region. We reproduce that, and check each region against the academic version directly.
![Per-region oil-supply risk: the published academic version in amber, the Nosible version in green, by producer region, monthly and z-scored](https://nosible.com/images/2026/06/nosible-oil-gpr-by-region.png)Per-region oil-supply risk: the published academic version in amber, the Nosible version in green, by producer region, monthly and z-scored
The major producers line up closely: the Middle East at 0.95, Venezuela at 0.96, the United States at 0.92, Russia at 0.91. Russia's signal jumps in 2022 with the invasion of Ukraine. Two regions stay weak: Africa at 0.44 and the North Sea at 0.04, where the published series is itself close to noise over this window.
We are careful about one claim. We match the academic risk index, not the oil price itself. The paper goes further and shows that an oil-supply shock pushes the oil price up and output down. We leave demonstrating that from the Nosible signal to future work.
**What you get:** an oil-supply risk signal that reproduces the academic benchmark, broken down by region.
## One method, many signals
This is not one index. It is a template, and it runs on two engines.
The first is the ontology over the events. The same recipe builds a signal for any subject the ontology classifies:
* Economic policy uncertainty.
* Stock-market volatility, against the VIX.
* Climate concern and pandemic risk.
The second is the meaning of the text. When a subject does not fit one topic, a phrase captures it instead, in every language at once. We have now done this twice, for oil supply and for trade coercion, with the same few lines of code. It extends to any traded asset:
* Gold, natural gas, wheat, copper.
* Freight and shipping.
* Individual currencies.
Each one is a use case: idea generation, risk monitoring, or a clean input for your own models. The number of signals you can build is effectively unlimited.
We rebuilt one of the most widely used risk measures in finance and matched it across every cut its authors publish: the global index, the country breakdown, the country pairs, and oil supply. We did it from one multilingual database, with no language model reading each article. The same approach now points at every other index on the shelf.
Nosible turns the world's news into a structured, multilingual, de-duplicated event database, and this post used one slice of it. If you want access to that database, or a signal like these built for your own models, [start a trial](https://nosible.com/start-trial). You can explore the live data at [nosible.world](https://nosible.world).
## References
* Caldara, Dario and Matteo Iacoviello. [Measuring Geopolitical Risk](https://www.federalreserve.gov/econres/ifdp/files/ifdp1222.pdf). Board of Governors of the Federal Reserve System, International Finance Discussion Paper No. 1222 (2018); published version in the *American Economic Review* 112(4), 2022. Index and data: [matteoiacoviello.com/gpr](https://www.matteoiacoviello.com/gpr.htm).
* Iacoviello, Matteo and Jonathan Tong (2026). [The AI-GPR Index: Measuring Geopolitical Risk using Artificial Intelligence](https://www.matteoiacoviello.com/research_files/AI_GPR_PAPER.pdf). Federal Reserve Board working paper. Overview and data: [matteoiacoviello.com/ai\_gpr](https://www.matteoiacoviello.com/ai_gpr.html).
* International Monetary Fund (2025). [Global Financial Stability Report, April 2025, Chapter 2: Geopolitical Risks and Their Implications for Asset Prices and Financial Stability](https://www.imf.org/-/media/files/publications/gfsr/2025/april/english/ch2.pdf). Measures geopolitical risk with the Caldara and Iacoviello (2022) indices, and estimates GPR betas, sovereign-yield and bank-exposure responses, and downside risk to stock returns.
* International Monetary Fund (2026). [World Economic Outlook, April 2026, Chapter 1](https://www.imf.org/-/media/files/publications/weo/2026/april/english/ch1.pdf). Uses the global and country-specific Caldara and Iacoviello geopolitical risk indices to track regional risk and estimate the macroeconomic effects of geopolitical shocks.
* European Central Bank and European Systemic Risk Board (2026). [Financial Stability Risks from Geoeconomic Fragmentation, January 2026](https://www.ecb.europa.eu/pub/pdf/other/ecb.report202601_financialstabilityrisks.en.pdf). Uses the Caldara and Iacoviello GPR index in its geopolitical-shock scenarios and VAR models.
[All Research](https://nosible.com/blog)
Related Research
![NOSIBLE World knowledge graph showing entity connections over a decade](https://nosible.com/images/2026/07/kg-hero-decade.png)
[Research](https://nosible.com/blog/tag/research)[Knowledge Graphs](https://nosible.com/blog/tag/knowledge-graphs)[Nosible World](https://nosible.com/blog/tag/nosible-world)
### [Point-in-Time Knowledge Graphs over Named Entities with NOSIBLE World](https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities)
2026-07-168 min read
![Signed contrast number line separating systemic and idiosyncratic risk](https://nosible.com/images/2026/06/walkthrough_step9_number_line.png)
[Research](https://nosible.com/blog/tag/research)[Technical](https://nosible.com/blog/tag/technical)[Classification](https://nosible.com/blog/tag/classification)[Nosible World](https://nosible.com/blog/tag/nosible-world)
### [Two Tricks for Turning Sentence Embeddings into Clean Features](https://nosible.com/blog/the-contrastive-geometry-of-risk)
2026-06-1814 min read
![Daily NOSIBLE Trade Policy Uncertainty index compared with the published index](https://nosible.com/images/2026/06/nosible-tpu-daily.png)
[Research](https://nosible.com/blog/tag/research)[Trade Policy](https://nosible.com/blog/tag/trade-policy)[Nosible World](https://nosible.com/blog/tag/nosible-world)
### [An Embedding-Based Approach to Trade and Economic Policy Uncertainty](https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty)
2026-06-1723 min read
