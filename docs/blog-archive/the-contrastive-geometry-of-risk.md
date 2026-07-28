---
title: "Two Tricks for Clean Features from Sentence Embeddings"
source: https://nosible.com/blog/the-contrastive-geometry-of-risk
published: 2026-06-18T00:00:00.000Z
fetched: 2026-07-28
archived-by: exa /contents (educational archive)
---

# Two Tricks for Clean Features from Sentence Embeddings

> Archived from <https://nosible.com/blog/the-contrastive-geometry-of-risk> for educational study. All content © NOSIBLE.

Two Tricks for Clean Features from Sentence Embeddings | NOSIBLE

Loading

ASK NOSIBLE

Two Tricks for Clean Features from Sentence Embeddings | NOSIBLE
[Skip to content](#main)
NEWAn Embedding-Based Approach to Trade and Economic Policy Uncertainty.[Read the post→](https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty)
[![NOSIBLE](https://nosible.com/logos/nosible-logo.svg)](https://nosible.com/#top)
[](https://nosible.com/search)[ENTER WORLD](https://nosible.world/world)[START TRIAL](https://nosible.com/start-trial)
Loading
ASK NOSIBLE
[Research](https://nosible.com/blog/tag/research)[Technical](https://nosible.com/blog/tag/technical)[Classification](https://nosible.com/blog/tag/classification)[Nosible World](https://nosible.com/blog/tag/nosible-world)
# Two Tricks for Turning Sentence Embeddings into Clean Features
[Stuart Reid](https://www.linkedin.com/in/stuartgordonreid/)
2026-06-18
14 min read
Copy as Markdown
![Signed contrast number line separating systemic and idiosyncratic risk](https://nosible.com/images/2026/06/walkthrough_step9_number_line.png)
NOSIBLE is a search engine, so naturally we've spent a lot of time staring at vectors. Today we are going to share two tricks that can be used to amplify signal and squash noise.
Let's imagine you are working with NOSIBLE World and you want to organize events into three geographic buckets: local, national, or global, then classify whether they are systemic (about the whole market) or idiosyncratic (about specific companies). World does not come with these dimensions, so how would you solve this? You'd reach out to an LLM right? That would work but it's slow, expensive, and massively increases the risk of foreknowledge bias. Today I'm going to show you how you can get this done using nothing more than geometry.
We will carry one event through the whole pipeline.
>
**> Input sentence:
**>
*> "Central banks across every major economy warn that a synchronized financial shock is freezing credit and driving stock markets lower worldwide."
*
>
It should come out **global** and **systemic**. Here is how, in ten steps.
## Step 1: write reference texts for your geographic buckets
A bucket is defined by a set of example sentences. We wrote twenty per bucket as near mirror images of one another: the same declarative frames, repeated across all three sets, with only the scale phrase changing from one place, to a whole country, to the whole world. Holding the topic and the sentence style constant means they cancel across the sets, and the only thing left to separate one set from another is geographic scale. Here are the three sets we authored for this post.
|Local|National|Global|
|A worsening crisis is now affecting a single town.|A worsening crisis is now affecting the whole country.|A worsening crisis is now affecting the entire world.|
|Disruption is being felt across one neighborhood.|Disruption is being felt across the entire nation.|Disruption is being felt across every continent.|
|The impact has spread across a single local community.|The impact has spread across the entire country.|The impact has spread across countries around the globe.|
|Day by day, the emergency is engulfing one city.|Day by day, the emergency is engulfing the whole nation.|Day by day, the emergency is engulfing nations worldwide.|
|Damage now reaches a single district.|Damage now reaches the country from coast to coast.|Damage now reaches the whole planet.|
|What began as a small problem is spreading across one village.|What began as a small problem is spreading across the country as a whole.|What began as a small problem is spreading across the entire globe.|
|Already, the shock has touched a single locality.|Already, the shock has touched the nation as a whole.|Already, the shock has touched every country on Earth.|
|Consequences now extend across one local area.|Consequences now extend across the country at large.|Consequences now extend across the global community.|
|The situation continues to escalate across a single town.|The situation continues to escalate across the whole country.|The situation continues to escalate across the entire world.|
|Fallout from the event now reaches one neighborhood.|Fallout from the event now reaches the entire nation.|Fallout from the event now reaches every continent.|
|Its effects are rippling across a single local community.|Its effects are rippling across the entire country.|Its effects are rippling across countries around the globe.|
|A growing disturbance is being felt throughout one city.|A growing disturbance is being felt throughout the whole nation.|A growing disturbance is being felt throughout nations worldwide.|
|The threat now spans a single district.|The threat now spans the country from coast to coast.|The threat now spans the whole planet.|
|By every measure, the incident is affecting one village.|By every measure, the incident is affecting the country as a whole.|By every measure, the incident is affecting the entire globe.|
|Upheaval is playing out across a single locality.|Upheaval is playing out across the nation as a whole.|Upheaval is playing out across every country on Earth.|
|The disruption stretches across one local area.|The disruption stretches across the country at large.|The disruption stretches across the global community.|
|There are serious consequences for a single town.|There are serious consequences for the whole country.|There are serious consequences for the entire world.|
|Mounting strain is being felt across one neighborhood.|Mounting strain is being felt across the entire nation.|Mounting strain is being felt across every continent.|
|The downturn has now reached a single local community.|The downturn has now reached the entire country.|The downturn has now reached countries around the globe.|
|The crisis is reverberating across one city.|The crisis is reverberating across the whole nation.|The crisis is reverberating across nations worldwide.|
## Step 2: embed your geographic reference texts with OpenAI
An embedding turns each sentence into a list of numbers that captures its meaning. We use OpenAI's `text-embedding-3-large` (3,072 dimensions) for a specific reason: it is frozen, with a September 2021 knowledge cutoff. Foreknowledge bias is the error of scoring a past event using information about how it actually turned out. Because this model's knowledge stops in September 2021, any event after that date is scored with no idea of what came next, so a backtest over recent history carries none of it. A live LLM, retrained on everything since, gives you no such guarantee.
## Step 3: compute cosine similarity to the geographic reference texts
Cosine similarity measures how close two embeddings point in the same direction: high means similar, low means different. Score the input against all sixty references. The raw numbers are noisy and overlap heavily across buckets, so you cannot just read off an answer.
![Strip plot of the input sentence's 60 raw cosines, 20 per bucket, colored local, national, global. The clouds are noisy and overlap in the middle.](https://nosible.com/images/2026/06/walkthrough_step3_geo_raw.png)Strip plot of the input sentence's 60 raw cosines, 20 per bucket, colored local, national, global. The clouds are noisy and overlap in the middle.
## Step 4: apply the tanh trick to squash background noise
Even two unrelated sentences will score around 0.3 just for sharing the same clipped, declarative register. Pass a tanh gate over the cosines, with a band from 0.25 to 0.50: anything below collapses toward 0, anything above saturates toward 1. The background noise is squashed and the real matches stand out.
![The same 60 cosines after gating. The background is squashed toward 0 and the strong global matches push toward 1.](https://nosible.com/images/2026/06/walkthrough_step4_geo_gated.png)The same 60 cosines after gating. The background is squashed toward 0 and the strong global matches push toward 1.
## Step 5: use a cubic mean to amplify the strongest signal
A bucket is many sentences, so turn each bucket's gated cosines into one number with a cubic power mean (p=3), which leans the aggregate toward the strongest matches instead of letting weak ones drag it down. The input scores **local 0.315, national 0.417, global 0.739**. The argmax is **global**. A dead-zone floor would return a clean "none" if no bucket cleared it, so off-topic events are left unassigned rather than forced.
![Bar chart of the three bucket scores: local 0.315, national 0.417, global 0.739. The argmax is global.](https://nosible.com/images/2026/06/walkthrough_step5_geo_scores.png)Bar chart of the three bucket scores: local 0.315, national 0.417, global 0.739. The argmax is global.
## Step 6: write mirror texts for systemic and idiosyncratic labels
The second job is a single axis with two poles, so write the two sets as **mirror images**: identical sentences except for the words that flip single-company to whole-market. These are the real production pairs. Topics vary across the pairs on purpose, so that averaging cancels the topic and leaves only the single-company versus whole-market flip.
|Idiosyncratic (one company)|Systemic (whole market)|
|Analysts warn that a single company could default on its debt|Analysts warn that companies across the economy could default on their debt|
|Investors caution that one firm's share price may collapse|Investors caution that the entire stock market may collapse|
|Economists predict that a single business might be forced into bankruptcy|Economists predict that businesses nationwide might be forced into bankruptcy|
|Analysts fear that one company may miss its profit forecast|Analysts fear that companies across every sector may miss their profit forecasts|
|Officials forecast that a single bank may soon fail|Officials forecast that the whole banking system may soon fail|
|Strategists believe that an individual company could soon see its credit rating cut|Strategists believe that issuers across the market could soon see their credit ratings cut|
|Analysts flag that a single retailer could run out of cash|Analysts flag that retailers throughout the industry could run out of cash|
|Investors fret that one firm's bonds may become worthless|Investors fret that corporate bonds across the market may become worthless|
|Economists say a single manufacturer might halt production|Economists say manufacturers nationwide might halt production|
|Analysts expect that a particular company may cut thousands of jobs|Analysts expect that companies across the economy may cut millions of jobs|
|Officials warn that one firm could soon be hit with a record fine|Officials warn that firms across the industry could soon be hit with record fines|
|Investors caution that a single stock could be wiped out|Investors caution that the broad equity market could be wiped out|
|Analysts predict that one company's profits may evaporate|Analysts predict that corporate profits across the economy may evaporate|
|Strategists fear that a single firm might lose its market access|Strategists fear that firms everywhere might lose their market access|
|Economists forecast that a single borrower may be unable to refinance|Economists forecast that borrowers across the market may be unable to refinance|
|Analysts believe that one company may soon face a crippling lawsuit|Analysts believe that companies across the sector may soon face crippling lawsuits|
|Officials flag that a single firm could soon be caught in a fraud scandal|Officials flag that fraud scandals could soon spread across the whole industry|
|Investors fret that one company could slash its dividend|Investors fret that companies across the market could slash their dividends|
|Analysts say a single business may lose its biggest customer|Analysts say businesses throughout the economy may lose their biggest customers|
|Strategists expect that an individual stock might plunge overnight|Strategists expect that the entire market might plunge overnight|
## Step 7: embed your mirror texts with OpenAI again
Embed both sets the same way. If you project the raw embeddings down to two dimensions, the two sets sit on top of each other. Raw cosine cannot separate them, because both sets are written in the same financial register and that shared style dominates the vectors.
![A 2D projection of the raw mirror-set embeddings. The idiosyncratic and systemic clouds overlap heavily.](https://nosible.com/images/2026/06/walkthrough_step7_scope_raw_2d.png)A 2D projection of the raw mirror-set embeddings. The idiosyncratic and systemic clouds overlap heavily.
## Step 8: compute the systemic and idiosyncratic contrast vectors
The fix is common-mode neutralization. Subtract the mean of all the reference vectors from each one, then renormalize. That removes the shared component the two sets carry. The shared coordinate drops from **+0.71 to +0.005**, and viewed along the contrast direction the method actually uses, the two sets now split cleanly.
![The same sets after neutralization, projected onto the contrast direction. Idiosyncratic on the right, systemic on the left, cleanly separated.](https://nosible.com/images/2026/06/walkthrough_step8_scope_neut_2d.png)The same sets after neutralization, projected onto the contrast direction. Idiosyncratic on the right, systemic on the left, cleanly separated.
## Step 9: compute the similarity to the contrast vectors
Score the input as a signed cosine contrast: its mean cosine to the neutralized idiosyncratic set minus its mean cosine to the neutralized systemic set. Positive means idiosyncratic (one company), negative means systemic (the whole market). Our input scores **−0.068**, landing on the systemic side of zero, which is right: a synchronized shock across every major economy is the whole market, not one company.
![Signed-contrast number line. Our running event sits at −0.068, left of zero on the systemic side of an axis that runs from systemic (whole market) on the left to idiosyncratic (one company) on the right.](https://nosible.com/images/2026/06/walkthrough_step9_number_line.png)Signed-contrast number line. Our running event sits at −0.068, left of zero on the systemic side of an axis that runs from systemic (whole market) on the left to idiosyncratic (one company) on the right.
## Step 10: fuse the signals together to get the full picture
The two reads are independent, so place each event on a plane: geographic scale on one axis, systemic versus idiosyncratic on the other. The input lands firmly in the global and systemic corner.
![A 2D scatter of the verification events plus the running input, each labelled by name and positioned by geographic scale and scope. The running input sits in the global, systemic corner.](https://nosible.com/images/2026/06/walkthrough_step10_fusion.png)A 2D scatter of the verification events plus the running input, each labelled by name and positioned by geographic scale and scope. The running input sits in the global, systemic corner.
## Verification
Let's apply the method to 18 sentences to check that it works: three geographies times two scopes, three samples each. The national and global cases are real warnings pulled straight from the NOSIBLE World leaderboard; the local cases are authored, because the leaderboard has nothing at town scale. We score every one through the full pipeline and compare the prediction to a label assigned by hand. The last column shows what a language model (`gemini-2.5-flash`, temperature 0) returns for the same sentence.
|Sentence|Intended|Method|LLM (gemini-2.5-flash)|scope score|
|Every employer in a small mill town warned of mass layoffs after the factory anchoring the local economy announced its closure.|local / systemic|local / idiosyncratic|local / systemic|+0.010|
|Shopkeepers across a small coastal town warned that a collapsed tourist season had pushed the whole local economy to the brink.|local / systemic|local / systemic|local / systemic|−0.057|
|Officials in a small farming town warned that a failed harvest would ripple through every business on the main street.|local / systemic|local / systemic|local / systemic|−0.029|
|The owner of a family bakery in a small town warned that surging rents would force the century-old shop to close.|local / idiosyncratic|local / idiosyncratic|local / idiosyncratic|+0.017|
|A single hardware store in a small town warned it would shut for good after a fraud drained its accounts.|local / idiosyncratic|local / idiosyncratic|local / idiosyncratic|+0.051|
|A local manufacturer in one town warned of layoffs after losing its only major contract.|local / idiosyncratic|local / idiosyncratic|local / idiosyncratic|+0.065|
|Analysts and consumer groups warn that persistent inflation and high energy costs will strain households and create fragility in Italy's housing market.|national / systemic|national / systemic|national / systemic|−0.027|
|Economic forecasters warn that UK unemployment will rise to a multi-year high as economic growth stalls in the coming months.|national / systemic|national / systemic|national / systemic|−0.054|
|Economists warn that Canada's economy will face sustained slower growth and a potential recession as artificial supports fade.|national / systemic|global / systemic|national / systemic|−0.059|
|Industry leaders warn that the liquidation of sugar producer Tongaat Hulett could trigger a failure of South Africa's sugar industry.|national / idiosyncratic|national / idiosyncratic|national / systemic|+0.022|
|Modella Capital warns that retailer TG Jones faces administration and collapse unless its lenders approve a restructuring plan.|national / idiosyncratic|local / idiosyncratic|national / idiosyncratic|+0.038|
|The Postmaster General warns that the United States Postal Service will run out of cash and be unable to pay its workers within a year.|national / idiosyncratic|national / systemic|national / systemic|−0.016|
|The IMF and a former central-bank governor warn that the global economy is unprepared for increasingly frequent and unpredictable shocks.|global / systemic|global / systemic|global / systemic|−0.031|
|Bond strategists warn that persistent inflation and geopolitical tension will keep government-bond yields elevated and push up borrowing costs worldwide.|global / systemic|global / systemic|global / systemic|−0.050|
|Economists warn that a synchronized slowdown will drag down corporate earnings across every major economy.|global / systemic|global / systemic|global / systemic|−0.058|
|Analysts warn that Tesla's stock price could plunge by more than 60% over the next year.|global / idiosyncratic|global / idiosyncratic|national / idiosyncratic|+0.043|
|Analysts warn that the highly anticipated IPO of SpaceX will struggle to outperform the market after its debut.|global / idiosyncratic|global / idiosyncratic|national / idiosyncratic|+0.028|
|Samsung's leadership and labour unions warn that planned strikes over pay will disrupt global semiconductor supply chains.|global / idiosyncratic|global / systemic|global / systemic|−0.005|
**Method: geography 16 of 18, scope 15 of 18, and both axes right at once on 13 of 18.** These are messy real-world warnings, not toy sentences, and the misses are the genuinely hard ones. The hardest are single companies whose trouble bleeds into a whole sector: Samsung's strike "disrupting global semiconductor supply chains" and the US Postal Service running out of cash both read as systemic, not idiosyncratic. On geography, Canada's macro warning tips global and the UK retailer TG Jones reads local rather than national.
A current frontier model, Google's `gemini-2.5-flash`, lands in exactly the same place: **geography 16 of 18, scope 15 of 18, and both at once 13 of 18**, a dead heat with the geometry on all three. It trips on the same Samsung and Postal Service scope calls, also reads the Tongaat Hulett warning as systemic, and drops the two unmistakably global companies, Tesla and SpaceX, into the national bucket. A frozen embedding and a handful of cosines hold their own against the model they are meant to replace, while staying deterministic, auditable, and effectively free.
## Why not just ask an LLM?
A language model can label an event too. For doing this at scale, on a corpus you have to stand behind, the geometry wins on five counts:
* **Deterministic and reproducible.** The same sentence always yields the same numbers. An LLM's answer drifts between calls and model versions, and you cannot audit why it chose a label.
* **Defensible.** Every output is a cosine, a mean, or a subtraction. You can re-derive any score by hand and explain it to a regulator or a client.
* **Cheap and fast at scale.** One embedding per document, then linear algebra for every bucket and every axis. The LLM route is one call per document per question, which does not scale across millions of documents and many features.
* **No training and no labelled data.** You write reference sentences. That is the entire setup.
* **No foreknowledge bias.** The model's knowledge stops in September 2021, so when you backtest on events after that date it cannot have seen how they played out, and the features leak nothing from the future.
## Why this works
Every number here is a cosine, a mean, or a subtraction. There is no training set to curate and nothing deciding the answer; the embedding is frozen, so the same text always yields the same features, reproducible and auditable by anyone. Two tricks cover the whole problem: a multiclass relevance score that sorts events into buckets (one set per bucket, gate, cubic mean, argmax), and a binary contrast that buckets the risk (two mirror sets, neutralize, signed cosine). Pick your problem, write the sentences, run the trick.
*Every figure was generated from real `text-embedding-3-large` vectors using the operations above. The geographic reference texts are authored for this post; the verification set pairs authored local examples with real warnings from the NOSIBLE World leaderboard; the systemic and idiosyncratic mirror sets are taken straight from production.*
[All Research](https://nosible.com/blog)
Related Research
![NOSIBLE World knowledge graph showing entity connections over a decade](https://nosible.com/images/2026/07/kg-hero-decade.png)
[Research](https://nosible.com/blog/tag/research)[Knowledge Graphs](https://nosible.com/blog/tag/knowledge-graphs)[Nosible World](https://nosible.com/blog/tag/nosible-world)
### [Point-in-Time Knowledge Graphs over Named Entities with NOSIBLE World](https://nosible.com/blog/point-in-time-knowledge-graphs-over-named-entities)
2026-07-168 min read
![Daily NOSIBLE Trade Policy Uncertainty index compared with the published index](https://nosible.com/images/2026/06/nosible-tpu-daily.png)
[Research](https://nosible.com/blog/tag/research)[Trade Policy](https://nosible.com/blog/tag/trade-policy)[Nosible World](https://nosible.com/blog/tag/nosible-world)
### [An Embedding-Based Approach to Trade and Economic Policy Uncertainty](https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty)
2026-06-1723 min read
![S&P 500 news-stress overlay with drawdown and signal thresholds](https://nosible.com/images/2026/06/news-stress-overlay-hero.png)
[Research](https://nosible.com/blog/tag/research)[Quantitative Strategy](https://nosible.com/blog/tag/quantitative-strategy)[Nosible World](https://nosible.com/blog/tag/nosible-world)
### [Turning News into a Risk-On/Risk-Off Equity Signal](https://nosible.com/blog/turning-news-into-a-risk-on-risk-off-equity-signal)
2026-06-169 min read
