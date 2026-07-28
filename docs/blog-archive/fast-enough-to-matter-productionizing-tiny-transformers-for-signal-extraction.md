---
title: "Matching GPT-5.1 Financial Sentiment with Qwen3"
source: https://nosible.com/blog/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction
published: 2025-12-12T00:00:00.000Z
fetched: 2026-07-28
archived-by: exa /contents (educational archive)
---

# Matching GPT-5.1 Financial Sentiment with Qwen3

> Archived from <https://nosible.com/blog/fast-enough-to-matter-productionizing-tiny-transformers-for-signal-extraction> for educational study. All content © NOSIBLE.

Matching GPT-5.1 Financial Sentiment with Qwen3 | NOSIBLE

Loading

ASK NOSIBLE

Matching GPT-5.1 Financial Sentiment with Qwen3 | NOSIBLE
[Skip to content](#main)
NEWAn Embedding-Based Approach to Trade and Economic Policy Uncertainty.[Read the post→](https://nosible.com/blog/an-embedding-based-approach-to-trade-and-economic-policy-uncertainty)
[![NOSIBLE](https://nosible.com/logos/nosible-logo.svg)](https://nosible.com/#top)
[](https://nosible.com/search)[ENTER WORLD](https://nosible.world/world)[START TRIAL](https://nosible.com/start-trial)
Loading
ASK NOSIBLE
[Technical](https://nosible.com/blog/tag/technical)[Sentiment](https://nosible.com/blog/tag/sentiment)[Signals](https://nosible.com/blog/tag/signals)[Classification](https://nosible.com/blog/tag/classification)
# Matching GPT-5.1 at Financial Sentiment with Active Learning and Qwen3
[Simon van Dyk](https://www.linkedin.com/in/simon-van-dyk/)
2025-12-12
27 min read
Copy as Markdown
![Running sprinter illustration representing efficient financial-sentiment model training](https://nosible.com/blog/illustrations/the-sprinter.png)
Every hedge fund we meet asks the same question: do you offer aspect-based financial sentiment? Why? Because sentiment moves markets. Especially when it's a forward-looking statement about a material event. They know it. We know it. It's how RavenPack grew into a 200+ person company.
Today we're sharing how we built and open sourced a financial sentiment model that beats FinBERT[1](#user-content-fn-1) and matches GPT-5.1's accuracy at a fraction of the cost. The results: **87.34% accuracy on real-world data** and **86.4% on Financial PhraseBank[2](#user-content-fn-2)**, outperforming FinBERT on both datasets while running orders of magnitude faster and cheaper than frontier LLMs.
Do we really need another sentiment model? Didn't FinBERT solve this in 2022? In a nutshell: no it's not solved, because while FinBERT performs well on the popular Financial PhraseBank dataset, it sucks on real-world data. The Financial PhraseBank dataset is contrived. It doesn't resemble what real-world data looks like, and models trained on it are overfit. LLMs, on the other hand, generalize well and are surprisingly good at labeling textual data, but they're intractably expensive to use at scale.
Therefore, over the past few weeks, we have trained and productionized three text classifiers. This post explains how. One classifier predicts the financial sentiment of a text snippet. Another determines whether the text is a forward-looking statement or not. The third determines whether the text contains a prediction or not. Combining [these signals along with other dimensions in NOSIBLE's data](https://nosible.com/files/2025/12/tesla-signals-sample.xlsx) enables us to do powerful aspect based analysis to answer questions like:
* Which retail companies show persistent negative sentiment in forward-looking statements about margin pressure, despite positive sentiment about revenue growth?
* Show me the correlation between negative forward-looking statements about regulatory risk and subsequent stock volatility for pharma companies in Q3 2024.
* How is sentiment diverging between forward-looking guidance versus actual results across semiconductor companies this quarter?
We've [open sourced these models on HuggingFace](https://huggingface.co/NOSIBLE) along with the datasets they were trained on. We also share source code for your own projects.
Wait, but you're a search engine, right? NOSIBLE is already a world-class search engine, but more importantly it's **incredibly fast**, which unlocks the ability to build [near real-time datafeeds](https://x.com/StuartReid1929/status/1920720746415849703). Our Search Feed product lets you turn any search about any topic into a point-in-time, backtest-friendly, deduplicated time series of data. [Here's an example for Tesla](https://nosible.com/files/2025/12/tesla-top-30.xlsx).
We've leaned even harder in this direction. Our Search Feeds product is now better described as **web-scale surveillance**. To prove this isn't just a marketing term, we're extracting and including these signals in our Search Feeds for our customers, and **we're just getting started**.
Let's get to it.
## Methodology
Our approach combines three ingredients: real-world data, active learning for label refinement, and fine-tuning a small causal LLM for production inference.
### Why Real-World Data Generalizes Better
A quality dataset is key to building a good ML model. The combination of compute, a return to MLPs, and a much larger labeled dataset contributed significantly to what made AlexNet so successful back in 2012.
Using the vast amount of real-world data indexed by NOSIBLE, we sampled a Search Feed centered around financial news. The texts are varied and contain a ton of signal. Here's what they look like:
>
> Some peers in the therapeutics segment have already reported their Q3 results. Gilead Sciences posted year-on-year revenue growth of 3%, beating expectations by 3.7%, and Biogen reported revenue up 2.8%, topping estimates by 8.2%. Following their reports, Gilead Sciences's stock traded up 1.2% and Biogen's stock was up 4.1%.
>
There are fun ones too:
>
> If you thought the wait for Kingdom Hearts III was ridiculous, you ain't see nothing yet. Now that we have an official release date for the game, Square Enix has already announced two collectors editions of the game. The first is your average run of the mill collectors edition, priced at $80 dollars. That's not the one you want. No, the big daddy collectors edition itself, the deluxe edition is the one you're going to want.
>
This variety makes them difficult to label accurately. It's also what makes them excellent training data. The nuanced examples force models to learn robust patterns rather than memorize artifacts.
Labeling 100,000 examples by hand is intractable. Instead, we used a combination of hand-labeling, LLM ensemble labeling, and active learning to produce a high-quality labeled dataset.
### Why Better Labels Are (Almost) All You Need
ML practitioners can unintentionally obsess over model architecture and hyperparameters, but often all you need is better quality data. We took this approach: invest heavily in label quality, then starting with a simple baseline model, iteratively test larger and more complex models.
The key insight is that correct labels on difficult examples matter more than thousands of mediocre labels. LLMs are excellent labelers, but they make mistakes on edge cases. Active learning helps you find and fix those mistakes systematically. Here's the four-step process:
1. **Look At Your Data**: Understand what makes classification difficult
2. **Label with LLMs**: Ensemble LLMs to label at scale
3. **Train Classifiers**: Build baselines to find signal
4. **Relabel Hard Texts**: Use active learning to improve label quality
#### Step 1: Look At Your Data
Don't vibe think. Go stare at the data. What makes it difficult?
Sample 200 texts uniformly from your dataset and label them by hand: **negative**, **neutral**, or **positive**. This process isn't glamorous, but it's essential. You need to understand what makes classification difficult before you can build a good classifier.
As you label, ask yourself:
* What patterns distinguish **negative**, **neutral**, and **positive** sentiment?
* Which examples are ambiguous or require domain knowledge?
* Are any classes under-represented?
Check your class distribution. Your sample doesn't need to match real-world distribution, it needs sufficient examples of each class for a model to learn from. If one class is under-represented (we needed more negative samples), sample more until you have enough examples to understand its patterns. This small investment pays dividends: you'll understand which edge cases will trip up your models later.
The goal isn't just labels. It's understanding. What do easy-to-label samples look like? What makes the difficult ones difficult? This knowledge will guide your prompt engineering in the next step.
#### Step 2: Label with LLMs
Now that you understand your data, write a prompt YOU could follow to label it consistently. Find the smartest LLMs. Ensemble them together.
Your prompt should be unambiguous. If you're labeling financial sentiment, define exactly what "negative," "neutral," and "positive" mean in your domain. Include examples of edge cases you discovered in Step 1. Make it a decision tree if that helps remove ambiguity.
We used eight models:
* [xAI: Grok 4 Fast](https://openrouter.ai/x-ai/grok-4-fast)
* [xAI: Grok 4 Fast (reasoning enabled)](https://openrouter.ai/x-ai/grok-4-fast)
* [Google: Gemini 2.5 Flash](https://openrouter.ai/google/gemini-2.5-flash)
* [OpenAI: GPT-5 Nano](https://openrouter.ai/openai/gpt-5-nano)
* [OpenAI: GPT-4.1 Mini](https://openrouter.ai/openai/gpt-4.1-mini)
* [OpenAI: gpt-oss-120b](https://openrouter.ai/openai/gpt-oss-120b)
* [Meta: Llama 4 Maverick](https://openrouter.ai/meta-llama/llama-4-maverick)
* [Qwen: Qwen3 32B](https://openrouter.ai/qwen/qwen3-32b)
Ensembling reduces individual model biases. To ensemble their labels, use majority vote. Start with your hand-labeled 200 samples to validate the prompt works before scaling to thousands.
Here are the prompts we used for each classification task:
Financial sentiment prompt
```python
f"""
# TASK DESCRIPTION

Read through the following snippet of text carefully and classify the **financial sentiment** as
either negative, neutral, or positive. You must also provide a short rationale for why you
assigned the financial sentiment you did.

For clarity here are the definitions negative, neutral, and positive sentiments:

   - **Negative**: The snippet describes an event or development that has had, is having, or is
       expected to have a material negative impact on the company's financial performance, share price, reputation,
       or outlook.

   - **Neutral**: The snippet is informational/descriptive and is not expected to have a material positive or
       negative impact on the company.

   - **Positive**: The snippet describes an event or development that has had, is having, or is expected to have
       a material positive impact on the company's financial performance, share price, reputation, or outlook.

Materiality note:
- “Material impact” includes likely effects on share price, revenue, costs, profitability, cash flow, guidance,
regulatory exposure, reputation, risk exposure, or competitive position.

# TASK GUIDELINES

For the avoidance of doubt here is a decision tree that you can follow to arrive at the most appropriate
sentiment classification for the snippet. Pay careful attention to the logic. Don't deviate.

START
│
├── Step 1: Carefully read and understand the snippet.
│
├── Step 2: Check for sentiment indicators:
│
├── Is the snippet clearly NEGATIVE?
│     (share price decline, losses, scandals, lawsuits, layoffs, product recalls, regulatory fines,
│      leadership resignations, declining sales, market-share losses, reputational damage etc.)
│    │
│    ├── YES → Classify as "negative"
│    │      └── Provide a rationale by summarizing WHY the snippet is negative.
│    │
│    └── NO → Continue below
│
├── Is the snippet clearly POSITIVE?
│     (share price increases, strong earnings, favorable partnerships, successful product launches, awards,
│      expansion plans, positive analyst coverage, reputational enhancement, etc.)
│    │
│    ├── YES → Classify as "positive"
│    │      └── Provide a rationale by summarizing WHY the snippet is positive.
│    │
│    └── NO → Continue below
│
└── If neither clearly positive nor negative → Classify as "neutral"
         (routine product announcements without performance implications, leadership appointments,
          scheduled reports, factual statements, general industry overviews, etc.)
       └── Provide a rationale by summarizing the WHY the snippet is neutral.

If there is conflicting sentiment in the snippet pick the most dominant one, otherwise default to **neutral**.

# RESPONSE FORMAT

You must respond with ONLY a valid JSON object formatted as follows. DO NOT WRITE ANY PREAMBLE JUST RETURN JSON.

{{
   "rationale": "A one-sentence rationale for your classification",
   "financial_sentiment": "either negative, neutral, or positive"
}}

# SNIPPET TO LABEL

Here is the snippet we would like you to assign a negative, neutral, or positive financial sentiment label to:

{text}

P.S. REMEMBER TO READ THE SNIPPET CAREFULLY AND FOLLOW THE GUIDELINES TO ARRIVE AT THE MOST APPROPRIATE
FINANCIAL SENTIMENT CLASSIFICATION. WHEN IN DOUBT, YOU SHOULD DEFER TO A "neutral" CLASSIFICATION FOR THE SNIPPET.
GOOD LUCK!
"""
```
Forward-looking prompt
```python
f"""
# TASK DESCRIPTION

You will be given a text snippet. Your task is to determine the **temporal
orientation** of the main event or topic in the text, classifying it as either
"forward" (forward-looking) or "not-forward" (backward-looking or neutral).

# GUIDELINES

For the avoidance of doubt here is a decision tree that you can follow to arrive
at the most appropriate temporal orientation classification for the snippet. Pay
careful attention to the logic. Don't deviate.

START
│
├── Step 1: Carefully read and identify the MAIN event or topic in the text.
│           (Ignore supporting details, commentary, or verb tenses of reporting)
│
├── Step 2: Determine the temporal orientation of this main event/topic:
│
├── Is the main event/topic FORWARD LOOKING?
│     (Will the event occur in the future or is it planned/expected?)
│     Examples: future launches, upcoming announcements, expansion plans, projections,
│              forecasts, guidance, targets, goals, roadmaps
│     Note: News about future plans (even if reported in past/present tense) = forward
│    │
│    ├── YES → Classify as "forward"
│    │      └── Provide a rationale explaining what future event the text focuses on.
│    │
│    └── NO → Classify as "not-forward"
│           └── This includes:
│               • Past events (announcements made yesterday, completed mergers, reported earnings)
│               • Current states (ongoing situations, present trading activity, existing conditions)
│               • Timeless facts or general statements
│               └── Provide a rationale explaining why the event is not forward-looking.
│
END

IMPORTANT: When uncertain about temporal orientation → Default to "not-forward"

# DISAMBIGUATION RULES

When the temporal orientation is unclear, apply these rules:

1. **Reporting Verb vs. Main Event Rule**
   - Ignore the tense of reporting verbs (said, announced, reported)
   - Focus on what is being reported about
   - Example: "CEO said yesterday the company will expand" → "forward" (expansion is future)

2. **Plans and Intentions Rule**
   - Any plans, intentions, targets, or forward guidance = "forward" (even if approved/decided in past)
   - Example: "Board approved new product launch" → "forward" (launch is future event)

3. **When in Doubt → Not-Forward**
   - If temporal orientation remains ambiguous → classify as "not-forward"

# TEXT SNIPPET TO LABEL
{text}

# RESPONSE FORMAT

You must respond with ONLY a valid JSON object formatted as follows. DO NOT WRITE ANY PREAMBLE JUST RETURN JSON.

{{
    "tense": "forward | not-forward",
    "rationale": "A one-sentence rationale for your classification"
}}

P.S. REMEMBER TO READ THE SNIPPET CAREFULLY AND FOLLOW THE GUIDELINES TO ARRIVE AT THE MOST APPROPRIATE
CLASSIFICATION. WHEN IN DOUBT, YOU SHOULD DEFER TO A "not-forward" CLASSIFICATION FOR THE SNIPPET. GOOD LUCK!
"""
```
Prediction prompt
```python
f"""
# TASK DESCRIPTION

Read the following text snippet carefully and classify its **causal structure** as either
**predictive** or **not-predictive**. You must also provide a short rationale for
why you assigned the label you did.

For clarity, here are the definitions:

    1. Predictive: makes a concrete claim, forecast, prediction or estimate about a specific event.

    2. Not Predictive: only reports or explains past or present facts. It does not contain ANY
       predictions, estimates, or forecasts. Plans, schedules, hopes, retrospectives or non-concreate
       predictions or estimates means it is **not-predictive**.

# TASK GUIDELINES

For the avoidance of doubt, follow this decision tree exactly. Don’t deviate.

START
 │
 ├── Step 1: Carefully read and understand the snippet.
 │
 ├── Step 2: Check for causal structure indicators:
 │
 ├── Is the snippet clearly PREDICTIVE?
 │     (contains explicit forecasts or predictions about the future or
 │      numerical estimates about current or future events.)
 │    │
 │    ├── YES → Classify as "predictive"
 │    │      └── Provide a rationale summarizing WHAT future outcome or effect is being
 │    │          forecast or expected.
 │    │
 │    └── NO → Continue below
 │
 └── If it is not clearly predictive → Classify as "not-predictive"
        └── Provide a rationale summarizing WHY it is 'not-predictive'.

# DISAMBIGUATION RULES.

1. Predictive
    - If a snippet contains ANY predictive text, label it **"predictive"**.
    - Numerical estimates about CURRENT or FUTURE statistics are to be considered **predictive**.
    - Analyst estimates and ratings are **predictive** even if stated in the past tense.

2. Not predictive
    - Forward-looking plans or hopes without a claimed outcome remain **"not-predictive"**.
    - Schedules / announcements of events are not a forecast about an uncertain outcome or effect so **not-predictive**.
    - Retrospective narratives about past events should be classified as **not-predictive**.
    - Potential future actions without concrete predictions are **not-predictive**.
    - Future intent without a concrete prediction is **not-predictive**.

# RESPONSE FORMAT

You must respond with ONLY a valid JSON object formatted as follows. DO NOT WRITE ANY PREAMBLE JUST RETURN JSON.

{{
    "rationale": "A one-sentence rationale for your classification",
    "causal": "predictive | not-predictive"
}}

SNIPPET TO LABEL

Here is the snippet we would like you to label:

{text}

P.S. REMEMBER TO READ THE SNIPPET CAREFULLY AND FOLLOW THE GUIDELINES TO ARRIVE AT THE MOST APPROPRIATE
CAUSAL CLASSIFICATION. WHEN IN DOUBT, YOU SHOULD DEFER TO A "not-predictive" CLASSIFICATION FOR THE SNIPPET.
GOOD LUCK!
"""
```
Validate your prompt on your hand-labeled samples. Where do they disagree with the ensemble? Use those disagreements and the LLMs rationale to re-label examples and refine your prompt. We discovered earnings reports were frequently labeled **neutral** when they actually contained material outcomes, so we emphasized financial materiality over tone.
Once validated, scale up to thousands of samples.
Dataset transformations from 200 hand labeled samples to 10,000 LLM labeled samples
```
┌────────┐
│ text   │  shape (200, 1)
│ str    │
└────────┘
     │
     │ +1 human label
     ▼
┌───────┬─────────────┐
│ text  │ human_label │  shape (200, 2)
│ str   │ str         │
└───────┴─────────────┘
     │
     │ +10,000 samples, +8 LLM labels, -1 Human label
     ▼
┌───────┬─────────────┬─────┬───────────┐
│ text  │ grok_4_fast │ ... │ qwen3_32b │  shape (10_000, 9)
│ str   │ str         │     │ str       │    
└───────┴─────────────┴─────┴───────────┘  *10_200? Nope.
     │                                     We kept the samples for the human set separate
     │ +majority vote
     ▼
┌───────┬─────────────┬─────┬───────────┬───────────────┐
│ text  │ grok_4_fast │ ... │ qwen3_32b │ majority_vote │  shape (10_000, 10)
│ str   │ str         │ str │ str       │ str           │   
└───────┴─────────────┴─────┴───────────┴───────────────┘
```
#### Step 3: Train Classifiers
A baseline classifier is an accuracy reference, but can also tell you:
* There is "signal": Verifies there are learnable features in the dataset.
* Labels are good: Some features have a linear relationship with the labels we've assigned.
* Benchmark for improvement: If a more complex model does poorly, you know the issue lies in something like the model architecture, training procedure, or a bug, not the dataset itself.
Training classifiers in a nutshell:
**1. Find a good representation:** Use text embeddings. Embedding models produce dense vectors that capture semantic meaning, including sentiment features. We ensembled six embedding models for more diverse representations.
**2. Train baseline classifiers:** Start simple. Train a linear classifier (we used SGDClassifier with hinge loss) on your embeddings. If it performs well, you have signal. If it fails, investigate your features, labels, or task difficulty.
**3. Check for overfitting:** Split your data (80/20 train/val). If training and validation accuracy are close, you're not overfit. Our baseline: 86.65% training accuracy, 85.93% validation accuracy. That's good signal with no overfitting.
Text classification using embeddings
```
                
                ┌───────────┐                ┌────────────┐
                │           │  Embeddings    │            │
    Text        │ Embedding │ representation │ Classifier │  Targets    
 ─────────────▶ │   Model   │ ────────────▶  │    Model   │ ─────────▶  Label = "negative"
 ["Miss..."]    │           │  [[0.42...]]   │            │ [0,-1,...] 
                └───────────┘                └────────────┘
                
 Target mapping: {0: neutral, 1: positive, -1: negative}
```
The embedding models we used:
* [Qwen3-Embedding-8B](https://openrouter.ai/qwen/qwen3-embedding-8b)
* [Qwen3-Embedding-4B](https://openrouter.ai/qwen/qwen3-embedding-4b)
* [Qwen3-Embedding-0.6B](https://openrouter.ai/qwen/qwen3-embedding-0.6b)
* [OpenAI: Text Embedding 3 Large](https://openrouter.ai/openai/text-embedding-3-large)
* [Google: Gemini Embedding 001](https://openrouter.ai/google/gemini-embedding-001)
* [Mistral: Mistral Embed 2312](https://openrouter.ai/mistralai/mistral-embed-2312)
Scaling to 100,000 samples added 1-2% accuracy giving us a final baseline: **86.65% training accuracy** and **85.93% validation accuracy** on the 20,000 validation samples.
Code to train baseline classifier
```python
import os
import datetime as dt
import polars as pl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def embedding_model(data: pl.DataFrame, model: str, embeddings_path: str) -> tuple:
    """
    Train a linear classifier on text embeddings for financial sentiment classification.
    
    Generates or loads embeddings, trains an SGDClassifier with hinge loss, and evaluates
    performance on train/validation splits. Returns predictions, accuracy metrics, and
    confusion matrices for both splits.
    
    :param data: DataFrame containing 'text' and 'majority_vote' columns with samples and labels
    :param model: Embedding model identifier for OpenRouter API
    :param embeddings_path: Path to cache/load embeddings (created if doesn't exist)
    :return: Returns a tuple of predictions, accuracy metrics, and confusion matrices for both splits.
    """
    # Memoize the embeddings for subsequent training runs.
    if not os.path.exists(embeddings_path):
        # Generate the embeddings using OpenRouter and store as a numpy array file.
        embed_data(data=data, out_file=embeddings_path, model=model)

    # Load the embeddings, only as many samples as this training run.
    embeddings = np.load(embeddings_path)
    embeddings = embeddings[:data.shape[0]]

    # Fetch labels using majority vote and map to model targets.
    label_to_target={"positive": 1, "neutral": 0, "negative": -1}
    labels = list(label_to_target.keys())
    targets = [label_to_target[row['majority_vote']] for row in data.to_dicts()]

    # Samples already shuffled, simple subscript split is sufficient.
    train_pct = 0.8
    n_train = int(data.shape[0] * train_pct)
    x_train, y_train = embeddings[:n_train], targets[:n_train]
    x_val, y_val = embeddings[n_train:], targets[n_train:]

    # Get the training and validation text.
    train_text = data[:n_train].select("text").to_numpy().flatten().tolist()
    val_text = data[n_train:].select("text").to_numpy().flatten().tolist()

    classifier = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=0.00001,
    )
    classifier.fit(x_train, y_train)

    # Make predictions.
    train_preds = classifier.predict(x_train)
    val_preds = classifier.predict(x_val)

    # Score the preds.
    train_acc = accuracy_score(y_true=y_train, y_pred=train_preds)
    val_acc = accuracy_score(y_true=y_val, y_pred=val_preds)

    confusion_train = confusion_matrix(y_true=y_train, y_pred=train_preds)
    confusion_train = {
        r: {
            c: int(confusion_train[i, j])
            for j, c in enumerate(labels)
        }
        for i, r in enumerate(labels)
    }
    confusion_val = confusion_matrix(y_true=y_val, y_pred=val_preds)
    confusion_val = {
        r: {
            c: int(confusion_val[i, j])
            for j, c in enumerate(labels)
        }
        for i, r in enumerate(labels)
    }

    print(f"{dt.datetime.utcnow()} - Train Accuracy {train_acc:.4f}")
    print(f"{dt.datetime.utcnow()} - Val Accuracy {val_acc:.4f}")
    print(f"{dt.datetime.utcnow()} - Confusion Matrix")
    print("Train", simplejson.dumps(confusion_train, indent=4))
    print("Val", simplejson.dumps(confusion_val, indent=4))

    return (
        (y_train, train_preds, train_text),
        (y_val, val_preds, val_text),
        (train_acc, val_acc),
        (confusion_train, confusion_val)
    )
```
A fun aside: including [some optional headers](https://openrouter.ai/docs/api/reference/overview#headers) in our embeddings requests to OpenRouter enabled them to measure our usage. Surprisingly, embedding a 100,000 sample dataset over 6 embedding models put us on the OpenRouter leaderboard for embeddings. Once we've operationalized this process into our Search Feeds product, we will sit in first place. In fact as of this writing we're in 2nd.
OpenRouter leaderboard for Qwen3-Embedding-8B usage, November 2025
![OpenRouter leaderboard showing NOSIBLE ranking 6th for Qwen3-Embedding-8B usage after the project](https://nosible.com/images/2025/12/openrouter-qwen3-embedding-8b-leaderboard.png)OpenRouter leaderboard showing NOSIBLE ranking 6th for Qwen3-Embedding-8B usage after the project
#### Step 4: Relabel Hard Texts
tldr; Use baselines to find difficult examples. Are they difficult? Or just wrong? Relabel the wrong ones.
**What is Active Learning?** In traditional supervised learning, labels are fixed ground truth. But LLM generated labels aren't infallible. Active learning treats labels as mutable: find examples where your model struggles, investigate if the label is wrong, and fix it. Iterate until convergence.
**Use baselines to find difficult examples:** Train your ensemble of linear models. Where do they all agree but the LLM ensemble label (majority vote) disagrees? These are your candidates for relabeling. When all your baseline models reach consensus, that's a strong signal the original label might be wrong.
**Are they difficult? Or just wrong?** Not every disagreement means a bad label. Some examples are genuinely ambiguous. Consult a stronger LLM (we used OpenAI's GPT-5.1) as an oracle to make these decisions. If the oracle agrees with your baselines, relabel. If not, keep the original label.
**Relabel iteratively:** Fix the labels, retrain your models, find new disagreements. The set of disagreements shrinks each iteration. Repeat until convergence. Iterate until no new disagreements appear.
The basic outline of our full training algorithm with relabeling is as follows:
1. Label a set of 100k samples with the LLM labelers and compute majority vote.
2. Train multiple linear models on different embeddings of the same text to predict the majority vote of the LLM labelers.
3. Perform iterative relabeling:
* Compare all the linear models' predictions to the majority vote label.
* Identify disagreements where all linear models agree but the majority vote label does not.
* Consult an oracle ([OpenAI: GPT-5.1](https://openrouter.ai/openai/gpt-5.1), a large LLM acting as our active-learning expert) to evaluate disagreements and relabel samples when appropriate.
* Drop the worst performing linear model on the validation set from the ensemble.
* Repeat until no additional samples require relabeling.
* This is the final dataset used for training the classification models.
The accuracy improvement was significant.
\~3.5% Validation accuracy improvement as a result relabeling.
|Split|Before|After|Improvement|
|Train|86.65%|90.03%|+3.38%|
|Validation|85.93%|89.48%|+3.55%|
Prompt used to consult the oracle
```python
"""
Consult the oracle to evaluate whether it agrees with the linear models prediction, or if we need to relabel this text sample.
"""
import textwrap

labels = ["negative", "neutral", "positive"]
labelling_prompt = textwrap.dedent(financial_sentiment_prompt(text))
label_options = " | ".join([f'"{label}"' for label in labels])
# The prediction variable is one of the labels above, where all linear models agreed.

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text":  textwrap.dedent(
                    f"""
                    # TASK DESCRIPTION
                    
                    Given the following LLM prompt, another model labelled this text as \"{prediction}\".
                    Is this model correct? Provide the correct label and a rationale for your answer.
                    
                    # RESPONSE FORMAT
                    
                    You must respond using this exact format. DO NOT RESPOND IN ANY OTHER WAY:

                    ```json
                    {{
                        "correct": "true" | "false",
                        "label": {label_options},
                        "reason": "A rationale for the correctness or incorrectness of your label."
                    }}
                    ```
 
 P.S. Think fast there is no time to waste. 
 """ 
 ) 
 } 
 ] 
 }, 
 { 
 "role": "user", 
 "content": [{ "type": "text", "text": labelling_prompt }] 
 } 
] 
 
# Send these messages to OpenRouter using the OpenAI client.
```
### Hijacking Qwen3 0.6B For Sentiment Analysis
With 90% baseline accuracy and 100,000 clean labels, we had a production-quality dataset. Now for the final step: training a model fast and cheap enough to run at scale.
We tested ModernBERT, DeBERTa, and several causal LLMs. The winner? Qwen3 0.6B, a tiny 600M parameter model that matched GPT-5.1's accuracy at orders of magnitude lower cost and latency. It's so small it's not even hosted on OpenRouter. You can run it on a phone.
Why did Qwen3 0.6B work so well? Modern causal LLMs are pre-trained on massive text corpora, giving them strong language understanding out of the box. Fine-tuning them for classification is elegant: treat classification as next-token prediction. Given a prompt with the text snippet, the model predicts the label token (**negative**, **neutral**, or **positive**). No classification head, no special architecture, just next-token prediction.
This means the representation changes too. Our baseline classifiers used embeddings, fine-tuning uses the tokenized text directly. The model learns to associate patterns in the raw token sequence with the correct label token.
To ensure the model didn't overfit, we compared training loss with validation loss and monitored Financial PhraseBank accuracy during training.
The implementation is straightforward, but three details are critical:
**1. Label masking:** Only compute loss on the label token, not the prompt. This forces the model to learn classification, not memorize prompts. Set prompt tokens to -100 to tell PyTorch to ignore them:
```python
labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]
```
**2. Left-padding:** Causal models need left-padding for batching. Right-padding makes the model attend to padding tokens when predicting, which breaks everything:
```python
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
```
**3. EOS token handling:** During evaluation, exclude end-of-sequence tokens (`<|im\_end|>` for Qwen 0.6B) from accuracy calculations. Only measure whether the model predicted the correct label token (positive/negative/neutral), not conversational markup. We initially got suspiciously good results because we included EOS tokens, and fixing this gave us the true accuracy, which we verified after training on an out-of-sample dataset.
To put this all together, here's the data flow for fine-tuning Qwen3 0.6B for financial sentiment classification:
Fine-tuning data flow
```
┌──────────────────────────────────────────────────────────┐
│ Training Example                                         │
├──────────────────────────────────────────────────────────┤
│  Text: "Tesla reported record deliveries..."             │
│  Label: "positive"                                       │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ Build Messages                                           │
├──────────────────────────────────────────────────────────┤
│  System: "Classify the financial sentiment as positive,  │
│           negative, or neutral."                         │
│  User: "Tesla reported record deliveries..."             │
│  Assistant: "positive" (target label)                    │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ Tokenize (apply chat template)                           │
├──────────────────────────────────────────────────────────┤
│  Prompt tokens: [51234, 8273, ..., 19283]                │
│  Answer tokens: [73421, 151645]  (positive + EOS)        │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ Combine + Label Masking                                  │
├──────────────────────────────────────────────────────────┤
│  input_ids:  [51234, 8273, ..., 19283, 73421, 151645]    │
│  labels:     [-100,  -100, ..., -100,  73421, 151645]    │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ Qwen3 0.6B → Predict next tokens in sequence             │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│ Loss computed only on non-masked tokens (73421, 151645)  │
│ → Backprop → Update weights                              │
└──────────────────────────────────────────────────────────┘
```
Here's our full training script
```python
import os
import datetime as dt
import random
import textwrap
from itertools import chain

import polars as pl
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding,
)
from functools import partial


def eval_finbert():
    """
    Evaluate the finbert model.

    :return:
    """
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Prefer Macro F1.
        f1 = f1_score(labels, predictions, average="macro")
        accuracy = accuracy_score(labels, predictions)

        return {"f1": f1, "accuracy": accuracy}

    finbert_model_id = "ProsusAI/finbert"

    finbert_label2id = {
        "positive": 0,
        "negative": 1,
        "neutral": 2,
    }
    finbert_id2label = {v: k for k, v in finbert_label2id.items()}
    finbert_num_labels = len(finbert_label2id)

    finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_id)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(
        finbert_model_id,
        num_labels=finbert_num_labels,
        label2id=finbert_label2id,
        id2label=finbert_id2label,
    )

    def tokenize_finbert(batch):
        return finbert_tokenizer(batch["text"], truncation=True, max_length=512)

    # PhraseBank with FinBERT tokenizer.
    finbert_phrase_ds = fin_bank_ds.map(tokenize_finbert, batched=True)
    val_ds_for_finbert = val_ds.map(tokenize_finbert, batched=True)

    finbert_collator = DataCollatorWithPadding(tokenizer=finbert_tokenizer)

    finbert_eval_args = TrainingArguments(
        output_dir="finbert_baseline_eval",
        per_device_eval_batch_size=32,
        do_train=False,
        do_eval=True,
        logging_strategy="no",
    )

    # Evaluate FinBERT on PhraseBank.
    finbert_trainer_phrase = Trainer(
        model=finbert_model,
        args=finbert_eval_args,
        data_collator=finbert_collator,
        eval_dataset=finbert_phrase_ds,
        compute_metrics=compute_metrics,
    )

    print("\n---")
    print("Evaluating FinBERT on Financial PhraseBank...")
    finbert_phrase_metrics = finbert_trainer_phrase.evaluate()
    print("FinBERT metrics on PhraseBank:", finbert_phrase_metrics)
    print("---")

    # Evaluate FinBERT on 20% of data_labels.
    finbert_trainer_val = Trainer(
        model=finbert_model,
        args=finbert_eval_args,
        data_collator=finbert_collator,
        eval_dataset=val_ds_for_finbert,
        compute_metrics=compute_metrics,
    )

    print("\n---")
    print("Evaluating FinBERT on 20% held-out data_labels...")
    finbert_val_metrics = finbert_trainer_val.evaluate()
    print("FinBERT metrics on 20% data_labels held-out set:", finbert_val_metrics)
    print("---\n")

    # Move the finbert model off the gpu.
    finbert_model.to(device="cpu")


def compute_metrics_without_eos(eval_pred, eos_token_id):
    """
    Compute the f1 score and the accuracy based on TOKEN classification. Given we are predicting a single token
    this is a good approximation of the actual scores.

    Make sure to remove eos token.

    :param eval_pred: Predictions.
    :param eos_token_id: The eos token id.
    :return:
    """
    predictions, labels = eval_pred
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]

    # Exclude Prompt (-100) AND EOS.
    valid_mask = (labels != -100) & (labels != eos_token_id)

    pred_flat = predictions[valid_mask]
    label_flat = labels[valid_mask]

    f1 = f1_score(label_flat, pred_flat, average="macro")
    accuracy = accuracy_score(label_flat, pred_flat)
    return {"f1": f1, "accuracy": accuracy}


def preprocess_logits_for_metrics(logits, labels):
    """
    Original logits are (Batch, Seq, Vocab).
    We only need (Batch, Seq) containing the indices of the max logit.
    """
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


if __name__ == "__main__":

    # ----------------------------------------------------
    # STEP 1 - Load financial sentiment dataset
    # ----------------------------------------------------

    task = "financial_sentiment"

    # Target TEXT strings for the Guard model.
    label_map = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral"
    }

    modeling_dir = os.path.join("/", "workspace", ".nosible")

    data_labels = pl.read_ipc(os.path.join(modeling_dir, f"{task}_100.0k_iter_18.ipc"))
    fin_bank = pl.read_ndjson(os.path.join(modeling_dir, "financial_phrase_bank.ndjson"))

    # Select text and text-labels.
    # Check if 'majority_vote' exists (it does in your IPC), otherwise rename or use 'labels'.
    if "majority_vote" in data_labels.columns:
        data = data_labels.select(["text", "majority_vote"]).rename({"majority_vote": "labels"})
    else:
        data = data_labels.select(["text", "labels"])

    fin_bank = fin_bank.select(["text", "labels"])

    print(f"Data shape: {data.shape}")

    # ----------------------------------------------------
    # STEP 2 - Create train/val splits
    # ----------------------------------------------------

    # Slice to 100k only if we have that much data, otherwise take all.
    limit = min(100_000, len(data))
    data = data[:limit]

    full_ds = Dataset.from_polars(df=data)
    fin_bank_ds = Dataset.from_polars(df=fin_bank)

    # Now this will succeed because we have at least 20 rows (16 train, 4 test)
    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    ds = DatasetDict({
        "train": train_ds,
        "val": val_ds,
        "phrasebank": fin_bank_ds,
    })

    # ----------------------------------------------------
    # STEP 3 - Fine-tune Qwen 3 (Guard Approach)
    # ----------------------------------------------------

    # UPDATED: Use Qwen 3 (0.6B).
    model_id = "Qwen/Qwen3-0.6B"

    # Important: trust_remote_code=True is essential for Qwen3.
    # Make sure to set left padding.
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load Model.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    def tokenize(batch):
        """
        Tokenize a batch.

        :param batch: Batch to tokenize.
        :return:
        """

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for text, label in zip(batch['text'], batch['labels']):

            # 1. Build the prompt (User part)
            system = "Classify the financial sentiment as positive, negative, or neutral."
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ]
            prompt_str = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False, # By default, this is true.
            )

            # 2. Build the answer (Assistant part)
            answer_str = f"{label}<|im_end|>" # Standard Qwen end token.

            # 3. Tokenize separately to know lengths
            prompt_tokens = tokenizer(prompt_str, add_special_tokens=False)
            answer_tokens = tokenizer(answer_str, add_special_tokens=False)

            assert len(answer_tokens["input_ids"]) == 2

            # 4. Combine
            input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
            attention_mask = prompt_tokens["attention_mask"] + answer_tokens["attention_mask"]

            # 5. CREATE LABELS WITH MASKING
            # -100 tells PyTorch to IGNORE these tokens during training.
            labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]

            # Truncate if necessary (simplified for brevity).
            if len(input_ids) > 2048:
                input_ids = input_ids[:2048]
                attention_mask = attention_mask[:2048]
                labels = labels[:2048]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }

    # Tokenize the datasets.
    t_ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names, num_proc=8)

    # Set the collator for padding.
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    task_output = f"{task}_qwen3_0.6B_{timestamp}"

    training_args = TrainingArguments(
        output_dir=task_output,

        # Qwen3 0.6B is very small, we might be able to increase batch size slightly.
        gradient_accumulation_steps=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=5,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        weight_decay=0.1,

        # Improvements for instruction fine-tuning.
        neftune_noise_alpha=5,

        # Optimizations.
        bf16=True,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=10,
        logging_dir=task_output,
        logging_first_step=True,
        group_by_length=True,
        torch_compile=False,

        # Eval params.
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_loss",
        report_to="none",
        save_strategy="steps",
        save_steps=500,

    )

    n_eval_train = int(0.05 * len(t_ds["train"]))
    chat_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=t_ds["train"],
        eval_dataset={
            "train": t_ds["train"].select(list(range(n_eval_train))),
            "val": t_ds["val"],
            "phrasebank": t_ds["phrasebank"],
        },
        compute_metrics=partial(compute_metrics_without_eos, eos_token_id=chat_end_token_id),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
```
## Results
How did our fine-tuned Qwen3 0.6B compare to FinBERT and frontier LLMs?
### Accuracy on Financial PhraseBank (Fake News)
We compare the accuracy of our fine-tuned Qwen3 0.6B model against FinBERT and a variety of LLMs prompted for zero-shot classification on the Financial PhraseBank dataset by Malo, P. et al. (2014). This dataset contains 4,840 financial news snippets labeled for sentiment by human experts.
![Financial sentiment accuracy comparison on PhraseBank dataset](https://nosible.com/images/2025/12/financial_sentiment_accuracy_on_phrasebank_dataset.png)Financial sentiment accuracy comparison on PhraseBank dataset
### Accuracy on Real-World Data (Real News)
Crucially, our model performs well on real world data too, not just benchmark datasets. Here are the accuracy results on the out of sample validation set from our 100,000 dataset.
![Financial sentiment accuracy comparison on NOSIBLE financial sentiment dataset](https://nosible.com/images/2025/12/financial_sentiment_accuracy_on_nosible_fin_sent_dataset.png)Financial sentiment accuracy comparison on NOSIBLE financial sentiment dataset
### Cost vs Accuracy tradeoff
Finally, we compared the accuracy vs cost tradeoff of our model. This is important because while large LLMs can achieve high accuracy, their inference costs can be prohibitive at scale. Our fine-tuned model strikes a good balance between accuracy and cost.
Although GPT-5.1 topped us by a margin, it's orders of magnitude more expensive, so in practical terms it's not feasible at our (NOSIBLE's) scale.
![Accuracy vs Cost plot: Financial sentiment accuracy on Financial PhraseBank vs Cost](https://nosible.com/images/2025/12/financial_sentiment_accuracy_vs_cost_on_phrasebank_dataset.png)Accuracy vs Cost plot: Financial sentiment accuracy on Financial PhraseBank vs Cost
![Accuracy vs Cost plot: Financial sentiment accuracy on NOSIBLE financial sentiment dataset vs Cost](https://nosible.com/images/2025/12/financial_sentiment_accuracy_vs_cost_on_nosible_dataset.png)Accuracy vs Cost plot: Financial sentiment accuracy on NOSIBLE financial sentiment dataset vs Cost
**Cost calculation methodology:**
**For the LLMs:** We summed input and output token costs per million in a 10:1 ratio, which reflects the typical input/output length relationship when prompted with our labeling prompt.
**For the fine-tuned model:** Since Qwen3 0.6B isn't hosted on OpenRouter, we estimated cost at a conservative 100:1 ratio given the minimal output—just the label token and EOS token.
## Conclusion
Circling back to the question &quot;do we really need another sentiment model?":
#### FinBERT didn't solve financial sentiment in 2022
FinBERT, the established benchmark model, performs poorly on our real-world data despite strong performance on Financial PhraseBank. This suggests overfitting to cleaner, more structured benchmark datasets rather than the messy, varied text found in actual web data.
#### LLMs are good at sentiment but don't scale
Large LLMs like GPT-5.1 excel at financial sentiment classification but are prohibitively expensive at production scale. We show they are orders of magnitude more costly than our fine-tuned Qwen3 0.6B model at similar accuracy levels.
#### Active learning makes auto-labeling possible at scale
Active learning with our relabeling algorithm is key to creating a large, real-world and high-quality dataset from which models can learn effectively.
One final insight from the project: causal models aren't just for text generation. When you understand how to leverage [logprobs](https://cookbook.openai.com/examples/using_logprobs), we show they can be powerful, production ready classification models too.
### Try it out!
As a reminder, we've open sourced everything you need to build this yourself:
* Financial Sentiment:
* [NOSIBLE Financial Sentiment](https://huggingface.co/datasets/NOSIBLE/financial-sentiment) dataset
* [NOSIBLE Financial Sentiment v1.1 Base](https://huggingface.co/NOSIBLE/financial-sentiment-v1.1-base) model
* Forward Looking:
* [NOSIBLE Forward-Looking](https://huggingface.co/datasets/NOSIBLE/forward-looking) dataset
* [NOSIBLE Forward-Looking v1.2 Base](https://huggingface.co/NOSIBLE/forward-looking-v1.2-base) model
* Prediction:
* [NOSIBLE Prediction](https://huggingface.co/datasets/NOSIBLE/prediction) dataset
* [NOSIBLE Prediction v1.1 Base](https://huggingface.co/NOSIBLE/prediction-v1.1-base) model
**Quickstart on HuggingFace**
1. Visit our [model page](https://huggingface.co/NOSIBLE/financial-sentiment-v1.1-base)
2. Click **Deploy** -> **HF Inference Endpoints**
3. Configure the endpoint:
* For Hardware choose a beefy GPU like a L40S
* For Inference Engine choose SGLang and set Max Prefill Tokens: `65536`
* For Container add container args: `--dtype float16 --cuda-graph-max-bs 128 --disable-radix-cache`
* For Advanced Settings choose Download Pattern: **Download Everything**
* Copy your endpoint URL and bring your HuggingFace API KEY.
* Use the code below to classify some text.
* Profit 🐒
Client code to interact with your HF Inference Endpoint.
```python
import math
from openai import OpenAI

# Initialize the client pointing to your vLLM server
client = OpenAI(
    base_url="YOUR_ENDPOINT_URL_HERE/v1",
    api_key="YOUR_API_KEY_HERE"
)

model_id = "NOSIBLE/financial-sentiment-v1.1-base"

# Input text to classify
text = "The company reported a record profit margin of 15% this quarter."

# Define the classification labels
labels = ["positive", "negative", "neutral"]

# Prepare the conversation
messages = [
    {"role": "system", "content": "Classify the financial sentiment as positive, neutral, or negative."},
    {"role": "user", "content": text},
]

# Make the API call
chat_completion = client.chat.completions.create(
    model=model_id,
    messages=messages,
    temperature=0,
    max_tokens=1,
    stream=False,
    logprobs=True,              # Enable log probabilities to calculate confidence
    top_logprobs=len(labels),   # Ensure we capture logprobs for our choices
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}, # Must be set to false.
        "regex": "(positive|neutral|negative)",
    },
)

# Extract the response content
response_label = chat_completion.choices[0].message.content

# Extract the logprobs for the generated token to calculate confidence
first_token_logprobs = chat_completion.choices[0].logprobs.content[0].top_logprobs

print(f"--- Classification Results ---")
print(f"Input: {text}")
print(f"Predicted Label: {response_label}\n")

print("--- Label Confidence ---")
for lp in first_token_logprobs:
    # Convert log probability to percentage
    probability = math.exp(lp.logprob)
    print(f"Token: '{lp.token}' | Probability: {probability:.2%}")
```
## Acknowledgments
The team involved in the project includes:
* [**Matthew Dicks**](https://www.linkedin.com/in/matthewdicks98/)
* [**Simon van Dyk**](https://www.linkedin.com/in/simon-van-dyk/)
* [**Gareth Warburton**](https://www.linkedin.com/in/garethwarburton/)
* [**Stuart Reid**](https://www.linkedin.com/in/stuartgordonreid/)
## Citations
**Datasets:**
* [**NOSIBLE Financial Sentiment:**](https://huggingface.co/datasets/NOSIBLE/financial-sentiment) Nosible Inc.
* [**Financial PhraseBank:**](https://huggingface.co/datasets/takala/financial_phrasebank) Malo, P. et al. (2014).
**Research:**
* **Qwen3:** *Qwen3 Technical Report*. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
* **Qwen3 Guard:** *Qwen3Guard Technical Report*. [arXiv:2510.14276v1](https://arxiv.org/html/2510.14276v1)
**NOSIBLE Search Feed Examples:**
* [Tesla Feed sample top-30](https://nosible.com/files/2025/12/tesla-top-30.xlsx)
* [Tesla Feed signals sample](https://nosible.com/files/2025/12/tesla-signals-sample.xlsx)
**Footnotes:**
## Footnotes
1. **FinBERT:** Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. [arXiv:1908.10063](https://arxiv.org/abs/1908.10063) [↩](#user-content-fnref-1)
2. **Financial PhraseBank:** [Malo, P. et al. (2014)](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) [↩](#user-content-fnref-2)
[All Research](https://nosible.com/blog)
Related Research
![3D cube illustration representing LLM ensemble distillation](https://nosible.com/blog/illustrations/cube.png)
[Technical](https://nosible.com/blog/tag/technical)[Sentiment](https://nosible.com/blog/tag/sentiment)[Signals](https://nosible.com/blog/tag/signals)
### [A Pattern for Scaling the Value Proposition of LLMs: Ensemble and Distil 🚀](https://nosible.com/blog/ensemble-and-distil)
2024-02-0612 min read
![Signed contrast number line separating systemic and idiosyncratic risk](https://nosible.com/images/2026/06/walkthrough_step9_number_line.png)
[Research](https://nosible.com/blog/tag/research)[Technical](https://nosible.com/blog/tag/technical)[Classification](https://nosible.com/blog/tag/classification)[Nosible World](https://nosible.com/blog/tag/nosible-world)
### [Two Tricks for Turning Sentence Embeddings into Clean Features](https://nosible.com/blog/the-contrastive-geometry-of-risk)
2026-06-1814 min read
![Abstract eye illustration representing financial-news sentiment analysis](https://nosible.com/blog/illustrations/eye.png)
[Technical](https://nosible.com/blog/tag/technical)[Sentiment](https://nosible.com/blog/tag/sentiment)
### [News Sentiment Showdown: Who Checks Vibes Best?](https://nosible.com/blog/news-sentiment-showdown-who-checks-vibes-best)
2024-01-2813 min read
