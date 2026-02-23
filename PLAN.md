## Prerequisites

### Dependencies

```bash
pip install rank_bm25 datasets rouge_score scipy matplotlib pandas requests tqdm
```

### OpenRouter Setup

- Sign up at https://openrouter.ai and obtain an API key
- Set your key as an environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

- Recommended model: `meta-llama/llama-3.2-8b-instruct` (cheap, fast)
- Alternative: `mistralai/mistral-7b-instruct`

### OpenRouter LLM Call Helper

```python
import os, requests

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-oss-20b"

def call_llm(prompt: str) -> str:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.0  # deterministic for reproducibility
        }
    )
    return response.json()["choices"][0]["message"]["content"].strip()
```

---

## Phase 1: Environment & Data Setup

### Step 1.1 — Load the Dataset

```python
from datasets import load_dataset

# Load Natural Questions (validation split is smaller and faster)
dataset = load_dataset("natural_questions", split="validation")

# Filter to questions with short answers only
def has_short_answer(example):
    annotations = example["annotations"]
    return any(len(a["short_answers"]) > 0 for a in annotations)

filtered = [ex for ex in dataset if has_short_answer(ex)]

# Sample 150 queries
import random
random.seed(42)
sample = random.sample(filtered, 150)
```

### Step 1.2 — Extract Queries and Build Corpus

```python
queries = []
corpus = []  # flat list of all document passages

for ex in sample:
    question = ex["question"]["text"]
    # Each NQ example has one long document — split into passages of ~100 words
    doc_text = ex["document"]["tokens"]["token"]
    doc_text = " ".join(doc_text)

    # Split document into passages (~100 words each)
    words = doc_text.split()
    passages = [" ".join(words[i:i+100]) for i in range(0, len(words), 100)]

    queries.append({
        "query": question,
        "passages": passages[:15]  # cap at 15 passages per query document
    })
```

### Step 1.3 — Build BM25 Index Per Query

```python
from rank_bm25 import BM25Okapi

def build_bm25(passages):
    tokenized = [p.lower().split() for p in passages]
    return BM25Okapi(tokenized)
```

> **Note:** BM25 is built per-query over that query's associated passages.
> This simulates a closed-corpus RAG setup and keeps the experiment controlled.

---

## Phase 2: Retrieval Pipeline (Per Query)

### Step 2.1 — Retrieve Top-10 Documents

```python
def retrieve_top_k(query: str, passages: list, k: int = 10):
    bm25 = build_bm25(passages)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices sorted by score descending
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "doc_id": idx,
            "text": passages[idx],
            "retrieval_score": scores[idx],
            "retrieval_rank": rank + 1  # 1 = best
        })
    return results
```

### Step 2.2 — Construct the RAG Prompt

```python
def build_rag_prompt(query: str, docs: list) -> str:
    context_block = ""
    for i, doc in enumerate(docs):
        context_block += f"[Document {i+1}]: {doc['text']}\n\n"

    prompt = f"""You are a helpful assistant. Answer the question below using ONLY the provided documents.
Be concise. If the documents do not contain the answer, say "I don't know."

{context_block}
Question: {query}
Answer:"""
    return prompt
```

### Step 2.3 — Generate Baseline Answer

```python
def generate_baseline(query: str, docs: list) -> str:
    prompt = build_rag_prompt(query, docs)
    return call_llm(prompt)
```

---

## Phase 3: Ablation Loop (Per Query)

For each of the 10 retrieved documents, remove it from the context, regenerate the answer,
and measure how much the output changes compared to the baseline.

### Step 3.1 — Run Ablation

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def run_ablation(query: str, docs: list, baseline_answer: str) -> list:
    influence_results = []

    for i in range(len(docs)):
        # Remove document i from context
        ablated_docs = [d for j, d in enumerate(docs) if j != i]

        # Generate answer without doc i
        prompt = build_rag_prompt(query, ablated_docs)
        ablated_answer = call_llm(prompt)

        # Compute ROUGE-L between baseline and ablated answer
        score = scorer.score(baseline_answer, ablated_answer)
        rouge_l = score["rougeL"].fmeasure

        # Influence = how much output CHANGED when doc was removed
        # Higher = doc had more influence
        influence_score = 1.0 - rouge_l

        influence_results.append({
            "doc_id": docs[i]["doc_id"],
            "retrieval_rank": docs[i]["retrieval_rank"],
            "retrieval_score": docs[i]["retrieval_score"],
            "ablated_answer": ablated_answer,
            "rouge_l": rouge_l,
            "influence_score": influence_score
        })

    return influence_results
```

### Step 3.2 — Assign Influence Ranks

```python
def assign_influence_ranks(ablation_results: list) -> list:
    # Sort by influence score descending (most influential = rank 1)
    sorted_results = sorted(ablation_results, key=lambda x: x["influence_score"], reverse=True)
    for rank, result in enumerate(sorted_results):
        result["influence_rank"] = rank + 1
    return ablation_results  # now has influence_rank field on each entry
```

---

## Phase 4: Metric Computation (Per Query)

### Step 4.1 — Spearman Correlation

```python
from scipy import stats

def compute_spearman(ablation_results: list) -> float:
    retrieval_ranks = [r["retrieval_rank"] for r in ablation_results]
    influence_ranks = [r["influence_rank"] for r in ablation_results]
    rho, pvalue = stats.spearmanr(retrieval_ranks, influence_ranks)
    return rho, pvalue
```

### Step 4.2 — Dominance Ratio

```python
def compute_dominance_ratio(ablation_results: list) -> float:
    scores = [r["influence_score"] for r in ablation_results]
    total = sum(scores)
    if total == 0:
        return 0.0
    return max(scores) / total
```

### Step 4.3 — Influence Entropy

```python
import numpy as np

def compute_influence_entropy(ablation_results: list) -> float:
    scores = np.array([r["influence_score"] for r in ablation_results])
    total = scores.sum()
    if total == 0:
        return 0.0
    probs = scores / total
    # Avoid log(0)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return entropy
```

### Step 4.4 — Divergence Flag

```python
def is_divergent(rho: float, threshold: float = 0.7) -> bool:
    return rho < threshold
```

---

## Phase 5: Full Pipeline Loop

```python
import pandas as pd
from tqdm import tqdm

results_log = []

for entry in tqdm(queries):
    query = entry["query"]
    passages = entry["passages"]

    # Skip if fewer than 10 passages available
    if len(passages) < 10:
        continue

    # Step 1: Retrieve top-10
    docs = retrieve_top_k(query, passages, k=10)

    # Step 2: Baseline generation
    baseline_answer = generate_baseline(query, docs)

    # Step 3: Ablation loop
    ablation_results = run_ablation(query, docs, baseline_answer)

    # Step 4: Assign influence ranks
    ablation_results = assign_influence_ranks(ablation_results)

    # Step 5: Compute metrics
    rho, pvalue = compute_spearman(ablation_results)
    dominance = compute_dominance_ratio(ablation_results)
    entropy = compute_influence_entropy(ablation_results)
    divergent = is_divergent(rho)

    results_log.append({
        "query": query,
        "baseline_answer": baseline_answer,
        "spearman_rho": rho,
        "spearman_pvalue": pvalue,
        "dominance_ratio": dominance,
        "influence_entropy": entropy,
        "is_divergent": divergent,
        "ablation_details": ablation_results
    })

# Save to CSV (flatten for export)
flat_rows = []
for r in results_log:
    for doc in r["ablation_details"]:
        flat_rows.append({
            "query": r["query"],
            "spearman_rho": r["spearman_rho"],
            "dominance_ratio": r["dominance_ratio"],
            "influence_entropy": r["influence_entropy"],
            "is_divergent": r["is_divergent"],
            **doc
        })

df = pd.DataFrame(flat_rows)
df.to_csv("rag_influence_results.csv", index=False)
print("Results saved to rag_influence_results.csv")
```

---

## Phase 6: Aggregation & Statistical Analysis

### Step 6.1 — Summary Statistics

```python
summary_df = pd.DataFrame([{
    "query": r["query"],
    "spearman_rho": r["spearman_rho"],
    "dominance_ratio": r["dominance_ratio"],
    "influence_entropy": r["influence_entropy"],
    "is_divergent": r["is_divergent"]
} for r in results_log])

print("=== Summary Statistics ===")
print(f"Total queries evaluated: {len(summary_df)}")
print(f"% Divergent (rho < 0.7): {summary_df['is_divergent'].mean() * 100:.1f}%")
print(f"Mean Spearman rho: {summary_df['spearman_rho'].mean():.3f}")
print(f"Median Spearman rho: {summary_df['spearman_rho'].median():.3f}")
print(f"Mean Dominance Ratio: {summary_df['dominance_ratio'].mean():.3f}")
print(f"Mean Influence Entropy: {summary_df['influence_entropy'].mean():.3f}")
```

### Step 6.2 — Identify Failure Modes

```python
# High retrieval rank (1-3) but low influence rank (8-10) = failure mode
failure_modes = []
for r in results_log:
    for doc in r["ablation_details"]:
        if doc["retrieval_rank"] <= 3 and doc["influence_rank"] >= 8:
            failure_modes.append({
                "query": r["query"],
                "doc_id": doc["doc_id"],
                "retrieval_rank": doc["retrieval_rank"],
                "influence_rank": doc["influence_rank"],
                "influence_score": doc["influence_score"]
            })

print(f"\nFailure modes detected: {len(failure_modes)}")
```

### Step 6.3 — Statistical Significance Test

```python
from scipy.stats import mannwhitneyu

divergent_rhos = summary_df[summary_df["is_divergent"]]["spearman_rho"]
non_divergent_rhos = summary_df[~summary_df["is_divergent"]]["spearman_rho"]

stat, p = mannwhitneyu(divergent_rhos, non_divergent_rhos, alternative="two-sided")
print(f"\nMann-Whitney U test p-value: {p:.4f}")
print("Significant difference" if p < 0.05 else "No significant difference")
```

---

## Phase 7: Visualizations

```python
import matplotlib.pyplot as plt

# Plot 1: Distribution of Spearman rho
plt.figure(figsize=(8, 5))
plt.hist(summary_df["spearman_rho"], bins=20, color="steelblue", edgecolor="black")
plt.axvline(x=0.7, color="red", linestyle="--", label="Divergence threshold (ρ=0.7)")
plt.xlabel("Spearman ρ")
plt.ylabel("Number of Queries")
plt.title("Distribution of Retrieval–Influence Rank Correlation")
plt.legend()
plt.savefig("spearman_distribution.png", dpi=150)
plt.show()

# Plot 2: Retrieval Rank vs Influence Rank scatter (sample query)
sample_result = results_log[0]
ret_ranks = [d["retrieval_rank"] for d in sample_result["ablation_details"]]
inf_ranks = [d["influence_rank"] for d in sample_result["ablation_details"]]

plt.figure(figsize=(6, 6))
plt.scatter(ret_ranks, inf_ranks, color="coral", s=100, edgecolors="black")
plt.plot([1, 10], [1, 10], "k--", alpha=0.4, label="Perfect agreement")
plt.xlabel("Retrieval Rank")
plt.ylabel("Influence Rank")
plt.title(f"Rank Agreement — Example Query\n\"{sample_result['query'][:60]}...\"")
plt.legend()
plt.savefig("rank_scatter_example.png", dpi=150)
plt.show()

# Plot 3: Dominance ratio distribution
plt.figure(figsize=(8, 5))
plt.hist(summary_df["dominance_ratio"], bins=20, color="mediumseagreen", edgecolor="black")
plt.xlabel("Dominance Ratio")
plt.ylabel("Number of Queries")
plt.title("Document Dominance Ratio Distribution")
plt.savefig("dominance_distribution.png", dpi=150)
plt.show()
```

---

## Key Variables Reference

| Variable            | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `retrieval_score`   | Raw BM25 score for a document                          |
| `retrieval_rank`    | Rank by BM25 score (1 = best)                          |
| `influence_score`   | `1 - ROUGE-L(baseline, ablated)`                       |
| `influence_rank`    | Rank by influence score (1 = most influential)         |
| `spearman_rho`      | Rank correlation between retrieval and influence ranks |
| `dominance_ratio`   | `max(influence) / sum(influence)`                      |
| `influence_entropy` | Shannon entropy of normalized influence scores         |
| `is_divergent`      | `True` if `spearman_rho < 0.7`                         |

---

## Expected Output Files

| File                         | Description                          |
| ---------------------------- | ------------------------------------ |
| `rag_influence_results.csv`  | Per-document metrics for all queries |
| `spearman_distribution.png`  | Histogram of Spearman ρ values       |
| `rank_scatter_example.png`   | Retrieval vs. influence rank scatter |
| `dominance_distribution.png` | Dominance ratio distribution         |

---

## Cost Estimate (OpenRouter)

Using `meta-llama/llama-3.2-8b-instruct` at ~$0.06/1M tokens:

- 150 queries × 11 LLM calls (1 baseline + 10 ablations) = **1,650 total calls**
- Average prompt ~800 tokens + response ~100 tokens ≈ 900 tokens/call
- Total: ~1.5M tokens ≈ **~$0.09 total** ✅

---
