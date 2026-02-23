# Influence-Aware RAG 🔬

This project implements an **Influence-Aware Retrieval-Augmented Generation (RAG)** pipeline. It analyzes how individual retrieved documents contribute to an LLM's final answer, using an ablation-based methodology to quantify and rank document influence. It also includes a robust Streamlit dashboard for visualizing the results.

## Overview

Traditional RAG systems retrieve top-k documents and feed them to an LLM. However, it's often unclear _which_ of those documents the LLM actually relied upon. This project solves that by:

1. Generating a baseline answer using all retrieved documents.
2. Systematically removing (ablating) one document at a time.
3. Measuring how much the answer changes (drops in ROUGE-L score) when a document is removed.
4. Comparing the retriever's ranking against the LLM's actual usage (influence ranking) using Spearman's rank correlation.

## Key Features

- **End-to-End Pipeline**: From dataset streaming and BM25 retrieval to LLM ablation and statistical analysis.
- **Robust LLM Integration**: Uses the OpenRouter API with built-in retry logic and exponential backoff.
- **Metrics Engine**: Computes Spearman correlation, Dominance Ratio, and Influence Entropy.
- **Interactive Dashboard**: A custom-themed Streamlit app (`app.py`) for visualising pipeline results and identifying failure modes.

## Project Structure

```
minor-project/
├── app.py                   # Streamlit dashboard
├── run_pipeline.py          # CLI entry point to execute the pipeline
├── requirements.txt         # Project dependencies
├── PLAN.md                  # Original architecture and methodology spec
├── outputs/                 # CSV and visualization outputs
└── pipeline/                # Core modular package
    ├── __init__.py
    ├── ablation.py          # Core ablation tracking logic
    ├── analysis.py          # Summary statistics & failure mode detection
    ├── config.py            # Centralised configuration
    ├── data.py              # Dataset streaming (Google Natural Questions)
    ├── llm.py               # OpenRouter API integration
    ├── metrics.py           # Spearman, dominance, entropy calculations
    ├── retrieval.py         # BM25 indexing and retrieving
    ├── runner.py            # Main pipeline orchestrator
    └── visualization.py     # Static matplotlib charts
```

## How It Works (Code Snippets)

### 1. Retrieval (BM25)

The pipeline splits documents into ~100-word passages and uses BM25 to find the top-10 most relevant passages for a given query.

```python
# From pipeline/retrieval.py
def retrieve_top_k(query: str, passages: list[str], k: int = 10):
    bm25 = build_bm25(passages)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    # Sort and pick top k...
```

### 2. The Baseline Answer

We ask the LLM to answer the query using all top-k retrieved documents.

```python
# From pipeline/ablation.py
prompt = build_rag_prompt(query, docs)
baseline_answer = call_llm(prompt, config)
```

### 3. The Ablation Loop

We loop through each document, remove it from the context, generate a new answer, and see how much the answer degraded compared to the baseline using ROUGE-L.

```python
# From pipeline/ablation.py
for i in range(len(docs)):
    # Remove doc i
    ablated_docs = [d for j, d in enumerate(docs) if j != i]

    # Generate new answer
    prompt = build_rag_prompt(query, ablated_docs)
    ablated_answer = call_llm(prompt, config)

    # Calculate influence: 1.0 - ROUGE-L(baseline, ablated)
    score = _scorer.score(baseline_answer, ablated_answer)
    rouge_l = score["rougeL"].fmeasure
    influence_score = 1.0 - rouge_l
```

### 4. Evaluation (Spearman's Rho)

We compare the original BM25 ranks (e.g., this doc was #1) against the influence ranks (e.g., this doc actually caused the biggest change when removed).

```python
# From pipeline/metrics.py
rho, pvalue = stats.spearmanr(retrieval_ranks, influence_ranks)
```

## Usage

### Run the Pipeline

Note: You must set the `OPENROUTER_API_KEY` environment variable.

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
export OPENROUTER_API_KEY="sk-or-..."
python run_pipeline.py --n-samples 50 --model openai/gpt-oss-20b
```

### Run the Dashboard

```bash
streamlit run app.py
```

You can toggle between **Simulation** mode (auto-generates data for demo purposes) and **Load Results (CSV)** to view the exact output of your pipeline run.
