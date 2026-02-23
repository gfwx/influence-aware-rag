"""
Ablation loop and influence scoring (PLAN.md Phase 3).
"""

from __future__ import annotations

import logging
from typing import Any

from rouge_score import rouge_scorer

from pipeline.config import PipelineConfig
from pipeline.llm import call_llm
from pipeline.retrieval import build_rag_prompt

logger = logging.getLogger(__name__)

# Module-level scorer — instantiated once, reused across calls.
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def generate_baseline(
    query: str,
    docs: list[dict[str, Any]],
    config: PipelineConfig | None = None,
) -> str:
    """Generate the baseline answer with all documents in context."""
    prompt = build_rag_prompt(query, docs)
    return call_llm(prompt, config)


def run_ablation(
    query: str,
    docs: list[dict[str, Any]],
    baseline_answer: str,
    config: PipelineConfig | None = None,
) -> list[dict[str, Any]]:
    """
    For each document, remove it from context and regenerate the answer.
    Compute ROUGE-L between baseline and ablated answer to derive influence.

    Returns
    -------
    list[dict]
        One entry per document with keys:
          doc_id, retrieval_rank, retrieval_score,
          ablated_answer, rouge_l, influence_score.
    """
    influence_results: list[dict[str, Any]] = []

    for i in range(len(docs)):
        # Remove document i
        ablated_docs = [d for j, d in enumerate(docs) if j != i]

        prompt = build_rag_prompt(query, ablated_docs)
        ablated_answer = call_llm(prompt, config)

        # ROUGE-L between baseline and ablated
        score = _scorer.score(baseline_answer, ablated_answer)
        rouge_l = score["rougeL"].fmeasure

        # Influence = magnitude of change (higher → more influential)
        influence_score = 1.0 - rouge_l

        influence_results.append(
            {
                "doc_id": docs[i]["doc_id"],
                "passage": docs[i].get("passage", ""),
                "retrieval_rank": docs[i]["retrieval_rank"],
                "retrieval_score": docs[i]["retrieval_score"],
                "ablated_answer": ablated_answer,
                "rouge_l": rouge_l,
                "influence_score": influence_score,
            }
        )

    return influence_results


def assign_influence_ranks(
    ablation_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Sort by influence score descending and assign 1-based influence ranks.
    Modifies the dicts in-place and returns the same list.
    """
    sorted_results = sorted(
        ablation_results,
        key=lambda x: x["influence_score"],
        reverse=True,
    )
    for rank, result in enumerate(sorted_results):
        result["influence_rank"] = rank + 1
    return ablation_results
