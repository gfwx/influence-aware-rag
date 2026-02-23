"""
BM25 retrieval and RAG prompt construction (PLAN.md Phase 2).
"""

from __future__ import annotations

import logging
from typing import Any

from rank_bm25 import BM25Okapi

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def build_bm25(passages: list[str]) -> BM25Okapi:
    """Build a BM25Okapi index over a list of passage strings."""
    tokenized = [p.lower().split() for p in passages]
    return BM25Okapi(tokenized)


def retrieve_top_k(
    query: str,
    passages: list[str],
    k: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-k passages for a query using BM25.

    Returns
    -------
    list[dict]
        Each dict has keys:
          ``doc_id``, ``text``, ``retrieval_score``, ``retrieval_rank``.
    """
    bm25 = build_bm25(passages)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:k]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices):
        results.append(
            {
                "doc_id": idx,
                "text": passages[idx],
                "retrieval_score": float(scores[idx]),
                "retrieval_rank": rank + 1,  # 1 = best
            }
        )
    return results


def build_rag_prompt(query: str, docs: list[dict[str, Any]]) -> str:
    """
    Construct a RAG prompt that asks the LLM to answer using only
    the provided documents.
    """
    context_block = ""
    for i, doc in enumerate(docs):
        context_block += f"[Document {i + 1}]: {doc['text']}\n\n"

    prompt = (
        "You are a helpful assistant. Answer the question below using ONLY "
        "the provided documents.\n"
        'Be concise. If the documents do not contain the answer, say "I don\'t know."\n\n'
        f"{context_block}"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt
