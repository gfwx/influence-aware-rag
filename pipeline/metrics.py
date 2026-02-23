"""
Per-query metric computation (PLAN.md Phase 4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def compute_spearman(ablation_results: list[dict[str, Any]]) -> tuple[float, float]:
    """
    Compute Spearman rank correlation between retrieval ranks
    and influence ranks.

    Returns
    -------
    (rho, pvalue)
    """
    retrieval_ranks = [r["retrieval_rank"] for r in ablation_results]
    influence_ranks = [r["influence_rank"] for r in ablation_results]
    rho, pvalue = stats.spearmanr(retrieval_ranks, influence_ranks)
    return float(rho), float(pvalue)


def compute_dominance_ratio(ablation_results: list[dict[str, Any]]) -> float:
    """max(influence) / sum(influence)."""
    scores = [r["influence_score"] for r in ablation_results]
    total = sum(scores)
    if total == 0:
        return 0.0
    return max(scores) / total


def compute_influence_entropy(ablation_results: list[dict[str, Any]]) -> float:
    """Shannon entropy of normalised influence scores."""
    scores = np.array([r["influence_score"] for r in ablation_results])
    total = scores.sum()
    if total == 0:
        return 0.0
    probs = scores / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def is_divergent(rho: float, threshold: float = 0.7) -> bool:
    """Flag a query as divergent if Spearman ρ < threshold."""
    return rho < threshold
