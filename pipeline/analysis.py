"""
Aggregation and statistical analysis (PLAN.md Phase 6).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)


def compute_summary_stats(results_log: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Build a summary DataFrame with one row per query, then print
    aggregate statistics to the logger.

    Returns
    -------
    pd.DataFrame
        Columns: query, spearman_rho, dominance_ratio, influence_entropy,
        is_divergent.
    """
    summary_df = pd.DataFrame(
        [
            {
                "query": r["query"],
                "spearman_rho": r["spearman_rho"],
                "spearman_pvalue": r["spearman_pvalue"],
                "dominance_ratio": r["dominance_ratio"],
                "influence_entropy": r["influence_entropy"],
                "is_divergent": r["is_divergent"],
            }
            for r in results_log
        ]
    )

    total = len(summary_df)
    pct_div = summary_df["is_divergent"].mean() * 100
    mean_rho = summary_df["spearman_rho"].mean()
    median_rho = summary_df["spearman_rho"].median()
    mean_dom = summary_df["dominance_ratio"].mean()
    mean_ent = summary_df["influence_entropy"].mean()

    logger.info("=== Summary Statistics ===")
    logger.info("Total queries evaluated: %d", total)
    logger.info("%% Divergent (rho < 0.7): %.1f%%", pct_div)
    logger.info("Mean Spearman rho: %.3f", mean_rho)
    logger.info("Median Spearman rho: %.3f", median_rho)
    logger.info("Mean Dominance Ratio: %.3f", mean_dom)
    logger.info("Mean Influence Entropy: %.3f", mean_ent)

    return summary_df


def identify_failure_modes(results_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Find documents with high retrieval rank (1-3) but low influence
    rank (≥ 8). These are cases where the retriever surfaces content
    the LLM ignores.
    """
    failure_modes: list[dict[str, Any]] = []
    for r in results_log:
        for doc in r["ablation_details"]:
            if doc["retrieval_rank"] <= 3 and doc["influence_rank"] >= 8:
                failure_modes.append(
                    {
                        "query": r["query"],
                        "doc_id": doc["doc_id"],
                        "retrieval_rank": doc["retrieval_rank"],
                        "influence_rank": doc["influence_rank"],
                        "influence_score": doc["influence_score"],
                    }
                )

    logger.info("Failure modes detected: %d", len(failure_modes))
    return failure_modes


def run_significance_test(
    summary_df: pd.DataFrame,
) -> tuple[float, float, bool]:
    """
    Run a Mann-Whitney U test comparing Spearman ρ values for
    divergent vs. non-divergent queries.

    Returns
    -------
    (statistic, p_value, is_significant)
    """
    divergent = summary_df[summary_df["is_divergent"]]["spearman_rho"]
    non_divergent = summary_df[~summary_df["is_divergent"]]["spearman_rho"]

    if len(divergent) == 0 or len(non_divergent) == 0:
        logger.warning(
            "Cannot run significance test — one group is empty "
            "(divergent=%d, non-divergent=%d).",
            len(divergent),
            len(non_divergent),
        )
        return 0.0, 1.0, False

    stat, p = mannwhitneyu(divergent, non_divergent, alternative="two-sided")
    significant = p < 0.05
    logger.info("Mann-Whitney U p-value: %.4f (%s)", p,
                "significant" if significant else "not significant")
    return float(stat), float(p), significant
