"""
Full pipeline loop (PLAN.md Phase 5).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline.retrieval import retrieve_top_k
from pipeline.ablation import generate_baseline, run_ablation, assign_influence_ranks
from pipeline.metrics import (
    compute_spearman,
    compute_dominance_ratio,
    compute_influence_entropy,
    is_divergent,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    queries: list[dict[str, Any]],
    config: PipelineConfig | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Execute the full ablation pipeline over all queries.

    Parameters
    ----------
    queries : list[dict]
        Each has keys "query" (str) and "passages" (list[str]).
    config : PipelineConfig, optional
    progress_callback : callable, optional
        Called with (current_index, total, status_message) after each query.

    Returns
    -------
    list[dict]
        One entry per query with all metrics and ablation details.
    """
    if config is None:
        config = PipelineConfig()
    config.validate()

    results_log: list[dict[str, Any]] = []
    # Filter queries with enough passages
    eligible = [q for q in queries if len(q["passages"]) >= config.top_k]
    skipped = len(queries) - len(eligible)
    if skipped:
        logger.info(
            "Skipped %d queries with fewer than %d passages.",
            skipped,
            config.top_k,
        )

    total = len(eligible)
    for idx, entry in enumerate(tqdm(eligible, desc="Pipeline")):
        query = entry["query"]
        passages = entry["passages"]

        if progress_callback:
            progress_callback(idx, total, f"Processing query {idx + 1}/{total}")

        # Step 1: Retrieve top-k
        docs = retrieve_top_k(query, passages, k=config.top_k)

        # Step 2: Baseline generation
        baseline_answer = generate_baseline(query, docs, config)

        # Step 3: Ablation loop
        ablation_results = run_ablation(query, docs, baseline_answer, config)

        # Step 4: Assign influence ranks
        ablation_results = assign_influence_ranks(ablation_results)

        # Step 5: Compute metrics
        rho, pvalue = compute_spearman(ablation_results)
        dominance = compute_dominance_ratio(ablation_results)
        entropy = compute_influence_entropy(ablation_results)
        divergent = is_divergent(rho, config.divergence_threshold)

        results_log.append(
            {
                "query": query,
                "baseline_answer": baseline_answer,
                "spearman_rho": rho,
                "spearman_pvalue": pvalue,
                "dominance_ratio": dominance,
                "influence_entropy": entropy,
                "is_divergent": divergent,
                "ablation_details": ablation_results,
            }
        )

    if progress_callback:
        progress_callback(total, total, "Pipeline complete")

    return results_log


def save_results_csv(
    results_log: list[dict[str, Any]],
    path: str = "rag_influence_results.csv",
) -> pd.DataFrame:
    """
    Flatten ``results_log`` into one row per document and save as CSV.

    Returns the flattened DataFrame.
    """
    flat_rows: list[dict[str, Any]] = []
    for r in results_log:
        for doc in r["ablation_details"]:
            row = {
                "query": r["query"],
                "baseline_answer": r["baseline_answer"],
                "spearman_rho": r["spearman_rho"],
                "spearman_pvalue": r["spearman_pvalue"],
                "dominance_ratio": r["dominance_ratio"],
                "influence_entropy": r["influence_entropy"],
                "is_divergent": r["is_divergent"],
            }
            # Merge doc-level fields (exclude ablated_answer to keep CSV small)
            for k, v in doc.items():
                if k != "ablated_answer":
                    row[k] = v
            flat_rows.append(row)

    df = pd.DataFrame(flat_rows)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Results saved to %s (%d rows).", path, len(df))
    return df
