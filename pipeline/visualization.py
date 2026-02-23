"""
Matplotlib visualisations (PLAN.md Phase 7).
"""

from __future__ import annotations

import os
import logging

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_spearman_distribution(
    summary_df: pd.DataFrame,
    threshold: float = 0.7,
    save_path: str = "spearman_distribution.png",
) -> str:
    """
    Histogram of Spearman ρ values with a divergence threshold line.

    Returns the path to the saved image.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        summary_df["spearman_rho"],
        bins=20,
        color="#2563eb",
        edgecolor="black",
    )
    ax.axvline(
        x=threshold,
        color="red",
        linestyle="--",
        label=f"Divergence threshold (ρ={threshold})",
    )
    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Distribution of Retrieval–Influence Rank Correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved Spearman distribution plot → %s", save_path)
    return save_path


def plot_rank_scatter(
    sample_result: dict,
    save_path: str = "rank_scatter_example.png",
) -> str:
    """
    Scatter plot of retrieval rank vs. influence rank for one query.

    Returns the path to the saved image.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    ablation = sample_result["ablation_details"]
    ret_ranks = [d["retrieval_rank"] for d in ablation]
    inf_ranks = [d["influence_rank"] for d in ablation]
    k = len(ret_ranks)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ret_ranks, inf_ranks, color="#888888", s=100, edgecolors="#000000")
    ax.plot([1, k], [1, k], "k--", alpha=0.4, label="Perfect agreement")
    ax.set_xlabel("Retrieval Rank")
    ax.set_ylabel("Influence Rank")
    query_short = sample_result["query"][:60]
    ax.set_title(f'Rank Agreement — Example Query\n"{query_short}..."')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved rank scatter plot → %s", save_path)
    return save_path


def plot_dominance_distribution(
    summary_df: pd.DataFrame,
    save_path: str = "dominance_distribution.png",
) -> str:
    """
    Histogram of dominance ratios across all queries.

    Returns the path to the saved image.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        summary_df["dominance_ratio"],
        bins=20,
        color="#888888",
        edgecolor="black",
    )
    ax.set_xlabel("Dominance Ratio")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Document Dominance Ratio Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved dominance distribution plot → %s", save_path)
    return save_path
