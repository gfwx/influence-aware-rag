"""
Influence-Aware RAG Pipeline
=============================
Modular package for running the ablation-based document influence
analysis pipeline on top of a BM25 retriever + LLM generator.
"""

from pipeline.config import PipelineConfig
from pipeline.llm import call_llm
from pipeline.data import load_and_sample_dataset, extract_queries_and_passages
from pipeline.retrieval import build_bm25, retrieve_top_k, build_rag_prompt
from pipeline.ablation import generate_baseline, run_ablation, assign_influence_ranks
from pipeline.metrics import (
    compute_spearman,
    compute_dominance_ratio,
    compute_influence_entropy,
    is_divergent,
)
from pipeline.runner import run_pipeline, save_results_csv
from pipeline.analysis import (
    compute_summary_stats,
    identify_failure_modes,
    run_significance_test,
)
from pipeline.visualization import (
    plot_spearman_distribution,
    plot_rank_scatter,
    plot_dominance_distribution,
)

__all__ = [
    "PipelineConfig",
    "call_llm",
    "load_and_sample_dataset",
    "extract_queries_and_passages",
    "build_bm25",
    "retrieve_top_k",
    "build_rag_prompt",
    "generate_baseline",
    "run_ablation",
    "assign_influence_ranks",
    "compute_spearman",
    "compute_dominance_ratio",
    "compute_influence_entropy",
    "is_divergent",
    "run_pipeline",
    "save_results_csv",
    "compute_summary_stats",
    "identify_failure_modes",
    "run_significance_test",
    "plot_spearman_distribution",
    "plot_rank_scatter",
    "plot_dominance_distribution",
]
