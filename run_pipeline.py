#!/usr/bin/env python3
"""
run_pipeline.py — CLI entry point for the Influence-Aware RAG pipeline.

Usage
-----
  # Full run (requires OPENROUTER_API_KEY)
  python run_pipeline.py

  # Custom sample size
  python run_pipeline.py --n-samples 50

  # Dry run — tests imports and config, no API calls
  python run_pipeline.py --dry-run

  # Use a specific model
  python run_pipeline.py --model mistralai/mistral-7b-instruct
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Influence-Aware RAG ablation pipeline."
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=150,
        help="Number of NQ examples to sample (default: 150).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Documents per query (default: 10).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="OpenRouter model identifier.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output CSV and plots (default: outputs/).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without making LLM API calls.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from pipeline.config import PipelineConfig

    config = PipelineConfig(
        n_samples=args.n_samples,
        top_k=args.top_k,
        model=args.model,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Dry-run mode ─────────────────────────────────────────────────
    if args.dry_run:
        logger.info("=== DRY RUN — validating setup ===")

        # Check API key
        if config.openrouter_api_key:
            logger.info("✓ OPENROUTER_API_KEY is set.")
        else:
            logger.warning("✗ OPENROUTER_API_KEY is NOT set.")

        # Check imports
        try:
            from pipeline import (  # noqa: F401
                call_llm,
                load_and_sample_dataset,
                extract_queries_and_passages,
                retrieve_top_k,
                run_ablation,
                compute_spearman,
                run_pipeline,
                save_results_csv,
                compute_summary_stats,
                identify_failure_modes,
                run_significance_test,
                plot_spearman_distribution,
                plot_rank_scatter,
                plot_dominance_distribution,
            )
            logger.info("✓ All pipeline modules import successfully.")
        except ImportError as exc:
            logger.error("✗ Import error: %s", exc)
            sys.exit(1)

        logger.info(
            "Config: model=%s  n_samples=%d  top_k=%d  seed=%d",
            config.model,
            config.n_samples,
            config.top_k,
            config.random_seed,
        )
        logger.info("=== Dry run complete. ===")
        return

    # ── Full pipeline ────────────────────────────────────────────────
    config.validate()  # will raise if API key is missing

    from pipeline.data import load_and_sample_dataset, extract_queries_and_passages
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

    # Phase 1 — Data
    logger.info("Phase 1: Loading and processing dataset...")
    sample = load_and_sample_dataset(config)
    queries = extract_queries_and_passages(sample, config)

    # Phase 2–5 — Retrieval + Ablation + Metrics
    logger.info("Phase 2-5: Running pipeline (%d queries)...", len(queries))
    results_log = run_pipeline(queries, config)

    # Save CSV
    csv_path = os.path.join(config.output_dir, config.output_csv)
    df = save_results_csv(results_log, csv_path)
    logger.info("✓ CSV saved: %s", csv_path)

    # Phase 6 — Analysis
    logger.info("Phase 6: Statistical analysis...")
    summary_df = compute_summary_stats(results_log)
    failures = identify_failure_modes(results_log)
    stat, p_val, sig = run_significance_test(summary_df)

    # Phase 7 — Visualisations
    logger.info("Phase 7: Generating plots...")
    plot_spearman_distribution(
        summary_df,
        config.divergence_threshold,
        os.path.join(config.output_dir, "spearman_distribution.png"),
    )
    if results_log:
        plot_rank_scatter(
            results_log[0],
            os.path.join(config.output_dir, "rank_scatter_example.png"),
        )
    plot_dominance_distribution(
        summary_df,
        os.path.join(config.output_dir, "dominance_distribution.png"),
    )

    # ── Final summary ────────────────────────────────────────────────
    n_queries = len(results_log)
    n_div = sum(1 for r in results_log if r["is_divergent"])
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Queries processed :  {n_queries}")
    print(f"  Divergent queries :  {n_div} ({n_div / n_queries * 100:.1f}%)")
    print(f"  Mean Spearman ρ   :  {summary_df['spearman_rho'].mean():.3f}")
    print(f"  Mann-Whitney p    :  {p_val:.4f} ({'significant' if sig else 'not significant'})")
    print(f"  Failure modes     :  {len(failures)}")
    print(f"  Output directory  :  {config.output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
