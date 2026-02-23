"""
Pipeline configuration and constants.
"""

import os
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Central configuration for the influence-aware RAG pipeline."""

    # ── OpenRouter / LLM ─────────────────────────────────────────────
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    model: str = "openai/gpt-oss-20b"
    max_tokens: int = 300
    temperature: float = 0.0  # deterministic for reproducibility

    # ── Dataset ──────────────────────────────────────────────────────
    dataset_name: str = "natural_questions"
    dataset_split: str = "validation"
    n_samples: int = 150
    random_seed: int = 42

    # ── Retrieval ────────────────────────────────────────────────────
    top_k: int = 10
    passage_word_size: int = 100
    max_passages_per_query: int = 15

    # ── Metrics ──────────────────────────────────────────────────────
    divergence_threshold: float = 0.7

    # ── Output ───────────────────────────────────────────────────────
    output_csv: str = "rag_influence_results.csv"
    output_dir: str = "outputs"

    # ── Rate limiting ────────────────────────────────────────────────
    llm_retry_max: int = 3
    llm_retry_delay: float = 2.0  # seconds between retries

    def validate(self) -> None:
        """Raise if the config is missing required values."""
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Export it as an environment variable or pass it to PipelineConfig."
            )
