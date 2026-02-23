"""
Dataset loading and preprocessing (PLAN.md Phase 1).

Uses HuggingFace streaming mode to avoid downloading the full ~40 GB
Natural Questions dataset.  Only the required number of examples are
pulled from the stream.
"""

import random
import logging
from typing import Any

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def load_and_sample_dataset(
    config: PipelineConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Stream the Natural Questions dataset and collect `n_samples` examples
    that have at least one short answer.

    Uses ``streaming=True`` so we never download the full dataset to disk.

    Returns
    -------
    list[dict]
        A list of raw NQ examples.
    """
    from datasets import load_dataset

    if config is None:
        config = PipelineConfig()

    logger.info(
        "Streaming dataset '%s' (split=%s) — collecting %d examples with short answers...",
        config.dataset_name,
        config.dataset_split,
        config.n_samples,
    )

    # Stream so we don't download tens of GB
    dataset = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        streaming=True,
    )

    def has_short_answer(example: dict) -> bool:
        annotations = example.get("annotations", {})
        short = annotations.get("short_answers", [])
        # short_answers is a list of lists (one per annotator)
        if isinstance(short, list):
            return any(
                (len(a) > 0 if isinstance(a, list) else bool(a))
                for a in short
            )
        return False

    # Collect candidates from the stream
    candidates: list[dict[str, Any]] = []
    seen = 0
    target = config.n_samples * 3  # over-collect, then sample

    for example in dataset:
        seen += 1
        if has_short_answer(example):
            candidates.append(example)
        if len(candidates) >= target:
            break
        # Safety: don't scan the entire dataset
        if seen >= target * 5:
            break

    logger.info(
        "Scanned %d examples, found %d with short answers.",
        seen,
        len(candidates),
    )

    # Sample the desired number
    random.seed(config.random_seed)
    n = min(config.n_samples, len(candidates))
    sample = random.sample(candidates, n)
    logger.info("Sampled %d examples.", n)
    return sample


def extract_queries_and_passages(
    sample: list[dict[str, Any]],
    config: PipelineConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Extract query strings and split each document into ~100-word passages.

    Parameters
    ----------
    sample : list[dict]
        Raw NQ examples from `load_and_sample_dataset`.
    config : PipelineConfig, optional

    Returns
    -------
    list[dict]
        Each entry has keys ``"query"`` (str) and ``"passages"`` (list[str]).
    """
    if config is None:
        config = PipelineConfig()

    queries: list[dict[str, Any]] = []
    for ex in sample:
        # NQ question field — handle both dict and plain string formats
        question = ex.get("question", {})
        if isinstance(question, dict):
            question = question.get("text", str(question))

        # NQ document tokens — handle nested dict or flat list
        doc = ex.get("document", {})
        if isinstance(doc, dict):
            tokens = doc.get("tokens", {})
            if isinstance(tokens, dict):
                token_list = tokens.get("token", [])
            elif isinstance(tokens, list):
                token_list = tokens
            else:
                token_list = []
        else:
            token_list = []

        if not token_list:
            # Fallback: try document_text or other fields
            doc_text = ex.get("document_text", "")
            if not doc_text:
                continue
        else:
            # Filter out HTML tags using the is_html mask
            is_html = []
            if isinstance(doc, dict):
                t = doc.get("tokens", {})
                if isinstance(t, dict):
                    is_html = t.get("is_html", [])

            if is_html and len(is_html) == len(token_list):
                # Keep only non-HTML tokens
                text_tokens = [
                    str(tok) for tok, html in zip(token_list, is_html)
                    if not html
                ]
            else:
                text_tokens = [str(t) for t in token_list]

            doc_text = " ".join(text_tokens)

        # Split into chunks of ~passage_word_size words
        words = doc_text.split()
        if len(words) < 20:
            continue  # skip extremely short documents

        chunk_size = config.passage_word_size
        passages = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        queries.append(
            {
                "query": question,
                "passages": passages[: config.max_passages_per_query],
            }
        )

    logger.info("Extracted %d query-passage pairs.", len(queries))
    return queries
