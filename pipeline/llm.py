"""
OpenRouter LLM call helper with retry logic.
"""

import time
import logging

import requests

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def call_llm(prompt: str, config: PipelineConfig | None = None) -> str:
    """
    Send a prompt to the OpenRouter chat-completions endpoint and return
    the assistant's reply.

    Parameters
    ----------
    prompt : str
        The user-role message content.
    config : PipelineConfig, optional
        Pipeline configuration. Constructed from env vars if not supplied.

    Returns
    -------
    str
        The LLM's response text, stripped of leading/trailing whitespace.
    """
    if config is None:
        config = PipelineConfig()
    config.validate()

    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    prompt_chars = len(prompt)
    logger.debug("LLM prompt length: %d chars (~%d tokens)", prompt_chars, prompt_chars // 4)

    last_error: Exception | None = None
    for attempt in range(1, config.llm_retry_max + 1):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if response.status_code != 200:
                body = response.text[:500]
                logger.warning(
                    "LLM HTTP %d — %s", response.status_code, body,
                )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError) as exc:
            last_error = exc
            logger.warning(
                "LLM call attempt %d/%d failed: %s",
                attempt,
                config.llm_retry_max,
                exc,
            )
            if attempt < config.llm_retry_max:
                time.sleep(config.llm_retry_delay * attempt)  # linear backoff

    raise RuntimeError(
        f"LLM call failed after {config.llm_retry_max} attempts: {last_error}"
    )
