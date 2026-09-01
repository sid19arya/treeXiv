"""Runtime configuration for treexiv, sourced from environment variables.

Defaults mirror `scratch/treexiv-mvp-openalex-prd.md` Section 4 ("Open decisions").
Nothing here is a persistent store — this is per-run, query-time configuration only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

SamplingStrategy = Literal["top_cited", "random"]

DEFAULT_TOTAL_CORPUS_CAP = 500
DEFAULT_PER_NODE_FANOUT_CAP = 100
DEFAULT_BM25_TOP_K = 40
DEFAULT_SAMPLING_STRATEGY: SamplingStrategy = "top_cited"
DEFAULT_BASE_URL = "https://api.openalex.org"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3

# Step 0 (`treexiv identify-seed`): an OpenRouter chat model, web-search
# grounded, that turns a vague description into a concrete seed-paper lead.
# Only touched when that subcommand runs — the rest of the pipeline never
# imports an LLM client.
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_LLM_WEB_SEARCH = True
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """All tunable knobs for one treexiv run.

    Construct via `Settings.from_env()` in normal use; the explicit constructor
    is what tests use to avoid touching real environment variables.
    """

    base_url: str = DEFAULT_BASE_URL
    mailto: str | None = None
    api_key: str | None = None
    total_corpus_cap: int = DEFAULT_TOTAL_CORPUS_CAP
    per_node_fanout_cap: int = DEFAULT_PER_NODE_FANOUT_CAP
    sampling_strategy: SamplingStrategy = DEFAULT_SAMPLING_STRATEGY
    bm25_top_k: int = DEFAULT_BM25_TOP_K
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    cache_dir: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    llm_web_search: bool = DEFAULT_LLM_WEB_SEARCH
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables (loading `.env` if present).

        Recognized variables: OPENALEX_API_KEY, OPENALEX_MAILTO,
        TREEXIV_TOTAL_CORPUS_CAP, TREEXIV_FANOUT_CAP, TREEXIV_SAMPLING_STRATEGY,
        TREEXIV_BM25_TOP_K, TREEXIV_CACHE_DIR, OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL, OPENROUTER_MODEL, TREEXIV_LLM_WEB_SEARCH.
        """
        load_dotenv()
        return cls(
            api_key=os.getenv("OPENALEX_API_KEY") or None,
            mailto=os.getenv("OPENALEX_MAILTO") or None,
            total_corpus_cap=int(
                os.getenv("TREEXIV_TOTAL_CORPUS_CAP", DEFAULT_TOTAL_CORPUS_CAP)
            ),
            per_node_fanout_cap=int(
                os.getenv("TREEXIV_FANOUT_CAP", DEFAULT_PER_NODE_FANOUT_CAP)
            ),
            sampling_strategy=os.getenv(  # type: ignore[arg-type]
                "TREEXIV_SAMPLING_STRATEGY", DEFAULT_SAMPLING_STRATEGY
            ),
            bm25_top_k=int(os.getenv("TREEXIV_BM25_TOP_K", DEFAULT_BM25_TOP_K)),
            cache_dir=os.getenv("TREEXIV_CACHE_DIR") or None,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            openrouter_base_url=(
                os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
            ),
            openrouter_model=os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
            llm_web_search=_env_bool("TREEXIV_LLM_WEB_SEARCH", DEFAULT_LLM_WEB_SEARCH),
        )
