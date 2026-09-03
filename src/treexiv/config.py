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
CurationMode = Literal["auto", "llm", "bm25"]
SourceMode = Literal["auto", "s2", "openalex"]

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

# Step 4 (`curate.py`): an LLM picks which papers actually belong in the tree
# and groups them into concept clusters, replacing the plain BM25 top-K cut.
# BM25 still runs first as a cheap prefilter, bounding how many abstracts the
# model has to read. "auto" means curate when an OpenRouter key is available
# and fall back to BM25 otherwise; "bm25" pins the old deterministic path.
DEFAULT_CURATION_MODE: CurationMode = "auto"
DEFAULT_CURATION_PREFILTER = 120
DEFAULT_CURATION_MAX_NODES = 35
# Step 4b (`synthesis.py`): a second, much smaller call that writes the lineage
# story for an already-curated graph. Only runs when curation itself ran.
DEFAULT_NARRATIVE = True

# Semantic Scholar supplies the two things OpenAlex can't: citation intents and
# plain-text abstracts. It is used only for the seed paper and its direct
# references/citations — unauthenticated S2 shares a rate-limit pool and 429s
# readily, so the bulk two-hop crawl stays on OpenAlex. "auto" tries S2 and
# carries on without it when unavailable; "openalex" skips it entirely.
DEFAULT_SOURCE_MODE: SourceMode = "auto"
DEFAULT_S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
# ~1 request/second unauthenticated. An S2_API_KEY lifts this; override with
# TREEXIV_S2_MIN_INTERVAL.
DEFAULT_S2_MIN_INTERVAL = 1.1
DEFAULT_S2_KEYED_MIN_INTERVAL = 0.1
# A ceiling on how much of a run can be spent waiting on S2.
DEFAULT_S2_REQUEST_BUDGET = 12


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an int-valued env var, falling back to `default` when it is unset,
    empty, or not a number — an empty override shouldn't crash a run."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


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
    curation_mode: CurationMode = DEFAULT_CURATION_MODE
    curation_prefilter: int = DEFAULT_CURATION_PREFILTER
    curation_max_nodes: int = DEFAULT_CURATION_MAX_NODES
    curation_model: str | None = None
    narrative: bool = DEFAULT_NARRATIVE
    source_mode: SourceMode = DEFAULT_SOURCE_MODE
    s2_api_key: str | None = None
    s2_base_url: str = DEFAULT_S2_BASE_URL
    s2_min_interval: float = DEFAULT_S2_MIN_INTERVAL
    s2_request_budget: int = DEFAULT_S2_REQUEST_BUDGET

    @property
    def resolved_curation_model(self) -> str:
        """The model the curation call uses — `OPENROUTER_MODEL` unless
        `TREEXIV_CURATION_MODEL` names a different one."""
        return self.curation_model or self.openrouter_model

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables (loading `.env` if present).

        Recognized variables: OPENALEX_API_KEY, OPENALEX_MAILTO,
        TREEXIV_TOTAL_CORPUS_CAP, TREEXIV_FANOUT_CAP, TREEXIV_SAMPLING_STRATEGY,
        TREEXIV_BM25_TOP_K, TREEXIV_CACHE_DIR, OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL, OPENROUTER_MODEL, TREEXIV_LLM_WEB_SEARCH,
        TREEXIV_CURATION, TREEXIV_CURATION_PREFILTER, TREEXIV_CURATION_MAX_NODES,
        TREEXIV_CURATION_MODEL, TREEXIV_NARRATIVE, TREEXIV_SOURCE, S2_API_KEY,
        S2_BASE_URL, TREEXIV_S2_MIN_INTERVAL, TREEXIV_S2_REQUEST_BUDGET.
        """
        load_dotenv()
        s2_key = os.getenv("S2_API_KEY") or None
        default_interval = (
            DEFAULT_S2_KEYED_MIN_INTERVAL if s2_key else DEFAULT_S2_MIN_INTERVAL
        )
        return cls(
            api_key=os.getenv("OPENALEX_API_KEY") or None,
            mailto=os.getenv("OPENALEX_MAILTO") or None,
            total_corpus_cap=_env_int("TREEXIV_TOTAL_CORPUS_CAP", DEFAULT_TOTAL_CORPUS_CAP),
            per_node_fanout_cap=_env_int("TREEXIV_FANOUT_CAP", DEFAULT_PER_NODE_FANOUT_CAP),
            sampling_strategy=_env_str(  # type: ignore[arg-type]
                "TREEXIV_SAMPLING_STRATEGY", DEFAULT_SAMPLING_STRATEGY
            ),
            bm25_top_k=_env_int("TREEXIV_BM25_TOP_K", DEFAULT_BM25_TOP_K),
            cache_dir=os.getenv("TREEXIV_CACHE_DIR") or None,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            openrouter_base_url=(
                os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
            ),
            openrouter_model=os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
            llm_web_search=_env_bool("TREEXIV_LLM_WEB_SEARCH", DEFAULT_LLM_WEB_SEARCH),
            curation_mode=_env_str(  # type: ignore[arg-type]
                "TREEXIV_CURATION", DEFAULT_CURATION_MODE
            ),
            curation_prefilter=_env_int(
                "TREEXIV_CURATION_PREFILTER", DEFAULT_CURATION_PREFILTER
            ),
            curation_max_nodes=_env_int(
                "TREEXIV_CURATION_MAX_NODES", DEFAULT_CURATION_MAX_NODES
            ),
            curation_model=os.getenv("TREEXIV_CURATION_MODEL") or None,
            narrative=_env_bool("TREEXIV_NARRATIVE", DEFAULT_NARRATIVE),
            source_mode=_env_str("TREEXIV_SOURCE", DEFAULT_SOURCE_MODE),  # type: ignore[arg-type]
            s2_api_key=s2_key,
            s2_base_url=_env_str("S2_BASE_URL", DEFAULT_S2_BASE_URL),
            s2_min_interval=_env_float("TREEXIV_S2_MIN_INTERVAL", default_interval),
            s2_request_budget=_env_int("TREEXIV_S2_REQUEST_BUDGET", DEFAULT_S2_REQUEST_BUDGET),
        )
