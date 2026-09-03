"""Shared fixtures: sample OpenAlex work payloads and a test-only Settings instance."""

from __future__ import annotations

import pytest

from treexiv.config import Settings


def make_work_payload(
    work_id: str,
    title: str,
    *,
    year: int = 2020,
    cited_by_count: int = 0,
    referenced_works: list[str] | None = None,
    abstract_words: dict[str, list[int]] | None = None,
    authors: list[str] | None = None,
    venue: str | None = "Test Venue",
) -> dict:
    """Build a minimal but realistic OpenAlex `/works` item payload."""
    return {
        "id": f"https://openalex.org/{work_id}",
        "display_name": title,
        "publication_year": year,
        "cited_by_count": cited_by_count,
        "authorships": [
            {"author": {"display_name": name}} for name in (authors or ["Alice Example"])
        ],
        "primary_location": {"source": {"display_name": venue}} if venue else {},
        "doi": f"https://doi.org/10.0000/{work_id.lower()}",
        "referenced_works": [
            f"https://openalex.org/{rid}" for rid in (referenced_works or [])
        ],
        "abstract_inverted_index": abstract_words,
    }


@pytest.fixture
def settings() -> Settings:
    return Settings(
        mailto="test@example.com",
        api_key=None,
        total_corpus_cap=500,
        per_node_fanout_cap=100,
        bm25_top_k=40,
        max_retries=1,
        cache_dir=None,
    )


@pytest.fixture(autouse=True)
def isolate_llm_env(monkeypatch) -> None:
    """Keep the developer's real `.env` from steering the suite.

    `Settings.from_env()` calls `load_dotenv()`, so a local `OPENROUTER_API_KEY`
    would otherwise flip curation-mode "auto" into a real LLM call, and the
    default source mode would reach for Semantic Scholar, in tests that only
    meant to exercise the BM25/OpenAlex path. Setting each var to "" (rather than
    deleting it) wins, because `load_dotenv` doesn't override vars already set.
    Tests that want a key set one explicitly after this fixture runs.
    """
    for var in (
        "OPENROUTER_API_KEY",
        "S2_API_KEY",
        "TREEXIV_CURATION",
        "TREEXIV_CURATION_PREFILTER",
        "TREEXIV_CURATION_MAX_NODES",
        "TREEXIV_CURATION_MODEL",
    ):
        monkeypatch.setenv(var, "")
    # Semantic Scholar is off by default in tests for the same reason: it is an
    # extra live dependency the BM25/OpenAlex paths shouldn't quietly acquire.
    # Tests that exercise it set TREEXIV_SOURCE themselves.
    monkeypatch.setenv("TREEXIV_SOURCE", "openalex")
