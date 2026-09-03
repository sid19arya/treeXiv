"""Step 0: turn a vague description into a concrete seed-paper lead.

`treexiv identify-seed "<free text>"` calls an OpenRouter chat model — web-search
grounded by default — to guess *which published paper* the user is describing,
and hands back something the rest of the pipeline can act on: a best-guess
title, an arXiv ID / DOI if the model is confident, and a suggested query string
for `search-seed`.

This is deliberately a **lead, not a resolution**. `search-seed` still runs
afterwards, and the usual disambiguation (web-search cross-check, asking the
user when candidates are close) still applies — see
`.claude/skills/treexiv-lineage/SKILL.md`. The LLM here just gets you from "that
paper about predicting protein structures with attention" to a title worth
searching, which OpenAlex title search alone can't do.

The HTTP/retry/JSON-parsing mechanics live in `llm.py`, shared with the Step 4
curation call; this module is just the prompt and the response shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from treexiv.config import Settings
from treexiv.exceptions import SeedIdentificationError
from treexiv.llm import chat_json, clean_str, coerce_int

_SYSTEM_PROMPT = """\
You identify academic papers from informal descriptions.

Given a user's rough description of a paper (topic, findings, rough era, authors,
or venue — any subset), determine the single most likely real, published paper
they mean. Use web search to confirm the exact title, authors, year, and an
arXiv ID or DOI where one exists. Do not invent identifiers: leave a field null
rather than guessing it.

Respond with ONLY a JSON object, no prose or code fences, with exactly these keys:
- "search_query": string. The best short query to find this paper via a
  title-search API (usually the exact title; fall back to distinctive title
  words plus first author if you're unsure of the exact title). Never null.
- "title": string or null. The exact paper title if you're confident.
- "arxiv_id": string or null. e.g. "1706.03762" (no "arXiv:" prefix, no version).
- "doi": string or null. Bare DOI, e.g. "10.1038/nature14539".
- "authors": array of strings. Author names, best effort, may be empty.
- "year": integer or null. Publication/preprint year.
- "confidence": "high" | "medium" | "low". How sure you are this is the paper.
- "reasoning": string. 1-3 sentences on why this is the match and any doubt.
- "alternatives": array of up to 3 objects {"title": string, "note": string}
  for other plausible papers, "note" saying how it differs. May be empty.

If the description is too vague to land on one paper, still return your best
single guess with "confidence": "low" and say so in "reasoning"."""


@dataclass(slots=True)
class SeedGuess:
    """The LLM's best guess at which paper a description refers to (Step 0 output).

    A lead for `search-seed`, not a resolved work — nothing downstream keys off
    this directly.
    """

    search_query: str
    title: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    confidence: str = "low"
    reasoning: str = ""
    alternatives: list[dict[str, str]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_query": self.search_query,
            "title": self.title,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "authors": self.authors,
            "year": self.year,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "sources": self.sources,
            "model": self.model,
        }


def identify_seed(
    description: str,
    settings: Settings,
    *,
    web_search: bool | None = None,
    http_client: httpx.Client | None = None,
) -> SeedGuess:
    """Ask the configured OpenRouter model which paper `description` refers to.

    `web_search` overrides `settings.llm_web_search` when not None. Raises
    `SeedIdentificationError` if the key is missing, the API keeps failing, or
    the model's reply can't be parsed into a `SeedGuess`.
    """
    description = description.strip()
    if not description:
        raise SeedIdentificationError("Empty description — nothing to identify.")
    if not settings.openrouter_api_key:
        raise SeedIdentificationError(
            "OPENROUTER_API_KEY is not set — needed for `treexiv identify-seed`. "
            "See .env.example."
        )

    use_web = settings.llm_web_search if web_search is None else web_search
    # OpenRouter's built-in web plugin (same thing the ":online" model suffix
    # enables), kept as an explicit param so the model slug in config stays clean.
    plugins = [{"id": "web", "max_results": 5}] if use_web else None

    result = chat_json(
        settings,
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        plugins=plugins,
        http_client=http_client,
        error_cls=SeedIdentificationError,
    )
    return _parse_guess(result.data, result.message, result.model)


def _parse_guess(parsed: dict[str, Any], message: dict[str, Any], model: str) -> SeedGuess:
    query = parsed.get("search_query")
    if not isinstance(query, str) or not query.strip():
        raise SeedIdentificationError(
            f"Model reply had no usable 'search_query': {str(parsed)[:300]}"
        )

    return SeedGuess(
        search_query=query.strip(),
        title=clean_str(parsed.get("title")),
        arxiv_id=clean_str(parsed.get("arxiv_id")),
        doi=clean_str(parsed.get("doi")),
        authors=[str(a) for a in parsed.get("authors") or [] if str(a).strip()],
        year=_valid_year(parsed.get("year")),
        confidence=clean_str(parsed.get("confidence")) or "low",
        reasoning=clean_str(parsed.get("reasoning")) or "",
        alternatives=_clean_alternatives(parsed.get("alternatives")),
        sources=_extract_sources(message),
        model=model,
    )


def _valid_year(value: Any) -> int | None:
    year = coerce_int(value)
    return year if year is not None and 1500 <= year <= 2100 else None


def _clean_alternatives(value: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        title = clean_str(item.get("title"))
        if not title:
            continue
        out.append({"title": title, "note": clean_str(item.get("note")) or ""})
    return out[:3]


def _extract_sources(message: dict[str, Any]) -> list[str]:
    """Pull URLs the web plugin cited, from OpenRouter's annotation list."""
    urls: list[str] = []
    for ann in message.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        citation = ann.get("url_citation") or {}
        url = citation.get("url") if isinstance(citation, dict) else None
        if isinstance(url, str) and url not in urls:
            urls.append(url)
    return urls
