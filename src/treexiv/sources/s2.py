"""Semantic Scholar Graph API client — the seed-resolution and citation-intent source.

S2 knows two things OpenAlex doesn't: *why* one paper cites another
(`intents`: background / methodology / result, plus `isInfluential`), and it
returns real abstract text rather than an inverted index to reassemble. Both
matter for curation quality, so S2 is asked first for the calls where they pay
off most — matching the seed paper, and reading the seed's own references and
citations.

It is *not* asked for the bulk two-hop crawl. Unauthenticated S2 is a shared
rate-limit pool: two back-to-back requests are enough to draw a 429 in
practice, so a few hundred traversal calls through it would be slow and
unreliable. OpenAlex, which has no such gate, keeps that job. See
`enrich.py` for how the two are stitched together.

Every failure here raises `SourceUnavailable`, which callers treat as "carry
on with OpenAlex alone" rather than as a fatal error. Setting `S2_API_KEY`
lifts the throttle (`TREEXIV_S2_MIN_INTERVAL`) without any other change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from treexiv.config import Settings
from treexiv.exceptions import SourceUnavailable
from treexiv.models import Work

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRY_AFTER_SECONDS = 10.0

# Everything the pipeline needs from a paper record, in one request.
PAPER_FIELDS = "paperId,externalIds,title,abstract,year,venue,citationCount,authors"


def _nested_fields(prefix: str) -> str:
    """Field list for a references/citations call: the intent flags plus the
    neighbouring paper's own fields, which S2 wants dotted (`citedPaper.title`)."""
    nested = ",".join(f"{prefix}.{name}" for name in PAPER_FIELDS.split(","))
    return f"intents,isInfluential,{nested}"


_REFERENCE_FIELDS = _nested_fields("citedPaper")
_CITATION_FIELDS = _nested_fields("citingPaper")


@dataclass(slots=True, frozen=True)
class CitationRef:
    """A neighbouring paper plus what S2 knows about *why* the citation exists.

    `intents` is S2's classification of the citation context ("background",
    "methodology", "result"); `is_influential` is its judgement that the cited
    work materially shaped the citing one. Both are empty/False when the
    reference came from a source that doesn't classify citations.
    """

    work: Work
    intents: list[str] = field(default_factory=list)
    is_influential: bool = False


class SemanticScholarClient:
    """A rate-limited S2 Graph API client for seed resolution and intent lookup.

    Requests are spaced by `settings.s2_min_interval` and capped at
    `settings.s2_request_budget` per run — this client is meant to be used for
    a handful of high-value calls, and the budget makes that structural rather
    than a matter of discipline.
    """

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=settings.s2_base_url, timeout=settings.timeout_seconds
        )
        self._last_request_at = 0.0
        self._requests_made = 0

    def __enter__(self) -> SemanticScholarClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def requests_made(self) -> int:
        return self._requests_made

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._settings.s2_api_key} if self._settings.s2_api_key else {}

    def _throttle(self) -> None:
        interval = self._settings.s2_min_interval
        if interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < interval:
            time.sleep(interval - elapsed)

    def _get(self, path: str, params: dict[str, Any]) -> dict:
        if self._requests_made >= self._settings.s2_request_budget:
            raise SourceUnavailable(
                f"Semantic Scholar request budget ({self._settings.s2_request_budget}) exhausted."
            )
        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries):
            self._throttle()
            self._last_request_at = time.monotonic()
            self._requests_made += 1
            try:
                response = self._client.get(path, params=params, headers=self._headers())
            except httpx.TransportError as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue
            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_error = SourceUnavailable(
                    f"Semantic Scholar returned {response.status_code} for {path}"
                )
                time.sleep(_retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise SourceUnavailable(
                    f"Semantic Scholar request to {path} failed: "
                    f"{response.status_code} {response.text[:200]}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise SourceUnavailable(
                    f"Semantic Scholar response was not JSON: {response.text[:200]}"
                ) from exc
            if not isinstance(payload, dict):
                raise SourceUnavailable(f"Unexpected Semantic Scholar payload for {path}")
            return payload
        raise SourceUnavailable(
            f"Semantic Scholar request to {path} failed after "
            f"{self._settings.max_retries} attempts: {last_error}"
        )

    def match_paper(self, query: str) -> Work | None:
        """S2's best single title match for `query`, or None if it has none.

        `/paper/search/match` is a title matcher, not a relevance search — it
        returns at most one paper and is the right call for "the user named a
        paper, which one is it?".
        """
        try:
            payload = self._get(
                "/paper/search/match", {"query": query, "fields": PAPER_FIELDS}
            )
        except SourceUnavailable as exc:
            # A genuine "no title matched" comes back as a 404 with an error
            # body; that is an answer, not an outage.
            if "404" in str(exc):
                return None
            raise
        results = payload.get("data") or []
        return work_from_s2(results[0]) if results else None

    def search_papers(self, query: str, limit: int = 5) -> list[Work]:
        """Relevance search, for when the caller wants candidates to choose between."""
        payload = self._get(
            "/paper/search", {"query": query, "limit": limit, "fields": PAPER_FIELDS}
        )
        return [work_from_s2(item) for item in payload.get("data") or [] if item]

    def get_paper(self, paper_id: str) -> Work:
        """Fetch one paper. `paper_id` may be an S2 ID, `DOI:...`, or `ARXIV:...`."""
        payload = self._get(f"/paper/{s2_reference(paper_id)}", {"fields": PAPER_FIELDS})
        return work_from_s2(payload)

    def get_references(self, paper_id: str, limit: int = 500) -> list[CitationRef]:
        """Papers `paper_id` cites, with citation intents."""
        payload = self._get(
            f"/paper/{s2_reference(paper_id)}/references",
            {"fields": _REFERENCE_FIELDS, "limit": limit},
        )
        return _parse_refs(payload, "citedPaper")

    def get_citations(self, paper_id: str, limit: int = 500) -> list[CitationRef]:
        """Papers that cite `paper_id`, with citation intents.

        S2 has no server-side sort here, so callers that want the most-cited
        first must sort the result themselves.
        """
        payload = self._get(
            f"/paper/{s2_reference(paper_id)}/citations",
            {"fields": _CITATION_FIELDS, "limit": limit},
        )
        return _parse_refs(payload, "citingPaper")


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Honour `Retry-After` when S2 sends one, else exponential backoff."""
    raw = response.headers.get("Retry-After")
    if raw:
        try:
            return min(float(raw), _MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
    return float(2**attempt)


def _parse_refs(payload: dict, paper_key: str) -> list[CitationRef]:
    refs: list[CitationRef] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        paper = item.get(paper_key)
        if not isinstance(paper, dict) or not paper.get("paperId"):
            continue
        intents = [str(i) for i in item.get("intents") or [] if str(i).strip()]
        refs.append(
            CitationRef(
                work=work_from_s2(paper),
                intents=intents,
                is_influential=bool(item.get("isInfluential")),
            )
        )
    return refs


def s2_reference(identifier: str) -> str:
    """Turn a bare DOI/arXiv ID into the prefixed form the S2 path expects.

    Already-prefixed identifiers keep their value but get an uppercased
    prefix, so a user-written "arXiv:2406.18665" and an "ARXIV:..." built here
    produce the same URL (and so the same cache key upstream).
    """
    value = identifier.strip()
    prefix, _, rest = value.partition(":")
    if rest and prefix.upper() in {"DOI", "ARXIV", "CORPUSID", "MAG", "PMID", "PMCID", "URL"}:
        return f"{prefix.upper()}:{rest}"
    if value.startswith(("https://doi.org/", "http://doi.org/")):
        return "DOI:" + value.split("doi.org/", 1)[1]
    if value.startswith("10."):
        return "DOI:" + value
    return value


def work_from_s2(payload: dict) -> Work:
    """Parse an S2 paper record into the shared `Work` shape."""
    external = payload.get("externalIds") or {}
    doi = external.get("DOI")
    authors = [
        a["name"] for a in payload.get("authors") or [] if isinstance(a, dict) and a.get("name")
    ]
    external_ids = {
        key.lower(): str(value)
        for key, value in external.items()
        if value not in (None, "")
    }
    return Work(
        id=f"S2:{payload['paperId']}",
        title=payload.get("title") or "(untitled)",
        publication_year=payload.get("year"),
        cited_by_count=payload.get("citationCount") or 0,
        authors=authors,
        venue=payload.get("venue") or None,
        doi=f"https://doi.org/{doi}" if doi else None,
        abstract_text=payload.get("abstract") or "",
        external_ids=external_ids,
        source="s2",
    )
