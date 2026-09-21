"""Exhaustive mode's harvest: scholarly search past the citation graph.

Two-hop citation traversal only reaches papers connected to the seed by a
reference. The parallel line of work that never cited it is invisible to it.
This module runs plain programmatic search across several scholarly indexes
and merges what comes back into the traversal's `ExpansionResult`, so Jev
(`exhaustive.py`) judges one combined corpus.

Sources, each best-effort, so one being down or rate-limited only costs its hits:

- **OpenAlex** `/works?search=`, reusing the traversal's client. Its hits carry
  OpenAlex IDs, so they are the best records to keep when a paper turns up twice.
- **Semantic Scholar** `/paper/search`, when the source mode allows S2 at all.
  It shares the per-run S2 request budget.
- **arXiv** API (Atom XML, parsed with the stdlib). It covers the newest
  preprints, which the other indexes pick up late.
- **Crossref** `/works?query.bibliographic=`. It has broad DOI coverage, but
  abstracts are often missing, in which case Jev judges on the title alone.

Queries are the stated idea and the seed title. There is no LLM rephrasing,
on purpose: this mode's premise is programmatic search plus an evaluation
model, not a generative one.

Merging is by identity key: DOI, arXiv ID (including arXiv's own
`10.48550/arxiv.*` DOIs), then normalized title. A paper already in the
citation expansion keeps its node, hop, and edges. A search hit only fills in
an abstract the node lacked. A paper found only by search enters at
`SEARCH_HOP`.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from treexiv.config import Settings
from treexiv.corpus import tokenize
from treexiv.exceptions import OpenAlexAPIError, SourceUnavailable
from treexiv.models import SEARCH_HOP, ExpansionResult, Node, Work, normalize_doi
from treexiv.openalex import OpenAlexClient
from treexiv.sources.s2 import SemanticScholarClient

WarningSink = Callable[[str], None]

ARXIV_API_URL = "https://export.arxiv.org/api/query"
CROSSREF_API_URL = "https://api.crossref.org/works"
_ARXIV_DOI_PREFIX = "10.48550/arxiv."
# arXiv asks for no more than one request every three seconds.
_ARXIV_MIN_INTERVAL = 3.0
_ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_VERSION_SUFFIX = re.compile(r"v\d+$")
_JATS_TAG = re.compile(r"<[^>]+>")
# Titles shorter than this ("Introduction", "Attention") are too generic to
# merge two records on.
_MIN_TITLE_KEY_CHARS = 24
# arXiv's query language ANDs terms, so a long idea sentence needs trimming to
# its content words or it matches nothing.
_MAX_ARXIV_TERMS = 6
_STOPWORDS = frozenset(
    "a an and are as at be by can for from how in into is it its of on or that the "
    "their this to via we what when which with without using use based new towards".split()
)
_CROSSREF_TYPES = "type:journal-article,type:proceedings-article,type:posted-content"


@dataclass(slots=True)
class HarvestReport:
    """What each source contributed, for the run summary."""

    hits: dict[str, int] = field(default_factory=dict)
    added: int = 0
    merged: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        per_source = ", ".join(f"{name} {count}" for name, count in self.hits.items()) or "none"
        return (
            f"Search harvest: {per_source} hits -> {self.added} new papers, "
            f"{self.merged} already in the citation expansion"
        )


def content_terms(text: str) -> list[str]:
    """The distinctive words of a query, in order, deduplicated."""
    seen: list[str] = []
    for token in tokenize(text):
        if len(token) > 2 and token not in _STOPWORDS and token not in seen:
            seen.append(token)
    return seen


def normalize_title(title: str) -> str:
    return "".join(tokenize(title))


def _strip_arxiv_version(arxiv_id: str) -> str:
    return _VERSION_SUFFIX.sub("", arxiv_id.strip().lower())


def identity_keys(doi: str | None, external_ids: dict[str, str], title: str) -> set[str]:
    """Every key under which two records of the same paper should collide."""
    keys: set[str] = set()
    bare = normalize_doi(doi or external_ids.get("doi"))
    if bare:
        if bare.startswith(_ARXIV_DOI_PREFIX):
            keys.add("arxiv:" + _strip_arxiv_version(bare[len(_ARXIV_DOI_PREFIX) :]))
        else:
            keys.add("doi:" + bare)
    arxiv = external_ids.get("arxiv")
    if arxiv:
        keys.add("arxiv:" + _strip_arxiv_version(arxiv))
    norm = normalize_title(title)
    if len(norm) >= _MIN_TITLE_KEY_CHARS:
        keys.add("title:" + norm)
    return keys


def _node_keys(node: Node) -> set[str]:
    return identity_keys(node.doi, node.external_ids, node.title)


def _work_keys(work: Work) -> set[str]:
    return identity_keys(work.doi, work.external_ids, work.title)


class ArxivClient:
    """Relevance search over the arXiv API, spaced to arXiv's stated limit."""

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=settings.timeout_seconds, follow_redirects=True
        )
        self._last_request_at = 0.0

    def __enter__(self) -> ArxivClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def search(self, query: str, limit: int) -> list[Work]:
        terms = content_terms(query)[:_MAX_ARXIV_TERMS]
        if not terms:
            return []
        works = self._query(terms, limit)
        # Six ANDed terms can over-constrain; retry on the leading three.
        if not works and len(terms) > 3:
            works = self._query(terms[:3], limit)
        return works

    def _query(self, terms: list[str], limit: int) -> list[Work]:
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < _ARXIV_MIN_INTERVAL:
            time.sleep(_ARXIV_MIN_INTERVAL - elapsed)
        self._last_request_at = time.monotonic()
        params: dict[str, str | int] = {
            "search_query": " AND ".join(f"all:{t}" for t in terms),
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        }
        # arXiv answers bursts with 429/503 and recovers within seconds, so
        # back off and retry before calling the source unavailable.
        status: int | str = "no response"
        for attempt in range(max(1, self._settings.max_retries)):
            if attempt:
                time.sleep(_ARXIV_MIN_INTERVAL * 2**attempt)
            try:
                response = self._client.get(ARXIV_API_URL, params=params)
            except httpx.TransportError as exc:
                status = str(exc)
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                status = response.status_code
                continue
            if response.status_code >= 400:
                raise SourceUnavailable(f"arXiv search failed: {response.status_code}")
            self._last_request_at = time.monotonic()
            return parse_arxiv_feed(response.text)
        raise SourceUnavailable(f"arXiv search failed: {status}")


def parse_arxiv_feed(xml_text: str) -> list[Work]:
    """Parse an arXiv API Atom feed into `Work`s."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SourceUnavailable(f"arXiv returned unparseable XML: {exc}") from exc
    works: list[Work] = []
    for entry in root.findall("atom:entry", _ATOM):
        raw_id = (entry.findtext("atom:id", "", _ATOM) or "").strip()
        if "/abs/" not in raw_id:
            continue
        arxiv_id = _strip_arxiv_version(raw_id.split("/abs/", 1)[1])
        title = " ".join((entry.findtext("atom:title", "", _ATOM) or "").split())
        published = entry.findtext("atom:published", "", _ATOM) or ""
        journal_doi = (entry.findtext("arxiv:doi", "", _ATOM) or "").strip()
        external = {"arxiv": arxiv_id}
        if journal_doi:
            external["doi"] = journal_doi.lower()
        works.append(
            Work(
                id=f"arxiv:{arxiv_id}",
                title=title or "(untitled)",
                publication_year=int(published[:4]) if published[:4].isdigit() else None,
                cited_by_count=0,
                authors=[
                    " ".join((a.findtext("atom:name", "", _ATOM) or "").split())
                    for a in entry.findall("atom:author", _ATOM)
                ],
                venue="arXiv",
                # arXiv's own DataCite DOI, which resolves and is also what
                # OpenAlex records preprints under, so the two records merge.
                doi=f"https://doi.org/{_ARXIV_DOI_PREFIX}{arxiv_id}",
                abstract_text=" ".join((entry.findtext("atom:summary", "", _ATOM) or "").split()),
                external_ids=external,
                source="arxiv",
            )
        )
    return works


class CrossrefClient:
    """Bibliographic search over Crossref's REST API."""

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=settings.timeout_seconds)

    def __enter__(self) -> CrossrefClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def search(self, query: str, limit: int) -> list[Work]:
        params: dict[str, Any] = {
            "query.bibliographic": query,
            "rows": min(limit, 1000),
            "filter": _CROSSREF_TYPES,
            "select": "DOI,title,author,issued,abstract,is-referenced-by-count,container-title",
        }
        if self._settings.mailto:
            params["mailto"] = self._settings.mailto
        try:
            response = self._client.get(CROSSREF_API_URL, params=params)
        except httpx.TransportError as exc:
            raise SourceUnavailable(f"Crossref search failed: {exc}") from exc
        if response.status_code >= 400:
            raise SourceUnavailable(f"Crossref search failed: {response.status_code}")
        try:
            items = response.json()["message"]["items"]
        except (ValueError, KeyError, TypeError) as exc:
            raise SourceUnavailable("Crossref returned an unexpected payload") from exc
        return [w for w in (work_from_crossref(item) for item in items) if w is not None]


def work_from_crossref(item: dict) -> Work | None:
    doi = normalize_doi(item.get("DOI"))
    titles = item.get("title") or []
    if not doi or not titles:
        return None
    parts = (item.get("issued") or {}).get("date-parts") or [[None]]
    year = parts[0][0] if parts and parts[0] else None
    authors = [
        " ".join(p for p in (a.get("given"), a.get("family")) if p)
        for a in item.get("author") or []
        if isinstance(a, dict) and (a.get("family") or a.get("given"))
    ]
    abstract = " ".join(_JATS_TAG.sub(" ", item.get("abstract") or "").split())
    if abstract.lower().startswith("abstract "):
        abstract = abstract[len("abstract ") :]
    venues = item.get("container-title") or []
    return Work(
        id=f"doi:{doi}",
        title=" ".join(str(titles[0]).split()),
        publication_year=year if isinstance(year, int) else None,
        cited_by_count=item.get("is-referenced-by-count") or 0,
        authors=authors,
        venue=venues[0] if venues else None,
        doi=f"https://doi.org/{doi}",
        abstract_text=abstract,
        external_ids={"doi": doi},
        source="crossref",
    )


SearchFn = Callable[[str, int], list[Work]]


def harvest(
    queries: Iterable[str],
    sources: dict[str, SearchFn],
    limit: int,
    *,
    on_warning: WarningSink | None = None,
) -> tuple[list[Work], dict[str, int]]:
    """Run every query against every source, in source order.

    A source that fails is reported and skipped for the rest of the run, so
    a rate-limited index costs one warning, not one per query.
    """
    works: list[Work] = []
    hits: dict[str, int] = {}
    query_list = [q.strip() for q in queries if q and q.strip()]
    for name, search in sources.items():
        hits[name] = 0
        for query in query_list:
            try:
                found = search(query, limit)
            except (SourceUnavailable, OpenAlexAPIError) as exc:
                if on_warning:
                    on_warning(f"{name} search unavailable ({exc}) — skipping it.")
                break
            hits[name] += len(found)
            works.extend(found)
    return works, hits


def merge_into(expansion: ExpansionResult, works: Iterable[Work]) -> tuple[int, int]:
    """Add search hits to `expansion` in place, returning (added, merged).

    Earlier works win a collision, so callers pass the source they trust most
    first. Existing nodes are never replaced. They only gain an abstract they
    were missing.
    """
    index: dict[str, Node] = {}
    for node in expansion.nodes.values():
        for key in _node_keys(node):
            index.setdefault(key, node)

    added = merged = 0
    for work in works:
        keys = _work_keys(work)
        existing = next((index[k] for k in keys if k in index), None)
        if existing is not None:
            merged += 1
            if not existing.abstract and work.abstract:
                existing.abstract = work.abstract
            for key in keys:
                index.setdefault(key, existing)
            continue
        if work.id in expansion.nodes:
            continue
        node = Node.from_work(work, hop=SEARCH_HOP)
        expansion.nodes[node.id] = node
        added += 1
        for key in keys:
            index.setdefault(key, node)
    return added, merged


def harvest_into(
    expansion: ExpansionResult,
    seed_title: str,
    idea_text: str,
    settings: Settings,
    openalex: OpenAlexClient,
    *,
    s2: SemanticScholarClient | None = None,
    arxiv: ArxivClient | None = None,
    crossref: CrossrefClient | None = None,
    on_warning: WarningSink | None = None,
) -> HarvestReport:
    """Search every available source for the idea and the seed title, and
    merge the hits into `expansion`."""
    sources: dict[str, SearchFn] = {"openalex": openalex.search_works}
    if s2 is not None:
        sources["s2"] = s2.search_papers
    owned: list[ArxivClient | CrossrefClient] = []
    if arxiv is None:
        arxiv = ArxivClient(settings)
        owned.append(arxiv)
    if crossref is None:
        crossref = CrossrefClient(settings)
        owned.append(crossref)
    sources["arxiv"] = arxiv.search
    sources["crossref"] = crossref.search

    report = HarvestReport()

    def warn(message: str) -> None:
        report.warnings.append(message)
        if on_warning:
            on_warning(message)

    try:
        works, report.hits = harvest(
            [idea_text, seed_title], sources, settings.exhaustive_search_limit, on_warning=warn
        )
    finally:
        for client in owned:
            client.close()
    report.added, report.merged = merge_into(expansion, works)
    return report
