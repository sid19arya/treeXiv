"""Stitching Semantic Scholar's citation intents onto an OpenAlex expansion.

The two sources are used for what each is good at (see `s2.py`): OpenAlex does
the bulk two-hop crawl and owns the graph's IDs; S2 answers "what does the seed
cite, what cites the seed, and *why*" for the seed paper alone. This module is
the join between them, and it runs in three steps:

1. Ask S2 for the seed's references and citations, with `intents` and
   `isInfluential`.
2. Cross-walk those papers to OpenAlex IDs by DOI — the only identifier both
   sources reliably agree on. Papers that don't match are dropped rather than
   guessed at.
3. Annotate the expansion's edges: any edge between the seed and a matched
   paper picks up its intents, and any node whose OpenAlex abstract came back
   empty is backfilled with S2's plain-text one.

Nothing here is load-bearing. Every step is wrapped so that a rate-limited or
absent S2 leaves the expansion exactly as OpenAlex produced it — an
un-annotated graph, not a failed run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from treexiv.config import Settings
from treexiv.exceptions import SourceUnavailable, TreeXivError
from treexiv.models import ExpansionResult, Work
from treexiv.openalex import OpenAlexClient
from treexiv.sources.s2 import CitationRef, SemanticScholarClient

WarningSink = Callable[[str], None]


@dataclass(slots=True)
class EnrichmentReport:
    """What the S2 pass actually managed to add, for reporting to the user."""

    attempted: bool = False
    references_seen: int = 0
    citations_seen: int = 0
    edges_annotated: int = 0
    abstracts_filled: int = 0
    # Neighbours S2 knows about that this expansion doesn't contain — usually
    # because the traversal caps cut them, not because the DOI lookup failed.
    unmatched: int = 0
    s2_requests: int = 0
    note: str = ""

    @property
    def succeeded(self) -> bool:
        return self.attempted and not self.note

    def summary(self) -> str:
        if not self.attempted:
            return "Semantic Scholar: not used"
        if self.note:
            return f"Semantic Scholar: unavailable ({self.note})"
        return (
            f"Semantic Scholar: {self.edges_annotated} edges labelled with citation intent, "
            f"{self.abstracts_filled} abstracts filled "
            f"({self.references_seen + self.citations_seen} neighbours seen, "
            f"{self.unmatched} of them not in this expansion, {self.s2_requests} requests)"
        )


@dataclass(slots=True)
class SeedLookup:
    """An S2 record for the seed paper, and how it was found."""

    work: Work
    matched_by: str = "title"
    alternatives: list[Work] = field(default_factory=list)


def find_seed(
    query: str, settings: Settings, *, client: SemanticScholarClient | None = None
) -> SeedLookup | None:
    """Ask S2 for the paper `query` names, or None if it can't say.

    Used ahead of OpenAlex search because S2's `/paper/search/match` is a title
    matcher rather than a relevance ranker: given something close to a real
    title it returns that paper, where a relevance search returns a plausible
    neighbourhood. Returns None (never raises) when S2 has no match or is
    unavailable, so the caller falls through to OpenAlex search.
    """
    owns = client is None
    s2 = client or SemanticScholarClient(settings)
    try:
        work = s2.match_paper(query)
        return SeedLookup(work=work) if work else None
    except TreeXivError:
        return None
    finally:
        if owns:
            s2.close()


def enrich_expansion(
    expansion: ExpansionResult,
    seed_work: Work,
    openalex: OpenAlexClient,
    settings: Settings,
    *,
    s2_client: SemanticScholarClient | None = None,
    on_warning: WarningSink | None = None,
) -> EnrichmentReport:
    """Annotate `expansion` in place with S2 citation intents and abstracts.

    Returns a report of what was added. Never raises for an S2 problem: the
    expansion is usable without it.
    """
    report = EnrichmentReport()
    if settings.source_mode == "openalex":
        return report

    report.attempted = True
    owns = s2_client is None
    s2 = s2_client or SemanticScholarClient(settings)
    try:
        seed_ref = _s2_reference_for(seed_work)
        if seed_ref is None:
            report.note = "seed paper has no DOI or arXiv ID to look it up by"
            return report
        references = s2.get_references(seed_ref)
        citations = s2.get_citations(seed_ref)
        report.references_seen = len(references)
        report.citations_seen = len(citations)
        _apply(expansion, seed_work, references, citations, openalex, report)
    except SourceUnavailable as exc:
        report.note = str(exc)
    finally:
        report.s2_requests = s2.requests_made
        if owns:
            s2.close()

    if report.note and on_warning:
        on_warning(
            f"Semantic Scholar enrichment skipped ({report.note}) — "
            "the graph is unlabelled but otherwise complete."
        )
    return report


def _s2_reference_for(work: Work) -> str | None:
    """The identifier to look the seed up by on S2 — DOI first, then arXiv."""
    doi = work.normalized_doi
    if doi:
        return f"DOI:{doi}"
    arxiv = work.external_ids.get("arxiv")
    return f"ARXIV:{arxiv}" if arxiv else None


def _apply(
    expansion: ExpansionResult,
    seed_work: Work,
    references: list[CitationRef],
    citations: list[CitationRef],
    openalex: OpenAlexClient,
    report: EnrichmentReport,
) -> None:
    """Cross-walk S2 neighbours to OpenAlex IDs and fold what they know in."""
    by_doi: dict[str, CitationRef] = {}
    directions: dict[str, str] = {}
    for ref, direction in [(r, "cited") for r in references] + [
        (c, "citing") for c in citations
    ]:
        doi = ref.work.normalized_doi
        if not doi:
            continue
        by_doi.setdefault(doi, ref)
        directions.setdefault(doi, direction)

    if not by_doi:
        report.unmatched = len(references) + len(citations)
        return

    # Only DOIs already present in the expansion are worth resolving: the point
    # is to label edges we have, not to widen the graph.
    present_by_doi = {
        node.external_ids.get("doi", "").lower(): node_id
        for node_id, node in expansion.nodes.items()
        if node.external_ids.get("doi")
    }
    matched: dict[str, str] = {}  # doi -> openalex node id
    unresolved: list[str] = []
    for doi in by_doi:
        node_id = present_by_doi.get(doi)
        if node_id:
            matched[doi] = node_id
        else:
            unresolved.append(doi)

    if unresolved:
        try:
            found = openalex.get_works_by_doi(unresolved)
        except TreeXivError:
            found = {}
        for doi, work in found.items():
            if work.id in expansion.nodes:
                matched[doi] = work.id

    report.unmatched = len(by_doi) - len(matched)
    seed_id = seed_work.id

    intents_by_pair: dict[tuple[str, str], CitationRef] = {}
    for doi, node_id in matched.items():
        ref = by_doi[doi]
        # Edge.source cites Edge.target: the seed cites its references, its
        # citing papers cite the seed.
        pair = (seed_id, node_id) if directions[doi] == "cited" else (node_id, seed_id)
        intents_by_pair[pair] = ref

    annotated = 0
    for i, edge in enumerate(expansion.edges):
        labelled = intents_by_pair.get((edge.source, edge.target))
        if labelled is None or not (labelled.intents or labelled.is_influential):
            continue
        expansion.edges[i] = edge.with_intents(
            tuple(labelled.intents), labelled.is_influential
        )
        annotated += 1
    report.edges_annotated = annotated

    filled = 0
    for doi, node_id in matched.items():
        node = expansion.nodes.get(node_id)
        s2_abstract = by_doi[doi].work.abstract
        if node is not None and not node.abstract and s2_abstract:
            node.abstract = s2_abstract
            filled += 1
    report.abstracts_filled = filled
