"""Tests for the S2-to-OpenAlex join: cross-walking by DOI, labelling edges
with citation intents, and backfilling empty abstracts. Both APIs are mocked."""

from __future__ import annotations

import dataclasses

import httpx
import pytest
import respx

from treexiv.models import Edge, ExpansionResult, Node, Work
from treexiv.openalex import OpenAlexClient
from treexiv.sources.enrich import enrich_expansion, find_seed
from treexiv.sources.s2 import SemanticScholarClient

_S2 = "https://api.semanticscholar.org/graph/v1"
_OA = "https://api.openalex.org"


@pytest.fixture
def s2_settings(settings):
    return dataclasses.replace(
        settings, s2_min_interval=0.0, s2_request_budget=10, source_mode="auto"
    )


def _node(node_id: str, title: str, *, doi: str | None, abstract: str = "abstract") -> Node:
    return Node(
        id=node_id,
        title=title,
        publication_year=2023,
        cited_by_count=5,
        authors=["Ada Example"],
        venue="Venue",
        abstract=abstract,
        hop=1,
        doi=f"https://doi.org/{doi}" if doi else None,
        external_ids={"doi": doi} if doi else {},
    )


@pytest.fixture
def expansion() -> ExpansionResult:
    return ExpansionResult(
        seed_id="W_SEED",
        nodes={
            "W_SEED": _node("W_SEED", "Seed", doi="10.0/seed"),
            "W_OLD": _node("W_OLD", "Cited work", doi="10.0/old"),
            "W_NEW": _node("W_NEW", "Citing work", doi="10.0/new", abstract=""),
            "W_FAR": _node("W_FAR", "Two hops out", doi="10.0/far"),
        },
        edges=[Edge("W_SEED", "W_OLD"), Edge("W_NEW", "W_SEED"), Edge("W_FAR", "W_NEW")],
    )


@pytest.fixture
def seed_work() -> Work:
    return Work(
        id="W_SEED",
        title="Seed",
        publication_year=2024,
        cited_by_count=100,
        doi="https://doi.org/10.0/seed",
        external_ids={"doi": "10.0/seed"},
    )


def _s2_paper(paper_id: str, doi: str, *, abstract: str = "") -> dict:
    return {
        "paperId": paper_id,
        "externalIds": {"DOI": doi},
        "title": f"S2 record {paper_id}",
        "abstract": abstract,
        "year": 2023,
        "citationCount": 7,
        "authors": [{"name": "Ada Example"}],
        "venue": "Venue",
    }


def _mock_s2(references: list[dict], citations: list[dict]) -> None:
    respx.get(f"{_S2}/paper/DOI:10.0/seed/references").mock(
        return_value=httpx.Response(200, json={"data": references})
    )
    respx.get(f"{_S2}/paper/DOI:10.0/seed/citations").mock(
        return_value=httpx.Response(200, json={"data": citations})
    )


@respx.mock
def test_intents_land_on_the_seeds_own_edges(s2_settings, expansion, seed_work) -> None:
    _mock_s2(
        references=[
            {
                "intents": ["methodology"],
                "isInfluential": True,
                "citedPaper": _s2_paper("s1", "10.0/old"),
            }
        ],
        citations=[
            {
                "intents": ["background"],
                "isInfluential": False,
                "citingPaper": _s2_paper("s2", "10.0/new"),
            }
        ],
    )
    with OpenAlexClient(s2_settings) as oa:
        report = enrich_expansion(expansion, seed_work, oa, s2_settings)

    by_pair = {(e.source, e.target): e for e in expansion.edges}
    assert by_pair[("W_SEED", "W_OLD")].intents == ("methodology",)
    assert by_pair[("W_SEED", "W_OLD")].is_influential is True
    assert by_pair[("W_NEW", "W_SEED")].intents == ("background",)
    # An edge S2 was never asked about stays unlabelled, not falsely labelled.
    assert by_pair[("W_FAR", "W_NEW")].intents == ()
    assert report.edges_annotated == 2
    assert report.succeeded


@respx.mock
def test_empty_abstracts_are_backfilled_from_s2(s2_settings, expansion, seed_work) -> None:
    """OpenAlex omits abstracts for plenty of works; an empty one contributes
    nothing to BM25 or curation, so S2's text is worth taking."""
    _mock_s2(
        references=[],
        citations=[
            {
                "intents": [],
                "isInfluential": False,
                "citingPaper": _s2_paper("s2", "10.0/new", abstract="The real abstract."),
            }
        ],
    )
    with OpenAlexClient(s2_settings) as oa:
        report = enrich_expansion(expansion, seed_work, oa, s2_settings)

    assert expansion.nodes["W_NEW"].abstract == "The real abstract."
    assert report.abstracts_filled == 1


@respx.mock
def test_existing_abstracts_are_left_alone(s2_settings, expansion, seed_work) -> None:
    _mock_s2(
        references=[
            {
                "intents": [],
                "isInfluential": False,
                "citedPaper": _s2_paper("s1", "10.0/old", abstract="S2 version."),
            }
        ],
        citations=[],
    )
    with OpenAlexClient(s2_settings) as oa:
        enrich_expansion(expansion, seed_work, oa, s2_settings)
    assert expansion.nodes["W_OLD"].abstract == "abstract"


@respx.mock
def test_papers_not_in_the_expansion_are_not_added(s2_settings, expansion, seed_work) -> None:
    """Enrichment labels the graph we have; it must not widen it."""
    respx.get(f"{_OA}/works").mock(return_value=httpx.Response(200, json={"results": []}))
    _mock_s2(
        references=[
            {
                "intents": ["result"],
                "isInfluential": True,
                "citedPaper": _s2_paper("sX", "10.0/not-in-graph"),
            }
        ],
        citations=[],
    )
    before = set(expansion.nodes)
    with OpenAlexClient(s2_settings) as oa:
        report = enrich_expansion(expansion, seed_work, oa, s2_settings)
    assert set(expansion.nodes) == before
    assert report.unmatched == 1
    assert report.edges_annotated == 0


@respx.mock
def test_rate_limited_s2_leaves_the_expansion_untouched(
    s2_settings, expansion, seed_work, monkeypatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    respx.get(f"{_S2}/paper/DOI:10.0/seed/references").mock(
        return_value=httpx.Response(429, json={"code": "429"})
    )
    warnings: list[str] = []
    with OpenAlexClient(s2_settings) as oa:
        report = enrich_expansion(
            expansion, seed_work, oa, s2_settings, on_warning=warnings.append
        )
    assert not report.succeeded
    assert "429" in report.note
    assert all(e.intents == () for e in expansion.edges)
    assert "unlabelled but otherwise complete" in warnings[0]


def test_source_mode_openalex_skips_s2_entirely(s2_settings, expansion, seed_work) -> None:
    off = dataclasses.replace(s2_settings, source_mode="openalex")
    # respx isn't active: any HTTP call at all would raise.
    with OpenAlexClient(off) as oa:
        report = enrich_expansion(expansion, seed_work, oa, off)
    assert not report.attempted
    assert report.summary() == "Semantic Scholar: not used"


def test_a_seed_with_no_doi_or_arxiv_id_cannot_be_looked_up(s2_settings, expansion) -> None:
    anonymous = Work(id="W_SEED", title="Seed", publication_year=2024, cited_by_count=1)
    with OpenAlexClient(s2_settings) as oa:
        report = enrich_expansion(expansion, anonymous, oa, s2_settings)
    assert "no DOI or arXiv ID" in report.note


@respx.mock
def test_unmatched_papers_are_resolved_through_openalex_by_doi(
    s2_settings, expansion, seed_work
) -> None:
    """A neighbour already in the graph under a different key is still matched,
    via an OpenAlex DOI lookup, rather than silently dropped."""
    del expansion.nodes["W_OLD"].external_ids["doi"]  # graph no longer indexes it by DOI
    oa_route = respx.get(f"{_OA}/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W_OLD",
                        "display_name": "Cited work",
                        "publication_year": 2023,
                        "cited_by_count": 5,
                        "doi": "https://doi.org/10.0/old",
                    }
                ]
            },
        )
    )
    _mock_s2(
        references=[
            {
                "intents": ["methodology"],
                "isInfluential": False,
                "citedPaper": _s2_paper("s1", "10.0/old"),
            }
        ],
        citations=[],
    )
    with OpenAlexClient(s2_settings) as oa:
        report = enrich_expansion(expansion, seed_work, oa, s2_settings)
    assert oa_route.called
    assert report.edges_annotated == 1


@respx.mock
def test_find_seed_returns_the_s2_match(s2_settings) -> None:
    respx.get(f"{_S2}/paper/search/match").mock(
        return_value=httpx.Response(200, json={"data": [_s2_paper("s1", "10.0/seed")]})
    )
    lookup = find_seed("some title", s2_settings)
    assert lookup is not None
    assert lookup.work.normalized_doi == "10.0/seed"


@respx.mock
def test_find_seed_is_silent_when_s2_is_down(s2_settings, monkeypatch) -> None:
    """Seed lookup is an optimization; an S2 outage must fall through to
    OpenAlex search rather than fail the command."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    respx.get(f"{_S2}/paper/search/match").mock(return_value=httpx.Response(503))
    assert find_seed("some title", s2_settings) is None


@respx.mock
def test_enrichment_shares_one_client_when_given_one(s2_settings, expansion, seed_work) -> None:
    _mock_s2(references=[], citations=[])
    with SemanticScholarClient(s2_settings) as s2, OpenAlexClient(s2_settings) as oa:
        report = enrich_expansion(expansion, seed_work, oa, s2_settings, s2_client=s2)
        assert s2.requests_made == 2
    assert report.s2_requests == 2
