"""Tests for the Semantic Scholar client. All S2 traffic is mocked with respx."""

from __future__ import annotations

import dataclasses

import httpx
import pytest
import respx

from treexiv.exceptions import SourceUnavailable
from treexiv.sources.s2 import SemanticScholarClient, s2_reference, work_from_s2

_BASE = "https://api.semanticscholar.org/graph/v1"


@pytest.fixture
def s2_settings(settings):
    # Zero interval: the throttle's timing is tested separately, and every
    # other test would otherwise pay a real second per request.
    return dataclasses.replace(settings, s2_min_interval=0.0, s2_request_budget=10)


def _paper(paper_id: str = "abc123", **overrides) -> dict:
    payload = {
        "paperId": paper_id,
        "externalIds": {"DOI": "10.48550/arXiv.2406.18665", "ArXiv": "2406.18665"},
        "title": "RouteLLM: Learning to Route LLMs with Preference Data",
        "abstract": "We present RouteLLM, a router trained on preference data.",
        "year": 2024,
        "venue": "arXiv.org",
        "citationCount": 644,
        "authors": [
            {"authorId": "1", "name": "Isaac Ong"},
            {"authorId": "2", "name": "Ion Stoica"},
        ],
    }
    payload.update(overrides)
    return payload


def test_work_from_s2_maps_every_field_we_rely_on() -> None:
    work = work_from_s2(_paper())
    assert work.id == "S2:abc123"
    assert work.source == "s2"
    assert work.publication_year == 2024
    assert work.cited_by_count == 644
    assert work.authors == ["Isaac Ong", "Ion Stoica"]
    assert work.doi == "https://doi.org/10.48550/arXiv.2406.18665"
    assert work.normalized_doi == "10.48550/arxiv.2406.18665"
    assert work.external_ids["arxiv"] == "2406.18665"
    # S2 hands back real abstract text — no inverted index to reassemble.
    assert work.abstract.startswith("We present RouteLLM")


def test_work_from_s2_tolerates_a_sparse_record() -> None:
    work = work_from_s2({"paperId": "x"})
    assert work.title == "(untitled)"
    assert work.cited_by_count == 0
    assert work.doi is None
    assert work.abstract == ""
    assert work.dedupe_key == "id:S2:x"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1234/abc", "DOI:10.1234/abc"),
        ("https://doi.org/10.1234/abc", "DOI:10.1234/abc"),
        ("arXiv:2406.18665", "ARXIV:2406.18665"),
        ("ARXIV:2406.18665", "ARXIV:2406.18665"),
        ("DOI:10.1234/abc", "DOI:10.1234/abc"),
        ("649def34f8be52c8b66281af98ae884c09aef38b", "649def34f8be52c8b66281af98ae884c09aef38b"),
    ],
)
def test_s2_reference_prefixes_identifiers(raw: str, expected: str) -> None:
    assert s2_reference(raw) == expected


@respx.mock
def test_match_paper_returns_the_single_best_title_match(s2_settings) -> None:
    route = respx.get(f"{_BASE}/paper/search/match").mock(
        return_value=httpx.Response(200, json={"data": [_paper()]})
    )
    with SemanticScholarClient(s2_settings) as client:
        work = client.match_paper("RouteLLM")
    assert route.called
    assert work is not None and work.title.startswith("RouteLLM")


@respx.mock
def test_match_paper_returns_none_when_nothing_matches(s2_settings) -> None:
    respx.get(f"{_BASE}/paper/search/match").mock(
        return_value=httpx.Response(404, json={"error": "Title match not found"})
    )
    with SemanticScholarClient(s2_settings) as client:
        assert client.match_paper("a paper that does not exist") is None


@respx.mock
def test_get_references_carries_intents_and_influence(s2_settings) -> None:
    respx.get(f"{_BASE}/paper/DOI:10.1/x/references").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "intents": ["methodology", "background"],
                        "isInfluential": True,
                        "citedPaper": _paper("ref1"),
                    },
                    {"intents": [], "isInfluential": False, "citedPaper": _paper("ref2")},
                    {"intents": ["background"], "citedPaper": None},
                ]
            },
        )
    )
    with SemanticScholarClient(s2_settings) as client:
        refs = client.get_references("10.1/x")
    assert [r.work.id for r in refs] == ["S2:ref1", "S2:ref2"]  # the null paper is skipped
    assert refs[0].intents == ["methodology", "background"]
    assert refs[0].is_influential is True
    assert refs[1].intents == []


@respx.mock
def test_get_citations_reads_the_citing_paper_side(s2_settings) -> None:
    respx.get(f"{_BASE}/paper/DOI:10.1/x/citations").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"intents": ["result"], "isInfluential": False, "citingPaper": _paper("c1")}
                ]
            },
        )
    )
    with SemanticScholarClient(s2_settings) as client:
        refs = client.get_citations("10.1/x")
    assert refs[0].work.id == "S2:c1"
    assert refs[0].intents == ["result"]


@respx.mock
def test_api_key_is_sent_when_configured(settings) -> None:
    keyed = dataclasses.replace(settings, s2_api_key="s2-key", s2_min_interval=0.0)
    route = respx.get(f"{_BASE}/paper/search/match").mock(
        return_value=httpx.Response(200, json={"data": [_paper()]})
    )
    with SemanticScholarClient(keyed) as client:
        client.match_paper("RouteLLM")
    assert route.calls[0].request.headers["x-api-key"] == "s2-key"


@respx.mock
def test_no_api_key_header_when_unset(s2_settings) -> None:
    route = respx.get(f"{_BASE}/paper/search/match").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    with SemanticScholarClient(s2_settings) as client:
        client.match_paper("x")
    assert "x-api-key" not in route.calls[0].request.headers


@respx.mock
def test_rate_limiting_is_retried_then_reported_as_unavailable(s2_settings, monkeypatch) -> None:
    """A 429 is the normal failure mode for unauthenticated S2, so it must
    surface as SourceUnavailable — the signal callers fall back on."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    retrying = dataclasses.replace(s2_settings, max_retries=3)
    route = respx.get(f"{_BASE}/paper/DOI:10.1/x/references").mock(
        return_value=httpx.Response(429, json={"code": "429"})
    )
    with SemanticScholarClient(retrying) as client:
        with pytest.raises(SourceUnavailable, match="after 3 attempts"):
            client.get_references("10.1/x")
    assert route.call_count == 3


@respx.mock
def test_retry_after_header_is_honoured(s2_settings, monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    retrying = dataclasses.replace(s2_settings, max_retries=2)
    respx.get(f"{_BASE}/paper/search/match").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "3"}),
            httpx.Response(200, json={"data": [_paper()]}),
        ]
    )
    with SemanticScholarClient(retrying) as client:
        assert client.match_paper("RouteLLM") is not None
    assert 3.0 in slept


@respx.mock
def test_absurd_retry_after_is_capped(s2_settings, monkeypatch) -> None:
    """S2 has been known to ask for very long waits; a run shouldn't stall."""
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    retrying = dataclasses.replace(s2_settings, max_retries=2)
    respx.get(f"{_BASE}/paper/search/match").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "3600"}),
            httpx.Response(200, json={"data": [_paper()]}),
        ]
    )
    with SemanticScholarClient(retrying) as client:
        client.match_paper("RouteLLM")
    assert max(slept) <= 10.0


@respx.mock
def test_request_budget_stops_the_client(s2_settings) -> None:
    """The budget is what keeps S2 from turning into the slow path of a run."""
    budgeted = dataclasses.replace(s2_settings, s2_request_budget=2)
    respx.get(f"{_BASE}/paper/search/match").mock(
        return_value=httpx.Response(200, json={"data": [_paper()]})
    )
    with SemanticScholarClient(budgeted) as client:
        client.match_paper("one")
        client.match_paper("two")
        with pytest.raises(SourceUnavailable, match="budget"):
            client.match_paper("three")
    assert client.requests_made == 2


@respx.mock
def test_non_json_response_is_unavailable_not_a_crash(s2_settings) -> None:
    respx.get(f"{_BASE}/paper/search/match").mock(
        return_value=httpx.Response(200, text="<html>maintenance</html>")
    )
    with SemanticScholarClient(s2_settings) as client:
        with pytest.raises(SourceUnavailable, match="not JSON"):
            client.match_paper("x")


@respx.mock
def test_requests_are_spaced_by_the_throttle(settings, monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    throttled = dataclasses.replace(settings, s2_min_interval=1.1)
    respx.get(f"{_BASE}/paper/search/match").mock(
        return_value=httpx.Response(200, json={"data": [_paper()]})
    )
    with SemanticScholarClient(throttled) as client:
        client.match_paper("one")
        client.match_paper("two")
    # First request goes immediately; the second waits out the interval.
    assert len(slept) == 1
    assert 0 < slept[0] <= 1.1
