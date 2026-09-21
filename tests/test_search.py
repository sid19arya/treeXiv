"""Tests for exhaustive mode's search harvest. arXiv and Crossref are mocked
with respx or replaced by plain functions; nothing here touches the network."""

from __future__ import annotations

import dataclasses

import httpx
import respx

from treexiv.exceptions import SourceUnavailable
from treexiv.models import SEARCH_HOP, Edge, ExpansionResult, Node, Work
from treexiv.sources.search import (
    ARXIV_API_URL,
    CROSSREF_API_URL,
    ArxivClient,
    CrossrefClient,
    content_terms,
    harvest,
    identity_keys,
    merge_into,
    parse_arxiv_feed,
    work_from_crossref,
)

_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2512.24601v2</id>
    <title>Recursive Language
      Models</title>
    <summary>  We study   recursion. </summary>
    <published>2025-12-30T10:00:00Z</published>
    <arxiv:doi>10.1000/JOURNAL.1</arxiv:doi>
    <author><name>Alex Zhang</name></author>
    <author><name>Omar Khattab</name></author>
  </entry>
</feed>"""


def _node(id_: str, title: str, *, doi: str | None = None, abstract: str = "", hop: int = 1):
    return Node(
        id=id_,
        title=title,
        publication_year=2020,
        cited_by_count=1,
        authors=[],
        venue=None,
        abstract=abstract,
        hop=hop,
        doi=doi,
    )


def _work(id_: str, title: str, *, doi: str | None = None, abstract: str = "", source="arxiv"):
    return Work(
        id=id_,
        title=title,
        publication_year=2021,
        cited_by_count=0,
        doi=doi,
        abstract_text=abstract,
        source=source,
    )


def test_parse_arxiv_feed_reads_entries() -> None:
    [work] = parse_arxiv_feed(_FEED)
    assert work.id == "arxiv:2512.24601"
    assert work.title == "Recursive Language Models"
    assert work.publication_year == 2025
    assert work.authors == ["Alex Zhang", "Omar Khattab"]
    assert work.abstract == "We study recursion."
    assert work.doi == "https://doi.org/10.48550/arxiv.2512.24601"
    assert work.external_ids == {"arxiv": "2512.24601", "doi": "10.1000/journal.1"}


def test_arxiv_doi_and_arxiv_id_share_an_identity_key() -> None:
    openalex_style = identity_keys("https://doi.org/10.48550/arXiv.2512.24601", {}, "x")
    arxiv_style = identity_keys(None, {"arxiv": "2512.24601v3"}, "y")
    assert openalex_style & arxiv_style == {"arxiv:2512.24601"}


def test_short_titles_are_not_identity_keys() -> None:
    assert identity_keys(None, {}, "Attention") == set()
    assert identity_keys(None, {}, "A sufficiently long distinctive title") == {
        "title:asufficientlylongdistinctivetitle"
    }


def test_content_terms_drop_stopwords_and_duplicates() -> None:
    assert content_terms("The recursion of models, and recursion in the models") == [
        "recursion",
        "models",
    ]


@respx.mock
def test_arxiv_search_retries_with_fewer_terms_when_empty(settings, monkeypatch) -> None:
    monkeypatch.setattr("treexiv.sources.search._ARXIV_MIN_INTERVAL", 0.0)
    empty = "<feed xmlns='http://www.w3.org/2005/Atom'></feed>"
    route = respx.get(ARXIV_API_URL).mock(
        side_effect=[httpx.Response(200, text=empty), httpx.Response(200, text=_FEED)]
    )
    with ArxivClient(settings) as client:
        works = client.search("alpha beta gamma delta epsilon", 10)
    assert len(works) == 1
    first, second = (call.request.url.params["search_query"] for call in route.calls)
    assert first.count("AND") == 4
    assert second == "all:alpha AND all:beta AND all:gamma"


def test_work_from_crossref_strips_jats_and_needs_a_doi() -> None:
    work = work_from_crossref(
        {
            "DOI": "10.1/ABC",
            "title": ["A  Paper"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "issued": {"date-parts": [[2019, 5]]},
            "abstract": "<jats:p>Abstract We did <jats:italic>things</jats:italic>.</jats:p>",
            "is-referenced-by-count": 7,
            "container-title": ["Journal"],
        }
    )
    assert work is not None
    assert (work.id, work.title, work.publication_year) == ("doi:10.1/abc", "A Paper", 2019)
    assert work.abstract == "We did things ."
    assert work.authors == ["Ada Lovelace"]
    assert work_from_crossref({"title": ["No DOI"]}) is None


@respx.mock
def test_crossref_search_sends_mailto_and_parses_items(settings) -> None:
    route = respx.get(CROSSREF_API_URL).mock(
        return_value=httpx.Response(
            200, json={"message": {"items": [{"DOI": "10.1/x", "title": ["T"]}, {"DOI": "10.1/y"}]}}
        )
    )
    with CrossrefClient(settings) as client:
        works = client.search("idea", 5)
    assert [w.id for w in works] == ["doi:10.1/x"]
    assert route.calls.last.request.url.params["mailto"] == "test@example.com"


def test_harvest_skips_a_failing_source_after_one_warning() -> None:
    calls: list[str] = []
    warnings: list[str] = []

    def broken(query: str, limit: int) -> list[Work]:
        calls.append(query)
        raise SourceUnavailable("down")

    def working(query: str, limit: int) -> list[Work]:
        return [_work(f"arxiv:{query}", query)]

    works, hits = harvest(
        ["idea", "seed title"],
        {"broken": broken, "arxiv": working},
        10,
        on_warning=warnings.append,
    )
    assert calls == ["idea"]
    assert len(warnings) == 1
    assert hits == {"broken": 0, "arxiv": 2}
    assert [w.id for w in works] == ["arxiv:idea", "arxiv:seed title"]


def test_merge_into_dedupes_fills_abstracts_and_marks_search_hop() -> None:
    expansion = ExpansionResult(
        seed_id="W0",
        nodes={
            "W0": _node("W0", "The seed paper", hop=0),
            "W1": _node(
                "W1", "Recursive Language Models", doi="https://doi.org/10.48550/arxiv.2512.24601"
            ),
        },
        edges=[Edge("W0", "W1")],
    )
    works = [
        # Same paper as W1 via the arXiv DOI: fills its missing abstract.
        _work(
            "arxiv:2512.24601",
            "Recursive LMs",
            doi="https://doi.org/10.48550/arxiv.2512.24601",
            abstract="from arxiv",
        ),
        # New paper, found twice (Crossref then arXiv title match): added once.
        _work(
            "doi:10.1/new", "A brand new parallel line of work", doi="10.1/new", source="crossref"
        ),
        _work("arxiv:9999.0001", "A Brand New Parallel Line of Work!"),
    ]
    added, merged = merge_into(expansion, works)
    assert (added, merged) == (1, 2)
    assert expansion.nodes["W1"].abstract == "from arxiv"
    assert expansion.nodes["W1"].hop == 1
    assert expansion.nodes["doi:10.1/new"].hop == SEARCH_HOP
    assert "arxiv:9999.0001" not in expansion.nodes
    assert expansion.edges == [Edge("W0", "W1")]


@respx.mock
def test_arxiv_search_backs_off_on_rate_limits(settings, monkeypatch) -> None:
    monkeypatch.setattr("treexiv.sources.search._ARXIV_MIN_INTERVAL", 0.0)
    route = respx.get(ARXIV_API_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, text=_FEED)]
    )
    retrying = dataclasses.replace(settings, max_retries=2)
    with ArxivClient(retrying) as client:
        assert len(client.search("recursion", 5)) == 1
    assert route.call_count == 2
