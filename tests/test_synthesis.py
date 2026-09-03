"""Tests for Step 4b (lineage synthesis). OpenRouter is mocked with respx."""

from __future__ import annotations

import dataclasses
import json

import httpx
import pytest
import respx

from treexiv.exceptions import SynthesisError
from treexiv.models import Cluster, Edge, FilteredGraph, Node, ScoredNode
from treexiv.synthesis import MAX_BEATS, build_prompt, synthesize_lineage

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _scored(id_: str, title: str, *, year: int, cluster: str | None, why: str = "") -> ScoredNode:
    return ScoredNode(
        node=Node(
            id=id_,
            title=title,
            publication_year=year,
            cited_by_count=10,
            authors=["Ada Example"],
            venue="Venue",
            abstract="abstract",
            hop=1,
        ),
        score=1.0,
        cluster_id=cluster,
        importance=4,
        why=why,
    )


@pytest.fixture
def curated_graph() -> FilteredGraph:
    return FilteredGraph(
        seed_id="SEED",
        idea_text="routing queries between strong and weak models",
        top_k=2,
        nodes=[
            _scored("SEED", "RouteLLM", year=2024, cluster=None, why="The seed paper."),
            _scored("OLD", "FrugalGPT", year=2023, cluster="c1", why="Defined the cost framing."),
            _scored("NEW", "GPT-5 System Card", year=2025, cluster="c2", why="Productionized it."),
        ],
        edges=[Edge("SEED", "OLD"), Edge("NEW", "SEED")],
        clusters=[
            Cluster(id="c1", name="Cascade precursors", summary="Where it came from.",
                    role="ancestor"),
            Cluster(id="c2", name="Deployment", summary="What it became.", role="descendant"),
        ],
        curation="llm",
    )


@pytest.fixture
def llm_settings(settings):
    return dataclasses.replace(settings, openrouter_api_key="sk-or-test")


def _reply(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


_GOOD = {
    "headline": "From cost-aware cascades to production routers.",
    "overview": "Paragraph one about the roots.\n\nParagraph two about the descendants.",
    "beats": [
        {"title": "Cascades arrive", "text": "FrugalGPT framed the tradeoff.", "papers": [1]},
        {"title": "Routing ships", "text": "It reached production.", "papers": [2]},
    ],
}


def test_prompt_numbers_papers_and_lists_clusters(curated_graph) -> None:
    messages, index_to_id = build_prompt(curated_graph, "RouteLLM")
    user = messages[1]["content"]
    assert "SEED PAPER: RouteLLM" in user
    assert "Cascade precursors (ancestor)" in user
    assert "[1]" in user and "[2]" in user
    # The seed is context, not one of the numbered papers to write beats about.
    assert set(index_to_id.values()) == {"OLD", "NEW"}
    assert "role: Defined the cost framing." in user


@respx.mock
def test_synthesize_lineage_parses_headline_overview_and_beats(
    llm_settings, curated_graph
) -> None:
    route = respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(_GOOD)))
    narrative = synthesize_lineage(curated_graph, llm_settings)

    assert route.called
    assert narrative.headline.startswith("From cost-aware cascades")
    assert "\n\n" in narrative.overview
    assert [b.title for b in narrative.beats] == ["Cascades arrive", "Routing ships"]
    assert narrative.beats[0].node_ids == ["OLD"]
    assert narrative.beats[1].node_ids == ["NEW"]
    assert narrative.model == "z-ai/glm-5.3-flash"


@respx.mock
def test_synthesize_lineage_drops_invented_paper_numbers(llm_settings, curated_graph) -> None:
    reply = {
        "overview": "A story.",
        "beats": [{"title": "Beat", "text": "Text.", "papers": [1, 99, "2", None]}],
    }
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(reply)))
    narrative = synthesize_lineage(curated_graph, llm_settings)
    assert narrative.beats[0].node_ids == ["OLD", "NEW"]


@respx.mock
def test_synthesize_lineage_skips_beats_with_no_text(llm_settings, curated_graph) -> None:
    reply = {
        "overview": "A story.",
        "beats": [{"title": "Empty"}, "not-an-object", {"text": "Real beat."}],
    }
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(reply)))
    narrative = synthesize_lineage(curated_graph, llm_settings)
    assert [b.text for b in narrative.beats] == ["Real beat."]


@respx.mock
def test_synthesize_lineage_caps_beat_count(llm_settings, curated_graph) -> None:
    reply = {
        "overview": "A story.",
        "beats": [{"title": f"B{i}", "text": f"Text {i}.", "papers": []} for i in range(20)],
    }
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(reply)))
    narrative = synthesize_lineage(curated_graph, llm_settings)
    assert len(narrative.beats) == MAX_BEATS


@respx.mock
def test_synthesize_lineage_requires_an_overview(llm_settings, curated_graph) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_reply({"headline": "Just a headline"}))
    )
    with pytest.raises(SynthesisError, match="no 'overview'"):
        synthesize_lineage(curated_graph, llm_settings)


def test_synthesize_lineage_refuses_a_seed_only_graph(llm_settings) -> None:
    graph = FilteredGraph(
        seed_id="SEED",
        idea_text="idea",
        top_k=0,
        nodes=[_scored("SEED", "RouteLLM", year=2024, cluster=None)],
        edges=[],
        curation="llm",
    )
    with pytest.raises(SynthesisError, match="only the seed"):
        synthesize_lineage(graph, llm_settings)


def test_synthesize_lineage_refuses_a_graph_missing_its_seed(llm_settings, curated_graph) -> None:
    orphaned = dataclasses.replace(curated_graph, seed_id="MISSING")
    with pytest.raises(SynthesisError, match="no seed node"):
        synthesize_lineage(orphaned, llm_settings)
