"""Tests for Step 4's LLM curation path. OpenRouter is mocked with respx; no
real LLM traffic happens in the suite."""

from __future__ import annotations

import dataclasses
import json

import httpx
import pytest
import respx

from treexiv.curate import build_prompt, classify_directions, curate_graph, prefilter
from treexiv.exceptions import CurationError
from treexiv.models import Edge, ExpansionResult, Node

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _node(id_: str, title: str, *, abstract: str = "", year: int = 2020, cites: int = 5) -> Node:
    return Node(
        id=id_,
        title=title,
        publication_year=year,
        cited_by_count=cites,
        authors=["Ada Example", "Bo Example"],
        venue="Venue",
        abstract=abstract,
        hop=1,
    )


@pytest.fixture
def curation_settings(settings):
    return dataclasses.replace(
        settings,
        openrouter_api_key="sk-or-test",
        openrouter_model="z-ai/glm-5.3-flash",
        curation_mode="llm",
        curation_prefilter=10,
        curation_max_nodes=3,
    )


@pytest.fixture
def expansion():
    """Seed cites OLD (ancestor); NEW cites seed (descendant); FAR cites NEW."""
    nodes = {
        "SEED": _node("SEED", "Recursive language models", abstract="recursion in models"),
        "OLD": _node("OLD", "Early recursion work", abstract="recursion foundations", year=2015),
        "NEW": _node("NEW", "Recursive models at scale", abstract="scaling recursion", year=2023),
        "FAR": _node("FAR", "Downstream application", abstract="applies recursion", year=2024),
        "OFF": _node("OFF", "Pasta cooking techniques", abstract="boil water", year=2021),
    }
    edges = [
        Edge("SEED", "OLD"),
        Edge("NEW", "SEED"),
        Edge("FAR", "NEW"),
        Edge("SEED", "OFF"),
    ]
    return ExpansionResult(seed_id="SEED", nodes=nodes, edges=edges)


def _reply(payload: dict) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}]}


_GOOD_REPLY = {
    "clusters": [
        {"id": 1, "name": "Foundations", "role": "ancestor", "summary": "Where it came from."},
        {"id": 2, "name": "Scaling follow-ups", "role": "descendant", "summary": "What it became."},
    ],
    "keep": [
        {"i": 1, "cluster": 1, "importance": 5, "why": "Introduced the core recursion idea."},
        {"i": 2, "cluster": 2, "importance": 4, "why": "Scaled it up."},
    ],
    "dropped_summary": "Cut incidental citations that only share vocabulary.",
}


def test_classify_directions_splits_ancestors_from_descendants(expansion) -> None:
    labels = classify_directions("SEED", expansion.edges, set(expansion.nodes))
    assert labels["OLD"] == "ancestor"
    assert labels["OFF"] == "ancestor"
    assert labels["NEW"] == "descendant"
    assert labels["FAR"] == "descendant"  # two hops out, still on the citing side
    assert "SEED" not in labels


def test_classify_directions_marks_disconnected_nodes_unrelated() -> None:
    labels = classify_directions("SEED", [Edge("A", "B")], {"SEED", "A", "B"})
    assert labels["A"] == "unrelated"
    assert labels["B"] == "unrelated"


def test_classify_directions_handles_cycles() -> None:
    """A citation cycle must not send the traversal into an infinite loop."""
    edges = [Edge("SEED", "A"), Edge("A", "B"), Edge("B", "SEED")]
    labels = classify_directions("SEED", edges, {"SEED", "A", "B"})
    assert labels["A"] in {"ancestor", "mixed"}
    assert labels["B"] == "mixed"


def test_prefilter_excludes_seed_and_respects_limit(expansion) -> None:
    candidates = prefilter(expansion, "recursion", limit=2)
    assert len(candidates) == 2
    assert all(c.node.id != "SEED" for c in candidates)
    assert [c.index for c in candidates] == [1, 2]


def test_prompt_lists_candidates_with_indices_and_directions(expansion) -> None:
    candidates = prefilter(expansion, "recursion", limit=5)
    messages = build_prompt(expansion.nodes["SEED"], "recursion", candidates, max_keep=10)
    user = messages[1]["content"]
    assert "SEED PAPER: Recursive language models" in user
    assert "recursion" in user
    for candidate in candidates:
        assert f"[{candidate.index}]" in user
    assert "(descendant)" in user and "(ancestor)" in user


@respx.mock
def test_curate_graph_keeps_selected_papers_and_clusters(curation_settings, expansion) -> None:
    route = respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(_GOOD_REPLY)))
    graph = curate_graph(expansion, "recursive language models", curation_settings)

    assert route.called
    assert graph.curation == "llm"
    assert graph.nodes[0].node.id == "SEED"  # seed always first, always retained
    assert len(graph.nodes) == 3
    assert {c.name for c in graph.clusters} == {"Foundations", "Scaling follow-ups"}
    assert graph.curation_notes.startswith("Cut incidental")
    kept = {sn.node.id: sn for sn in graph.nodes if sn.node.id != "SEED"}
    assert all(sn.why for sn in kept.values())
    assert all(sn.cluster_id for sn in kept.values())


@respx.mock
def test_curate_graph_sends_the_curation_model(curation_settings, expansion) -> None:
    settings = dataclasses.replace(curation_settings, curation_model="anthropic/claude-sonnet-5")
    route = respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(_GOOD_REPLY)))
    curate_graph(expansion, "recursion", settings)
    assert json.loads(route.calls[0].request.content)["model"] == "anthropic/claude-sonnet-5"


@respx.mock
def test_curate_graph_drops_edges_to_uncurated_papers(curation_settings, expansion) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(_GOOD_REPLY)))
    graph = curate_graph(expansion, "recursion", curation_settings)
    kept_ids = {sn.node.id for sn in graph.nodes}
    for edge in graph.edges:
        assert edge.source in kept_ids and edge.target in kept_ids


@respx.mock
def test_curate_graph_ignores_hallucinated_indices(curation_settings, expansion) -> None:
    reply = {
        "clusters": [{"id": 1, "name": "Foundations", "role": "ancestor", "summary": ""}],
        "keep": [
            {"i": 999, "cluster": 1, "importance": 5, "why": "does not exist"},
            {"i": 1, "cluster": 1, "importance": 5, "why": "real one"},
        ],
    }
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(reply)))
    graph = curate_graph(expansion, "recursion", curation_settings)
    assert len(graph.nodes) == 2  # seed + the one real pick


@respx.mock
def test_curate_graph_deduplicates_repeated_picks(curation_settings, expansion) -> None:
    reply = {
        "clusters": [{"id": 1, "name": "Foundations", "role": "ancestor", "summary": ""}],
        "keep": [
            {"i": 1, "cluster": 1, "importance": 5, "why": "first"},
            {"i": 1, "cluster": 1, "importance": 2, "why": "again"},
        ],
    }
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(reply)))
    graph = curate_graph(expansion, "recursion", curation_settings)
    assert len(graph.nodes) == 2


@respx.mock
def test_curate_graph_rehomes_papers_put_in_undefined_clusters(
    curation_settings, expansion
) -> None:
    reply = {
        "clusters": [{"id": 1, "name": "Foundations", "role": "ancestor", "summary": ""}],
        "keep": [{"i": 1, "cluster": 7, "importance": 3, "why": "cluster 7 was never defined"}],
    }
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(reply)))
    graph = curate_graph(expansion, "recursion", curation_settings)
    kept = [sn for sn in graph.nodes if sn.node.id != "SEED"]
    assert len(kept) == 1
    assert kept[0].cluster_id == "other"
    assert [c.id for c in graph.clusters] == ["other"]


@respx.mock
def test_curate_graph_clamps_importance_and_tolerates_string_numbers(
    curation_settings, expansion
) -> None:
    reply = {
        "clusters": [{"id": "1", "name": "Foundations", "role": "nonsense", "summary": ""}],
        "keep": [{"i": "1", "cluster": "1", "importance": "99", "why": "over the top"}],
    }
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(reply)))
    graph = curate_graph(expansion, "recursion", curation_settings)
    kept = [sn for sn in graph.nodes if sn.node.id != "SEED"][0]
    assert kept.importance == 5
    assert graph.clusters[0].role == "contemporary"  # unknown role normalized


@respx.mock
def test_curate_graph_caps_selection_at_max_nodes(curation_settings, expansion) -> None:
    settings = dataclasses.replace(curation_settings, curation_max_nodes=1)
    reply = {
        "clusters": [{"id": 1, "name": "Foundations", "role": "ancestor", "summary": ""}],
        "keep": [
            {"i": 1, "cluster": 1, "importance": 2, "why": "less central"},
            {"i": 2, "cluster": 1, "importance": 5, "why": "most central"},
        ],
    }
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(reply)))
    graph = curate_graph(expansion, "recursion", settings)
    kept = [sn for sn in graph.nodes if sn.node.id != "SEED"]
    assert len(kept) == 1
    assert kept[0].why == "most central"  # the cap keeps the most important, not the first


@respx.mock
def test_curate_graph_raises_when_nothing_recognizable_was_kept(
    curation_settings, expansion
) -> None:
    reply = {"clusters": [], "keep": [{"i": 500, "cluster": 1, "importance": 5, "why": "nope"}]}
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_reply(reply)))
    with pytest.raises(CurationError, match="no recognizable papers"):
        curate_graph(expansion, "recursion", curation_settings)


@respx.mock
def test_curate_graph_raises_on_unparseable_reply(curation_settings, expansion) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "I cannot help with that."}}]}
        )
    )
    with pytest.raises(CurationError, match="not JSON"):
        curate_graph(expansion, "recursion", curation_settings)


def test_curate_graph_raises_when_expansion_has_only_the_seed(curation_settings) -> None:
    only_seed = ExpansionResult(seed_id="SEED", nodes={"SEED": _node("SEED", "Seed")}, edges=[])
    with pytest.raises(CurationError, match="only the seed"):
        curate_graph(only_seed, "recursion", curation_settings)


def test_curate_graph_raises_when_seed_missing_from_expansion(curation_settings) -> None:
    orphaned = ExpansionResult(seed_id="SEED", nodes={"W1": _node("W1", "Other")}, edges=[])
    with pytest.raises(CurationError, match="no seed node"):
        curate_graph(orphaned, "recursion", curation_settings)


def test_curate_graph_requires_an_api_key(settings, expansion) -> None:
    keyless = dataclasses.replace(settings, openrouter_api_key=None, curation_mode="llm")
    with pytest.raises(CurationError, match="OPENROUTER_API_KEY"):
        curate_graph(expansion, "recursion", keyless)
