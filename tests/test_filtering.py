"""Tests for Step 4: the deterministic BM25 filter and the `build_graph`
entry point that chooses between it and the LLM curation path."""

from __future__ import annotations

import dataclasses
import json

import httpx
import pytest
import respx

from treexiv.exceptions import CurationError
from treexiv.filtering import build_graph, filter_by_idea
from treexiv.models import Edge, ExpansionResult, Node


def _node(id_: str, title: str, abstract: str = "", cited_by_count: int = 1) -> Node:
    return Node(id=id_, title=title, publication_year=2020, cited_by_count=cited_by_count,
                authors=[], venue=None, abstract=abstract, hop=1)


def test_seed_always_retained_even_with_low_score() -> None:
    seed = _node("SEED", "Completely unrelated gardening tips")
    relevant = _node("W1", "Recursive language models and reasoning")
    expansion = ExpansionResult(
        seed_id="SEED",
        nodes={"SEED": seed, "W1": relevant},
        edges=[Edge("SEED", "W1")],
    )
    filtered = filter_by_idea(expansion, "recursive language models", top_k=1)
    ids = {n.node.id for n in filtered.nodes}
    assert "SEED" in ids


def test_keeps_only_top_k_non_seed_nodes() -> None:
    seed = _node("SEED", "Recursive language models overview")
    nodes = {"SEED": seed}
    for i in range(5):
        nodes[f"W{i}"] = _node(f"W{i}", "Recursive language models extension paper")
    expansion = ExpansionResult(seed_id="SEED", nodes=nodes, edges=[])
    filtered = filter_by_idea(expansion, "recursive language models", top_k=2)
    assert len(filtered.nodes) == 3  # seed + top 2


def test_drops_edges_with_filtered_out_endpoint() -> None:
    seed = _node("SEED", "Recursive language models")
    kept = _node("W1", "Recursive language models extension")
    dropped = _node("W2", "Totally unrelated topic about cooking pasta")
    expansion = ExpansionResult(
        seed_id="SEED",
        nodes={"SEED": seed, "W1": kept, "W2": dropped},
        edges=[Edge("SEED", "W1"), Edge("SEED", "W2")],
    )
    filtered = filter_by_idea(expansion, "recursive language models", top_k=1)
    kept_ids = {n.node.id for n in filtered.nodes}
    assert "W2" not in kept_ids
    assert Edge("SEED", "W2") not in filtered.edges
    assert Edge("SEED", "W1") in filtered.edges


def test_nodes_sorted_by_score_descending() -> None:
    seed = _node("SEED", "Recursive language models")
    high = _node("W1", "Recursive language models recursive language models")
    low = _node("W2", "Recursive")
    expansion = ExpansionResult(
        seed_id="SEED", nodes={"SEED": seed, "W1": high, "W2": low}, edges=[]
    )
    filtered = filter_by_idea(expansion, "recursive language models", top_k=2)
    scores = [n.score for n in filtered.nodes]
    assert scores == sorted(scores, reverse=True)


def test_top_k_larger_than_corpus_keeps_everything() -> None:
    seed = _node("SEED", "Seed paper")
    other = _node("W1", "Other paper")
    expansion = ExpansionResult(seed_id="SEED", nodes={"SEED": seed, "W1": other}, edges=[])
    filtered = filter_by_idea(expansion, "seed", top_k=100)
    assert len(filtered.nodes) == 2


# --- build_graph: which of the two Step 4 paths actually runs -----------------

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_CURATION_REPLY = {
    "choices": [
        {
            "message": {
                "content": json.dumps(
                    {
                        "clusters": [
                            {"id": 1, "name": "Foundations", "role": "ancestor", "summary": "s"}
                        ],
                        "keep": [{"i": 1, "cluster": 1, "importance": 5, "why": "central"}],
                    }
                )
            }
        }
    ]
}


@pytest.fixture
def two_node_expansion():
    seed = _node("SEED", "Recursive language models")
    other = _node("W1", "Recursive language models extension")
    return ExpansionResult(
        seed_id="SEED", nodes={"SEED": seed, "W1": other}, edges=[Edge("W1", "SEED")]
    )


def test_build_graph_bm25_mode_never_calls_the_llm(settings, two_node_expansion) -> None:
    keyed = dataclasses.replace(settings, curation_mode="bm25", openrouter_api_key="sk-or-test")
    # respx isn't active here: any HTTP call at all would raise.
    graph = build_graph(two_node_expansion, "recursive language models", keyed)
    assert graph.curation == "bm25"
    assert graph.clusters == []


def test_build_graph_auto_falls_back_to_bm25_without_a_key(settings, two_node_expansion) -> None:
    keyless = dataclasses.replace(settings, curation_mode="auto", openrouter_api_key=None)
    warnings: list[str] = []
    graph = build_graph(
        two_node_expansion, "recursive language models", keyless, on_warning=warnings.append
    )
    assert graph.curation == "bm25"
    assert "OPENROUTER_API_KEY" in warnings[0]


@respx.mock
def test_build_graph_auto_curates_when_a_key_is_available(settings, two_node_expansion) -> None:
    keyed = dataclasses.replace(settings, curation_mode="auto", openrouter_api_key="sk-or-test")
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_CURATION_REPLY))
    graph = build_graph(two_node_expansion, "recursive language models", keyed)
    assert graph.curation == "llm"
    assert [c.name for c in graph.clusters] == ["Foundations"]


@respx.mock
def test_build_graph_auto_falls_back_when_curation_fails(settings, two_node_expansion) -> None:
    keyed = dataclasses.replace(settings, curation_mode="auto", openrouter_api_key="sk-or-test")
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(500))
    warnings: list[str] = []
    graph = build_graph(
        two_node_expansion, "recursive language models", keyed, on_warning=warnings.append
    )
    assert graph.curation == "bm25"
    assert "falling back" in warnings[0]


@respx.mock
def test_build_graph_llm_mode_surfaces_the_failure(settings, two_node_expansion) -> None:
    keyed = dataclasses.replace(settings, curation_mode="llm", openrouter_api_key="sk-or-test")
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(500))
    with pytest.raises(CurationError):
        build_graph(two_node_expansion, "recursive language models", keyed)
