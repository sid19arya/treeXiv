"""Tests for exhaustive mode's Jev-driven selection, relations and clustering.

Jev is replaced by a deterministic fake evaluator, and OpenRouter (the
summariser) by respx, so nothing here touches the network.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import httpx
import pytest
import respx

from treexiv import jev
from treexiv.exceptions import JevError
from treexiv.exhaustive import (
    RELATIONS,
    build_exhaustive_graph,
    candidate_pairs,
    cluster_papers,
    cluster_role,
    keyword_name,
    order_pair,
    select_kept,
)
from treexiv.models import EDGE_SEMANTIC, SEARCH_HOP, Edge, ExpansionResult, Node

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _node(id_: str, title: str, year: int | None, *, hop: int = 1, abstract: str = "") -> Node:
    return Node(
        id=id_,
        title=title,
        publication_year=year,
        cited_by_count=5,
        authors=["Ada Example"],
        venue=None,
        abstract=abstract or title,
        hop=hop,
    )


@pytest.fixture
def corpus() -> ExpansionResult:
    """SEED cites OLD; NEW cites SEED; LATE cites NEW. SIB was found only by
    search. OFF is off-topic and should never survive."""
    nodes = {
        "SEED": _node("SEED", "Recursive language models", 2020, hop=0),
        "OLD": _node("OLD", "Early recursion in neural models", 2015),
        "NEW": _node("NEW", "Recursive language models at scale", 2023),
        "LATE": _node("LATE", "Recursive models for long documents", 2024, hop=2),
        "SIB": _node("SIB", "Recursion as an alternative to long context", 2022, hop=SEARCH_HOP),
        "OFF": _node("OFF", "Pasta cooking techniques", 2021),
    }
    edges = [Edge("SEED", "OLD"), Edge("NEW", "SEED"), Edge("LATE", "NEW"), Edge("SEED", "OFF")]
    return ExpansionResult(seed_id="SEED", nodes=nodes, edges=edges)


def _fake_evaluator(calls: list[dict] | None = None):
    """Relevance: 3.5 for anything about recursion, 0.5 otherwise. Relations:
    SIB is an 'alternative' to everything, other pairs 'extends'."""

    def evaluate(state: Any, questions: dict[str, jev.Question]) -> dict[str, jev.Answer]:
        if calls is not None:
            calls.append(questions)
        answers = {}
        for key, question in questions.items():
            instructions = question["instructions"]
            if question["type"] == "score":
                title = instructions["paper"]["title"].lower()
                value = 3.5 if "recurs" in title else 0.5
                answers[key] = jev.Answer(type="score", value=value, probabilities={})
            elif "paper" in instructions:
                # Lineage role: the oldest paper is the foundation.
                old = instructions["paper"]["year"] < 2020
                picked = "foundation" if old else "incremental"
                answers[key] = jev.Answer(
                    type="choice", value=picked, probabilities={picked: 0.8, "tangential": 0.2}
                )
            else:
                titles = {instructions["earlier"]["title"], instructions["later"]["title"]}
                picked = "alternative" if any("alternative" in t for t in titles) else "extends"
                probabilities = dict.fromkeys(RELATIONS, 0.025)
                probabilities[picked] = 0.9
                answers[key] = jev.Answer(type="choice", value=picked, probabilities=probabilities)
        return answers

    return evaluate


@pytest.fixture
def ex_settings(settings):
    return dataclasses.replace(
        settings, exhaustive_keep=10, exhaustive_neighbours=2, ai_gateway_api_key="vck-test"
    )


def test_selects_relevant_papers_and_keeps_edge_kinds_apart(corpus, ex_settings) -> None:
    calls: list[dict] = []
    graph = build_exhaustive_graph(
        corpus, "recursive language models", ex_settings, evaluator=_fake_evaluator(calls)
    )
    ids = {sn.node.id for sn in graph.nodes}
    assert ids == {"SEED", "OLD", "NEW", "LATE", "SIB"}
    assert graph.curation == "jev"

    # Relevance (5 candidates, no work IDs in the questions), lineage role for
    # the 4 that pass the gate, then relations.
    assert len(calls) == 3
    assert all(key.startswith("p") for key in calls[0])
    assert "SEED" not in json.dumps(calls[0])
    assert sorted(calls[1]) == ["l0", "l1", "l2", "l3"]

    citations = [e for e in graph.edges if e.is_citation]
    semantic = [e for e in graph.edges if not e.is_citation]
    # Real citations survive (the one to OFF doesn't) and carry Jev's reading.
    assert {(e.source, e.target) for e in citations} == {
        ("SEED", "OLD"),
        ("NEW", "SEED"),
        ("LATE", "NEW"),
    }
    assert all(e.relation == "extends" for e in citations)
    # SIB has no citation at all, so its relation to the seed is a semantic
    # edge, oriented later -> earlier like a citation would be.
    assert Edge("SIB", "SEED", kind=EDGE_SEMANTIC, relation="alternative", confidence=0.9) in (
        semantic
    )
    # No semantic edge duplicates a citation pair.
    cited = {frozenset((e.source, e.target)) for e in citations}
    assert not any(frozenset((e.source, e.target)) in cited for e in semantic)


def test_every_kept_paper_is_clustered_with_a_why(corpus, ex_settings) -> None:
    graph = build_exhaustive_graph(corpus, "recursion", ex_settings, evaluator=_fake_evaluator())
    cluster_ids = {c.id for c in graph.clusters}
    for scored in graph.nodes:
        if scored.node.id == "SEED":
            assert scored.cluster_id is None
            continue
        assert scored.cluster_id in cluster_ids
        assert scored.why.startswith("Jev relevance 3.5/4. Lineage role: ")
        assert scored.importance == 5
    assert "Jev scored 5 candidates (4 from citation traversal, 1 from search); 4 reached" in (
        graph.curation_notes
    )


def test_without_an_llm_key_strands_get_keyword_names(corpus, ex_settings) -> None:
    warnings: list[str] = []
    graph = build_exhaustive_graph(
        corpus, "recursion", ex_settings, evaluator=_fake_evaluator(), on_warning=warnings.append
    )
    assert graph.narrative is None
    assert all(c.name for c in graph.clusters)
    assert any("OPENROUTER_API_KEY" in w for w in warnings)


@respx.mock
def test_summariser_names_strands_and_writes_the_story(corpus, ex_settings) -> None:
    settings = dataclasses.replace(ex_settings, openrouter_api_key="sk-or-test")
    graph_ids = build_exhaustive_graph(
        corpus, "recursion", ex_settings, evaluator=_fake_evaluator()
    )
    strand_count = len([c for c in graph_ids.clusters if c.id != "other"])
    naming = {"clusters": [{"id": i, "name": f"Strand {i}", "summary": "s"} for i in range(1, 9)]}
    story = {"headline": "h", "overview": "o", "beats": []}
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})
            for body in (naming, story)
        ]
    )
    graph = build_exhaustive_graph(corpus, "recursion", settings, evaluator=_fake_evaluator())
    assert route.call_count == 2
    named = [c for c in graph.clusters if c.id != "other"]
    assert [c.name for c in named] == [f"Strand {i}" for i in range(1, strand_count + 1)]
    assert graph.narrative is not None and graph.narrative.overview == "o"


def test_needs_a_gateway_key_without_an_injected_evaluator(corpus, settings) -> None:
    with pytest.raises(JevError, match="AI_GATEWAY_API_KEY"):
        build_exhaustive_graph(corpus, "recursion", settings)


def test_nothing_relevant_enough_raises(corpus, ex_settings) -> None:
    strict = dataclasses.replace(ex_settings, exhaustive_min_relevance=3.9)
    with pytest.raises(JevError, match="no candidate"):
        build_exhaustive_graph(corpus, "recursion", strict, evaluator=_fake_evaluator())


def test_order_pair_prefers_citation_over_year() -> None:
    a, b = _node("A", "a", 2020), _node("B", "b", 2018)
    assert order_pair(a, b, set()) == (b, a)
    # B cites A, so A is the earlier work whatever the metadata says.
    assert order_pair(a, b, {("B", "A")}) == (a, b)
    assert order_pair(_node("N", "n", None), b, set())[0].id == "B"


def test_candidate_pairs_cover_seed_citations_and_neighbours() -> None:
    seed = _node("S", "graph neural networks", 2020, hop=0)
    kept = [
        _node("A", "graph neural message passing", 2019),
        _node("B", "graph neural attention", 2021),
        _node("C", "protein folding structures", 2022),
    ]
    pairs = candidate_pairs(seed, kept, [Edge("C", "A")], neighbours=1)
    assert {("A", "S"), ("B", "S"), ("C", "S")} <= set(pairs)
    assert ("A", "C") in pairs  # citation edge
    assert ("A", "B") in pairs  # BM25 neighbours
    assert pairs == sorted(set(pairs))


def test_cluster_papers_separates_components_and_collects_loose_papers() -> None:
    adjacency: dict[str, dict[str, float]] = {}

    def link(a: str, b: str, w: float = 1.0) -> None:
        adjacency.setdefault(a, {})[b] = w
        adjacency.setdefault(b, {})[a] = w

    for a, b in [("A", "B"), ("B", "C"), ("A", "C"), ("X", "Y"), ("Y", "Z")]:
        link(a, b)
    link("W", "A", 0.1)  # too weak to join by propagation, but joins as a singleton
    strands, loose = cluster_papers(["A", "X", "B", "Y", "C", "Z", "W", "LONE"], adjacency)
    assert strands == [["A", "B", "C", "W"], ["X", "Y", "Z"]]
    assert loose == ["LONE"]


def test_cluster_papers_folds_down_to_max_clusters() -> None:
    adjacency: dict[str, dict[str, float]] = {}
    ids = []
    for i in range(4):
        a, b = f"a{i}", f"b{i}"
        ids += [a, b]
        adjacency.setdefault(a, {})[b] = 1.0
        adjacency.setdefault(b, {})[a] = 1.0
    # Pair 3 is weakly tied to pair 0, so that's where it folds.
    adjacency["a3"]["a0"] = adjacency.setdefault("a0", {})["a3"] = 0.2
    strands, loose = cluster_papers(ids, adjacency, max_clusters=3)
    assert len(strands) == 3
    assert sorted(strands[0]) == ["a0", "a3", "b0", "b3"]
    assert loose == []


def test_cluster_role_votes_by_direction_then_year() -> None:
    members = [_node("A", "a", 2010), _node("B", "b", 2012), _node("C", "c", 2030)]
    assert cluster_role(members, {}, 2020) == "ancestor"
    assert cluster_role(members, {"A": "descendant", "B": "descendant"}, 2020) == "descendant"
    assert cluster_role(members[:2], {"A": "descendant"}, 2020) == "contemporary"


def test_keyword_name_uses_distinctive_title_words() -> None:
    members = [
        _node("A", "Sparse attention kernels", 2020),
        _node("B", "Sparse attention at scale", 2021),
    ]
    everyone = members + [_node("C", "Dense attention baselines", 2019)]
    assert keyword_name(members, everyone).startswith("Sparse")


def test_lineage_rank_puts_a_foundation_above_a_more_on_topic_variant() -> None:
    """The failure the role question exists for: Jev rates a recent variant
    as more relevant than the paper that started the line."""
    foundation = _node("F", "FrugalGPT", 2023)
    variant = _node("V", "SkewRoute", 2025)
    roles = {
        "F": jev.Answer("choice", "foundation", {"foundation": 0.52, "incremental": 0.2}),
        "V": jev.Answer("choice", "incremental", {"foundation": 0.0, "incremental": 0.98}),
    }
    kept = select_kept([variant, foundation], {"F": 2.77, "V": 2.98}, roles, keep=1)
    assert [n.id for n in kept] == ["F"]
