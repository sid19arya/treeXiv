from tests.conftest import make_work_payload
from treexiv.models import (
    Edge,
    ExpansionResult,
    FilteredGraph,
    Node,
    ScoredNode,
    Work,
    normalize_work_id,
)


def test_normalize_work_id_strips_url_prefix() -> None:
    assert normalize_work_id("https://openalex.org/W123") == "W123"


def test_normalize_work_id_passthrough_short_form() -> None:
    assert normalize_work_id("W123") == "W123"


def test_normalize_work_id_uppercases_w_prefix() -> None:
    assert normalize_work_id("w123") == "W123"


def test_work_from_api_parses_core_fields() -> None:
    payload = make_work_payload(
        "W1",
        "A Great Paper",
        year=2021,
        cited_by_count=42,
        referenced_works=["W2", "W3"],
        abstract_words={"Great": [0], "results": [1]},
        authors=["Alice", "Bob"],
        venue="NeurIPS",
    )
    work = Work.from_api(payload)
    assert work.id == "W1"
    assert work.title == "A Great Paper"
    assert work.publication_year == 2021
    assert work.cited_by_count == 42
    assert work.referenced_works == ["W2", "W3"]
    assert work.authors == ["Alice", "Bob"]
    assert work.venue == "NeurIPS"
    assert work.abstract == "Great results"


def test_work_from_api_handles_missing_optional_fields() -> None:
    payload = {"id": "https://openalex.org/W9", "display_name": "Bare Paper"}
    work = Work.from_api(payload)
    assert work.id == "W9"
    assert work.cited_by_count == 0
    assert work.authors == []
    assert work.venue is None
    assert work.abstract == ""


def test_node_round_trips_through_dict() -> None:
    node = Node(
        id="W1",
        title="Paper",
        publication_year=2020,
        cited_by_count=5,
        authors=["A"],
        venue="V",
        abstract="text",
        hop=1,
    )
    assert Node.from_dict(node.to_dict()) == node


def test_edge_round_trips_through_dict() -> None:
    edge = Edge(source="W1", target="W2")
    assert Edge.from_dict(edge.to_dict()) == edge


def test_expansion_result_round_trips_through_dict() -> None:
    seed = Node("W1", "Seed", 2020, 10, [], None, "", 0)
    child = Node("W2", "Child", 2019, 5, [], None, "", 1)
    result = ExpansionResult(
        seed_id="W1",
        nodes={"W1": seed, "W2": child},
        edges=[Edge("W1", "W2")],
        truncated=True,
    )
    restored = ExpansionResult.from_dict(result.to_dict())
    assert restored.seed_id == "W1"
    assert restored.truncated is True
    assert restored.nodes.keys() == {"W1", "W2"}
    assert restored.edges == [Edge("W1", "W2")]


def test_filtered_graph_round_trips_through_dict() -> None:
    node = Node("W1", "Seed", 2020, 10, [], None, "", 0)
    graph = FilteredGraph(
        seed_id="W1",
        idea_text="an idea",
        top_k=40,
        nodes=[ScoredNode(node=node, score=3.5)],
        edges=[],
    )
    restored = FilteredGraph.from_dict(graph.to_dict())
    assert restored.seed_id == "W1"
    assert restored.idea_text == "an idea"
    assert restored.nodes[0].score == 3.5
    assert restored.nodes[0].node.id == "W1"
