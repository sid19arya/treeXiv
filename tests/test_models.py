from tests.conftest import make_work_payload
from treexiv.models import (
    Cluster,
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


def test_filtered_graph_round_trips_clusters_and_curation_fields() -> None:
    graph = FilteredGraph(
        seed_id="SEED",
        idea_text="recursive language models",
        top_k=2,
        nodes=[
            ScoredNode(
                node=Node(
                    id="W1",
                    title="A paper",
                    publication_year=2021,
                    cited_by_count=3,
                    authors=["Ada Example"],
                    venue="Venue",
                    abstract="text",
                    hop=1,
                ),
                score=1.5,
                cluster_id="c1",
                importance=4,
                why="Extended the core idea.",
            )
        ],
        edges=[Edge("W1", "SEED")],
        clusters=[Cluster(id="c1", name="Follow-ups", summary="What grew out of it.",
                          role="descendant")],
        curation="llm",
        curation_notes="Cut incidental citations.",
    )

    restored = FilteredGraph.from_dict(graph.to_dict())

    assert restored.curation == "llm"
    assert restored.curation_notes == "Cut incidental citations."
    assert restored.clusters[0].name == "Follow-ups"
    assert restored.clusters[0].role == "descendant"
    assert restored.nodes[0].cluster_id == "c1"
    assert restored.nodes[0].importance == 4
    assert restored.nodes[0].why == "Extended the core idea."
    assert restored.nodes[0].node.title == "A paper"


def test_filtered_graph_reads_pre_curation_json() -> None:
    """JSON written before curation existed still loads, as the BM25 shape."""
    legacy = {
        "seed_id": "SEED",
        "idea_text": "idea",
        "top_k": 1,
        "nodes": [
            {
                "id": "SEED",
                "title": "Seed",
                "publication_year": 2020,
                "cited_by_count": 0,
                "authors": [],
                "venue": None,
                "abstract": "",
                "hop": 0,
                "doi": None,
                "bm25_score": 0.0,
            }
        ],
        "edges": [],
    }
    graph = FilteredGraph.from_dict(legacy)
    assert graph.curation == "bm25"
    assert graph.clusters == []
    assert graph.nodes[0].cluster_id is None
