from treexiv.models import Edge, Node
from treexiv.narrative import SEED_DESCRIPTION, describe_relationship


def _node(id_: str, title: str, hop: int) -> Node:
    return Node(
        id=id_, title=title, publication_year=2020, cited_by_count=1,
        authors=[], venue=None, abstract="", hop=hop,
    )


def test_seed_node_gets_seed_description() -> None:
    nodes = {"SEED": _node("SEED", "Seed Paper", 0)}
    assert describe_relationship("SEED", "SEED", nodes, []) == SEED_DESCRIPTION


def test_ancestor_direct_edge_seed_cites_node() -> None:
    nodes = {"SEED": _node("SEED", "Seed", 0), "A": _node("A", "Ancestor", 1)}
    edges = [Edge(source="SEED", target="A")]
    text = describe_relationship("A", "SEED", nodes, edges)
    assert "seed" in text.lower()
    assert "cites this one directly" in text


def test_descendant_direct_edge_node_cites_seed() -> None:
    nodes = {"SEED": _node("SEED", "Seed", 0), "D": _node("D", "Descendant", 1)}
    edges = [Edge(source="D", target="SEED")]
    text = describe_relationship("D", "SEED", nodes, edges)
    assert "cites the seed directly" in text


def test_two_hop_chain_mentions_bridge_title() -> None:
    nodes = {
        "SEED": _node("SEED", "Seed", 0),
        "MID": _node("MID", "Bridge Paper", 1),
        "FAR": _node("FAR", "Far Paper", 2),
    }
    edges = [Edge(source="SEED", target="MID"), Edge(source="MID", target="FAR")]
    text = describe_relationship("FAR", "SEED", nodes, edges)
    assert "Bridge Paper" in text
    assert "Two hops" in text


def test_two_hop_chain_in_descendant_direction() -> None:
    # FAR cites MID, MID cites SEED - a forward (descendant) chain, exercising
    # the "edge.source == id_a" branch of _edge_phrase rather than the
    # reversed one the ancestor-direction test above hits.
    nodes = {
        "SEED": _node("SEED", "Seed", 0),
        "MID": _node("MID", "Bridge Paper", 1),
        "FAR": _node("FAR", "Far Paper", 2),
    }
    edges = [Edge(source="FAR", target="MID"), Edge(source="MID", target="SEED")]
    text = describe_relationship("FAR", "SEED", nodes, edges)
    assert '"Far Paper" cites "Bridge Paper"' in text
    assert "Bridge Paper" in text and "cites" in text


def test_skips_bridge_candidate_not_itself_connected_to_seed() -> None:
    # FAR connects to both DEAD_END (no path to seed) and MID (which does) -
    # the first candidate tried must not short-circuit the search.
    nodes = {
        "SEED": _node("SEED", "Seed", 0),
        "DEAD_END": _node("DEAD_END", "Dead End", 1),
        "MID": _node("MID", "Bridge Paper", 1),
        "FAR": _node("FAR", "Far Paper", 2),
    }
    edges = [
        Edge(source="FAR", target="DEAD_END"),
        Edge(source="SEED", target="MID"),
        Edge(source="MID", target="FAR"),
    ]
    text = describe_relationship("FAR", "SEED", nodes, edges)
    assert "Bridge Paper" in text


def test_no_traceable_path_falls_back_to_hop_count() -> None:
    nodes = {"SEED": _node("SEED", "Seed", 0), "ORPHAN": _node("ORPHAN", "Orphan", 2)}
    text = describe_relationship("ORPHAN", "SEED", nodes, [])
    assert "2 hop" in text


def test_ignores_edges_to_nodes_outside_the_filtered_set() -> None:
    # Edge references a bridge node that isn't in nodes_by_id (filtered out) -
    # must not raise KeyError, and should fall back gracefully.
    nodes = {"SEED": _node("SEED", "Seed", 0), "FAR": _node("FAR", "Far Paper", 2)}
    edges = [Edge(source="FAR", target="GONE"), Edge(source="GONE", target="SEED")]
    text = describe_relationship("FAR", "SEED", nodes, edges)
    assert "2 hop" in text
