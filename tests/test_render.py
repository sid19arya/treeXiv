import json
import re

from treexiv.models import (
    Cluster,
    Edge,
    FilteredGraph,
    LineageNarrative,
    NarrativeBeat,
    Node,
    ScoredNode,
)
from treexiv.render import render_html


def _node(id_: str, hop: int, year: int = 2020, title: str | None = None, doi=None) -> Node:
    return Node(
        id=id_,
        title=title or f"Paper {id_}" * 3,  # long enough to exercise label truncation
        publication_year=year,
        cited_by_count=1,
        authors=["Ada Lovelace", "Alan Turing"],
        venue="Venue",
        abstract="abstract text",
        hop=hop,
        doi=doi,
    )


def _graph(nodes: list[Node], edges: list[Edge], seed_id: str = "SEED") -> FilteredGraph:
    return FilteredGraph(
        seed_id=seed_id,
        idea_text="an idea",
        top_k=10,
        nodes=[ScoredNode(n, score=1.0) for n in nodes],
        edges=edges,
    )


def _extract_json(html: str, var_name: str) -> object:
    match = re.search(rf"var {var_name} = (.*?);\n", html)
    assert match, f"{var_name} not found in rendered HTML"
    # Reverse the "</":"<\/" escaping applied before embedding.
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_render_html_writes_file(tmp_path) -> None:
    graph = _graph([_node("SEED", 0), _node("W1", 1)], [Edge("SEED", "W1")])
    out_path = tmp_path / "tree.html"
    result = render_html(graph, out_path)
    assert result == out_path
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "SEED" in content
    assert "W1" in content


def test_render_html_is_self_contained(tmp_path) -> None:
    """Must not depend on a sibling `lib/` directory or a CDN - a single
    portable file, per the PRD's "one HTML artifact per run" goal."""
    graph = _graph([_node("SEED", 0)], [])
    out_path = tmp_path / "tree.html"
    render_html(graph, out_path)
    content = out_path.read_text(encoding="utf-8")
    assert "lib/vis-9.1.2" not in content
    assert "cdn.jsdelivr" not in content
    assert "<script" in content
    assert "vis.Network" in content


def test_render_html_creates_parent_directories(tmp_path) -> None:
    out_path = tmp_path / "nested" / "dir" / "tree.html"
    render_html(_graph([_node("SEED", 0)], []), out_path)
    assert out_path.exists()


def test_render_html_drops_edges_with_missing_endpoint(tmp_path) -> None:
    graph = _graph([_node("SEED", 0)], [Edge("SEED", "NOT_IN_NODES")])
    out_path = tmp_path / "tree.html"
    render_html(graph, out_path)  # must not raise
    content = out_path.read_text(encoding="utf-8")
    edges = _extract_json(content, "EDGES")
    assert edges == []


def test_render_html_arrow_points_from_cited_to_citing(tmp_path) -> None:
    """Edge.source cites Edge.target (the data direction), but the rendered
    arrow should point the other way: from the earlier/cited paper to the
    later paper citing it, so "A -> B" reads as "A led to B"."""
    # SEED cites OLDER (SEED is the citing work, OLDER is what it cites).
    graph = _graph([_node("SEED", 0), _node("OLDER", 1)], [Edge("SEED", "OLDER")])
    out_path = tmp_path / "tree.html"
    render_html(graph, out_path)
    edges = _extract_json(out_path.read_text(encoding="utf-8"), "EDGES")
    assert edges == [{"from": "OLDER", "to": "SEED"}]


def test_render_html_embeds_relationship_and_idea_text(tmp_path) -> None:
    graph = _graph(
        [_node("SEED", 0), _node("W1", 1)], [Edge("SEED", "W1")]
    )
    out_path = tmp_path / "tree.html"
    render_html(graph, out_path)
    content = out_path.read_text(encoding="utf-8")
    nodes = _extract_json(content, "NODES")
    by_id = {n["id"]: n for n in nodes}
    assert by_id["SEED"]["is_seed"] is True
    assert by_id["W1"]["is_seed"] is False
    assert "seed" in by_id["W1"]["relationship"].lower()
    assert "an idea" in content  # embedded IDEA_TEXT


def test_render_html_positions_seed_at_origin(tmp_path) -> None:
    graph = _graph(
        [_node("SEED", 0, year=2020), _node("OLD", 1, year=2000)], [Edge("SEED", "OLD")]
    )
    out_path = tmp_path / "tree.html"
    render_html(graph, out_path)
    nodes = _extract_json(out_path.read_text(encoding="utf-8"), "NODES")
    by_id = {n["id"]: n for n in nodes}
    assert by_id["SEED"]["x"] == 0.0 and by_id["SEED"]["y"] == 0.0
    assert by_id["OLD"]["x"] < 0


def test_render_html_neutralizes_script_breakout_in_title(tmp_path) -> None:
    dangerous = 'Evil</script><script>alert(1)</script> Title'
    graph = _graph([_node("SEED", 0, title=dangerous)], [])
    out_path = tmp_path / "tree.html"
    render_html(graph, out_path)
    content = out_path.read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in content
    assert "alert(1)" in content  # content preserved, just neutralized


def test_render_html_includes_doi_link_when_present(tmp_path) -> None:
    graph = _graph([_node("SEED", 0, doi="https://doi.org/10.1/abc")], [])
    out_path = tmp_path / "tree.html"
    render_html(graph, out_path)
    nodes = _extract_json(out_path.read_text(encoding="utf-8"), "NODES")
    assert nodes[0]["doi"] == "https://doi.org/10.1/abc"


def test_render_html_larger_arrow_scale_factor(tmp_path) -> None:
    graph = _graph([_node("SEED", 0)], [])
    out_path = tmp_path / "tree.html"
    render_html(graph, out_path)
    content = out_path.read_text(encoding="utf-8")
    assert "scaleFactor: 1.6" in content


def test_render_html_fits_view_synchronously_not_via_deferred_event(tmp_path) -> None:
    """Regression test: fit() must run right after `new vis.Network(...)`,
    not inside a `network.once("afterDrawing", ...)` callback - with physics
    off, the only draw pass happens synchronously in the constructor, so a
    listener attached afterward never fires and the view stays at its
    default zoom (most nodes off-screen)."""
    graph = _graph([_node("SEED", 0)], [])
    out_path = tmp_path / "tree.html"
    render_html(graph, out_path)
    content = out_path.read_text(encoding="utf-8")
    assert 'network.once("afterDrawing"' not in content
    construct_idx = content.index("new vis.Network(container, data, options)")
    fit_idx = content.index("network.fit({ animation: false });")
    assert construct_idx < fit_idx < construct_idx + 600


def test_stats_line_names_the_curation_path(tmp_path) -> None:
    """A reader should be able to tell a curated set from a keyword top-K."""
    seed = Node(id="SEED", title="Seed", publication_year=2020, cited_by_count=1,
                authors=["Ada Example"], venue="Venue", abstract="", hop=0)
    other = Node(id="W1", title="Follow-up", publication_year=2022, cited_by_count=2,
                 authors=["Bo Example"], venue="Venue", abstract="", hop=1)
    curated = FilteredGraph(
        seed_id="SEED",
        idea_text="an idea",
        top_k=1,
        nodes=[ScoredNode(node=seed, score=0.0), ScoredNode(node=other, score=1.0,
                                                            cluster_id="c1", importance=4)],
        edges=[Edge("W1", "SEED")],
        clusters=[Cluster(id="c1", name="Follow-ups", role="descendant")],
        curation="llm",
    )
    html = render_html(curated, tmp_path / "curated.html").read_text(encoding="utf-8")
    assert "1 concept clusters" in html or "across 1 concept cluster" in html
    assert "top-1 by keyword relevance" not in html

    bm25 = FilteredGraph(
        seed_id="SEED",
        idea_text="an idea",
        top_k=1,
        nodes=[ScoredNode(node=seed, score=0.0), ScoredNode(node=other, score=1.0)],
        edges=[],
    )
    html = render_html(bm25, tmp_path / "bm25.html").read_text(encoding="utf-8")
    assert "top-1 by keyword relevance" in html


def _narrated_graph() -> FilteredGraph:
    seed = _node("SEED", 0, title="Seed paper")
    other = _node("W1", 1, title="Follow-up paper")
    return FilteredGraph(
        seed_id="SEED",
        idea_text="an idea",
        top_k=1,
        nodes=[
            ScoredNode(node=seed, score=0.0, why="The seed paper."),
            ScoredNode(node=other, score=1.0, cluster_id="c1", importance=4,
                       why="Scaled the idea up."),
        ],
        edges=[Edge("W1", "SEED")],
        clusters=[Cluster(id="c1", name="Follow-ups", summary="What grew out of it.",
                          role="descendant")],
        curation="llm",
        narrative=LineageNarrative(
            headline="A short lineage.",
            overview="First paragraph.\n\nSecond paragraph.",
            beats=[NarrativeBeat(title="The turn", text="It changed here.", node_ids=["W1"])],
        ),
    )


def test_render_embeds_the_lineage_story(tmp_path) -> None:
    html = render_html(_narrated_graph(), tmp_path / "tree.html").read_text(encoding="utf-8")
    narrative = _extract_json(html, "NARRATIVE")
    assert narrative["headline"] == "A short lineage."
    assert "Second paragraph." in narrative["overview"]
    assert narrative["beats"][0]["node_ids"] == ["W1"]


def test_render_carries_per_paper_reasons(tmp_path) -> None:
    html = render_html(_narrated_graph(), tmp_path / "tree.html").read_text(encoding="utf-8")
    nodes = _extract_json(html, "NODES")
    by_id = {n["id"]: n for n in nodes}
    assert by_id["W1"]["why"] == "Scaled the idea up."
    assert by_id["W1"]["cluster_id"] == "c1"
    assert by_id["W1"]["importance"] == 4


def test_render_without_a_narrative_has_no_story(tmp_path) -> None:
    """The BM25 path still renders; the sidebar just falls back to the seed."""
    graph = _graph([_node("SEED", 0), _node("W1", 1)], [Edge("SEED", "W1")])
    html = render_html(graph, tmp_path / "tree.html").read_text(encoding="utf-8")
    assert _extract_json(html, "NARRATIVE") is None


def _clustered_graph(n_per_cluster: int = 3) -> FilteredGraph:
    """A seed plus two clusters, with citations inside and across them."""
    seed = _node("SEED", 0, year=2020, title="Seed paper")
    nodes = [ScoredNode(node=seed, score=0.0, why="The seed.")]
    edges: list[Edge] = []
    for cluster_index, (cluster_id, base_year) in enumerate([("c1", 2015), ("c2", 2023)]):
        for i in range(n_per_cluster):
            node = _node(f"{cluster_id}_{i}", 1, year=base_year + i, title=f"Paper {cluster_id}{i}")
            nodes.append(
                ScoredNode(node=node, score=1.0, cluster_id=cluster_id, importance=3,
                           why=f"Role in {cluster_id}.")
            )
            # Within-cluster citation chain, plus an edge to the seed.
            if i:
                edges.append(Edge(f"{cluster_id}_{i}", f"{cluster_id}_{i - 1}"))
            if cluster_index == 0:
                edges.append(Edge("SEED", node.id))
            else:
                edges.append(Edge(node.id, "SEED"))
    # A cross-cluster citation: every c2 paper cites the first c1 paper.
    edges += [Edge(f"c2_{i}", "c1_0") for i in range(n_per_cluster)]
    return FilteredGraph(
        seed_id="SEED",
        idea_text="an idea",
        top_k=n_per_cluster * 2,
        nodes=nodes,
        edges=edges,
        clusters=[
            Cluster(id="c1", name="Roots", summary="Where it came from.", role="ancestor"),
            Cluster(id="c2", name="Follow-ups", summary="What it became.", role="descendant"),
        ],
        curation="llm",
    )


def test_clustered_graph_emits_one_payload_per_cluster(tmp_path) -> None:
    html = render_html(_clustered_graph(), tmp_path / "tree.html").read_text(encoding="utf-8")
    clusters = _extract_json(html, "CLUSTERS")
    assert [c["name"] for c in clusters] == ["Roots", "Follow-ups"]
    assert clusters[0]["id"] == "cluster:c1"
    assert clusters[0]["count"] == 3
    assert clusters[0]["span"] == "2015–2017"
    assert set(clusters[0]["member_ids"]) == {"c1_0", "c1_1", "c1_2"}
    assert clusters[0]["role_label"] == "Where the idea came from"
    assert clusters[1]["role_label"] == "What it grew into"


def test_cluster_members_are_positioned_inside_their_cluster(tmp_path) -> None:
    """Expanding must not move anything, so members are placed near their
    cluster's center up front."""
    html = render_html(_clustered_graph(), tmp_path / "tree.html").read_text(encoding="utf-8")
    clusters = {c["cluster_id"]: c for c in _extract_json(html, "CLUSTERS")}
    nodes = {n["id"]: n for n in _extract_json(html, "NODES")}
    for cluster_id, cluster in clusters.items():
        for member_id in cluster["member_ids"]:
            dx = abs(nodes[member_id]["x"] - cluster["x"])
            dy = abs(nodes[member_id]["y"] - cluster["y"])
            assert dx < 400 and dy < 400, f"{member_id} drifted from {cluster_id}"


def test_clusters_do_not_overlap_each_other(tmp_path) -> None:
    html = render_html(_clustered_graph(), tmp_path / "tree.html").read_text(encoding="utf-8")
    clusters = _extract_json(html, "CLUSTERS")
    (a, b) = clusters[0], clusters[1]
    distance = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
    assert distance > 500


def test_legend_switches_to_cluster_roles_when_clustered(tmp_path) -> None:
    html = render_html(_clustered_graph(), tmp_path / "tree.html").read_text(encoding="utf-8")
    labels = [entry["label"] for entry in _extract_json(html, "HOP_LEGEND")]
    assert labels == ["Seed paper", "Where the idea came from", "What it grew into"]


def test_legend_stays_hop_based_without_clusters(tmp_path) -> None:
    graph = _graph([_node("SEED", 0), _node("W1", 1)], [Edge("SEED", "W1")])
    html = render_html(graph, tmp_path / "tree.html").read_text(encoding="utf-8")
    labels = [entry["label"] for entry in _extract_json(html, "HOP_LEGEND")]
    assert labels == ["Seed", "1 hop away", "2 hops away"]


def test_flat_graph_emits_no_clusters(tmp_path) -> None:
    """A BM25 graph must render exactly as before: no cluster layer at all."""
    graph = _graph([_node("SEED", 0), _node("W1", 1)], [Edge("SEED", "W1")])
    html = render_html(graph, tmp_path / "tree.html").read_text(encoding="utf-8")
    assert _extract_json(html, "CLUSTERS") == []


def test_every_paper_and_edge_is_still_embedded_when_clustered(tmp_path) -> None:
    """Collapsing happens in the browser, not in the data: the page must ship
    the full graph so expanding a cluster needs no round trip."""
    graph = _clustered_graph()
    html = render_html(graph, tmp_path / "tree.html").read_text(encoding="utf-8")
    assert len(_extract_json(html, "NODES")) == len(graph.nodes)
    assert len(_extract_json(html, "EDGES")) == len(graph.edges)


def test_cluster_membership_partitions_the_papers(tmp_path) -> None:
    """The browser resolves each paper to "itself, or its cluster". That only
    works if every non-seed paper belongs to exactly one cluster's member list."""
    graph = _clustered_graph()
    html = render_html(graph, tmp_path / "tree.html").read_text(encoding="utf-8")
    clusters = _extract_json(html, "CLUSTERS")
    nodes = _extract_json(html, "NODES")

    seen: list[str] = []
    for cluster in clusters:
        seen.extend(cluster["member_ids"])
    assert len(seen) == len(set(seen)), "a paper appears in two clusters"
    assert set(seen) == {n["id"] for n in nodes if not n["is_seed"]}


def test_unclustered_papers_stay_visible_alongside_clusters(tmp_path) -> None:
    """A paper the curator left unclustered must not vanish: the browser only
    hides papers it can resolve to a cluster."""
    graph = _clustered_graph()
    stray = _node("STRAY", 1, year=2019, title="Unclustered paper")
    graph.nodes.append(ScoredNode(node=stray, score=0.5))
    graph.edges.append(Edge("STRAY", "SEED"))

    html = render_html(graph, tmp_path / "tree.html").read_text(encoding="utf-8")
    clusters = _extract_json(html, "CLUSTERS")
    assert all("STRAY" not in c["member_ids"] for c in clusters)
    assert any(n["id"] == "STRAY" for n in _extract_json(html, "NODES"))
