import json
import re

from treexiv.models import Cluster, Edge, FilteredGraph, Node, ScoredNode
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
