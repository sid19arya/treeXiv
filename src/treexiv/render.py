"""Step 5: render a `FilteredGraph` as a single self-contained, interactive HTML file.

No frontend framework, no build step, no CDN dependency (vis-network is
vendored in `treexiv/assets/vis-network/` and inlined) - just one HTML file
per run, per the PRD's non-goals.

Two views, chosen by whether the graph has concept clusters:

- **Clustered** (the curated path). The page opens showing one node per
  concept cluster, not per paper - a handful of named strands instead of
  thirty-odd dots. Clicking a cluster expands it in place into its papers;
  clicking again collapses it. Edges are aggregated to whatever is currently
  visible, so a citation between two collapsed clusters is drawn once, thick,
  between them.
- **Flat** (the BM25 path, or any graph with no clusters). Every paper on the
  year diagonal, as before.

Expansion is done by rebuilding the two vis-network DataSets from a visibility
map rather than by vis-network's own `cluster()` API. Positions are all
computed up front in `layout.py` and physics stays off, so nothing moves when
a cluster opens - which is the entire point of a deterministic layout, and
something the built-in clustering makes hard to guarantee.

Each node also carries a `narrative.describe_relationship` string, shown in
the left sidebar when that node is selected.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from treexiv.curate import seed_edge_intents
from treexiv.layout import Position, compute_cluster_layout, compute_positions
from treexiv.models import Cluster, FilteredGraph, LineageNarrative, Node, ScoredNode
from treexiv.narrative import describe_relationship

_HOP_COLORS = {0: "#f2a900", 1: "#2a9d8f", 2: "#8ecae6"}
_DEFAULT_COLOR = "#cbd5e1"
_HOP_LABELS = {0: "Seed", 1: "1 hop away", 2: "2 hops away"}
_SEED_COLOR = "#f2a900"
_SEED_BORDER = "#7c4a03"

# Clusters are coloured by their role in the lineage, which is the one thing a
# reader wants from the collapsed view: what came before, what came after.
_ROLE_COLORS = {
    "ancestor": "#2a9d8f",
    "descendant": "#8ecae6",
    "contemporary": "#c8b6e2",
}
_ROLE_LABELS = {
    "ancestor": "Where the idea came from",
    "descendant": "What it grew into",
    "contemporary": "Parallel work",
}


def _load_asset(name: str) -> str:
    """Read a vendored asset from `treexiv/assets/vis-network/` (not a Python
    subpackage - the hyphen in the directory name is deliberate and fine for
    a plain resource path)."""
    return (resources.files("treexiv") / "assets" / "vis-network" / name).read_text(
        encoding="utf-8"
    )


def _short_label(node: Node, max_title_len: int = 42) -> str:
    """A compact graph label: "Lastname et al. 2019" when we have authors
    and a year, otherwise a truncated title."""
    if node.authors and node.publication_year:
        last_name = node.authors[0].split()[-1] if node.authors[0].split() else node.authors[0]
        suffix = " et al." if len(node.authors) > 1 else ""
        return f"{last_name}{suffix} {node.publication_year}"
    title = node.title
    return title if len(title) <= max_title_len else title[: max_title_len - 1] + "…"


def _escape_for_inline_script(payload: str) -> str:
    """Neutralize `</script>` sequences that could appear inside embedded
    JSON (e.g. a paper title literally containing that substring)."""
    return payload.replace("</", "<\\/")


def _node_payload(
    scored: ScoredNode, is_seed: bool, relationship: str, position: Position, intents: str
) -> dict:
    node, score = scored.node, scored.score
    return {
        "id": node.id,
        "label": _short_label(node),
        "title": node.title,
        "abstract": node.abstract or "(no abstract available)",
        "authors": node.authors,
        "venue": node.venue,
        "publication_year": node.publication_year,
        "cited_by_count": node.cited_by_count,
        "doi": node.doi,
        "hop": node.hop,
        "score": score,
        "is_seed": is_seed,
        "relationship": relationship,
        "why": scored.why,
        "cluster_id": scored.cluster_id,
        "importance": scored.importance,
        "intents": intents,
        "x": position.x,
        "y": position.y,
        "color": {
            "background": _SEED_COLOR if is_seed else _HOP_COLORS.get(node.hop, _DEFAULT_COLOR),
            "border": _SEED_BORDER if is_seed else "#334155",
        },
        "size": 14 + min(node.cited_by_count, 5000) ** 0.5,
        "borderWidth": 3 if is_seed else 1.5,
        "font": {"size": 13 if is_seed else 11},
    }


def _narrative_payload(
    narrative: LineageNarrative | None, clusters: list[Cluster]
) -> dict | None:
    """The written lineage story, or None when a run produced no prose."""
    if narrative is None and not clusters:
        return None
    return {
        "headline": narrative.headline if narrative else "",
        "overview": narrative.overview if narrative else "",
        "beats": [
            {"title": b.title, "text": b.text, "node_ids": b.node_ids}
            for b in (narrative.beats if narrative else [])
        ],
    }


def _cluster_payloads(graph: FilteredGraph) -> tuple[list[dict], dict[str, Position]]:
    """One collapsed node per concept cluster, plus every member's position.

    Returns an empty list for a graph with no clusters, which is what puts the
    renderer into its flat view.
    """
    members_by_cluster: dict[str, list[Node]] = {}
    for scored in graph.nodes:
        if scored.node.id == graph.seed_id or not scored.cluster_id:
            continue
        members_by_cluster.setdefault(scored.cluster_id, []).append(scored.node)
    if not members_by_cluster:
        return [], {}

    seed_node = next((sn.node for sn in graph.nodes if sn.node.id == graph.seed_id), None)
    layouts = compute_cluster_layout(
        members_by_cluster, seed_node.publication_year if seed_node else None
    )

    positions: dict[str, Position] = {}
    for cluster_layout in layouts.values():
        positions.update(cluster_layout.members)

    payloads = []
    for cluster in graph.clusters:
        members = members_by_cluster.get(cluster.id)
        placed = layouts.get(cluster.id)
        if not members or placed is None:
            continue
        years = [n.publication_year for n in members if n.publication_year is not None]
        payloads.append(
            {
                "id": f"cluster:{cluster.id}",
                "cluster_id": cluster.id,
                "label": f"{cluster.name}\n({len(members)} papers)",
                "name": cluster.name,
                "summary": cluster.summary,
                "role": cluster.role,
                "role_label": _ROLE_LABELS.get(cluster.role, cluster.role),
                "member_ids": [n.id for n in members],
                "count": len(members),
                "span": f"{min(years)}\u2013{max(years)}" if years else "",
                "x": placed.center.x,
                "y": placed.center.y,
                "color": {
                    "background": _ROLE_COLORS.get(cluster.role, _DEFAULT_COLOR),
                    "border": "#334155",
                },
                "size": 26 + min(len(members), 40) ** 0.6 * 6,
                "font": {"size": 16, "face": "-apple-system, Segoe UI, Roboto, sans-serif"},
            }
        )
    return payloads, positions


def _legend(cluster_payloads: list[dict]) -> list[dict]:
    """Legend entries: cluster roles in the clustered view, hops in the flat one."""
    if not cluster_payloads:
        return [{"color": _HOP_COLORS[h], "label": _HOP_LABELS[h]} for h in (0, 1, 2)]
    entries = [{"color": _SEED_COLOR, "label": "Seed paper"}]
    seen: set[str] = set()
    for payload in cluster_payloads:
        role = payload["role"]
        if role in seen:
            continue
        seen.add(role)
        entries.append(
            {"color": _ROLE_COLORS.get(role, _DEFAULT_COLOR), "label": payload["role_label"]}
        )
    return entries


def _stats_text(graph: FilteredGraph, edge_payloads: list[dict]) -> str:
    """The one-line summary under the title, naming how the set was chosen.

    Worth being explicit: "top-40 by BM25" and "35 papers an LLM judged
    load-bearing" are very different claims about what the reader is looking at.
    """
    base = f"{len(graph.nodes)} papers shown · {len(edge_payloads)} citation edges · "
    if graph.curation == "llm":
        clusters = f" across {len(graph.clusters)} concept clusters" if graph.clusters else ""
        return base + f"selected as the lineage of the stated idea{clusters}"
    return base + f"top-{graph.top_k} by keyword relevance to the stated idea"


def render_html(graph: FilteredGraph, out_path: str | Path, title: str = "TreeXiv Lineage") -> Path:
    """Write `graph` to `out_path` as an interactive, self-contained HTML file.

    Returns the resolved output path.
    """
    out_path = Path(out_path)
    nodes_by_id = {sn.node.id: sn.node for sn in graph.nodes}
    cluster_payloads, member_positions = _cluster_payloads(graph)
    # Papers inside a cluster sit at their in-cluster position; the seed, and
    # anything the curator left unclustered, falls back to the flat diagonal.
    positions = compute_positions([sn.node for sn in graph.nodes], graph.seed_id)
    positions.update(member_positions)

    intents_by_node = seed_edge_intents(graph.seed_id, graph.edges)
    node_payloads = [
        _node_payload(
            sn,
            sn.node.id == graph.seed_id,
            describe_relationship(sn.node.id, graph.seed_id, nodes_by_id, graph.edges),
            positions[sn.node.id],
            intents_by_node.get(sn.node.id, ""),
        )
        for sn in graph.nodes
    ]
    node_ids = {n["id"] for n in node_payloads}
    # Edge.source cites Edge.target - that's the data/citation direction (see
    # models.Edge, narrative.py). The rendered arrow deliberately points the
    # other way, from the earlier/cited paper to the later paper that cites
    # it, so the arrow reads as "this led to that" rather than "this cites
    # that". That also lines up with the diagonal layout: arrows flow from
    # top-left toward bottom-right, same as the old -> new axis.
    edge_payloads = [
        {"from": e.target, "to": e.source}
        for e in graph.edges
        if e.source in node_ids and e.target in node_ids
    ]
    seed_node = nodes_by_id.get(graph.seed_id)

    html = _TEMPLATE
    html = html.replace("__PAGE_TITLE__", _escape_for_inline_script(json.dumps(title)))
    html = html.replace("__IDEA_TEXT__", _escape_for_inline_script(json.dumps(graph.idea_text)))
    html = html.replace("__SEED_ID__", _escape_for_inline_script(json.dumps(graph.seed_id)))
    html = html.replace(
        "__STATS_TEXT__", _escape_for_inline_script(json.dumps(_stats_text(graph, edge_payloads)))
    )
    html = html.replace(
        "__SEED_LABEL__",
        _escape_for_inline_script(json.dumps(seed_node.title if seed_node else "(seed)")),
    )
    html = html.replace(
        "__NODES_JSON__", _escape_for_inline_script(json.dumps(node_payloads))
    )
    html = html.replace(
        "__EDGES_JSON__", _escape_for_inline_script(json.dumps(edge_payloads))
    )
    html = html.replace(
        "__HOP_LEGEND_JSON__",
        _escape_for_inline_script(json.dumps(_legend(cluster_payloads))),
    )
    html = html.replace(
        "__NARRATIVE_JSON__",
        _escape_for_inline_script(json.dumps(_narrative_payload(graph.narrative, graph.clusters))),
    )
    html = html.replace(
        "__CLUSTERS_JSON__", _escape_for_inline_script(json.dumps(cluster_payloads))
    )
    html = html.replace("__VIS_NETWORK_JS__", _load_asset("vis-network.min.js"))
    html = html.replace("__VIS_NETWORK_CSS__", _load_asset("vis-network.css"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__PAGE_TITLE__</title>
<style>__VIS_NETWORK_CSS__</style>
<style>
  :root {
    --sidebar-w: 360px;
    --bg: #f8fafc;
    --panel-bg: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --accent: #f2a900;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  #app { display: flex; height: 100vh; width: 100vw; overflow: hidden; }

  #sidebar { width: var(--sidebar-w); flex: none; background: var(--panel-bg);
    border-right: 1px solid var(--border); padding: 20px; overflow-y: auto;
    box-shadow: 2px 0 8px rgba(15, 23, 42, 0.04); }
  #sidebar .badge { display: inline-block; font-size: 11px; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; color: #7c4a03;
    background: #fff3d6; border: 1px solid #f2d38a; border-radius: 999px;
    padding: 3px 10px; margin-bottom: 12px; }
  #sidebar h1 { font-size: 18px; line-height: 1.35; margin: 0 0 8px; }
  #sidebar .meta { font-size: 13px; color: var(--muted); margin-bottom: 4px; }
  #sidebar .relationship { margin: 14px 0; padding: 12px 14px; background: #f1f5f9;
    border-left: 3px solid var(--accent); border-radius: 4px; font-size: 13px; line-height: 1.5; }
  #sidebar .section-label { font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--muted); margin: 16px 0 6px; }
  #sidebar .abstract { font-size: 13px; line-height: 1.6; color: #1e293b; white-space: pre-wrap; }
  #sidebar a.doi { font-size: 12px; color: #1d4ed8; text-decoration: none; }
  #sidebar a.doi:hover { text-decoration: underline; }
  #back-to-seed { display: none; font-size: 12px; color: #1d4ed8; background: none;
    border: none; cursor: pointer; padding: 0 0 14px; text-decoration: underline; }
  #sidebar .intents { font-size: 11px; color: #475569; background: #eef2ff;
    border: 1px solid #c7d2fe; border-radius: 999px; padding: 4px 10px;
    display: inline-block; margin-bottom: 10px; }
  #sidebar .why { margin: 14px 0; padding: 12px 14px; background: #fffbeb;
    border-left: 3px solid var(--accent); border-radius: 4px; font-size: 13px;
    line-height: 1.5; }
  #sidebar .why b { display: block; font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.06em; color: #92700a; margin-bottom: 4px; }

  #story .headline { font-size: 16px; line-height: 1.4; font-weight: 600; margin: 0 0 12px; }
  #story .overview p { font-size: 13px; line-height: 1.65; color: #1e293b; margin: 0 0 10px; }
  #story .beat { width: 100%; text-align: left; display: block; background: #f8fafc;
    border: 1px solid var(--border); border-left: 3px solid #94a3b8; border-radius: 4px;
    padding: 10px 12px; margin-bottom: 8px; cursor: pointer; font: inherit; color: inherit; }
  #story .beat:hover { background: #f1f5f9; border-left-color: var(--accent); }
  #story .beat.active { background: #fff7e6; border-left-color: var(--accent); }
  #story .beat .beat-title { font-size: 12px; font-weight: 700; margin-bottom: 3px; }
  #story .beat .beat-text { font-size: 12px; line-height: 1.5; color: #475569; }
  #story .beat .beat-count { font-size: 11px; color: var(--muted); margin-top: 5px; }
  #story .cluster-row { display: flex; align-items: baseline; gap: 6px; font-size: 12px;
    line-height: 1.5; margin: 4px 0; }
  #story .cluster-row .role { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--muted); flex: none; padding-top: 2px; }

  #cluster .toggle { display: block; margin: 10px 0 2px; font: inherit; font-size: 12px;
    color: #1d4ed8; background: none; border: 1px solid var(--border); border-radius: 5px;
    padding: 5px 10px; cursor: pointer; }
  #cluster .toggle:hover { background: #f1f5f9; }
  #cluster .members { margin-top: 4px; }
  #cluster .member { width: 100%; text-align: left; display: block; background: none;
    border: none; border-bottom: 1px solid var(--border); padding: 8px 0; cursor: pointer;
    font: inherit; color: inherit; }
  #cluster .member:hover { background: #f8fafc; }
  #cluster .member .m-title { font-size: 12.5px; line-height: 1.4; }
  #cluster .member .m-meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
  #hint { position: absolute; top: 74px; left: 18px; z-index: 5; font-size: 11px;
    color: var(--muted); background: rgba(255,255,255,0.92); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; }
  #expand-all { margin-left: 8px; font: inherit; font-size: 11px; color: #1d4ed8;
    background: none; border: none; cursor: pointer; text-decoration: underline; padding: 0; }

  #graph-pane { flex: 1; position: relative; min-width: 0; }
  #topbar { position: absolute; top: 0; left: 0; right: 0; z-index: 5; padding: 14px 18px;
    background: linear-gradient(to bottom, rgba(248,250,252,.96), rgba(248,250,252,0)); }
  #topbar h2 { font-size: 14px; margin: 0 0 2px; color: var(--text); }
  #topbar .idea { font-size: 12px; color: var(--muted); font-style: italic; }
  #topbar .stats { font-size: 11px; color: var(--muted); margin-top: 4px; }
  #legend { position: absolute; bottom: 16px; left: 18px; z-index: 5; font-size: 11px;
    color: var(--muted); background: rgba(255,255,255,0.9); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px; }
  #legend .row { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
  #legend .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  #legend .axis-note { margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border);
    max-width: 220px; }
  #network { position: absolute; inset: 0; }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <div id="story" style="display:none;">
      <span class="badge">The story</span>
      <div class="headline" id="story-headline"></div>
      <div class="overview" id="story-overview"></div>
      <div class="section-label" id="story-beats-label" style="display:none;">How it unfolded</div>
      <div id="story-beats"></div>
      <div class="section-label" id="story-clusters-label" style="display:none;">Strands</div>
      <div id="story-clusters"></div>
    </div>
    <div id="cluster" style="display:none;">
      <span class="badge" id="cluster-badge">Strand</span>
      <button id="cluster-back">&larr; Back to the story</button>
      <h1 id="cluster-name"></h1>
      <div class="meta" id="cluster-meta"></div>
      <button id="cluster-toggle" class="toggle"></button>
      <div class="relationship" id="cluster-summary"></div>
      <div class="section-label">Papers in this strand</div>
      <div class="members" id="cluster-members"></div>
    </div>
    <div id="paper">
      <span class="badge" id="panel-badge">Seed paper</span>
      <button id="back-to-seed">&larr; Back to seed paper</button>
      <h1 id="panel-title"></h1>
      <div class="meta" id="panel-authors"></div>
      <div class="meta" id="panel-venue"></div>
      <div class="meta" id="panel-score"></div>
      <a class="doi" id="panel-doi" href="#" target="_blank" rel="noopener"></a>
      <div class="why" id="panel-why" style="display:none;"></div>
      <div class="relationship" id="panel-relationship" style="display:none;"></div>
      <div class="intents" id="panel-intents" style="display:none;"></div>
      <div class="section-label">Abstract</div>
      <div class="abstract" id="panel-abstract"></div>
    </div>
  </div>
  <div id="graph-pane">
    <div id="topbar">
      <h2 id="topbar-title"></h2>
      <div class="idea" id="topbar-idea"></div>
      <div class="stats" id="topbar-stats"></div>
    </div>
    <div id="network"></div>
    <div id="hint" style="display:none;"></div>
    <div id="legend"></div>
  </div>
</div>
<script>__VIS_NETWORK_JS__</script>
<script>
(function () {
  var NODES = __NODES_JSON__;
  var EDGES = __EDGES_JSON__;
  var SEED_ID = __SEED_ID__;
  var SEED_LABEL = __SEED_LABEL__;
  var IDEA_TEXT = __IDEA_TEXT__;
  var STATS_TEXT = __STATS_TEXT__;
  var HOP_LEGEND = __HOP_LEGEND_JSON__;
  var PAGE_TITLE = __PAGE_TITLE__;
  var NARRATIVE = __NARRATIVE_JSON__;
  var CLUSTERS = __CLUSTERS_JSON__;
  var CLUSTERED = CLUSTERS.length > 0;
  var HAS_STORY = !!(NARRATIVE && (NARRATIVE.overview || CLUSTERED));

  document.title = PAGE_TITLE;
  document.getElementById("topbar-title").textContent = "Lineage for: " + SEED_LABEL;
  document.getElementById("topbar-idea").textContent = "Core idea: “" + IDEA_TEXT + "”";
  document.getElementById("topbar-stats").textContent = STATS_TEXT;

  var legendEl = document.getElementById("legend");
  HOP_LEGEND.forEach(function (entry) {
    var row = document.createElement("div");
    row.className = "row";
    var dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = entry.color;
    row.appendChild(dot);
    row.appendChild(document.createTextNode(entry.label));
    legendEl.appendChild(row);
  });
  var axisNote = document.createElement("div");
  axisNote.className = "axis-note";
  axisNote.textContent = "Position: older papers top-left, newer bottom-right, seed centered. " +
    "Arrows point from earlier work to what it led to.";
  legendEl.appendChild(axisNote);

  var nodesById = {};
  NODES.forEach(function (n) { nodesById[n.id] = n; });

  var clustersById = {};
  var clusterOfPaper = {};   // paper id -> cluster id
  CLUSTERS.forEach(function (c) {
    clustersById[c.cluster_id] = c;
    c.member_ids.forEach(function (id) { clusterOfPaper[id] = c.cluster_id; });
  });

  // Which clusters are currently open. Everything else in the graph - the
  // seed, and any paper the curator left unclustered - is always visible.
  var expanded = {};

  function visibleIdFor(paperId) {
    var clusterId = clusterOfPaper[paperId];
    if (!clusterId || expanded[clusterId]) return paperId;
    return "cluster:" + clusterId;
  }

  function paperNodeData(n) {
    return {
      id: n.id, label: n.label, title: n.title, x: n.x, y: n.y,
      color: n.color, size: n.size, borderWidth: n.borderWidth, font: n.font,
      shape: "dot"
    };
  }

  function clusterNodeData(c) {
    return {
      id: c.id, label: c.label, title: c.name + " - " + c.count + " papers",
      x: c.x, y: c.y, color: c.color, size: c.size, font: c.font,
      borderWidth: 2, shape: "dot"
    };
  }

  function buildNodes() {
    var out = [];
    NODES.forEach(function (n) {
      if (visibleIdFor(n.id) === n.id) out.push(paperNodeData(n));
    });
    CLUSTERS.forEach(function (c) {
      if (!expanded[c.cluster_id]) out.push(clusterNodeData(c));
    });
    return out;
  }

  function buildEdges() {
    // Every citation is redrawn between whatever is visible at each end. Two
    // papers inside the same collapsed cluster produce a self-loop, which is
    // dropped; several citations between the same pair collapse into one
    // thicker edge rather than a bundle of parallel lines.
    var byPair = {};
    EDGES.forEach(function (e) {
      var from = visibleIdFor(e.from);
      var to = visibleIdFor(e.to);
      if (from === to) return;
      var key = from + "|" + to;
      if (!byPair[key]) byPair[key] = { from: from, to: to, weight: 0 };
      byPair[key].weight += 1;
    });
    return Object.keys(byPair).map(function (key) {
      var e = byPair[key];
      return {
        id: key,
        from: e.from,
        to: e.to,
        width: Math.min(1.5 + (e.weight - 1) * 0.9, 7),
        label: e.weight > 1 ? String(e.weight) : undefined,
        font: { size: 10, color: "#64748b", strokeWidth: 3, strokeColor: "#f8fafc" }
      };
    });
  }

  var visNodes = new vis.DataSet(buildNodes());
  var visEdges = new vis.DataSet(buildEdges());

  var container = document.getElementById("network");
  var data = { nodes: visNodes, edges: visEdges };
  var options = {
    physics: false,
    interaction: { hover: true, tooltipDelay: 150, dragView: true, zoomView: true },
    edges: {
      arrows: { to: { enabled: true, scaleFactor: 1.6 } },
      color: { color: "#94a3b8", highlight: "#334155", hover: "#475569" },
      width: 1.5,
      smooth: { type: "continuous" }
    },
    nodes: { shape: "dot" }
  };
  var network = new vis.Network(container, data, options);
  // Physics is off and every node has a static (x, y), so there is no
  // stabilization to wait for - fit() can run immediately. It must NOT be
  // deferred to a one-time "afterDrawing" listener registered after this
  // point: with physics disabled, vis-network's only draw pass happens
  // synchronously inside the constructor above, so a listener attached
  // afterward would miss it and never fire, leaving the view at its
  // default zoom (centered near the origin, most nodes off-screen).
  network.fit({ animation: false });

  function rebuild(focusIds) {
    // Positions are fixed up front, so a rebuild only changes which nodes are
    // drawn - nothing slides around, and the diagonal keeps its meaning.
    visNodes.clear();
    visEdges.clear();
    visNodes.add(buildNodes());
    visEdges.add(buildEdges());
    if (focusIds && focusIds.length) {
      network.fit({ nodes: focusIds, animation: { duration: 400 } });
    }
    updateHint();
  }

  function toggleCluster(clusterId, focus) {
    var cluster = clustersById[clusterId];
    if (!cluster) return;
    if (expanded[clusterId]) {
      delete expanded[clusterId];
      rebuild(focus === false ? null : ["cluster:" + clusterId]);
    } else {
      expanded[clusterId] = true;
      rebuild(focus === false ? null : cluster.member_ids);
    }
  }

  function openClustersFor(nodeIds) {
    // A beat or a search can name papers inside collapsed clusters; open
    // whatever is needed so the highlight has something to land on.
    var changed = false;
    nodeIds.forEach(function (id) {
      var clusterId = clusterOfPaper[id];
      if (clusterId && !expanded[clusterId]) {
        expanded[clusterId] = true;
        changed = true;
      }
    });
    if (changed) rebuild(null);
  }

  var hintEl = document.getElementById("hint");

  function updateHint() {
    if (!CLUSTERED) return;
    var openCount = Object.keys(expanded).length;
    hintEl.style.display = "block";
    hintEl.textContent = openCount
      ? openCount + " of " + CLUSTERS.length + " strands open. Close one from its panel."
      : "Each circle is a strand of the story. Click one to open it.";
    var button = document.createElement("button");
    button.id = "expand-all";
    button.type = "button";
    button.textContent = openCount === CLUSTERS.length ? "Collapse all" : "Expand all";
    button.addEventListener("click", function () {
      if (Object.keys(expanded).length === CLUSTERS.length) {
        expanded = {};
      } else {
        CLUSTERS.forEach(function (c) { expanded[c.cluster_id] = true; });
      }
      rebuild(null);
      network.fit({ animation: { duration: 400 } });
    });
    hintEl.appendChild(button);
  }

  function fmtAuthors(authors) {
    if (!authors || !authors.length) return "Authors unknown";
    if (authors.length <= 4) return authors.join(", ");
    return authors.slice(0, 4).join(", ") + ", et al.";
  }

  var storyEl = document.getElementById("story");
  var paperEl = document.getElementById("paper");
  var clusterEl = document.getElementById("cluster");

  function showPanel(which) {
    storyEl.style.display = which === "story" ? "block" : "none";
    paperEl.style.display = which === "paper" ? "block" : "none";
    clusterEl.style.display = which === "cluster" ? "block" : "none";
  }

  function showCluster(clusterId) {
    var cluster = clustersById[clusterId];
    if (!cluster) return;
    showPanel("cluster");
    document.getElementById("cluster-badge").textContent = cluster.role_label;
    document.getElementById("cluster-name").textContent = cluster.name;
    document.getElementById("cluster-meta").textContent =
      cluster.count + (cluster.count === 1 ? " paper" : " papers") +
      (cluster.span ? " \u00b7 " + cluster.span : "");
    var summaryEl = document.getElementById("cluster-summary");
    summaryEl.textContent = cluster.summary || "";
    summaryEl.style.display = cluster.summary ? "block" : "none";

    var toggle = document.getElementById("cluster-toggle");
    toggle.textContent = expanded[clusterId]
      ? "Collapse this strand" : "Open this strand in the graph";
    toggle.onclick = function () {
      toggleCluster(clusterId);
      showCluster(clusterId);
    };

    var membersEl = document.getElementById("cluster-members");
    membersEl.innerHTML = "";
    cluster.member_ids
      .map(function (id) { return nodesById[id]; })
      .filter(Boolean)
      .sort(function (a, b) {
        return (a.publication_year || 0) - (b.publication_year || 0);
      })
      .forEach(function (n) {
        var btn = document.createElement("button");
        btn.className = "member";
        btn.type = "button";
        var title = document.createElement("div");
        title.className = "m-title";
        title.textContent = n.title;
        var meta = document.createElement("div");
        meta.className = "m-meta";
        meta.textContent = [n.publication_year, n.cited_by_count + " citations"]
          .filter(Boolean).join(" \u00b7 ");
        btn.appendChild(title);
        btn.appendChild(meta);
        btn.addEventListener("click", function () {
          // The paper may be inside a still-closed strand; open it so there is
          // something on the canvas to select.
          openClustersFor([n.id]);
          network.selectNodes([n.id]);
          network.focus(n.id, { scale: 1, animation: { duration: 400 } });
          showNode(n.id);
        });
        membersEl.appendChild(btn);
      });
  }

  function buildStory() {
    if (!HAS_STORY) return;
    document.getElementById("story-headline").textContent = NARRATIVE.headline || "";
    var overviewEl = document.getElementById("story-overview");
    (NARRATIVE.overview || "").split(/\n\s*\n/).forEach(function (para) {
      if (!para.trim()) return;
      var p = document.createElement("p");
      p.textContent = para.trim();
      overviewEl.appendChild(p);
    });

    var beats = NARRATIVE.beats || [];
    var beatsEl = document.getElementById("story-beats");
    if (beats.length) {
      document.getElementById("story-beats-label").style.display = "block";
    }
    beats.forEach(function (beat) {
      var btn = document.createElement("button");
      btn.className = "beat";
      btn.type = "button";
      if (beat.title) {
        var t = document.createElement("div");
        t.className = "beat-title";
        t.textContent = beat.title;
        btn.appendChild(t);
      }
      var body = document.createElement("div");
      body.className = "beat-text";
      body.textContent = beat.text;
      btn.appendChild(body);
      var ids = (beat.node_ids || []).filter(function (id) { return !!nodesById[id]; });
      if (ids.length) {
        var count = document.createElement("div");
        count.className = "beat-count";
        count.textContent = "Highlights " + ids.length + (ids.length === 1 ? " paper" : " papers");
        btn.appendChild(count);
      }
      btn.addEventListener("click", function () {
        Array.prototype.forEach.call(beatsEl.children, function (el) {
          el.classList.remove("active");
        });
        btn.classList.add("active");
        if (!ids.length) return;
        // Open whichever strands hold this beat's papers, so the highlight
        // lands on the papers themselves rather than on a closed circle.
        openClustersFor(ids);
        network.selectNodes(ids);
        network.fit({ nodes: ids, animation: { duration: 400 } });
      });
      beatsEl.appendChild(btn);
    });

    if (CLUSTERS.length) {
      document.getElementById("story-clusters-label").style.display = "block";
      var clustersEl = document.getElementById("story-clusters");
      CLUSTERS.forEach(function (cluster) {
        var row = document.createElement("div");
        row.className = "cluster-row";
        var role = document.createElement("span");
        role.className = "role";
        role.textContent = cluster.role;
        var body = document.createElement("span");
        var name = document.createElement("b");
        name.textContent = cluster.name + ". ";
        body.appendChild(name);
        body.appendChild(document.createTextNode(cluster.summary || ""));
        row.appendChild(role);
        row.appendChild(body);
        row.style.cursor = "pointer";
        row.addEventListener("click", function () {
          network.selectNodes(
            expanded[cluster.cluster_id] ? cluster.member_ids : [cluster.id]
          );
          showCluster(cluster.cluster_id);
        });
        clustersEl.appendChild(row);
      });
    }
  }

  function showStory() {
    // The story is the resting state when there is one; without it the seed
    // paper keeps that role, exactly as before narratives existed.
    if (!HAS_STORY) { showNode(SEED_ID); return; }
    showPanel("story");
  }

  function showNode(id) {
    var n = nodesById[id];
    if (!n) return;
    showPanel("paper");
    var isSeed = id === SEED_ID;
    document.getElementById("panel-badge").textContent = isSeed
      ? "Seed paper" : "Hop " + n.hop + (n.hop === 1 ? " — direct neighbor" : " — indirect");
    document.getElementById("panel-title").textContent = n.title;
    document.getElementById("panel-authors").textContent = fmtAuthors(n.authors);
    var venueYear = [n.venue, n.publication_year].filter(Boolean).join(" · ");
    document.getElementById("panel-venue").textContent =
      venueYear + (venueYear ? " · " : "") + n.cited_by_count + " citations";
    var scoreEl = document.getElementById("panel-score");
    scoreEl.textContent = isSeed
      ? "Always included as the seed paper" : "Relevance score: " + n.score.toFixed(2);
    var doiEl = document.getElementById("panel-doi");
    if (n.doi) {
      doiEl.href = n.doi; doiEl.textContent = n.doi; doiEl.style.display = "inline";
    } else {
      doiEl.style.display = "none";
    }
    var whyEl = document.getElementById("panel-why");
    if (n.why) {
      whyEl.innerHTML = "";
      var label = document.createElement("b");
      label.textContent = "Why this paper is here";
      whyEl.appendChild(label);
      whyEl.appendChild(document.createTextNode(n.why));
      whyEl.style.display = "block";
    } else {
      whyEl.style.display = "none";
    }
    var relEl = document.getElementById("panel-relationship");
    if (isSeed) {
      relEl.style.display = "none";
    } else {
      relEl.textContent = n.relationship;
      relEl.style.display = "block";
    }
    var intentEl = document.getElementById("panel-intents");
    if (n.intents) {
      intentEl.textContent = "Citation intent (Semantic Scholar): " + n.intents;
      intentEl.style.display = "block";
    } else {
      intentEl.style.display = "none";
    }
    document.getElementById("panel-abstract").textContent = n.abstract;
    var back = document.getElementById("back-to-seed");
    back.textContent = HAS_STORY ? "← Back to the story" : "← Back to seed paper";
    back.style.display = (HAS_STORY || !isSeed) ? "block" : "none";
  }

  document.getElementById("back-to-seed").addEventListener("click", function () {
    network.unselectAll();
    showStory();
  });
  document.getElementById("cluster-back").addEventListener("click", function () {
    network.unselectAll();
    showStory();
  });

  network.on("click", function (params) {
    if (!params.nodes || !params.nodes.length) {
      network.unselectAll();
      showStory();
      return;
    }
    var id = params.nodes[0];
    if (id.indexOf("cluster:") === 0) {
      var clusterId = id.slice("cluster:".length);
      toggleCluster(clusterId);
      showCluster(clusterId);
      return;
    }
    showNode(id);
  });

  buildStory();
  updateHint();
  showStory();
})();
</script>
</body>
</html>
"""
