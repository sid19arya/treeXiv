"""Step 5: render a `FilteredGraph` as a single self-contained, interactive HTML file.

No frontend framework, no build step, no CDN dependency (vis-network is
vendored in `treexiv/assets/vis-network/` and inlined) - just one HTML file
per run, per the PRD's non-goals.

Layout: `layout.compute_positions` places nodes on a diagonal by publication
year (older top-left, newer bottom-right, seed centered) instead of letting
vis-network's physics engine decide - see that module for why. Each node
also carries a `narrative.describe_relationship` string, shown in the left
sidebar when that node is selected.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from treexiv.layout import Position, compute_positions
from treexiv.models import FilteredGraph, Node
from treexiv.narrative import describe_relationship

_HOP_COLORS = {0: "#f2a900", 1: "#2a9d8f", 2: "#8ecae6"}
_DEFAULT_COLOR = "#cbd5e1"
_HOP_LABELS = {0: "Seed", 1: "1 hop away", 2: "2 hops away"}


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
    node: Node, score: float, is_seed: bool, relationship: str, position: Position
) -> dict:
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
        "x": position.x,
        "y": position.y,
        "color": {
            "background": "#f2a900" if is_seed else _HOP_COLORS.get(node.hop, _DEFAULT_COLOR),
            "border": "#7c4a03" if is_seed else "#334155",
        },
        "size": 14 + min(node.cited_by_count, 5000) ** 0.5,
        "borderWidth": 3 if is_seed else 1.5,
        "font": {"size": 13 if is_seed else 11},
    }


def render_html(graph: FilteredGraph, out_path: str | Path, title: str = "TreeXiv Lineage") -> Path:
    """Write `graph` to `out_path` as an interactive, self-contained HTML file.

    Returns the resolved output path.
    """
    out_path = Path(out_path)
    nodes_by_id = {sn.node.id: sn.node for sn in graph.nodes}
    positions = compute_positions([sn.node for sn in graph.nodes], graph.seed_id)

    node_payloads = [
        _node_payload(
            sn.node,
            sn.score,
            sn.node.id == graph.seed_id,
            describe_relationship(sn.node.id, graph.seed_id, nodes_by_id, graph.edges),
            positions[sn.node.id],
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
        "__STATS_TEXT__",
        _escape_for_inline_script(
            json.dumps(
                f"{len(node_payloads)} papers shown · {len(edge_payloads)} citation edges "
                f"· top-{graph.top_k} by relevance to the stated idea"
            )
        ),
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
        _escape_for_inline_script(
            json.dumps([{"color": _HOP_COLORS[h], "label": _HOP_LABELS[h]} for h in (0, 1, 2)])
        ),
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
    <span class="badge" id="panel-badge">Seed paper</span>
    <button id="back-to-seed">&larr; Back to seed paper</button>
    <h1 id="panel-title"></h1>
    <div class="meta" id="panel-authors"></div>
    <div class="meta" id="panel-venue"></div>
    <div class="meta" id="panel-score"></div>
    <a class="doi" id="panel-doi" href="#" target="_blank" rel="noopener"></a>
    <div class="relationship" id="panel-relationship" style="display:none;"></div>
    <div class="section-label">Abstract</div>
    <div class="abstract" id="panel-abstract"></div>
  </div>
  <div id="graph-pane">
    <div id="topbar">
      <h2 id="topbar-title"></h2>
      <div class="idea" id="topbar-idea"></div>
      <div class="stats" id="topbar-stats"></div>
    </div>
    <div id="network"></div>
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

  var visNodes = new vis.DataSet(NODES.map(function (n) {
    return {
      id: n.id, label: n.label, title: n.title, x: n.x, y: n.y,
      color: n.color, size: n.size, borderWidth: n.borderWidth, font: n.font,
      shape: "dot"
    };
  }));
  var visEdges = new vis.DataSet(EDGES);

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

  function fmtAuthors(authors) {
    if (!authors || !authors.length) return "Authors unknown";
    if (authors.length <= 4) return authors.join(", ");
    return authors.slice(0, 4).join(", ") + ", et al.";
  }

  function showNode(id) {
    var n = nodesById[id];
    if (!n) return;
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
    var relEl = document.getElementById("panel-relationship");
    if (isSeed) {
      relEl.style.display = "none";
    } else {
      relEl.textContent = n.relationship;
      relEl.style.display = "block";
    }
    document.getElementById("panel-abstract").textContent = n.abstract;
    document.getElementById("back-to-seed").style.display = isSeed ? "none" : "block";
  }

  document.getElementById("back-to-seed").addEventListener("click", function () {
    network.unselectAll();
    showNode(SEED_ID);
  });

  network.on("click", function (params) {
    if (params.nodes && params.nodes.length) {
      showNode(params.nodes[0]);
    } else {
      network.unselectAll();
      showNode(SEED_ID);
    }
  });

  showNode(SEED_ID);
})();
</script>
</body>
</html>
"""
