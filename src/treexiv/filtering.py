"""Step 4: reduce the expanded graph to the papers worth putting in front of a reader.

Two paths, selected by `Settings.curation_mode` (`--curation` on the CLI):

- **llm** (`curate.py`) — a model picks which papers are load-bearing for the
  stated idea and groups them into concept clusters. This is the path that
  makes the output readable, and the default when an OpenRouter key exists.
- **bm25** — the original deterministic filter: score every node by BM25
  against the core-idea text, keep the top-K (the seed always survives), drop
  edges with a filtered-out endpoint. No API key, no network, no variance.

`build_graph` is the entry point; it runs the LLM path when asked and falls
back to BM25 when curation is unavailable or fails, so a missing key or a bad
model reply degrades the output rather than failing the run.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from treexiv.config import Settings
from treexiv.corpus import BM25Corpus
from treexiv.curate import curate_graph
from treexiv.exceptions import CurationError
from treexiv.models import Edge, ExpansionResult, FilteredGraph, ScoredNode

WarningSink = Callable[[str], None]


def filter_by_idea(expansion: ExpansionResult, idea_text: str, top_k: int) -> FilteredGraph:
    """Reduce an `ExpansionResult` to its top-K most idea-relevant nodes (BM25 path)."""
    node_list = list(expansion.nodes.values())
    corpus = BM25Corpus(node_list)
    scores = corpus.scores(idea_text)

    ranked_ids = sorted(
        (n.id for n in node_list if n.id != expansion.seed_id),
        key=lambda nid: scores.get(nid, 0.0),
        reverse=True,
    )
    selected_ids = {expansion.seed_id, *ranked_ids[:top_k]}

    selected_nodes = [
        ScoredNode(node=n, score=scores.get(n.id, 0.0))
        for n in node_list
        if n.id in selected_ids
    ]
    selected_nodes.sort(key=lambda sn: sn.score, reverse=True)

    selected_edges = [
        Edge(source=e.source, target=e.target)
        for e in expansion.edges
        if e.source in selected_ids and e.target in selected_ids
    ]

    return FilteredGraph(
        seed_id=expansion.seed_id,
        idea_text=idea_text,
        top_k=top_k,
        nodes=selected_nodes,
        edges=selected_edges,
        curation="bm25",
    )


def build_graph(
    expansion: ExpansionResult,
    idea_text: str,
    settings: Settings,
    *,
    http_client: httpx.Client | None = None,
    on_warning: WarningSink | None = None,
) -> FilteredGraph:
    """Filter `expansion` down to a renderable graph, per `settings.curation_mode`.

    With mode "llm" a curation failure propagates — the caller asked for
    curation specifically. With "auto" it is reported through `on_warning` and
    the BM25 result is returned instead.
    """
    mode = settings.curation_mode
    if mode == "bm25":
        return filter_by_idea(expansion, idea_text, top_k=settings.bm25_top_k)

    if mode == "llm":
        return curate_graph(expansion, idea_text, settings, http_client=http_client)

    if not settings.openrouter_api_key:
        if on_warning:
            on_warning(
                "OPENROUTER_API_KEY unset — falling back to the BM25 top-K filter. "
                "Set it (see .env.example) for LLM-curated, clustered output."
            )
        return filter_by_idea(expansion, idea_text, top_k=settings.bm25_top_k)

    try:
        return curate_graph(expansion, idea_text, settings, http_client=http_client)
    except CurationError as exc:
        if on_warning:
            on_warning(f"LLM curation failed ({exc}) — falling back to the BM25 top-K filter.")
        return filter_by_idea(expansion, idea_text, top_k=settings.bm25_top_k)
