"""Step 4: reduce the expanded graph to the papers worth putting in front of a reader.

Two paths, selected by `Settings.curation_mode` (`--curation` on the CLI):

- **llm** (`curate.py`, then `synthesis.py`) — a model picks which papers are
  load-bearing for the stated idea, groups them into concept clusters, and
  writes the story they add up to. This is the path that makes the output
  readable, and the default when an OpenRouter key exists.
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
from treexiv.exceptions import CurationError, SynthesisError
from treexiv.models import Edge, ExpansionResult, FilteredGraph, ScoredNode
from treexiv.synthesis import synthesize_lineage

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
        graph = curate_graph(expansion, idea_text, settings, http_client=http_client)
        return _with_narrative(graph, settings, http_client, on_warning)

    if not settings.openrouter_api_key:
        if on_warning:
            on_warning(
                "OPENROUTER_API_KEY unset — falling back to the BM25 top-K filter. "
                "Set it (see .env.example) for LLM-curated, clustered output."
            )
        return filter_by_idea(expansion, idea_text, top_k=settings.bm25_top_k)

    try:
        graph = curate_graph(expansion, idea_text, settings, http_client=http_client)
    except CurationError as exc:
        if on_warning:
            on_warning(f"LLM curation failed ({exc}) — falling back to the BM25 top-K filter.")
        return filter_by_idea(expansion, idea_text, top_k=settings.bm25_top_k)
    return _with_narrative(graph, settings, http_client, on_warning)


def _with_narrative(
    graph: FilteredGraph,
    settings: Settings,
    http_client: httpx.Client | None,
    on_warning: WarningSink | None,
) -> FilteredGraph:
    """Attach the written lineage story, if it's wanted and the call succeeds.

    Always non-fatal: a curated graph without prose is still the useful part
    of the output, so a synthesis failure is reported and dropped.
    """
    if not settings.narrative:
        return graph
    try:
        graph.narrative = synthesize_lineage(graph, settings, http_client=http_client)
    except SynthesisError as exc:
        if on_warning:
            on_warning(f"Lineage synthesis failed ({exc}) — rendering the graph without it.")
    return graph
