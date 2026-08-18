"""Step 4: filter the expanded graph down to nodes relevant to the stated core idea.

Per the PRD: score every node by BM25 against the user's core-idea text, keep
the top-K (the seed is always retained regardless of score), and drop any
edge whose endpoint didn't survive.
"""

from __future__ import annotations

from treexiv.corpus import BM25Corpus
from treexiv.models import Edge, ExpansionResult, FilteredGraph, ScoredNode


def filter_by_idea(expansion: ExpansionResult, idea_text: str, top_k: int) -> FilteredGraph:
    """Reduce an `ExpansionResult` to its top-K most idea-relevant nodes."""
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
    )
