"""Human-readable "how does this connect to the seed?" text for the sidebar.

Edge semantics throughout (see models.Edge): `source` cites `target`. So
`Edge(seed, X)` means the seed cites X (X is something the seed built on -
typically older), and `Edge(X, seed)` means X cites the seed (X grew out of
the seed - typically newer).

Exhaustive mode adds `kind="semantic"` edges: relations Jev inferred with no
citation behind them (same orientation, `source` is the later paper). Only
citation edges are ever phrased as "cites". A semantic edge is described as
an inference with its confidence, and multi-hop paths are built from
citations alone, so every "cites" in the sidebar is still a fact from the data.
"""

from __future__ import annotations

from treexiv.models import SEARCH_HOP, Edge, Node

SEED_DESCRIPTION = "This is the seed paper the whole tree is built from."

# Semantic Scholar's citation-intent vocabulary, in words a reader recognizes.
_INTENT_WORDS = {
    "background": "background",
    "methodology": "its method",
    "result": "its result",
}


# Third-person phrasing of a Jev relation, read as "<later> ___ <earlier>".
_RELATION_PHRASES = {
    "extends": "builds on",
    "applies": "applies",
    "alternative": "offers an alternative to",
}


def _find_edge(edges: list[Edge], a: str, b: str) -> Edge | None:
    """The edge directly connecting `a` and `b`, in either direction."""
    for edge in edges:
        if {edge.source, edge.target} == {a, b}:
            return edge
    return None


def describe_intents(edge: Edge) -> str:
    """Semantic Scholar's read on what a citation was *for*, as a clause.

    Empty when the edge carries no intent data — most edges don't, since only
    the seed's own references and citations are looked up (see
    `sources/enrich.py`). Absence means "not checked", not "incidental".
    """
    parts = []
    if edge.intents:
        readable = [_INTENT_WORDS.get(i, i) for i in edge.intents]
        parts.append("cited for " + _join(readable))
    if edge.is_influential:
        parts.append("flagged as an influential citation")
    return "; ".join(parts)


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _direct_relationship(edge: Edge, node_id: str, seed_id: str) -> str:
    if edge.source == seed_id and edge.target == node_id:
        base = "The seed paper cites this one directly - it's part of what the seed built on."
    else:
        base = "This paper cites the seed directly - it's part of what grew out of the seed."
    detail = describe_intents(edge)
    text = f"{base} ({detail.capitalize()}.)" if detail else base
    phrase = _RELATION_PHRASES.get(edge.relation)
    if phrase and edge.confidence is not None:
        later = "the seed" if edge.source == seed_id else "this paper"
        earlier = "this one" if edge.source == seed_id else "the seed"
        text += f" Jev reads it as: {later} {phrase} {earlier} ({edge.confidence:.0%})."
    return text


def _inferred_relationship(edge: Edge, node_id: str) -> str:
    """Phrase a Jev-inferred edge to the seed. Never as a citation."""
    phrase = _RELATION_PHRASES.get(edge.relation, "is related to")
    confidence = f" ({edge.confidence:.0%} confidence)" if edge.confidence is not None else ""
    if edge.source == node_id:
        claim = f"this paper {phrase} the seed paper"
    else:
        claim = f"the seed paper {phrase} this one"
    return (
        f"No citation links this paper to the seed. Jev infers that {claim}{confidence} "
        "- an inferred relation, not a reference."
    )


def _edge_phrase(edge: Edge, id_a: str, title_a: str, id_b: str, title_b: str) -> str:
    """Phrase one edge as a citation sentence, anchored on `id_a` first."""
    if edge.source == id_a:
        return f'"{title_a}" cites "{title_b}"'
    return f'"{title_b}" cites "{title_a}"'


def describe_relationship(
    node_id: str, seed_id: str, nodes_by_id: dict[str, Node], edges: list[Edge]
) -> str:
    """One or two sentences describing how `node_id` relates to the seed paper."""
    if node_id == seed_id:
        return SEED_DESCRIPTION

    citations = [e for e in edges if e.is_citation]
    direct = _find_edge(citations, node_id, seed_id)
    if direct is not None:
        return _direct_relationship(direct, node_id, seed_id)
    inferred = _find_edge([e for e in edges if not e.is_citation], node_id, seed_id)
    if inferred is not None:
        return _inferred_relationship(inferred, node_id)

    node = nodes_by_id[node_id]
    for edge in citations:
        if edge.source == node_id:
            bridge_id = edge.target
        elif edge.target == node_id:
            bridge_id = edge.source
        else:
            continue
        if bridge_id == seed_id or bridge_id not in nodes_by_id:
            continue
        bridge_edge = _find_edge(citations, bridge_id, seed_id)
        if bridge_edge is None:
            continue
        bridge = nodes_by_id[bridge_id]
        leg1 = _edge_phrase(edge, node_id, node.title, bridge_id, bridge.title)
        leg2 = _edge_phrase(bridge_edge, bridge_id, bridge.title, seed_id, "the seed paper")
        return f'Two hops from the seed via "{bridge.title}": {leg1}; {leg2}.'

    if node.hop == SEARCH_HOP:
        return (
            "Found by searching for the stated idea, not by following citations - no "
            "citation path to the seed is known. It is here because Jev judged it relevant."
        )
    return (
        f"Reached {node.hop} hop(s) from the seed during traversal, but the connecting "
        "paper(s) didn't make the relevance cutoff, so the direct path isn't shown."
    )
