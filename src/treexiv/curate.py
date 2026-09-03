"""Step 4 (LLM path): pick the papers actually worth showing, and group them.

The BM25 filter this replaces answers "which abstracts share words with the
user's idea?" — which is not the same question as "which papers form this
idea's lineage?", and at K=40 it produced a graph too dense to read and too
lexical to trust. This module asks a model the real question instead:

1. BM25 still runs first, purely as a **prefilter** (default: top 120), so the
   model reads a bounded number of abstracts rather than the whole corpus.
2. One chat call ranks those candidates as lineage, not as text: keep the
   papers that are load-bearing for the stated idea, drop the rest, group the
   survivors into a handful of named concept clusters, and say in one line why
   each survivor earned its place.

Candidates are presented to the model as small integer indices, not OpenAlex
work IDs. A cheap model copies `[42]` back reliably; long opaque strings like
`W2741809807` it will sometimes mangle, and a mangled ID is a silently dropped
paper. Indices are mapped back to IDs here, and anything unrecognizable is
discarded rather than guessed at.

Failure is expected to be survivable: every parse problem raises
`CurationError`, and `filtering.build_graph` falls back to the deterministic
BM25 path when it does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from treexiv.config import Settings
from treexiv.corpus import BM25Corpus
from treexiv.exceptions import CurationError
from treexiv.llm import chat_json, clean_str, coerce_int
from treexiv.models import Cluster, Edge, ExpansionResult, FilteredGraph, Node, ScoredNode

# How much of each abstract the model sees. Enough to tell what a paper did,
# short enough that ~120 of them stay a cheap prompt.
ABSTRACT_CHARS = 320
_VALID_ROLES = {"ancestor", "descendant", "contemporary"}
_OTHER_CLUSTER_ID = "other"

_SYSTEM_PROMPT = """\
You build citation-lineage maps: given a seed paper and the specific idea a
reader cares about, you decide which surrounding papers actually tell the story
of where that idea came from and what it grew into.

You will get the seed paper and a numbered list of candidate papers found by
traversing citations two hops out from the seed. Each candidate is labelled
"ancestor" (the seed's line of work cites it), "descendant" (it cites the seed's
line of work), or "contemporary". Some also carry a "citation intent" from
Semantic Scholar, saying what the citation was *for*: "methodology" or "result"
means the work was actually used or built on, "background" is often just
context, and "influential" is a strong signal to keep. Papers with no intent
shown were simply not checked — absence is not evidence against them.

Your job is to be RUTHLESS. Most candidates are incidental citations — shared
benchmarks, boilerplate references, tangentially related work — and including
them makes the map useless. Keep a paper only if a knowledgeable reader tracing
this specific idea would be worse off without it: the work the idea is built on,
the papers that first stated or named it, the ones that materially extended,
scaled, challenged, or superseded it.

Then group the papers you keep into 3-7 concept clusters. A cluster is a
distinct strand of the story ("early sparse-attention approximations",
"scaling-law follow-ups"), not a date range and not a topic label copied off a
title.

Respond with ONLY a JSON object, no prose or code fences:
{
  "clusters": [
    {"id": 1, "name": "short strand name (2-5 words)",
     "role": "ancestor" | "descendant" | "contemporary",
     "summary": "1-2 sentences: what this strand contributes to the idea's story"}
  ],
  "keep": [
    {"i": <candidate number>, "cluster": <cluster id>, "importance": 1-5,
     "why": "one line: this paper's specific role in the lineage of the stated idea"}
  ],
  "dropped_summary": "1-2 sentences on what kinds of papers you cut and why"
}

Rules:
- "i" must be a number from the candidate list. Never invent one.
- Keep between %(min_keep)d and %(max_keep)d papers. Fewer, better-chosen papers
  beat more.
- "importance" is 5 for papers central to this idea's story, 1 for supporting
  detail.
- Do not include the seed paper itself in "keep" — it is always in the map.
- Every kept paper needs a cluster that exists in "clusters"."""


@dataclass(slots=True)
class _Candidate:
    """One paper offered to the model, with the index it is shown under."""

    index: int
    node: Node
    score: float
    direction: str
    intent: str = ""


def classify_directions(seed_id: str, edges: list[Edge], node_ids: set[str]) -> dict[str, str]:
    """Label every node "ancestor", "descendant", "mixed", or "unrelated".

    `Edge.source` cites `Edge.target`, so walking source→target away from the
    seed reaches work the seed's line was built on (ancestors), and walking
    target→source reaches work built on top of it (descendants). A node
    reachable both ways is "mixed"; one reachable neither way (its connecting
    papers exist in the expansion but not on a directed path) is "unrelated".
    """
    cites: dict[str, list[str]] = {}
    cited_by: dict[str, list[str]] = {}
    for edge in edges:
        cites.setdefault(edge.source, []).append(edge.target)
        cited_by.setdefault(edge.target, []).append(edge.source)

    def reachable(adjacency: dict[str, list[str]]) -> set[str]:
        seen: set[str] = set()
        frontier = [seed_id]
        while frontier:
            current = frontier.pop()
            for neighbour in adjacency.get(current, ()):
                if neighbour not in seen and neighbour != seed_id:
                    seen.add(neighbour)
                    frontier.append(neighbour)
        return seen

    ancestors = reachable(cites)
    descendants = reachable(cited_by)

    labels: dict[str, str] = {}
    for node_id in node_ids:
        if node_id == seed_id:
            continue
        is_ancestor, is_descendant = node_id in ancestors, node_id in descendants
        if is_ancestor and is_descendant:
            labels[node_id] = "mixed"
        elif is_ancestor:
            labels[node_id] = "ancestor"
        elif is_descendant:
            labels[node_id] = "descendant"
        else:
            labels[node_id] = "unrelated"
    return labels


def prefilter(expansion: ExpansionResult, idea_text: str, limit: int) -> list[_Candidate]:
    """BM25-rank the expansion and take the top `limit` non-seed nodes."""
    node_list = list(expansion.nodes.values())
    scores = BM25Corpus(node_list).scores(idea_text)
    directions = classify_directions(
        expansion.seed_id, expansion.edges, set(expansion.nodes.keys())
    )
    intents = seed_edge_intents(expansion.seed_id, expansion.edges)
    ranked = sorted(
        (n for n in node_list if n.id != expansion.seed_id),
        key=lambda n: (scores.get(n.id, 0.0), n.cited_by_count),
        reverse=True,
    )
    return [
        _Candidate(
            index=i,
            node=node,
            score=scores.get(node.id, 0.0),
            direction=directions.get(node.id, "unrelated"),
            intent=intents.get(node.id, ""),
        )
        for i, node in enumerate(ranked[:limit], start=1)
    ]


def seed_edge_intents(seed_id: str, edges: list[Edge]) -> dict[str, str]:
    """Per-node summary of what Semantic Scholar said about its citation to or
    from the seed, e.g. "methodology, influential".

    Only edges touching the seed carry intents (see `sources/enrich.py`), which
    is exactly where they most affect the keep/drop call: a paper the seed
    cites *for its method* is lineage, one cited *for background* often isn't.
    """
    out: dict[str, str] = {}
    for edge in edges:
        if seed_id not in (edge.source, edge.target):
            continue
        other = edge.target if edge.source == seed_id else edge.source
        labels = list(edge.intents)
        if edge.is_influential:
            labels.append("influential")
        if labels:
            out[other] = ", ".join(labels)
    return out


def _describe_paper(node: Node) -> str:
    authors = node.authors[0].split()[-1] if node.authors else "unknown"
    if len(node.authors) > 1:
        authors += " et al."
    abstract = node.abstract.strip().replace("\n", " ")
    if len(abstract) > ABSTRACT_CHARS:
        abstract = abstract[:ABSTRACT_CHARS].rsplit(" ", 1)[0] + "…"
    year = node.publication_year or "n.d."
    head = f"{node.title} ({authors}, {year}, {node.cited_by_count} citations)"
    return f"{head}\n{abstract}" if abstract else head


def build_prompt(
    seed: Node, idea_text: str, candidates: list[_Candidate], max_keep: int
) -> list[dict[str, str]]:
    """The system + user messages for one curation call."""
    lines = [
        f"SEED PAPER: {_describe_paper(seed)}",
        "",
        f"THE IDEA THE READER CARES ABOUT: {idea_text}",
        "",
        f"CANDIDATES ({len(candidates)}):",
    ]
    for candidate in candidates:
        tags = candidate.direction
        if candidate.intent:
            tags += f", citation intent: {candidate.intent}"
        lines.append(f"[{candidate.index}] ({tags}) {_describe_paper(candidate.node)}")
    system = _SYSTEM_PROMPT % {"min_keep": max(5, max_keep // 3), "max_keep": max_keep}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _parse_clusters(raw: Any) -> dict[str, Cluster]:
    clusters: dict[str, Cluster] = {}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        cluster_id = clean_str(item.get("id"))
        name = clean_str(item.get("name"))
        if not cluster_id or not name:
            continue
        role = (clean_str(item.get("role")) or "").lower()
        clusters[cluster_id] = Cluster(
            id=cluster_id,
            name=name,
            summary=clean_str(item.get("summary")) or "",
            role=role if role in _VALID_ROLES else "contemporary",
        )
    return clusters


@dataclass(slots=True)
class _Kept:
    candidate: _Candidate
    cluster_id: str
    importance: int
    why: str


def _parse_keep(
    raw: Any, by_index: dict[int, _Candidate], clusters: dict[str, Cluster]
) -> list[_Kept]:
    kept: list[_Kept] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        index = coerce_int(item.get("i", item.get("index")))
        candidate = by_index.get(index) if index is not None else None
        if candidate is None or candidate.node.id in seen:
            continue
        seen.add(candidate.node.id)
        cluster_id = clean_str(item.get("cluster")) or _OTHER_CLUSTER_ID
        if cluster_id not in clusters:
            # The model named a cluster it never defined (or mistyped an id).
            # Better to keep the paper under a catch-all than to lose it.
            cluster_id = _OTHER_CLUSTER_ID
        importance = coerce_int(item.get("importance")) or 3
        kept.append(
            _Kept(
                candidate=candidate,
                cluster_id=cluster_id,
                importance=min(5, max(1, importance)),
                why=clean_str(item.get("why")) or "",
            )
        )
    return kept


def curate_graph(
    expansion: ExpansionResult,
    idea_text: str,
    settings: Settings,
    *,
    http_client: httpx.Client | None = None,
) -> FilteredGraph:
    """Run the LLM curation pass over `expansion`, returning the graph to render.

    Raises `CurationError` if there is nothing to curate, the API call fails,
    or the reply can't be turned into a usable selection.
    """
    seed = expansion.nodes.get(expansion.seed_id)
    if seed is None:
        raise CurationError(f"Expansion has no seed node {expansion.seed_id!r} to curate around.")

    candidates = prefilter(expansion, idea_text, settings.curation_prefilter)
    if not candidates:
        raise CurationError("Expansion contains only the seed paper — nothing to curate.")

    result = chat_json(
        settings,
        build_prompt(seed, idea_text, candidates, settings.curation_max_nodes),
        model=settings.resolved_curation_model,
        temperature=0.1,
        http_client=http_client,
        error_cls=CurationError,
    )

    clusters = _parse_clusters(result.data.get("clusters"))
    by_index = {c.index: c for c in candidates}
    kept = _parse_keep(result.data.get("keep"), by_index, clusters)
    if not kept:
        raise CurationError(
            f"Curation reply kept no recognizable papers: {json.dumps(result.data)[:300]}"
        )

    kept.sort(key=lambda k: (k.importance, k.candidate.score), reverse=True)
    kept = kept[: settings.curation_max_nodes]
    notes = clean_str(result.data.get("dropped_summary")) or ""
    return _assemble(expansion, seed, idea_text, kept, clusters, notes)


def _assemble(
    expansion: ExpansionResult,
    seed: Node,
    idea_text: str,
    kept: list[_Kept],
    clusters: dict[str, Cluster],
    notes: str,
) -> FilteredGraph:
    """Turn the model's selection into a `FilteredGraph` (seed pinned, edges pruned)."""
    nodes = [
        ScoredNode(
            node=seed,
            score=0.0,
            cluster_id=None,
            importance=5,
            why="The seed paper this map is built around.",
        )
    ]
    nodes += [
        ScoredNode(
            node=k.candidate.node,
            score=k.candidate.score,
            cluster_id=k.cluster_id,
            importance=k.importance,
            why=k.why,
        )
        for k in kept
    ]

    used_cluster_ids = {k.cluster_id for k in kept}
    ordered_clusters = [clusters[cid] for cid in clusters if cid in used_cluster_ids]
    if _OTHER_CLUSTER_ID in used_cluster_ids and _OTHER_CLUSTER_ID not in clusters:
        ordered_clusters.append(
            Cluster(id=_OTHER_CLUSTER_ID, name="Other related work", role="contemporary")
        )

    selected_ids = {sn.node.id for sn in nodes}
    edges = [e for e in expansion.edges if e.source in selected_ids and e.target in selected_ids]

    return FilteredGraph(
        seed_id=expansion.seed_id,
        idea_text=idea_text,
        top_k=len(kept),
        nodes=nodes,
        edges=edges,
        clusters=ordered_clusters,
        curation="llm",
        curation_notes=notes,
    )
