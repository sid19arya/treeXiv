"""Exhaustive mode: Jev decides what belongs and how it connects; an LLM only writes.

The curated path (`curate.py`) hands one chat model 120 BM25-shortlisted
abstracts and asks it to keep, drop, and cluster them in one reply. That takes
minutes, and it can only choose from what citation traversal collected.
This path is built from different parts:

1. **Corpus**: the citation expansion plus scholarly search hits
   (`sources/search.py`), merged by identity.
2. **Relevance gate (Jev)**: every candidate (BM25-ordered, capped at
   `exhaustive_max_candidates`) gets a 0-4 rubric score against the stated
   idea. Those at or above `exhaustive_min_relevance` pass. The score is only
   a gate. In practice Jev puts every on-topic paper near 3 and rates a
   narrow recent variant *above* the paper that started the line, because
   the variant is more squarely "about" the topic.
2b. **Lineage role (Jev)**: each paper that passes the gate is asked what role
   it plays: foundation / milestone / incremental / tangential. The
   `exhaustive_keep` papers kept are ranked by `lineage_rank`, which weights
   foundation + milestone probability most, then relevance, then citations.
   This is what keeps FrugalGPT in a RouteLLM map instead of fifty 2026
   router variants.
3. **Relations (Jev)**: survivors are paired with the seed, with the papers
   they share a citation edge with, and with their `exhaustive_neighbours`
   BM25 nearest neighbours. Never all pairs. For each pair Jev chooses how the
   later paper relates to the earlier one: extends / applies / alternative /
   related / unrelated.
4. **Clusters (no model)**: weighted label propagation over the relation
   probabilities, with citation edges as extra weight, then merged down to at
   most `MAX_CLUSTERS`. Roles (ancestor / descendant / contemporary) come from
   citation direction where it is known and publication year where it is not.
5. **Words (LLM, optional)**: `synthesis.name_clusters` names the strands Jev's
   relations formed, and `synthesis.synthesize_lineage` writes the story.
   Without an OpenRouter key, strands get keyword names and there is no story.

Edges keep the package's split between data and inference. A citation edge
is a real reference. Jev can annotate it with a relation but never creates
one. A relation Jev is confident about, between papers with no citation
between them, becomes a separate `kind="semantic"` edge, and everything
downstream labels it as inferred.

As in `curate.py`, questions carry no work IDs. Papers go into structured
`instructions` and answers come back under short keys (`p12`, `r40`) that
are mapped back here.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from treexiv import jev
from treexiv.config import Settings
from treexiv.corpus import BM25Corpus, document_text, tokenize
from treexiv.curate import classify_directions
from treexiv.exceptions import JevError, SynthesisError
from treexiv.models import (
    EDGE_SEMANTIC,
    SEARCH_HOP,
    Cluster,
    Edge,
    ExpansionResult,
    FilteredGraph,
    Node,
    ScoredNode,
)
from treexiv.synthesis import name_clusters, synthesize_lineage

WarningSink = Callable[[str], None]
Evaluator = Callable[[Any, Mapping[str, jev.Question]], dict[str, jev.Answer]]

PAPER_ABSTRACT_CHARS = 600
SEED_ABSTRACT_CHARS = 900
MAX_CLUSTERS = 7
# Relation-probability mass below this doesn't link two papers for clustering.
_MIN_CLUSTER_WEIGHT = 0.35
# A cluster holding more than this share of the papers (and more than
# `_SPLIT_MIN_SIZE`) is re-split at a stricter weight threshold.
_SPLIT_SHARE = 0.5
_SPLIT_MIN_SIZE = 12
_CITATION_WEIGHT = 0.5
_OTHER_CLUSTER_ID = "other"

RELEVANCE_LEVELS = [
    "Unrelated to the core idea",
    "Same broad field, but not about the core idea",
    "Related work a specialist might mention in passing",
    "Directly relevant: a precursor, a close variant, or a direct extension of the core idea",
    "Essential: the core idea's lineage cannot be told without this paper",
]
RELEVANCE_QUESTION = (
    "How central is `paper` to the lineage of the core idea in the state — "
    "the work that idea grew out of, or the work it grew into?"
)

LINEAGE_ROLES = {
    "foundation": (
        "an earlier work that `seed_paper` and the core idea directly build on or grew out of"
    ),
    "milestone": (
        "a landmark step that first stated, named, or substantially redefined the core idea, "
        "which later work builds on"
    ),
    "incremental": "one of many variants, applications, or refinements of the core idea",
    "tangential": "touches the core idea only in passing",
}
ROLE_QUESTION = "What role does `paper` play in the lineage of the core idea?"
# How much a paper's foundation-or-milestone probability outweighs its
# relevance (0-1 after scaling) and its citation count (0-1, log scale).
_LINEAGE_WEIGHT = 2.0
_CITATION_WEIGHT_RANK = 0.3

RELATIONS = {
    "extends": (
        "`later` directly builds on, improves, or generalizes the idea or method of `earlier`"
    ),
    "applies": "`later` takes the idea of `earlier` to a new task, domain, or setting",
    "alternative": (
        "`later` proposes a competing or alternative approach to the problem `earlier` addresses"
    ),
    "related": "same topic, but `later` owes no direct intellectual debt to `earlier`",
    "unrelated": "the two papers are about different things",
}
RELATION_QUESTION = "How does `later` relate to `earlier`?"
# Relations strong enough to draw as an edge. "related" only feeds clustering.
EDGE_RELATIONS = ("extends", "applies", "alternative")
# Infinitives, for "is judged to <verb> X".
RELATION_VERBS = {
    "extends": "build on",
    "applies": "apply",
    "alternative": "offer an alternative to",
}


@dataclass(slots=True, frozen=True)
class Relation:
    """Jev's read on one ordered pair: how `later` relates to `earlier`."""

    earlier: str
    later: str
    relation: str
    confidence: float
    affinity: float


def paper_view(node: Node, abstract_chars: int = PAPER_ABSTRACT_CHARS) -> dict[str, Any]:
    """What Jev sees of one paper."""
    abstract = " ".join(node.abstract.split())
    if len(abstract) > abstract_chars:
        abstract = abstract[:abstract_chars].rsplit(" ", 1)[0] + "…"
    view: dict[str, Any] = {"title": node.title, "year": node.publication_year}
    if node.authors:
        view["authors"] = node.authors[0] + (" et al." if len(node.authors) > 1 else "")
    if abstract:
        view["abstract"] = abstract
    return view


def idea_state(seed: Node, idea_text: str) -> dict[str, Any]:
    """The shared state every exhaustive-mode question is asked against."""
    return {
        "core_idea": idea_text,
        "seed_paper": paper_view(seed, SEED_ABSTRACT_CHARS),
    }


def shortlist(corpus: ExpansionResult, idea_text: str, limit: int) -> list[Node]:
    """Non-seed nodes, BM25-ordered against the idea, capped at `limit`.

    Only a bound on spend: Jev scores everything this returns.
    """
    others = [n for n in corpus.nodes.values() if n.id != corpus.seed_id]
    scores = BM25Corpus(others).scores(idea_text)
    ranked = sorted(others, key=lambda n: (scores.get(n.id, 0.0), n.cited_by_count), reverse=True)
    return ranked[:limit]


def score_relevance(state: Any, candidates: list[Node], evaluator: Evaluator) -> dict[str, float]:
    """Jev's 0-4 relevance score for each candidate, keyed by node ID."""
    key_to_id = {f"p{i}": node.id for i, node in enumerate(candidates)}
    questions = {
        key: jev.score(
            {"paper": paper_view(node), "question": RELEVANCE_QUESTION}, RELEVANCE_LEVELS
        )
        for key, node in zip(key_to_id, candidates, strict=True)
    }
    answers = evaluator(state, questions)
    scores: dict[str, float] = {}
    for key, node_id in key_to_id.items():
        answer = answers.get(key)
        if answer is None or answer.type != "score":
            raise JevError(f"Jev returned no relevance score for question {key!r}")
        scores[node_id] = float(answer.value)
    return scores


def gate(candidates: list[Node], relevance: dict[str, float], min_relevance: float) -> list[Node]:
    """Candidates at or above `min_relevance`, in their original order."""
    return [n for n in candidates if relevance.get(n.id, 0.0) >= min_relevance]


def score_roles(state: Any, candidates: list[Node], evaluator: Evaluator) -> dict[str, jev.Answer]:
    """Jev's lineage-role choice for each candidate, keyed by node ID."""
    key_to_id = {f"l{i}": node.id for i, node in enumerate(candidates)}
    questions = {
        key: jev.choice({"paper": paper_view(node), "question": ROLE_QUESTION}, LINEAGE_ROLES)
        for key, node in zip(key_to_id, candidates, strict=True)
    }
    answers = evaluator(state, questions)
    roles: dict[str, jev.Answer] = {}
    for key, node_id in key_to_id.items():
        answer = answers.get(key)
        if answer is None or answer.type != "choice":
            raise JevError(f"Jev returned no lineage role for question {key!r}")
        roles[node_id] = answer
    return roles


def lineage_weight(role: jev.Answer) -> float:
    """Probability the paper is a foundation or milestone of the idea."""
    if role.probabilities:
        return role.p("foundation") + role.p("milestone")
    return 1.0 if role.value in ("foundation", "milestone") else 0.0


def lineage_rank(node: Node, relevance: float, role: jev.Answer) -> float:
    """The keep-order key. Foundation/milestone probability dominates;
    relevance separates the incremental majority; citations break near-ties."""
    citations = min(1.0, math.log10(node.cited_by_count + 1) / 3)
    return (
        _LINEAGE_WEIGHT * lineage_weight(role) + relevance / 4 + _CITATION_WEIGHT_RANK * citations
    )


def select_kept(
    candidates: list[Node],
    relevance: dict[str, float],
    roles: dict[str, jev.Answer],
    keep: int,
) -> list[Node]:
    """The `keep` best candidates by `lineage_rank`."""
    ranked = sorted(
        candidates,
        key=lambda n: (lineage_rank(n, relevance[n.id], roles[n.id]), n.cited_by_count),
        reverse=True,
    )
    return ranked[:keep]


def order_pair(a: Node, b: Node, cites: set[tuple[str, str]]) -> tuple[Node, Node]:
    """(earlier, later). A citation between them decides; otherwise the year,
    with unknown years treated as later; otherwise the ID, just to be stable."""
    if (a.id, b.id) in cites:
        return b, a
    if (b.id, a.id) in cites:
        return a, b
    year_a = a.publication_year if a.publication_year is not None else 10**6
    year_b = b.publication_year if b.publication_year is not None else 10**6
    if year_a != year_b:
        return (a, b) if year_a < year_b else (b, a)
    return (a, b) if a.id < b.id else (b, a)


def candidate_pairs(
    seed: Node, kept: list[Node], edges: list[Edge], neighbours: int
) -> list[tuple[str, str]]:
    """Unordered pairs worth asking Jev about, deduplicated and stable-ordered."""
    ids = {n.id for n in kept} | {seed.id}
    pairs: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        if a != b:
            pairs.add((a, b) if a < b else (b, a))

    for node in kept:
        add(seed.id, node.id)
    for edge in edges:
        if edge.is_citation and edge.source in ids and edge.target in ids:
            add(edge.source, edge.target)
    if neighbours > 0 and len(kept) > 1:
        corpus = BM25Corpus(kept)
        for node in kept:
            ranked = corpus.top_k(document_text(node), neighbours + 1)
            for other, score in ranked:
                if other.id != node.id and score > 0:
                    add(node.id, other.id)
    return sorted(pairs)


def judge_relations(
    state: Any,
    pairs: list[tuple[str, str]],
    nodes_by_id: dict[str, Node],
    cites: set[tuple[str, str]],
    evaluator: Evaluator,
) -> list[Relation]:
    """Ask Jev how the later paper of each pair relates to the earlier one."""
    ordered: dict[str, tuple[Node, Node]] = {}
    questions: dict[str, jev.Question] = {}
    for i, (a, b) in enumerate(pairs):
        earlier, later = order_pair(nodes_by_id[a], nodes_by_id[b], cites)
        key = f"r{i}"
        ordered[key] = (earlier, later)
        questions[key] = jev.choice(
            {
                "earlier": paper_view(earlier),
                "later": paper_view(later),
                "question": RELATION_QUESTION,
            },
            RELATIONS,
        )
    answers = evaluator(state, questions)
    relations: list[Relation] = []
    for key, (earlier, later) in ordered.items():
        answer = answers.get(key)
        if answer is None or answer.type != "choice":
            raise JevError(f"Jev returned no relation for question {key!r}")
        picked = answer.value if answer.value in RELATIONS else "unrelated"
        affinity = sum(answer.p(r) for r in EDGE_RELATIONS) + 0.5 * answer.p("related")
        relations.append(
            Relation(
                earlier=earlier.id,
                later=later.id,
                relation=picked,
                confidence=answer.p(picked) if answer.probabilities else 1.0,
                affinity=affinity if answer.probabilities else float(picked in EDGE_RELATIONS),
            )
        )
    return relations


def _label_propagation(
    ids: list[str], adjacency: dict[str, dict[str, float]], min_weight: float
) -> dict[str, list[str]]:
    """Weighted label propagation, deterministic: nodes update in `ids` order
    and ties go to the label that ranks first in `ids`."""
    rank = {node_id: i for i, node_id in enumerate(ids)}
    labels = {node_id: node_id for node_id in ids}
    for _ in range(30):
        changed = False
        for node_id in ids:
            totals: dict[str, float] = defaultdict(float)
            for other, weight in adjacency.get(node_id, {}).items():
                if other in labels and weight >= min_weight:
                    totals[labels[other]] += weight
            if not totals:
                continue
            best = max(totals.values())
            tied = [label for label, total in totals.items() if total >= best - 1e-9]
            new = labels[node_id] if labels[node_id] in tied else min(tied, key=rank.__getitem__)
            if new != labels[node_id]:
                labels[node_id] = new
                changed = True
        if not changed:
            break
    groups: dict[str, list[str]] = {}
    for node_id in ids:
        groups.setdefault(labels[node_id], []).append(node_id)
    return groups


def _link_weight(a: list[str], b: list[str], adjacency: dict[str, dict[str, float]]) -> float:
    b_set = set(b)
    return sum(
        w for node_id in a for other, w in adjacency.get(node_id, {}).items() if other in b_set
    )


def cluster_papers(
    ids: list[str],
    adjacency: dict[str, dict[str, float]],
    max_clusters: int = MAX_CLUSTERS,
) -> tuple[list[list[str]], list[str]]:
    """Group `ids` (ordered most-relevant first) into at most `max_clusters`
    strands, plus the loose papers linked to no strand at all.

    Strands come back in order of their best member's relevance.
    """
    groups = list(_label_propagation(ids, adjacency, _MIN_CLUSTER_WEIGHT).values())

    # A strand that swallowed most of the map says little. Split it at a
    # stricter threshold while that still separates something.
    threshold = _MIN_CLUSTER_WEIGHT
    while threshold < 0.85:
        big = max(groups, key=len)
        if len(big) <= max(_SPLIT_MIN_SIZE, _SPLIT_SHARE * len(ids)):
            break
        threshold += 0.15
        sub = list(_label_propagation(big, adjacency, threshold).values())
        if len(sub) > 1:
            groups.remove(big)
            groups.extend(sub)

    # Singletons join the strand they're most linked to, or the catch-all.
    strands = [g for g in groups if len(g) > 1]
    loose: list[str] = []
    for group in groups:
        if len(group) > 1:
            continue
        best = max(strands, key=lambda s: _link_weight(group, s, adjacency), default=None)
        if best is not None and _link_weight(group, best, adjacency) > 0:
            best.append(group[0])
        else:
            loose.append(group[0])

    # Too many strands: fold the smallest into its closest neighbour.
    while len(strands) > max_clusters:
        strands.sort(key=len)
        smallest = strands.pop(0)
        target = max(strands, key=lambda s: _link_weight(smallest, s, adjacency))
        if _link_weight(smallest, target, adjacency) > 0:
            target.extend(smallest)
        else:
            loose.extend(smallest)

    rank = {node_id: i for i, node_id in enumerate(ids)}
    ordered = [sorted(s, key=rank.__getitem__) for s in strands]
    ordered.sort(key=lambda s: rank[s[0]])
    return ordered, sorted(loose, key=rank.__getitem__)


_NAME_STOPWORDS = frozenset(
    "a an and are as at be by can for from how in into is it its of on or that the their this to "
    "via we what when which with without using use based new towards toward study approach "
    "analysis paper model models method methods learning deep neural network networks".split()
)


def keyword_name(members: list[Node], everyone: list[Node], words: int = 3) -> str:
    """A no-LLM strand name: the title words most over-represented in `members`."""

    def title_terms(node: Node) -> set[str]:
        return {
            t
            for t in tokenize(node.title)
            if len(t) > 2 and t not in _NAME_STOPWORDS and not t.isdigit()
        }

    inside = Counter(t for n in members for t in title_terms(n))
    overall = Counter(t for n in everyone for t in title_terms(n))
    ranked = sorted(
        (t for t, c in inside.items() if c >= 2 or len(members) <= 2),
        key=lambda t: (inside[t] / (overall[t] + 1) * inside[t], t),
        reverse=True,
    )
    picked = ranked[:words]
    return " / ".join(picked).title() if picked else "Related work"


def cluster_role(members: list[Node], directions: dict[str, str], seed_year: int | None) -> str:
    """Where a strand sits relative to the seed: the members' majority vote,
    each voting by citation direction if known, by year if not."""
    votes: Counter[str] = Counter()
    for node in members:
        direction = directions.get(node.id)
        if direction in ("ancestor", "descendant"):
            votes[direction] += 1
        elif seed_year is not None and node.publication_year is not None:
            if node.publication_year < seed_year:
                votes["ancestor"] += 1
            elif node.publication_year > seed_year:
                votes["descendant"] += 1
            else:
                votes["contemporary"] += 1
        else:
            votes["contemporary"] += 1
    if not votes:
        return "contemporary"
    role, count = votes.most_common(1)[0]
    return role if count > len(members) / 2 else "contemporary"


def _short(title: str, limit: int = 70) -> str:
    return title if len(title) <= limit else title[: limit - 1] + "…"


def _title_or_seed(
    node_id: str, seed_id: str, nodes_by_id: dict[str, Node], seed_words: str
) -> str:
    return seed_words if node_id == seed_id else f'"{_short(nodes_by_id[node_id].title)}"'


def describe_node(
    node_id: str,
    relevance: float,
    relations: list[Relation],
    nodes_by_id: dict[str, Node],
    seed_id: str,
    role: jev.Answer | None = None,
) -> str:
    """The sidebar's 'why it's here' line: Jev's score and strongest relation,
    preferring the relation to the seed."""
    mine = [
        r for r in relations if node_id in (r.earlier, r.later) and r.relation in EDGE_RELATIONS
    ]
    mine.sort(key=lambda r: (seed_id in (r.earlier, r.later), r.confidence), reverse=True)
    text = f"Jev relevance {relevance:.1f}/4."
    if role is not None and role.value in LINEAGE_ROLES:
        confidence = f" ({role.p(role.value):.0%})" if role.probabilities else ""
        text += f" Lineage role: {role.value}{confidence}."
    if mine:
        rel = mine[0]
        verb = RELATION_VERBS[rel.relation]
        if rel.later == node_id:
            other = _title_or_seed(rel.earlier, seed_id, nodes_by_id, "the seed paper")
            text += f" Judged to {verb} {other} ({rel.confidence:.0%})."
        else:
            other = _title_or_seed(rel.later, seed_id, nodes_by_id, "The seed paper")
            text += f" {other} is judged to {verb} it ({rel.confidence:.0%})."
    return text


def build_edges(
    corpus_edges: list[Edge],
    relations: list[Relation],
    selected: set[str],
    threshold: float,
) -> list[Edge]:
    """Citation edges among `selected`, annotated with Jev's relation where it
    judged that pair, plus semantic edges for confident relations with no
    citation behind them."""
    by_pair = {frozenset((r.earlier, r.later)): r for r in relations}
    edges: list[Edge] = []
    cited_pairs: set[frozenset[str]] = set()
    for edge in corpus_edges:
        if not edge.is_citation or edge.source not in selected or edge.target not in selected:
            continue
        pair = frozenset((edge.source, edge.target))
        cited_pairs.add(pair)
        rel = by_pair.get(pair)
        edges.append(edge.with_relation(rel.relation, rel.confidence) if rel else edge)
    for rel in relations:
        pair = frozenset((rel.earlier, rel.later))
        if (
            pair in cited_pairs
            or rel.relation not in EDGE_RELATIONS
            or rel.confidence < threshold
            or not pair <= selected
        ):
            continue
        edges.append(
            Edge(
                source=rel.later,
                target=rel.earlier,
                kind=EDGE_SEMANTIC,
                relation=rel.relation,
                confidence=rel.confidence,
            )
        )
    return edges


def _adjacency(
    relations: list[Relation], edges: list[Edge], ids: set[str]
) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)

    def bump(a: str, b: str, weight: float) -> None:
        if a in ids and b in ids and a != b:
            adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
            adjacency[b][a] = adjacency[b].get(a, 0.0) + weight

    for rel in relations:
        bump(rel.earlier, rel.later, rel.affinity)
    for edge in edges:
        if edge.is_citation:
            bump(edge.source, edge.target, _CITATION_WEIGHT)
    return adjacency


def build_exhaustive_graph(
    corpus: ExpansionResult,
    idea_text: str,
    settings: Settings,
    *,
    http_client: httpx.Client | None = None,
    llm_client: httpx.Client | None = None,
    evaluator: Evaluator | None = None,
    on_warning: WarningSink | None = None,
) -> FilteredGraph:
    """Run steps 2-5 of the module docstring over an already-harvested corpus.

    Raises `JevError` if Jev is unavailable or its answers can't be used; the
    caller decides whether to fall back. LLM failures are never fatal.
    """
    seed = corpus.nodes.get(corpus.seed_id)
    if seed is None:
        raise JevError(f"Corpus has no seed node {corpus.seed_id!r}.")
    if evaluator is None:
        if not settings.ai_gateway_api_key:
            raise JevError("AI_GATEWAY_API_KEY is not set — exhaustive mode needs Jev.")

        def evaluator(state: Any, questions: Mapping[str, jev.Question]) -> dict[str, jev.Answer]:
            return jev.evaluate_many(settings, state, questions, http_client=http_client)

    state = idea_state(seed, idea_text)
    candidates = shortlist(corpus, idea_text, settings.exhaustive_max_candidates)
    if not candidates:
        raise JevError("Corpus contains only the seed paper — nothing to judge.")

    relevance = score_relevance(state, candidates, evaluator)
    passed = gate(candidates, relevance, settings.exhaustive_min_relevance)
    if not passed:
        raise JevError(
            f"Jev scored no candidate at or above relevance {settings.exhaustive_min_relevance}."
        )
    roles = score_roles(state, passed, evaluator)
    kept = select_kept(passed, relevance, roles, settings.exhaustive_keep)

    nodes_by_id = {n.id: n for n in [seed, *kept]}
    cites = {(e.source, e.target) for e in corpus.edges if e.is_citation}
    pairs = candidate_pairs(seed, kept, corpus.edges, settings.exhaustive_neighbours)
    relations = judge_relations(state, pairs, nodes_by_id, cites, evaluator)

    selected = set(nodes_by_id)
    edges = build_edges(corpus.edges, relations, selected, settings.jev_edge_threshold)

    kept_ids = [n.id for n in kept]
    strands, loose = cluster_papers(kept_ids, _adjacency(relations, edges, set(kept_ids)))
    directions = classify_directions(
        corpus.seed_id, [e for e in corpus.edges if e.is_citation], set(kept_ids)
    )
    groups = [(str(i), strand) for i, strand in enumerate(strands, start=1)]
    if loose:
        groups.append((_OTHER_CLUSTER_ID, loose))
    clusters: list[Cluster] = []
    cluster_of: dict[str, str] = {}
    for cluster_id, group in groups:
        members = [nodes_by_id[m] for m in group]
        clusters.append(
            Cluster(
                id=cluster_id,
                name=(
                    "Other related work"
                    if cluster_id == _OTHER_CLUSTER_ID
                    else keyword_name(members, kept)
                ),
                role=cluster_role(members, directions, seed.publication_year),
            )
        )
        cluster_of.update(dict.fromkeys(group, cluster_id))

    nodes = [
        ScoredNode(
            node=seed, score=0.0, importance=5, why="The seed paper this map is built around."
        )
    ]
    for node in kept:
        nodes.append(
            ScoredNode(
                node=node,
                score=relevance[node.id],
                cluster_id=cluster_of.get(node.id),
                importance=min(5, max(1, round(relevance[node.id]) + 1)),
                why=describe_node(
                    node.id, relevance[node.id], relations, nodes_by_id, seed.id, roles[node.id]
                ),
            )
        )

    searched = sum(1 for n in candidates if n.hop == SEARCH_HOP)
    semantic = sum(1 for e in edges if not e.is_citation)
    graph = FilteredGraph(
        seed_id=corpus.seed_id,
        idea_text=idea_text,
        top_k=len(kept),
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        curation="jev",
        curation_notes=(
            f"Jev scored {len(candidates)} candidates ({len(candidates) - searched} from citation "
            f"traversal, {searched} from search); {len(passed)} reached relevance >= "
            f"{settings.exhaustive_min_relevance:g}, and the {len(kept)} kept were ranked by "
            f"lineage role; it judged {len(pairs)} pairs, "
            f"{semantic} of which became inferred edges."
        ),
    )
    return _with_words(graph, settings, llm_client, on_warning)


def _with_words(
    graph: FilteredGraph,
    settings: Settings,
    llm_client: httpx.Client | None,
    on_warning: WarningSink | None,
) -> FilteredGraph:
    """Let the summariser name the strands and tell the story. Never fatal."""
    if not settings.openrouter_api_key:
        if on_warning:
            on_warning(
                "OPENROUTER_API_KEY unset — strands keep keyword names and there is no "
                "written story."
            )
        return graph
    try:
        named = name_clusters(graph, settings, http_client=llm_client)
    except SynthesisError as exc:
        if on_warning:
            on_warning(f"Cluster naming failed ({exc}) — keeping keyword names.")
    else:
        graph.clusters = [
            Cluster(id=c.id, name=named[c.id][0], summary=named[c.id][1], role=c.role)
            if c.id in named
            else c
            for c in graph.clusters
        ]
    if settings.narrative:
        try:
            graph.narrative = synthesize_lineage(graph, settings, http_client=llm_client)
        except SynthesisError as exc:
            if on_warning:
                on_warning(f"Lineage synthesis failed ({exc}) — rendering without it.")
    return graph
