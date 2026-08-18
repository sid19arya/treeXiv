"""Step 2: two-hop bidirectional expansion from a seed work.

Implements `scratch/treexiv-mvp-openalex-prd.md` Section 3, Step 2:

- Backward (papers the seed cites): read straight off `referenced_works`.
- Forward (papers citing the seed): `GET /works?filter=cites:{id}`.
- Repeat one hop further from every node collected in hop 1, both directions.
- Per-node fan-out cap: top-N by `cited_by_count` by default, or a random
  `sample=` when the caller wants divergent branches instead of prominence.
- Global total-corpus cap: once hit, stop adding nodes, prioritizing whichever
  candidates have the highest `cited_by_count` across the whole frontier.

Edges reflect only citations discovered during traversal (source cites
target) — this is not an exhaustive pairwise citation check across the final
node set. Step 4 (`filtering.py`) drops any edge whose endpoint didn't
survive relevance filtering.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from treexiv.cache import WorkCache
from treexiv.config import Settings
from treexiv.models import Edge, ExpansionResult, Node, Work
from treexiv.openalex import OpenAlexClient


@dataclass(slots=True)
class _Candidate:
    work: Work
    parent_id: str
    direction: str  # "backward" (parent cites work) or "forward" (work cites parent)


def _select_backward(
    referenced_ids: list[str],
    works_by_id: dict[str, Work],
    cap: int,
    strategy: str,
    rng: random.Random,
) -> list[Work]:
    """Apply the fan-out cap to a fixed, already-known list of referenced works."""
    candidates = [works_by_id[wid] for wid in referenced_ids if wid in works_by_id]
    if len(candidates) <= cap:
        return candidates
    if strategy == "random":
        return rng.sample(candidates, cap)
    return sorted(candidates, key=lambda w: w.cited_by_count, reverse=True)[:cap]


def _fetch_referenced_works(
    client: OpenAlexClient, cache: WorkCache, referenced_ids: list[str]
) -> dict[str, Work]:
    cached = cache.get_many(referenced_ids)
    missing = [wid for wid in referenced_ids if wid not in cached]
    fetched = client.get_works_by_ids(missing) if missing else {}
    for work in fetched.values():
        cache.put_work(work)
    return {**cached, **fetched}


def _fetch_citing_works(
    client: OpenAlexClient,
    cache: WorkCache,
    work_id: str,
    cap: int,
    strategy: str,
    sample_seed: int | None,
) -> list[Work]:
    """Fetch citing works live and warm the by-ID cache with the results.

    The list itself (which works cite `work_id`, in what order) isn't cached
    — `WorkCache` is keyed by work ID, not by query — but any work returned
    here is available to later `get_works_by_ids` lookups without a re-fetch.
    """
    works = client.get_citing_works(work_id, limit=cap, strategy=strategy, sample_seed=sample_seed)
    for work in works:
        cache.put_work(work)
    return works


def _add_candidates(
    nodes: dict[str, Node],
    works: dict[str, Work],
    edges: list[Edge],
    seen_edges: set[tuple[str, str]],
    candidates: list[_Candidate],
    hop: int,
    total_cap: int,
) -> bool:
    """Add as many candidates as the global cap allows, highest-cited first.

    Returns True if any candidate had to be dropped for lack of budget.
    """
    truncated = False
    by_id: dict[str, _Candidate] = {}
    for candidate in candidates:
        if candidate.work.id in nodes or candidate.work.id in by_id:
            continue
        by_id[candidate.work.id] = candidate

    ordered = sorted(by_id.values(), key=lambda c: c.work.cited_by_count, reverse=True)
    for candidate in ordered:
        if len(nodes) >= total_cap:
            truncated = True
            break
        nodes[candidate.work.id] = Node.from_work(candidate.work, hop=hop)
        works[candidate.work.id] = candidate.work

    for candidate in candidates:
        if candidate.work.id not in nodes:
            continue
        source, target = (
            (candidate.parent_id, candidate.work.id)
            if candidate.direction == "backward"
            else (candidate.work.id, candidate.parent_id)
        )
        pair = (source, target)
        if pair in seen_edges:
            continue
        seen_edges.add(pair)
        edges.append(Edge(source=source, target=target))

    return truncated


def _expand_one_node(
    client: OpenAlexClient,
    cache: WorkCache,
    settings: Settings,
    node_id: str,
    node_work: Work,
    rng: random.Random,
    sample_seed: int | None,
) -> list[_Candidate]:
    referenced = _fetch_referenced_works(client, cache, node_work.referenced_works)
    backward_selected = _select_backward(
        node_work.referenced_works,
        referenced,
        settings.per_node_fanout_cap,
        settings.sampling_strategy,
        rng,
    )
    forward_selected = _fetch_citing_works(
        client,
        cache,
        node_id,
        settings.per_node_fanout_cap,
        settings.sampling_strategy,
        sample_seed,
    )
    return [
        _Candidate(work=w, parent_id=node_id, direction="backward") for w in backward_selected
    ] + [_Candidate(work=w, parent_id=node_id, direction="forward") for w in forward_selected]


def expand_two_hop(
    client: OpenAlexClient,
    settings: Settings,
    seed_work: Work,
    cache: WorkCache | None = None,
    sample_seed: int | None = None,
) -> ExpansionResult:
    """Run the full two-hop bidirectional expansion described in the module docstring."""
    active_cache = cache or WorkCache(cache_dir=None, seed_id=seed_work.id)
    rng = random.Random(sample_seed)

    nodes: dict[str, Node] = {seed_work.id: Node.from_work(seed_work, hop=0)}
    works: dict[str, Work] = {seed_work.id: seed_work}
    edges: list[Edge] = []
    seen_edges: set[tuple[str, str]] = set()
    truncated = False

    hop1_candidates = _expand_one_node(
        client, active_cache, settings, seed_work.id, seed_work, rng, sample_seed
    )
    truncated |= _add_candidates(
        nodes, works, edges, seen_edges, hop1_candidates, hop=1, total_cap=settings.total_corpus_cap
    )

    hop1_node_ids = [nid for nid, n in nodes.items() if n.hop == 1]
    hop2_candidates: list[_Candidate] = []
    for node_id in hop1_node_ids:
        hop2_candidates.extend(
            _expand_one_node(
                client, active_cache, settings, node_id, works[node_id], rng, sample_seed
            )
        )
    truncated |= _add_candidates(
        nodes, works, edges, seen_edges, hop2_candidates, hop=2, total_cap=settings.total_corpus_cap
    )

    active_cache.save()
    return ExpansionResult(seed_id=seed_work.id, nodes=nodes, edges=edges, truncated=truncated)
