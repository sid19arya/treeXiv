"""Tests for two-hop bidirectional expansion (expand.py).

Uses a fake OpenAlex client (same call surface as OpenAlexClient) so these
tests exercise expansion/capping/dedup logic without any network access.
"""

from __future__ import annotations

import dataclasses

from treexiv.cache import WorkCache
from treexiv.expand import expand_two_hop
from treexiv.models import Work


class FakeOpenAlexClient:
    """In-memory stand-in for OpenAlexClient's `get_works_by_ids`/`get_citing_works`."""

    def __init__(self, works: dict[str, Work], citing_universe: dict[str, list[Work]]) -> None:
        self._works = works
        self._citing_universe = citing_universe

    def get_works_by_ids(self, work_ids):
        return {wid: self._works[wid] for wid in work_ids if wid in self._works}

    def get_citing_works(self, work_id, limit, strategy="top_cited", sample_seed=None):
        universe = self._citing_universe.get(work_id, [])
        if strategy == "random":
            import random

            rng = random.Random(sample_seed)
            pool = list(universe)
            rng.shuffle(pool)
            return pool[:limit]
        ranked = sorted(universe, key=lambda w: w.cited_by_count, reverse=True)
        return ranked[:limit]


def _work(id_: str, *, cited_by_count: int = 0, referenced_works=None) -> Work:
    return Work(
        id=id_,
        title=f"Paper {id_}",
        publication_year=2020,
        cited_by_count=cited_by_count,
        referenced_works=referenced_works or [],
    )


def test_seed_always_present_at_hop_zero(settings) -> None:
    seed = _work("SEED")
    client = FakeOpenAlexClient({"SEED": seed}, {})
    result = expand_two_hop(client, settings, seed)
    assert result.nodes["SEED"].hop == 0
    assert not result.truncated


def test_backward_hop1_adds_referenced_works(settings) -> None:
    r1 = _work("R1", cited_by_count=5)
    seed = _work("SEED", referenced_works=["R1"])
    client = FakeOpenAlexClient({"SEED": seed, "R1": r1}, {})
    result = expand_two_hop(client, settings, seed)
    assert "R1" in result.nodes
    assert result.nodes["R1"].hop == 1
    assert any(e.source == "SEED" and e.target == "R1" for e in result.edges)


def test_forward_hop1_adds_citing_works(settings) -> None:
    f1 = _work("F1", cited_by_count=8)
    seed = _work("SEED")
    client = FakeOpenAlexClient({"SEED": seed}, {"SEED": [f1]})
    result = expand_two_hop(client, settings, seed)
    assert "F1" in result.nodes
    assert result.nodes["F1"].hop == 1
    assert any(e.source == "F1" and e.target == "SEED" for e in result.edges)


def test_hop2_expands_from_hop1_nodes(settings) -> None:
    x = _work("X", cited_by_count=1)
    r1 = _work("R1", cited_by_count=5, referenced_works=["X"])
    seed = _work("SEED", referenced_works=["R1"])
    client = FakeOpenAlexClient({"SEED": seed, "R1": r1, "X": x}, {})
    result = expand_two_hop(client, settings, seed)
    assert "X" in result.nodes
    assert result.nodes["X"].hop == 2
    assert any(e.source == "R1" and e.target == "X" for e in result.edges)


def test_dedup_keeps_earliest_hop_for_repeated_node(settings) -> None:
    # F1 cites SEED (hop 1) AND F1 cites R1 too (would be hop 2 via R1) -
    # F1 must end up recorded once, at hop 1.
    f1 = _work("F1", cited_by_count=9)
    r1 = _work("R1", cited_by_count=5)
    seed = _work("SEED", referenced_works=["R1"])
    client = FakeOpenAlexClient(
        {"SEED": seed, "R1": r1, "F1": f1}, {"SEED": [f1], "R1": [f1]}
    )
    result = expand_two_hop(client, settings, seed)
    assert result.nodes["F1"].hop == 1


def test_per_node_fanout_cap_keeps_top_cited(settings) -> None:
    capped_settings = dataclasses.replace(settings, per_node_fanout_cap=2, total_corpus_cap=500)
    refs = [f"R{i}" for i in range(5)]
    works = {f"R{i}": _work(f"R{i}", cited_by_count=i) for i in range(5)}
    seed = _work("SEED", referenced_works=refs)
    works["SEED"] = seed
    client = FakeOpenAlexClient(works, {})
    result = expand_two_hop(client, capped_settings, seed)
    hop1_ids = {nid for nid, n in result.nodes.items() if n.hop == 1}
    assert hop1_ids == {"R4", "R3"}  # top 2 by cited_by_count


def test_global_total_cap_stops_adding_nodes_and_sets_truncated(settings) -> None:
    capped_settings = dataclasses.replace(
        settings, total_corpus_cap=3, per_node_fanout_cap=100
    )
    refs = [f"R{i}" for i in range(5)]
    works = {f"R{i}": _work(f"R{i}", cited_by_count=i) for i in range(5)}
    seed = _work("SEED", referenced_works=refs)
    works["SEED"] = seed
    client = FakeOpenAlexClient(works, {})
    result = expand_two_hop(client, capped_settings, seed)
    assert result.truncated is True
    assert len(result.nodes) == 3  # cap includes the seed itself
    # highest-cited candidates should win the remaining 2 slots
    kept_non_seed = {nid for nid in result.nodes if nid != "SEED"}
    assert kept_non_seed == {"R4", "R3"}


def test_cache_is_populated_after_expansion(settings, tmp_path) -> None:
    r1 = _work("R1", cited_by_count=5)
    seed = _work("SEED", referenced_works=["R1"])
    client = FakeOpenAlexClient({"SEED": seed, "R1": r1}, {})
    cache = WorkCache(cache_dir=tmp_path, seed_id="SEED")
    expand_two_hop(client, settings, seed, cache=cache)
    assert (tmp_path / "SEED.json").exists()
