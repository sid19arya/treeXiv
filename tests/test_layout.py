from treexiv.layout import Position, compute_positions
from treexiv.models import Node


def _node(id_: str, year: int | None) -> Node:
    return Node(
        id=id_, title=f"Paper {id_}", publication_year=year, cited_by_count=1,
        authors=[], venue=None, abstract="", hop=0,
    )


def test_seed_is_always_centered() -> None:
    nodes = [_node("SEED", 2020), _node("OLD", 2010), _node("NEW", 2023)]
    positions = compute_positions(nodes, "SEED")
    assert positions["SEED"].x == 0.0
    assert positions["SEED"].y == 0.0


def test_older_paper_is_top_left_of_seed() -> None:
    nodes = [_node("SEED", 2020), _node("OLD", 2010)]
    positions = compute_positions(nodes, "SEED")
    assert positions["OLD"].x < 0
    assert positions["OLD"].y < 0


def test_newer_paper_is_bottom_right_of_seed() -> None:
    nodes = [_node("SEED", 2020), _node("NEW", 2023)]
    positions = compute_positions(nodes, "SEED")
    assert positions["NEW"].x > 0
    assert positions["NEW"].y > 0


def test_relative_ordering_follows_year() -> None:
    nodes = [_node("SEED", 2020), _node("A", 2015), _node("B", 2005)]
    positions = compute_positions(nodes, "SEED")
    # B (2005) is older than A (2015), so further toward top-left.
    assert positions["B"].x < positions["A"].x < 0


def test_same_year_nodes_are_fanned_out_not_stacked() -> None:
    nodes = [_node("SEED", 2020), _node("A", 2015), _node("B", 2015)]
    positions = compute_positions(nodes, "SEED")
    assert (positions["A"].x, positions["A"].y) != (positions["B"].x, positions["B"].y)


def test_missing_year_node_lands_near_but_not_on_the_seed() -> None:
    nodes = [_node("SEED", 2020), _node("UNKNOWN", None)]
    positions = compute_positions(nodes, "SEED")
    assert abs(positions["UNKNOWN"].x) < 100
    assert (positions["UNKNOWN"].x, positions["UNKNOWN"].y) != (0.0, 0.0)


def test_node_sharing_seed_year_does_not_collide_with_seed() -> None:
    nodes = [_node("SEED", 2020), _node("TWIN", 2020)]
    positions = compute_positions(nodes, "SEED")
    assert positions["SEED"] != positions["TWIN"]


def test_middle_ranked_node_at_seed_year_gets_nudged_off_center() -> None:
    # A, B, SEED (alphabetical) at the same year: B lands as the exact
    # middle-ranked member of that three-way fan-out, which works out to a
    # raw offset of 0 - i.e. it would land exactly on the seed without the
    # "nudge lone zero-offset nodes" fix.
    nodes = [_node("SEED", 2020), _node("A", 2020), _node("B", 2020)]
    positions = compute_positions(nodes, "SEED")
    assert positions["B"] != Position(0.0, 0.0)
    assert positions["B"] != positions["SEED"]


def test_seed_missing_from_node_list_does_not_crash() -> None:
    nodes = [_node("A", 2015), _node("B", 2018)]
    positions = compute_positions(nodes, "SEED_NOT_PRESENT")
    assert set(positions.keys()) == {"A", "B"}


def test_empty_node_list_returns_empty() -> None:
    assert compute_positions([], "SEED") == {}


def test_all_same_year_does_not_divide_by_zero() -> None:
    nodes = [_node("SEED", 2020), _node("A", 2020), _node("B", 2020)]
    positions = compute_positions(nodes, "SEED")
    assert positions["SEED"].x == 0.0
    # A and B share the seed's year - no crash, and they're still distinguishable.
    assert (positions["A"].x, positions["A"].y) != (positions["B"].x, positions["B"].y)


def test_positions_are_deterministic_across_calls() -> None:
    nodes = [_node("SEED", 2020), _node("A", 2015), _node("B", 2015), _node("C", 2023)]
    first = compute_positions(nodes, "SEED")
    second = compute_positions(nodes, "SEED")
    assert first == second
