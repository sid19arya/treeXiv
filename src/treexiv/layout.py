"""Deterministic node layout for the rendered graph.

Position is driven by publication year, not by vis-network's physics engine:
older papers sit toward the top-left, newer papers toward the bottom-right,
and the seed paper is pinned at the center (regardless of its own year,
though in practice it lands there naturally since the diagonal coordinate is
computed relative to the seed's year). Nodes sharing a year are fanned out
along the perpendicular axis, by node ID order, so they don't stack exactly
on top of each other.

Two layouts live here, one per view:

- `compute_positions` — every paper on the year diagonal. Used for graphs with
  no concept clusters (the BM25 path), and unchanged from the original design
  except that same-year fan-out now widens with the size of the group instead
  of using a fixed step, which is what made dense years unreadable.
- `compute_cluster_layout` — one position per *cluster* on that same diagonal,
  by its members' median year, plus a fixed position for each member inside
  its cluster's own region. Both are computed up front, so expanding a cluster
  in the browser swaps which nodes are drawn without anything moving.

Physics stays off in the renderer - this layout is the whole point, not a
seed for a force simulation to immediately scramble.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from treexiv.models import Node

DIAGONAL_SCALE = 900.0
PERPENDICULAR_STEP = 70.0
# Clusters need much more room than single papers: each one has to hold its
# expanded members without colliding with its neighbours.
CLUSTER_DIAGONAL_SCALE = 1500.0
CLUSTER_LANE_STEP = 520.0
MEMBER_SPACING = 190.0
_SQRT_HALF = 1 / math.sqrt(2)
_PERPENDICULAR_UNIT = (_SQRT_HALF, -_SQRT_HALF)


@dataclass(frozen=True, slots=True)
class Position:
    x: float
    y: float


def _diagonal_coordinate(
    year: int | None, seed_year: int | None, min_year: int, max_year: int
) -> float:
    """Map a publication year to [-1, 1]: -1 is the oldest paper in the set,
    +1 is the newest, 0 is the seed's own year."""
    if year is None or seed_year is None:
        return 0.0
    if year <= seed_year:
        span = seed_year - min_year
        return -((seed_year - year) / span) if span > 0 else 0.0
    span = max_year - seed_year
    return (year - seed_year) / span if span > 0 else 0.0


@dataclass(frozen=True, slots=True)
class ClusterLayout:
    """Where a cluster sits, and where each of its papers sits inside it."""

    cluster_id: str
    center: Position
    members: dict[str, Position]


def _median_year(nodes: list[Node]) -> int | None:
    years = sorted(n.publication_year for n in nodes if n.publication_year is not None)
    return years[len(years) // 2] if years else None


def _member_positions(nodes: list[Node], center: Position) -> dict[str, Position]:
    """Lay a cluster's papers out around its center, oldest first.

    A near-square grid rather than a ring: it keeps the footprint compact (so
    an expanded cluster is less likely to reach its neighbours) and reading
    order left-to-right, top-to-bottom still tracks publication year.
    """
    ordered = sorted(nodes, key=lambda n: (n.publication_year or 0, n.id))
    if not ordered:
        return {}
    columns = max(1, math.ceil(math.sqrt(len(ordered))))
    rows = math.ceil(len(ordered) / columns)
    positions: dict[str, Position] = {}
    for i, node in enumerate(ordered):
        col, row = i % columns, i // columns
        positions[node.id] = Position(
            x=center.x + (col - (columns - 1) / 2) * MEMBER_SPACING,
            y=center.y + (row - (rows - 1) / 2) * MEMBER_SPACING,
        )
    return positions


def compute_cluster_layout(
    members_by_cluster: dict[str, list[Node]], seed_year: int | None
) -> dict[str, ClusterLayout]:
    """Place each cluster on the year diagonal, and its papers inside it.

    Clusters are ordered by median publication year, so the diagonal still
    reads oldest-to-newest at the top level. They're then pushed apart along
    the perpendicular axis in alternating lanes: two clusters with similar
    median years would otherwise sit on top of each other, and at this zoom
    level a collision costs far more than a slightly untrue position.
    """
    if not members_by_cluster:
        return {}

    medians = {cid: _median_year(nodes) for cid, nodes in members_by_cluster.items()}
    known = [y for y in medians.values() if y is not None]
    min_year = min(known) if known else (seed_year or 0)
    max_year = max(known) if known else (seed_year or 0)

    ordered_ids = sorted(
        members_by_cluster,
        key=lambda cid: (medians[cid] is None, medians[cid] or 0, cid),
    )

    layouts: dict[str, ClusterLayout] = {}
    for lane, cluster_id in enumerate(ordered_ids):
        d = _diagonal_coordinate(medians[cluster_id], seed_year, min_year, max_year)
        # Alternate above/below the diagonal: 0, +1, -1, +2, -2, ...
        step = ((lane + 1) // 2) * (1 if lane % 2 else -1)
        offset = step * CLUSTER_LANE_STEP
        center = Position(
            x=d * CLUSTER_DIAGONAL_SCALE + offset * _PERPENDICULAR_UNIT[0],
            y=d * CLUSTER_DIAGONAL_SCALE + offset * _PERPENDICULAR_UNIT[1],
        )
        layouts[cluster_id] = ClusterLayout(
            cluster_id=cluster_id,
            center=center,
            members=_member_positions(members_by_cluster[cluster_id], center),
        )
    return layouts


def compute_positions(nodes: list[Node], seed_id: str) -> dict[str, Position]:
    """Compute one (x, y) per node, per the module docstring's layout rule."""
    if not nodes:
        return {}

    seed_node = next((n for n in nodes if n.id == seed_id), None)
    seed_year = seed_node.publication_year if seed_node else None
    known_years = [n.publication_year for n in nodes if n.publication_year is not None]
    min_year = min(known_years) if known_years else (seed_year or 0)
    max_year = max(known_years) if known_years else (seed_year or 0)

    # Fan-out index: group by year (None grouped together) so same-year
    # nodes don't overlap, ordered by ID for a stable, reproducible layout.
    year_groups: dict[int | None, list[Node]] = {}
    for node in nodes:
        year_groups.setdefault(node.publication_year, []).append(node)
    for group in year_groups.values():
        group.sort(key=lambda n: n.id)

    # Crowded years get a wider step. With a fixed one, a year holding twenty
    # papers drew them all inside a couple of node diameters - the single
    # biggest reason the flat view was unreadable.
    busiest = max((len(g) for g in year_groups.values()), default=1)
    step = PERPENDICULAR_STEP * (1.0 + 0.08 * max(0, busiest - 4))

    positions: dict[str, Position] = {}
    for group in year_groups.values():
        n = len(group)
        for i, node in enumerate(group):
            d = _diagonal_coordinate(node.publication_year, seed_year, min_year, max_year)
            base_x = d * DIAGONAL_SCALE
            base_y = d * DIAGONAL_SCALE
            offset = (i - (n - 1) / 2) * step
            # A lone node at exactly the seed's diagonal coordinate (its own
            # year, or an unknown year) would otherwise get offset == 0 too,
            # landing it exactly on top of the seed. Nudge it off-center.
            if node.id != seed_id and base_x == 0.0 and base_y == 0.0 and offset == 0.0:
                offset = step * 0.5
            x = base_x + offset * _PERPENDICULAR_UNIT[0]
            y = base_y + offset * _PERPENDICULAR_UNIT[1]
            positions[node.id] = Position(x=x, y=y)

    if seed_node is not None:
        positions[seed_id] = Position(x=0.0, y=0.0)

    return positions
