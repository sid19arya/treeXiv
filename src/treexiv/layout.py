"""Deterministic node layout for the rendered graph.

Position is driven by publication year, not by vis-network's physics engine:
older papers sit toward the top-left, newer papers toward the bottom-right,
and the seed paper is pinned at the center (regardless of its own year,
though in practice it lands there naturally since the diagonal coordinate is
computed relative to the seed's year). Nodes sharing a year are fanned out
along the perpendicular axis, by node ID order, so they don't stack exactly
on top of each other.

Physics stays off in the renderer - this layout is the whole point, not a
seed for a force simulation to immediately scramble.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from treexiv.models import Node

DIAGONAL_SCALE = 900.0
PERPENDICULAR_STEP = 70.0
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

    positions: dict[str, Position] = {}
    for group in year_groups.values():
        n = len(group)
        for i, node in enumerate(group):
            d = _diagonal_coordinate(node.publication_year, seed_year, min_year, max_year)
            base_x = d * DIAGONAL_SCALE
            base_y = d * DIAGONAL_SCALE
            offset = (i - (n - 1) / 2) * PERPENDICULAR_STEP
            # A lone node at exactly the seed's diagonal coordinate (its own
            # year, or an unknown year) would otherwise get offset == 0 too,
            # landing it exactly on top of the seed. Nudge it off-center.
            if node.id != seed_id and base_x == 0.0 and base_y == 0.0 and offset == 0.0:
                offset = PERPENDICULAR_STEP * 0.5
            x = base_x + offset * _PERPENDICULAR_UNIT[0]
            y = base_y + offset * _PERPENDICULAR_UNIT[1]
            positions[node.id] = Position(x=x, y=y)

    if seed_node is not None:
        positions[seed_id] = Position(x=0.0, y=0.0)

    return positions
