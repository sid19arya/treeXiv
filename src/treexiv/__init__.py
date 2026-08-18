"""treexiv: trace a paper's citation lineage as an interactive HTML graph.

See the top-level README for usage. Primarily driven as a Claude Code skill
(`.claude/skills/treexiv-lineage/`), which handles seed-paper disambiguation
before calling into this package; the `treexiv` CLI it wraps is also usable
directly. Implementation spec: `scratch/treexiv-mvp-openalex-prd.md`.
"""

from __future__ import annotations

__version__ = "0.1.0"
