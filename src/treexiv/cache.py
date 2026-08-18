"""Optional per-run JSON cache for OpenAlex work records.

Per the PRD (Section 4): "a lightweight per-run cache... is a reasonable
optional addition, but it is not required and should not turn into a
database." This is exactly that — one JSON file per seed work ID, storing
raw work payloads keyed by normalized work ID, so a repeated run on the same
seed doesn't re-fetch. Nothing here is shared across seeds or queried; it's
a flat key-value dump.
"""

from __future__ import annotations

import json
from pathlib import Path

from treexiv.models import Work


class WorkCache:
    """A disabled-by-default, file-backed cache of `Work` records for one seed."""

    def __init__(self, cache_dir: str | Path | None, seed_id: str) -> None:
        self._enabled = cache_dir is not None
        self._path = Path(cache_dir) / f"{seed_id}.json" if cache_dir else None
        self._entries: dict[str, dict] = {}
        if self._enabled and self._path is not None and self._path.exists():
            self._entries = json.loads(self._path.read_text(encoding="utf-8"))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get(self, work_id: str) -> Work | None:
        if not self._enabled:
            return None
        payload = self._entries.get(work_id)
        return Work.from_api(payload) if payload is not None else None

    def get_many(self, work_ids: list[str]) -> dict[str, Work]:
        return {wid: work for wid in work_ids if (work := self.get(wid)) is not None}

    def put(self, work_id: str, raw_payload: dict) -> None:
        if not self._enabled:
            return
        self._entries[work_id] = raw_payload

    def put_work(self, work: Work) -> None:
        """Cache a parsed `Work` by round-tripping it through a minimal raw payload.

        Used when the source of a `Work` was itself a cache hit or an
        already-parsed object rather than a fresh API response.
        """
        if not self._enabled:
            return
        self._entries[work.id] = {
            "id": work.id,
            "display_name": work.title,
            "publication_year": work.publication_year,
            "cited_by_count": work.cited_by_count,
            "authorships": [{"author": {"display_name": a}} for a in work.authors],
            "primary_location": {"source": {"display_name": work.venue}},
            "doi": work.doi,
            "referenced_works": work.referenced_works,
            "abstract_inverted_index": work.abstract_inverted_index,
        }

    def save(self) -> None:
        if not self._enabled or self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries), encoding="utf-8")
