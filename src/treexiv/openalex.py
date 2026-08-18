"""Thin client for the OpenAlex `/works` API.

Only the calls this MVP needs: title search (seed resolution candidates),
fetching works by ID in batches, and fetching a work's citing papers with
either top-cited or random-sample prioritization. See
`scratch/treexiv-mvp-openalex-prd.md` Section 3, Steps 1-2.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import httpx

from treexiv.config import Settings
from treexiv.exceptions import OpenAlexAPIError
from treexiv.models import Work, normalize_work_id

_WORKS_PATH = "/works"
_ID_BATCH_SIZE = 50
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAlexClient:
    """A synchronous OpenAlex client with polite-pool params and basic retry.

    Use as a context manager, or call `.close()` when done, to release the
    underlying HTTP connection pool.
    """

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=settings.base_url, timeout=settings.timeout_seconds
        )

    def __enter__(self) -> OpenAlexClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _base_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self._settings.mailto:
            params["mailto"] = self._settings.mailto
        if self._settings.api_key:
            params["api_key"] = self._settings.api_key
        return params

    def _get(self, path: str, params: dict[str, Any]) -> dict:
        merged = {**self._base_params(), **params}
        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries):
            try:
                response = self._client.get(path, params=merged)
            except httpx.TransportError as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue
            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_error = OpenAlexAPIError(
                    f"OpenAlex request to {path} failed with retryable status"
                    f" {response.status_code}",
                    status_code=response.status_code,
                )
                time.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise OpenAlexAPIError(
                    f"OpenAlex request to {path} failed: "
                    f"{response.status_code} {response.text[:300]}",
                    status_code=response.status_code,
                )
            return response.json()
        raise OpenAlexAPIError(
            f"OpenAlex request to {path} failed after "
            f"{self._settings.max_retries} attempts: {last_error}"
        )

    def search_works(self, query: str, limit: int = 5) -> list[Work]:
        """Search for candidate works by title/free text (Step 1 candidate lookup)."""
        payload = self._get(_WORKS_PATH, {"search": query, "per_page": limit})
        return [Work.from_api(item) for item in payload.get("results", [])]

    def get_work(self, work_id: str) -> Work:
        """Fetch a single work by its OpenAlex ID."""
        normalized = normalize_work_id(work_id)
        payload = self._get(f"{_WORKS_PATH}/{normalized}", {})
        return Work.from_api(payload)

    def get_works_by_ids(self, work_ids: Iterable[str]) -> dict[str, Work]:
        """Fetch multiple works by ID, batched to respect OpenAlex's OR-filter limits.

        Used for hop-1 backward expansion, where `referenced_works` gives us
        IDs but not the full records (title/abstract/cited_by_count) needed
        for prioritization and the BM25 corpus.
        """
        ids = [normalize_work_id(w) for w in work_ids]
        results: dict[str, Work] = {}
        for start in range(0, len(ids), _ID_BATCH_SIZE):
            batch = ids[start : start + _ID_BATCH_SIZE]
            if not batch:
                continue
            filter_value = "openalex_id:" + "|".join(batch)
            payload = self._get(
                _WORKS_PATH, {"filter": filter_value, "per_page": len(batch)}
            )
            for item in payload.get("results", []):
                work = Work.from_api(item)
                results[work.id] = work
        return results

    def get_citing_works(
        self,
        work_id: str,
        limit: int,
        strategy: str = "top_cited",
        sample_seed: int | None = None,
    ) -> list[Work]:
        """Fetch works that cite `work_id` (forward direction), capped at `limit`.

        `strategy="top_cited"` sorts server-side by `cited_by_count` descending
        (prioritizes established, likely-core-lineage papers — the MVP
        default). `strategy="random"` uses OpenAlex's `sample=` parameter to
        deliberately catch less-cited, divergent branches instead.
        """
        normalized = normalize_work_id(work_id)
        params: dict[str, Any] = {
            "filter": f"cites:{normalized}",
            "per_page": limit,
        }
        if strategy == "random":
            params["sample"] = limit
            if sample_seed is not None:
                params["seed"] = sample_seed
        else:
            params["sort"] = "cited_by_count:desc"
        payload = self._get(_WORKS_PATH, params)
        return [Work.from_api(item) for item in payload.get("results", [])]
