"""Private web front-end for the treexiv pipeline.

This is deliberately outside the Phase 0 CLI/skill scope (see `CLAUDE.md`):
one module, opt-in via the `web` extra, reusing the exact same package
functions the CLI does. It exists only to run the pipeline behind a browser
form on Render's free tier — nothing here changes the core package.

Every route except ``/health`` sits behind HTTP Basic Auth
(``TREEXIV_WEB_USER`` / ``TREEXIV_WEB_PASSWORD`` env vars). Without valid
credentials a request gets a 401 and the pipeline never runs — no OpenAlex
calls, no data. ``/health`` is left open so Render's health check works.

Run locally:  ``uv run --extra web uvicorn treexiv.web:app --reload``
"""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from treexiv.config import Settings
from treexiv.exceptions import TreeXivError
from treexiv.expand import expand_two_hop
from treexiv.filtering import filter_by_idea
from treexiv.openalex import OpenAlexClient
from treexiv.render import render_html

# Hard ceilings on caller-supplied knobs. Even an authenticated request (or a
# leaked credential) can't turn one call into a multi-thousand-request
# OpenAlex crawl. These are above the usual config.py defaults, not a
# replacement for them.
_MAX_TOTAL_CAP = 500
_MAX_FANOUT_CAP = 100
_MAX_TOP_K = 100

_INDEX_HTML = (resources.files("treexiv") / "webassets" / "index.html").read_text(
    encoding="utf-8"
)

app = FastAPI(title="treexiv", docs_url=None, redoc_url=None, openapi_url=None)
_basic = HTTPBasic()


def _require_auth(
    credentials: Annotated[HTTPBasicCredentials, Depends(_basic)],
) -> None:
    """Reject any request whose Basic-Auth credentials don't match the env vars."""
    expected_user = os.environ.get("TREEXIV_WEB_USER")
    expected_password = os.environ.get("TREEXIV_WEB_PASSWORD")
    if not expected_user or not expected_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth is not configured (TREEXIV_WEB_USER / TREEXIV_WEB_PASSWORD).",
        )
    user_ok = secrets.compare_digest(credentials.username, expected_user)
    password_ok = secrets.compare_digest(credentials.password, expected_password)
    if not (user_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )


AuthDep = Annotated[None, Depends(_require_auth)]


def _openalex_client() -> Iterator[OpenAlexClient]:
    """Request-scoped OpenAlex client. Overridden in tests."""
    with OpenAlexClient(Settings.from_env()) as client:
        yield client


ClientDep = Annotated[OpenAlexClient, Depends(_openalex_client)]


class RunRequest(BaseModel):
    work_id: str = Field(min_length=1)
    idea: str = Field(min_length=1)
    total_cap: int | None = Field(default=None, ge=1, le=_MAX_TOTAL_CAP)
    fanout_cap: int | None = Field(default=None, ge=1, le=_MAX_FANOUT_CAP)
    top_k: int | None = Field(default=None, ge=1, le=_MAX_TOP_K)
    sampling: str | None = None
    sample_seed: int | None = None


def _settings_for(req: RunRequest) -> Settings:
    """Env settings with the request's caps applied, each clamped to a ceiling."""
    base = Settings.from_env()
    return Settings(
        base_url=base.base_url,
        mailto=base.mailto,
        api_key=base.api_key,
        total_corpus_cap=min(req.total_cap or base.total_corpus_cap, _MAX_TOTAL_CAP),
        per_node_fanout_cap=min(req.fanout_cap or base.per_node_fanout_cap, _MAX_FANOUT_CAP),
        sampling_strategy=(req.sampling or base.sampling_strategy),  # type: ignore[arg-type]
        bm25_top_k=min(req.top_k or base.bm25_top_k, _MAX_TOP_K),
        timeout_seconds=base.timeout_seconds,
        max_retries=base.max_retries,
        cache_dir=None,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Open, unauthenticated — Render pings this and it exposes nothing."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(_: AuthDep) -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML)


@app.get("/api/search")
def search(
    _: AuthDep,
    client: ClientDep,
    q: Annotated[str, Query(min_length=2, max_length=300)],
    limit: Annotated[int, Query(ge=1, le=10)] = 5,
) -> JSONResponse:
    """Candidate seed works for a free-text / title / DOI query."""
    try:
        candidates = client.search_works(q, limit=limit)
    except TreeXivError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAlex error: {exc}") from exc
    return JSONResponse(
        [
            {
                "id": w.id,
                "title": w.title,
                "publication_year": w.publication_year,
                "cited_by_count": w.cited_by_count,
                "authors": w.authors,
                "venue": w.venue,
                "doi": w.doi,
            }
            for w in candidates
        ]
    )


@app.post("/api/run")
def run(_: AuthDep, client: ClientDep, req: RunRequest) -> JSONResponse:
    """Expand -> filter -> render for a resolved work ID. Returns the HTML plus stats."""
    settings = _settings_for(req)
    try:
        seed_work = client.get_work(req.work_id)
        expansion = expand_two_hop(client, settings, seed_work, sample_seed=req.sample_seed)
        filtered = filter_by_idea(expansion, req.idea, top_k=settings.bm25_top_k)
        tmp_dir = Path(tempfile.mkdtemp(prefix="treexiv-"))
        try:
            html_path = render_html(
                filtered, tmp_dir / "tree.html", title=f"TreeXiv · {seed_work.title}"
            )
            html = html_path.read_text(encoding="utf-8")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except TreeXivError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAlex error: {exc}") from exc

    return JSONResponse(
        {
            "seed_id": seed_work.id,
            "seed_title": seed_work.title,
            "seed_year": seed_work.publication_year,
            "expanded": len(expansion.nodes),
            "kept": len(filtered.nodes),
            "edges": len(filtered.edges),
            "truncated": expansion.truncated,
            "html": html,
        }
    )
