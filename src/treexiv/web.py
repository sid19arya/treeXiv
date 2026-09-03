"""Private web front-end for the treexiv pipeline.

This is deliberately outside the Phase 0 CLI/skill scope (see `CLAUDE.md`):
one module, opt-in via the `web` extra, reusing the exact same package
functions the CLI does. It exists only to run the pipeline behind a browser
form on Render's free tier — nothing here changes the core package.

Three routes reach past OpenAlex, all of them optional and all of them
degrading rather than failing: ``/api/identify`` (Step 0, turns a vague
description into a seed-paper lead, 501 without ``OPENROUTER_API_KEY``),
``/api/search`` (Semantic Scholar's title matcher ahead of OpenAlex relevance
search), and ``/api/run`` (LLM curation and the lineage narrative, falling
back to the BM25 filter without a key). Anything that degrades says so in the
response's ``warnings``, since a browser user has no stderr to read.

Note that a curated run is *slow* — minutes, not seconds, dominated by the
curation call. ``_WEB_CURATION_PREFILTER`` trims the shortlist to keep that
in hand, and a deployment behind a proxy with a request timeout should either
raise that timeout or run with ``curation: "bm25"``.

Every route except ``/health`` sits behind HTTP Basic Auth
(``TREEXIV_WEB_USER`` / ``TREEXIV_WEB_PASSWORD`` env vars). Without valid
credentials a request gets a 401 and the pipeline never runs — no OpenAlex
calls, no data. ``/health`` is left open so Render's health check works.

Run locally:  ``uv run --extra web uvicorn treexiv.web:app --reload``
"""

from __future__ import annotations

import dataclasses
import os
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from treexiv.config import Settings
from treexiv.exceptions import TreeXivError
from treexiv.expand import expand_two_hop
from treexiv.filtering import build_graph
from treexiv.models import Work
from treexiv.openalex import OpenAlexClient
from treexiv.render import render_html
from treexiv.seed_llm import identify_seed
from treexiv.sources.enrich import enrich_expansion, find_seed
from treexiv.sources.s2 import SemanticScholarClient

# Hard ceilings on caller-supplied knobs. Even an authenticated request (or a
# leaked credential) can't turn one call into a multi-thousand-request
# OpenAlex crawl. These are above the usual config.py defaults, not a
# replacement for them.
_MAX_TOTAL_CAP = 500
_MAX_FANOUT_CAP = 100
_MAX_TOP_K = 100
_MAX_CURATION_NODES = 40
# Curation reads one abstract per shortlisted paper, and its wall time scales
# with that. The CLI's 120 is fine for a terminal you can leave running; a
# browser request waiting on a hosted worker is not, so the web caps it lower.
_WEB_CURATION_PREFILTER = 70

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


def _openrouter_http() -> Iterator[httpx.Client]:
    """Request-scoped HTTP client for the OpenRouter calls (Step 0 seed
    identification, and the curation/narrative pass). Overridden in tests."""
    settings = Settings.from_env()
    with httpx.Client(
        base_url=settings.openrouter_base_url, timeout=settings.llm_timeout_seconds
    ) as client:
        yield client


OpenRouterDep = Annotated[httpx.Client, Depends(_openrouter_http)]


def _s2_client() -> Iterator[SemanticScholarClient | None]:
    """Request-scoped Semantic Scholar client, or None when S2 is switched off.

    Injected rather than constructed inline for the same reason as the others:
    the tests swap in a mock transport, and this module's tests deliberately
    avoid respx's global patching, which collides with the in-process
    TestClient transport.
    """
    settings = Settings.from_env()
    if settings.source_mode == "openalex":
        yield None
        return
    with SemanticScholarClient(settings) as client:
        yield client


S2Dep = Annotated[SemanticScholarClient | None, Depends(_s2_client)]


class IdentifyRequest(BaseModel):
    description: str = Field(min_length=3, max_length=2000)
    web: bool | None = None


class RunRequest(BaseModel):
    work_id: str = Field(min_length=1)
    idea: str = Field(min_length=1)
    total_cap: int | None = Field(default=None, ge=1, le=_MAX_TOTAL_CAP)
    fanout_cap: int | None = Field(default=None, ge=1, le=_MAX_FANOUT_CAP)
    top_k: int | None = Field(default=None, ge=1, le=_MAX_TOP_K)
    sampling: str | None = None
    sample_seed: int | None = None
    curation: str | None = None
    max_nodes: int | None = Field(default=None, ge=1, le=_MAX_CURATION_NODES)
    narrative: bool | None = None


def _settings_for(req: RunRequest) -> Settings:
    """Env settings with the request's caps applied, each clamped to a ceiling."""
    base = Settings.from_env()
    return dataclasses.replace(
        base,
        total_corpus_cap=min(req.total_cap or base.total_corpus_cap, _MAX_TOTAL_CAP),
        per_node_fanout_cap=min(req.fanout_cap or base.per_node_fanout_cap, _MAX_FANOUT_CAP),
        sampling_strategy=(req.sampling or base.sampling_strategy),  # type: ignore[arg-type]
        bm25_top_k=min(req.top_k or base.bm25_top_k, _MAX_TOP_K),
        curation_mode=(req.curation or base.curation_mode),  # type: ignore[arg-type]
        curation_max_nodes=min(
            req.max_nodes or base.curation_max_nodes, _MAX_CURATION_NODES
        ),
        curation_prefilter=min(base.curation_prefilter, _WEB_CURATION_PREFILTER),
        narrative=base.narrative if req.narrative is None else req.narrative,
        cache_dir=None,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Open, unauthenticated — Render pings this and it exposes nothing."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(_: AuthDep) -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML)


@app.post("/api/identify")
def identify(_: AuthDep, http: OpenRouterDep, req: IdentifyRequest) -> JSONResponse:
    """Step 0: guess which paper a free-text description refers to (OpenRouter).

    Returns the same shape as `treexiv identify-seed` — a lead, not a
    resolution; the caller still runs `/api/search` on `search_query`.
    """
    settings = Settings.from_env()
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Seed identification is not configured (OPENROUTER_API_KEY unset).",
        )
    try:
        guess = identify_seed(
            req.description, settings, web_search=req.web, http_client=http
        )
    except TreeXivError as exc:
        raise HTTPException(status_code=502, detail=f"Identify error: {exc}") from exc
    return JSONResponse(guess.to_dict())


def _candidate(work: Work, matched_by: str) -> dict:
    return {
        "id": work.id,
        "title": work.title,
        "publication_year": work.publication_year,
        "cited_by_count": work.cited_by_count,
        "authors": work.authors,
        "venue": work.venue,
        "doi": work.doi,
        "matched_by": matched_by,
    }


@app.get("/api/search")
def search(
    _: AuthDep,
    client: ClientDep,
    s2: S2Dep,
    q: Annotated[str, Query(min_length=2, max_length=300)],
    limit: Annotated[int, Query(ge=1, le=10)] = 5,
) -> JSONResponse:
    """Candidate seed works for a free-text / title / DOI query.

    Semantic Scholar's title matcher runs first and its match, resolved into
    OpenAlex by DOI, leads the list — same as the CLI's `search-seed`. Every
    ID returned is still an OpenAlex work ID.
    """
    settings = Settings.from_env()
    candidates: list[dict] = []
    seen: set[str] = set()
    if s2 is not None:
        matched = _s2_match(q, settings, client, s2)
        if matched is not None:
            candidates.append(matched)
            seen.add(matched["id"])
    try:
        found = client.search_works(q, limit=limit)
    except TreeXivError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAlex error: {exc}") from exc
    for work in found:
        if work.id not in seen:
            candidates.append(_candidate(work, "openalex_search"))
            seen.add(work.id)
    return JSONResponse(candidates)


def _s2_match(
    query: str, settings: Settings, client: OpenAlexClient, s2: SemanticScholarClient
) -> dict | None:
    """S2's best title match, resolved into an OpenAlex work — None if any step
    doesn't pan out, since the OpenAlex search still runs either way."""
    lookup = find_seed(query, settings, client=s2)
    doi = lookup.work.normalized_doi if lookup else None
    if not doi:
        return None
    try:
        work = client.get_works_by_doi([doi]).get(doi)
    except TreeXivError:
        return None
    return _candidate(work, "semantic_scholar") if work else None


@app.post("/api/run")
def run(
    _: AuthDep, client: ClientDep, http: OpenRouterDep, s2: S2Dep, req: RunRequest
) -> JSONResponse:
    """Expand -> curate -> narrate -> render. Returns the HTML plus stats.

    Warnings from the pipeline (an LLM fallback, an S2 outage) are collected
    and returned rather than only logged: on the web there is no stderr for
    the user to read, and "this came back as a plain keyword filter" is
    something they should be told.
    """
    settings = _settings_for(req)
    warnings: list[str] = []
    try:
        seed_work = client.get_work(req.work_id)
        expansion = expand_two_hop(client, settings, seed_work, sample_seed=req.sample_seed)
        enrichment = enrich_expansion(
            expansion, seed_work, client, settings, s2_client=s2, on_warning=warnings.append
        )
        filtered = build_graph(
            expansion, req.idea, settings, http_client=http, on_warning=warnings.append
        )
        tmp_dir = Path(tempfile.mkdtemp(prefix="treexiv-"))
        try:
            html_path = render_html(
                filtered, tmp_dir / "tree.html", title=f"TreeXiv · {seed_work.title}"
            )
            html = html_path.read_text(encoding="utf-8")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except TreeXivError as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline error: {exc}") from exc

    narrative = filtered.narrative
    return JSONResponse(
        {
            "seed_id": seed_work.id,
            "seed_title": seed_work.title,
            "seed_year": seed_work.publication_year,
            "expanded": len(expansion.nodes),
            "kept": len(filtered.nodes),
            "edges": len(filtered.edges),
            "truncated": expansion.truncated,
            "curation": filtered.curation,
            "clusters": [
                {"name": c.name, "role": c.role, "summary": c.summary}
                for c in filtered.clusters
            ],
            "headline": narrative.headline if narrative else "",
            "intents_labelled": enrichment.edges_annotated,
            "warnings": warnings,
            "html": html,
        }
    )
