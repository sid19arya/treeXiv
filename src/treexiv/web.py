"""Web front-end for the treexiv pipeline: a public landing page, and the
pipeline itself behind invite-only accounts.

This is deliberately outside the Phase 0 CLI/skill scope (see `CLAUDE.md`):
opt-in via the `web` extra, reusing the exact same package functions the CLI
does. It exists only to run the pipeline behind a browser form on Render's
free tier — nothing here changes the core package.

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

Public: ``/`` (landing page), ``/example`` (one pre-rendered tree), the
sign-in/sign-up pages, and ``/health``. Everything that runs the pipeline —
``/app`` and every ``/api/*`` route — needs a signed-in account (see
`webauth.py`): without one ``/app`` redirects to ``/login`` and the API
answers 401, so no OpenAlex or LLM call is made for a stranger. Accounts need
``DATABASE_URL`` and ``TREEXIV_SESSION_SECRET``; with either unset the
account routes answer 503 and the public pages still load.

Run locally (any string works as the secret in dev):
``DATABASE_URL=sqlite:///web.sqlite3 TREEXIV_SESSION_SECRET=dev \\``
``uv run --extra web uvicorn treexiv.web:app --reload``
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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
from treexiv.webauth import (
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    AccountError,
    EmailTaken,
    InvalidInvite,
    LoginThrottle,
    StoreUnavailable,
    User,
    UserStore,
    open_store,
    read_session,
    sign_session,
)

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

_ASSETS = resources.files("treexiv") / "webassets"


def _asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


_INDEX_HTML = _asset("index.html")
_LANDING_HTML = _asset("landing.html")
_LOGIN_HTML = _asset("login.html")
_SIGNUP_HTML = _asset("signup.html")
_SITE_CSS = _asset("site.css")
# The public example tree is a committed render, not a live run: strangers
# get to see real output without spending an LLM call.
_EXAMPLE_HTML = _asset("example.html") if (_ASSETS / "example.html").is_file() else None

app = FastAPI(title="treexiv", docs_url=None, redoc_url=None, openapi_url=None)
_throttle = LoginThrottle()


def _user_store() -> UserStore:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Accounts are not configured (DATABASE_URL).",
        )
    return open_store(url)


@app.exception_handler(StoreUnavailable)
def _store_unavailable(request: Request, exc: StoreUnavailable) -> JSONResponse:
    return JSONResponse(
        {"detail": "The accounts database is unavailable — try again shortly."},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


StoreDep = Annotated[UserStore, Depends(_user_store)]


def _session_secret() -> str:
    secret = os.environ.get("TREEXIV_SESSION_SECRET")
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Accounts are not configured (TREEXIV_SESSION_SECRET).",
        )
    return secret


def _current_user(request: Request, store: StoreDep) -> User | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    user_id = read_session(token, _session_secret())
    return store.get_user(user_id) if user_id is not None else None


def _require_user(user: Annotated[User | None, Depends(_current_user)]) -> User:
    """No session, a forged one, or a deleted account: 401, and nothing runs."""
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in first.")
    return user


AuthDep = Annotated[User, Depends(_require_user)]


def _signed_in(request: Request) -> bool:
    """For pages that only choose between the app and the sign-in form —
    never an error, even with accounts unconfigured.

    A validly signed cookie counts as signed in while the database is
    unreachable: the page loads, and its API calls (which do check the
    account) wait out the outage instead of bouncing the user to /login.
    """
    token = request.cookies.get(SESSION_COOKIE)
    secret = os.environ.get("TREEXIV_SESSION_SECRET")
    if not token or not secret:
        return False
    user_id = read_session(token, secret)
    if user_id is None:
        return False
    try:
        return _user_store().get_user(user_id) is not None
    except StoreUnavailable:
        return True
    except HTTPException:
        return False


def _start_session(request: Request, response: Response, user: User) -> None:
    # Render terminates TLS at its proxy, so the app itself sees plain http;
    # the forwarded scheme is what the browser actually used.
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    response.set_cookie(
        SESSION_COOKIE,
        sign_session(user.id, _session_secret()),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        secure=scheme == "https",
        samesite="lax",
    )


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


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class SignupRequest(BaseModel):
    invite: str = Field(min_length=8, max_length=200)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


@app.get("/health")
def health() -> dict[str, str]:
    """Open, unauthenticated — Render pings this and it exposes nothing."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    return HTMLResponse(_LANDING_HTML)


@app.get("/site.css")
def site_css() -> Response:
    return Response(_SITE_CSS, media_type="text/css")


@app.get("/example", response_class=HTMLResponse)
def example() -> HTMLResponse:
    if _EXAMPLE_HTML is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No example yet.")
    return HTMLResponse(_EXAMPLE_HTML)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> Response:
    if _signed_in(request):
        return RedirectResponse("/app", status_code=status.HTTP_303_SEE_OTHER)
    return HTMLResponse(_LOGIN_HTML)


@app.get("/signup", response_class=HTMLResponse)
def signup_page() -> HTMLResponse:
    return HTMLResponse(_SIGNUP_HTML)


@app.get("/app", response_class=HTMLResponse)
def index(request: Request) -> Response:
    if not _signed_in(request):
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    return HTMLResponse(_INDEX_HTML)


@app.post("/auth/login")
def login(request: Request, store: StoreDep, req: LoginRequest) -> JSONResponse:
    key = req.email.strip().lower()
    if _throttle.blocked(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed sign-ins for that email — wait a few minutes.",
        )
    user = store.authenticate(req.email, req.password)
    if user is None:
        _throttle.failed(key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong email or password."
        )
    _throttle.reset(key)
    response = JSONResponse({"email": user.email})
    _start_session(request, response, user)
    return response


@app.post("/auth/signup")
def signup(request: Request, store: StoreDep, req: SignupRequest) -> JSONResponse:
    try:
        user = store.signup(req.invite, req.email, req.password)
    except InvalidInvite as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmailTaken as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except AccountError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    response = JSONResponse({"email": user.email})
    _start_session(request, response, user)
    return response


@app.post("/auth/logout")
def logout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/api/me")
def me(user: AuthDep) -> dict[str, str]:
    return {"email": user.email}


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
