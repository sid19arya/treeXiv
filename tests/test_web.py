"""Tests for the private web front-end (`treexiv.web`).

The OpenAlex client dependency is overridden with one wired to an
`httpx.MockTransport`, so no real network traffic happens and no respx
global patching collides with the in-process TestClient transport.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from tests.conftest import make_work_payload  # noqa: E402
from tests.test_webauth import add_invite  # noqa: E402
from treexiv import web  # noqa: E402
from treexiv.config import Settings  # noqa: E402
from treexiv.openalex import OpenAlexClient  # noqa: E402
from treexiv.sources.s2 import SemanticScholarClient  # noqa: E402
from treexiv.webauth import SESSION_COOKIE, UserStore, open_store  # noqa: E402

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'web.sqlite3'}")
    monkeypatch.setenv("TREEXIV_SESSION_SECRET", "test-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def _store() -> UserStore:
    return open_store(os.environ["DATABASE_URL"])


def _sign_up(tc: TestClient, email: str = "ada@example.com") -> None:
    code = f"invite-for-{email}"
    add_invite(_store(), code)
    r = tc.post(
        "/auth/signup", json={"invite": code, "email": email, "password": "a long password"}
    )
    assert r.status_code == 200, r.text


@pytest.fixture
def anon() -> TestClient:
    """A browser with no session."""
    return TestClient(web.app)


@pytest.fixture
def client() -> TestClient:
    """A browser signed in to a fresh account."""
    tc = TestClient(web.app)
    _sign_up(tc)
    return tc


def _use_openalex(handler: Handler) -> None:
    def _dep() -> OpenAlexClient:
        http = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://api.openalex.org"
        )
        return OpenAlexClient(Settings(max_retries=1), http_client=http)

    web.app.dependency_overrides[web._openalex_client] = _dep


def _use_s2(handler: Handler) -> None:
    def _dep() -> SemanticScholarClient:
        http = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="https://api.semanticscholar.org/graph/v1",
        )
        return SemanticScholarClient(
            Settings(max_retries=1, s2_min_interval=0.0), http_client=http
        )

    web.app.dependency_overrides[web._s2_client] = _dep


def _use_openrouter(handler: Handler) -> None:
    def _dep() -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="https://openrouter.ai/api/v1",
        )

    web.app.dependency_overrides[web._openrouter_http] = _dep


@pytest.fixture(autouse=True)
def _clear_overrides() -> None:
    yield
    web.app.dependency_overrides.clear()


def test_health_is_open_without_auth(anon: TestClient) -> None:
    assert anon.get("/health").json() == {"status": "ok"}


def test_public_pages_need_no_account(anon: TestClient) -> None:
    for path in ("/", "/login", "/signup", "/site.css"):
        assert anon.get(path).status_code == 200, path
    assert "TreeXiv" in anon.get("/").text


def test_example_tree_is_public(anon: TestClient) -> None:
    r = anon.get("/example")
    assert r.status_code == 200
    assert "Attention Is All You Need" in r.text


def test_public_pages_load_with_accounts_unconfigured(
    anon: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("TREEXIV_SESSION_SECRET", "")
    assert anon.get("/").status_code == 200
    assert anon.get("/login").status_code == 200
    r = anon.post("/auth/login", json={"email": "ada@example.com", "password": "x"})
    assert r.status_code == 503


def test_app_redirects_to_login_when_signed_out(anon: TestClient) -> None:
    r = anon.get("/app", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_app_loads_when_signed_in(client: TestClient) -> None:
    assert client.get("/app").status_code == 200
    r = client.get("/login", follow_redirects=False)
    assert r.headers["location"] == "/app"


def test_api_rejects_signed_out_requests_without_touching_openalex(anon: TestClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream call without an account")

    _use_openalex(handler)
    r = anon.get("/api/search", params={"q": "attention"})
    assert r.status_code == 401
    assert "www-authenticate" not in r.headers  # no browser Basic-Auth popup
    assert anon.post("/api/run", json={"work_id": "W1", "idea": "x"}).status_code == 401


def test_forged_session_is_rejected(anon: TestClient) -> None:
    anon.cookies.set(SESSION_COOKIE, "1.9999999999.deadbeef")
    assert anon.get("/api/me").status_code == 401


def test_me_login_logout(client: TestClient, anon: TestClient) -> None:
    assert client.get("/api/me").json() == {"email": "ada@example.com"}
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/api/me").status_code == 401

    bad = anon.post("/auth/login", json={"email": "ada@example.com", "password": "nope"})
    assert bad.status_code == 401
    ok = anon.post(
        "/auth/login", json={"email": "ADA@example.com", "password": "a long password"}
    )
    assert ok.status_code == 200, ok.text
    assert anon.get("/api/me").json() == {"email": "ada@example.com"}


def test_session_cookie_flags(anon: TestClient) -> None:
    add_invite(_store(), "invite-https")
    r = anon.post(
        "/auth/signup",
        json={"invite": "invite-https", "email": "bob@example.com", "password": "a long password"},
        headers={"x-forwarded-proto": "https"},
    )
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "secure" in cookie


def test_signup_errors(anon: TestClient, client: TestClient) -> None:
    r = anon.post(
        "/auth/signup",
        json={"invite": "not-a-real-invite", "email": "x@example.com", "password": "long enough!"},
    )
    assert r.status_code == 400
    assert "invite" in r.json()["detail"]

    add_invite(_store(), "invite-dupe")
    r = anon.post(
        "/auth/signup",
        json={"invite": "invite-dupe", "email": "ada@example.com", "password": "a long password"},
    )
    assert r.status_code == 409

    r = anon.post(
        "/auth/signup",
        json={"invite": "invite-dupe", "email": "new@example.com", "password": "short"},
    )
    assert r.status_code == 400
    assert "10 characters" in r.json()["detail"]


def test_login_is_throttled(client: TestClient, anon: TestClient) -> None:
    web._throttle.reset("ada@example.com")
    try:
        for _ in range(web._throttle.limit):
            anon.post("/auth/login", json={"email": "ada@example.com", "password": "nope"})
        r = anon.post(
            "/auth/login", json={"email": "ada@example.com", "password": "a long password"}
        )
        assert r.status_code == 429
    finally:
        web._throttle.reset("ada@example.com")


def test_deleted_account_loses_access(client: TestClient) -> None:
    with _store().connection() as conn:
        conn.execute("DELETE FROM users")
    assert client.get("/api/me").status_code == 401
    assert client.get("/app", follow_redirects=False).status_code == 303


def test_search_returns_candidates(client: TestClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works"
        return httpx.Response(
            200, json={"results": [make_work_payload("W1", "Attention Is All You Need")]}
        )

    _use_openalex(handler)
    r = client.get("/api/search", params={"q": "attention"})
    assert r.status_code == 200
    body = r.json()
    assert body[0]["id"] == "W1"
    assert body[0]["title"] == "Attention Is All You Need"


def test_run_returns_html_and_stats(client: TestClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/works/W1":
            return httpx.Response(
                200, json=make_work_payload("W1", "Attention Is All You Need", referenced_works=[])
            )
        return httpx.Response(200, json={"results": []})

    _use_openalex(handler)
    r = client.post(
        "/api/run",
        json={"work_id": "W1", "idea": "self-attention for sequence modeling"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seed_id"] == "W1"
    assert body["kept"] >= 1
    assert "<!doctype html>" in body["html"].lower()


def test_identify_returns_guess(client: TestClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == "Bearer sk-or-test"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"search_query": "Attention Is All You Need", '
                            '"title": "Attention Is All You Need", "confidence": "high"}',
                        }
                    }
                ]
            },
        )

    _use_openrouter(handler)
    r = client.post(
        "/api/identify",
        json={"description": "the 2017 transformer paper"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["search_query"] == "Attention Is All You Need"
    assert body["confidence"] == "high"


def test_identify_requires_auth(anon: TestClient) -> None:
    r = anon.post("/api/identify", json={"description": "something"})
    assert r.status_code == 401


def test_identify_501_when_openrouter_key_unset(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    r = client.post(
        "/api/identify", json={"description": "the 2017 transformer paper"}
    )
    assert r.status_code == 501
    assert "OPENROUTER_API_KEY" in r.json()["detail"]


def test_identify_rejects_too_short_description(client: TestClient) -> None:
    r = client.post("/api/identify", json={"description": "x"})
    assert r.status_code == 422


def test_run_clamps_oversized_caps(client: TestClient) -> None:
    r = client.post(
        "/api/run",
        json={"work_id": "W1", "idea": "x", "total_cap": 999999}
    )
    # 999999 is past the Field ceiling -> 422 before any OpenAlex call.
    assert r.status_code == 422


def _curation_reply() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "clusters": [
                                {"id": 1, "name": "Roots", "role": "ancestor",
                                 "summary": "Where it came from."}
                            ],
                            "keep": [
                                {"i": 1, "cluster": 1, "importance": 5, "why": "the origin"}
                            ],
                            "dropped_summary": "cut the noise",
                        }
                    )
                }
            }
        ]
    }


def _narrative_reply() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "headline": "A short lineage.",
                            "overview": "Where it came from.\n\nWhere it went.",
                            "beats": [{"title": "The turn", "text": "It changed.", "papers": [1]}],
                        }
                    )
                }
            }
        ]
    }


def _seed_and_one_reference(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/works/W1":
        return httpx.Response(
            200, json=make_work_payload("W1", "Seed paper", referenced_works=["R1"])
        )
    return httpx.Response(
        200, json={"results": [make_work_payload("R1", "Cited work", cited_by_count=9)]}
    )


def test_run_curates_and_returns_strands_and_headline(client: TestClient) -> None:
    _use_openalex(_seed_and_one_reference)
    replies = iter([_curation_reply(), _narrative_reply()])

    def llm(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(replies))

    _use_openrouter(llm)
    r = client.post(
        "/api/run",
        json={"work_id": "W1", "idea": "citation lineage"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["curation"] == "llm"
    assert body["clusters"][0]["name"] == "Roots"
    assert body["headline"] == "A short lineage."
    assert body["warnings"] == []


def test_run_reports_a_fallback_to_the_browser(client: TestClient) -> None:
    """There is no stderr in a browser: a run that quietly degraded to the
    keyword filter has to say so in the response."""
    _use_openalex(_seed_and_one_reference)
    _use_openrouter(lambda request: httpx.Response(500, text="model down"))
    r = client.post(
        "/api/run", json={"work_id": "W1", "idea": "citation lineage"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["curation"] == "bm25"
    assert body["clusters"] == []
    assert any("falling back" in w for w in body["warnings"])


def test_run_bm25_mode_never_calls_the_llm(client: TestClient) -> None:
    _use_openalex(_seed_and_one_reference)

    def llm(request: httpx.Request) -> httpx.Response:
        raise AssertionError("curation must not run in bm25 mode")

    _use_openrouter(llm)
    r = client.post(
        "/api/run",
        json={"work_id": "W1", "idea": "citation lineage", "curation": "bm25"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["curation"] == "bm25"


def test_run_clamps_the_curation_node_cap(client: TestClient) -> None:
    r = client.post(
        "/api/run",
        json={"work_id": "W1", "idea": "x", "max_nodes": 5000}
    )
    assert r.status_code == 422


def test_search_prefers_the_s2_title_match(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("TREEXIV_SOURCE", "auto")
    exact = make_work_payload("W_EXACT", "The Exact Paper", cited_by_count=500)
    exact["doi"] = "https://doi.org/10.0/exact"

    def openalex(request: httpx.Request) -> httpx.Response:
        # The DOI cross-walk and the relevance search hit the same path; the
        # filter param is what distinguishes them (and it arrives URL-encoded).
        if request.url.params.get("filter", "").startswith("doi:"):
            return httpx.Response(200, json={"results": [exact]})
        return httpx.Response(
            200, json={"results": [make_work_payload("W_OTHER", "Loosely related")]}
        )

    _use_openalex(openalex)
    _use_s2(
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "s1",
                        "externalIds": {"DOI": "10.0/exact"},
                        "title": "The Exact Paper",
                        "year": 2024,
                        "citationCount": 500,
                        "authors": [{"name": "Ada Example"}],
                    }
                ]
            },
        )
    )
    r = client.get("/api/search", params={"q": "The Exact Paper"})
    assert r.status_code == 200, r.text
    candidates = r.json()
    assert candidates[0]["id"] == "W_EXACT"
    assert candidates[0]["matched_by"] == "semantic_scholar"
