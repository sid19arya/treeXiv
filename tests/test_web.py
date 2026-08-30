"""Tests for the private web front-end (`treexiv.web`).

The OpenAlex client dependency is overridden with one wired to an
`httpx.MockTransport`, so no real network traffic happens and no respx
global patching collides with the in-process TestClient transport.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from tests.conftest import make_work_payload  # noqa: E402
from treexiv import web  # noqa: E402
from treexiv.config import Settings  # noqa: E402
from treexiv.openalex import OpenAlexClient  # noqa: E402

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TREEXIV_WEB_USER", "u")
    monkeypatch.setenv("TREEXIV_WEB_PASSWORD", "p")


@pytest.fixture
def client() -> TestClient:
    return TestClient(web.app)


def _use_openalex(handler: Handler) -> None:
    def _dep() -> OpenAlexClient:
        http = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://api.openalex.org"
        )
        return OpenAlexClient(Settings(max_retries=1), http_client=http)

    web.app.dependency_overrides[web._openalex_client] = _dep


@pytest.fixture(autouse=True)
def _clear_overrides() -> None:
    yield
    web.app.dependency_overrides.clear()


def test_health_is_open_without_auth(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_index_requires_auth(client: TestClient) -> None:
    assert client.get("/").status_code == 401
    assert client.get("/", auth=("u", "p")).status_code == 200


def test_api_rejects_wrong_password(client: TestClient) -> None:
    r = client.get("/api/search", params={"q": "attention"}, auth=("u", "nope"))
    assert r.status_code == 401


def test_missing_auth_env_returns_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TREEXIV_WEB_USER", raising=False)
    monkeypatch.delenv("TREEXIV_WEB_PASSWORD", raising=False)
    assert client.get("/", auth=("u", "p")).status_code == 503


def test_search_returns_candidates(client: TestClient) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works"
        return httpx.Response(
            200, json={"results": [make_work_payload("W1", "Attention Is All You Need")]}
        )

    _use_openalex(handler)
    r = client.get("/api/search", params={"q": "attention"}, auth=("u", "p"))
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
        json={"work_id": "W1", "idea": "self-attention for sequence modeling"},
        auth=("u", "p"),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seed_id"] == "W1"
    assert body["kept"] >= 1
    assert "<!doctype html>" in body["html"].lower()


def test_run_clamps_oversized_caps(client: TestClient) -> None:
    r = client.post(
        "/api/run",
        json={"work_id": "W1", "idea": "x", "total_cap": 999999},
        auth=("u", "p"),
    )
    # 999999 is past the Field ceiling -> 422 before any OpenAlex call.
    assert r.status_code == 422
