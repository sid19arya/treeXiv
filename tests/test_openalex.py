import dataclasses

import httpx
import pytest
import respx

from tests.conftest import make_work_payload
from treexiv.exceptions import OpenAlexAPIError
from treexiv.openalex import OpenAlexClient


@pytest.fixture
def client(settings) -> OpenAlexClient:
    return OpenAlexClient(settings)


@respx.mock
def test_search_works_parses_results(client: OpenAlexClient) -> None:
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200, json={"results": [make_work_payload("W1", "Recursive LMs", cited_by_count=10)]}
        )
    )
    results = client.search_works("recursive language models", limit=5)
    assert route.called
    assert len(results) == 1
    assert results[0].id == "W1"
    assert results[0].title == "Recursive LMs"
    request = route.calls[0].request
    assert request.url.params["search"] == "recursive language models"
    assert request.url.params["per_page"] == "5"


@respx.mock
def test_search_works_includes_polite_pool_params(client: OpenAlexClient) -> None:
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    client.search_works("query")
    assert route.calls[0].request.url.params["mailto"] == "test@example.com"


@respx.mock
def test_get_work_normalizes_id_in_path(client: OpenAlexClient) -> None:
    route = respx.get("https://api.openalex.org/works/W123").mock(
        return_value=httpx.Response(200, json=make_work_payload("W123", "A Paper"))
    )
    work = client.get_work("https://openalex.org/W123")
    assert route.called
    assert work.id == "W123"


@respx.mock
def test_get_works_by_ids_batches_over_limit(client: OpenAlexClient) -> None:
    ids = [f"W{i}" for i in range(120)]
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [make_work_payload("W1", "P1")]})
    )
    client.get_works_by_ids(ids)
    # 120 ids at batch size 50 -> 3 requests
    assert route.call_count == 3


@respx.mock
def test_get_works_by_ids_returns_keyed_by_id(client: OpenAlexClient) -> None:
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    make_work_payload("W1", "P1"),
                    make_work_payload("W2", "P2"),
                ]
            },
        )
    )
    result = client.get_works_by_ids(["W1", "W2"])
    assert set(result.keys()) == {"W1", "W2"}


@respx.mock
def test_get_citing_works_top_cited_sorts_server_side(client: OpenAlexClient) -> None:
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    client.get_citing_works("W1", limit=50, strategy="top_cited")
    params = route.calls[0].request.url.params
    assert params["filter"] == "cites:W1"
    assert params["sort"] == "cited_by_count:desc"
    assert "sample" not in params


@respx.mock
def test_get_citing_works_random_uses_sample_param(client: OpenAlexClient) -> None:
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    client.get_citing_works("W1", limit=50, strategy="random", sample_seed=7)
    params = route.calls[0].request.url.params
    assert params["sample"] == "50"
    assert params["seed"] == "7"
    assert "sort" not in params


@respx.mock
def test_retries_on_retryable_status_then_succeeds(settings, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    route = respx.get("https://api.openalex.org/works").mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"results": []})]
    )
    retrying_client = OpenAlexClient(dataclasses.replace(settings, max_retries=3))
    results = retrying_client.search_works("q")
    assert results == []
    assert route.call_count == 2


@respx.mock
def test_raises_on_non_retryable_error(client: OpenAlexClient) -> None:
    respx.get("https://api.openalex.org/works").mock(return_value=httpx.Response(404, text="nope"))
    with pytest.raises(OpenAlexAPIError):
        client.search_works("q")


@respx.mock
def test_raises_after_exhausting_retries(client: OpenAlexClient, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    respx.get("https://api.openalex.org/works").mock(return_value=httpx.Response(500))
    with pytest.raises(OpenAlexAPIError):
        client.search_works("q")


@respx.mock
def test_retries_on_transport_error_then_succeeds(settings, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    route = respx.get("https://api.openalex.org/works").mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={"results": []})]
    )
    retrying_client = OpenAlexClient(dataclasses.replace(settings, max_retries=3))
    results = retrying_client.search_works("q")
    assert results == []
    assert route.call_count == 2


@respx.mock
def test_raises_after_exhausting_retries_on_transport_error(settings, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    respx.get("https://api.openalex.org/works").mock(side_effect=httpx.ConnectError("boom"))
    failing_client = OpenAlexClient(dataclasses.replace(settings, max_retries=2))
    with pytest.raises(OpenAlexAPIError):
        failing_client.search_works("q")
