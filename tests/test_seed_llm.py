"""Tests for Step 0 (`identify-seed`). OpenRouter is mocked with respx; no real
LLM traffic happens in the suite."""

from __future__ import annotations

import dataclasses
import json

import httpx
import pytest
import respx

from treexiv.exceptions import SeedIdentificationError
from treexiv.seed_llm import identify_seed

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@pytest.fixture
def llm_settings(settings):
    return dataclasses.replace(
        settings, openrouter_api_key="sk-or-test", openrouter_model="z-ai/glm-5.3-flash"
    )


def _completion(content: str, *, annotations: list | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if annotations is not None:
        message["annotations"] = annotations
    return {"choices": [{"message": message}]}


_GOOD_JSON = json.dumps(
    {
        "search_query": "Attention Is All You Need",
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "doi": None,
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "confidence": "high",
        "reasoning": "The description matches the transformer paper.",
        "alternatives": [
            {
                "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
                "note": "Earlier attention work, RNN-based.",
            }
        ],
    }
)


@respx.mock
def test_identify_seed_parses_structured_reply(llm_settings) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion(_GOOD_JSON))
    )
    guess = identify_seed("the paper that introduced the transformer architecture", llm_settings)

    assert route.called
    assert guess.search_query == "Attention Is All You Need"
    assert guess.arxiv_id == "1706.03762"
    assert guess.year == 2017
    assert guess.confidence == "high"
    assert guess.authors[0] == "Ashish Vaswani"
    assert guess.alternatives[0]["note"]
    assert guess.model == "z-ai/glm-5.3-flash"


@respx.mock
def test_identify_seed_sends_auth_and_model_and_web_plugin(llm_settings) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion(_GOOD_JSON))
    )
    identify_seed("something", llm_settings, web_search=True)

    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer sk-or-test"
    body = json.loads(request.content)
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["plugins"] == [{"id": "web", "max_results": 5}]


@respx.mock
def test_identify_seed_no_web_omits_plugin(llm_settings) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion(_GOOD_JSON))
    )
    identify_seed("something", llm_settings, web_search=False)
    assert "plugins" not in json.loads(route.calls[0].request.content)


@respx.mock
def test_identify_seed_strips_code_fences(llm_settings) -> None:
    fenced = f"```json\n{_GOOD_JSON}\n```"
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_completion(fenced)))
    guess = identify_seed("x", llm_settings)
    assert guess.title == "Attention Is All You Need"


@respx.mock
def test_identify_seed_extracts_json_from_surrounding_prose(llm_settings) -> None:
    messy = f"Here is the answer:\n{_GOOD_JSON}\nHope that helps!"
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_completion(messy)))
    guess = identify_seed("x", llm_settings)
    assert guess.search_query == "Attention Is All You Need"


@respx.mock
def test_identify_seed_collects_web_sources(llm_settings) -> None:
    annotations = [
        {"type": "url_citation", "url_citation": {"url": "https://arxiv.org/abs/1706.03762"}},
        {"type": "url_citation", "url_citation": {"url": "https://arxiv.org/abs/1706.03762"}},
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion(_GOOD_JSON, annotations=annotations))
    )
    guess = identify_seed("x", llm_settings)
    assert guess.sources == ["https://arxiv.org/abs/1706.03762"]


def test_identify_seed_requires_api_key(settings) -> None:
    with pytest.raises(SeedIdentificationError, match="OPENROUTER_API_KEY"):
        identify_seed("x", settings)


def test_identify_seed_rejects_empty_description(llm_settings) -> None:
    with pytest.raises(SeedIdentificationError, match="Empty description"):
        identify_seed("   ", llm_settings)


@respx.mock
def test_identify_seed_raises_on_non_json_reply(llm_settings) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion("I could not find that paper."))
    )
    with pytest.raises(SeedIdentificationError, match="not JSON"):
        identify_seed("x", llm_settings)


@respx.mock
def test_identify_seed_raises_when_search_query_missing(llm_settings) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion(json.dumps({"title": "Something"})))
    )
    with pytest.raises(SeedIdentificationError, match="search_query"):
        identify_seed("x", llm_settings)


@respx.mock
def test_identify_seed_raises_on_http_error(llm_settings) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(401, text="bad key"))
    with pytest.raises(SeedIdentificationError, match="401"):
        identify_seed("x", llm_settings)


@respx.mock
def test_identify_seed_retries_then_succeeds(llm_settings, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    retrying = dataclasses.replace(llm_settings, max_retries=3)
    route = respx.post(_CHAT_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=_completion(_GOOD_JSON))]
    )
    guess = identify_seed("x", retrying)
    assert route.call_count == 2
    assert guess.confidence == "high"


@respx.mock
def test_identify_seed_exhausts_retries_on_transport_error(llm_settings, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    respx.post(_CHAT_URL).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(SeedIdentificationError, match="after 1 attempts"):
        identify_seed("x", llm_settings)


@respx.mock
def test_identify_seed_raises_on_non_json_http_body(llm_settings) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, text="not json at all"))
    with pytest.raises(SeedIdentificationError, match="not JSON"):
        identify_seed("x", llm_settings)


@respx.mock
def test_identify_seed_raises_on_missing_choices(llm_settings) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json={"id": "x"}))
    with pytest.raises(SeedIdentificationError, match="Unexpected OpenRouter response shape"):
        identify_seed("x", llm_settings)


@respx.mock
def test_identify_seed_raises_on_empty_completion(llm_settings) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_completion("   ")))
    with pytest.raises(SeedIdentificationError, match="empty completion"):
        identify_seed("x", llm_settings)


@respx.mock
def test_identify_seed_raises_when_reply_is_json_array(llm_settings) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_completion("[1, 2, 3]")))
    with pytest.raises(SeedIdentificationError, match="not a JSON object"):
        identify_seed("x", llm_settings)


@respx.mock
def test_identify_seed_drops_out_of_range_year_and_reuses_http_client(llm_settings) -> None:
    body = json.dumps({"search_query": "q", "year": 3500, "authors": ["A", "  "]})
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=_completion(body)))
    with httpx.Client(base_url=llm_settings.openrouter_base_url) as client:
        guess = identify_seed("x", llm_settings, http_client=client)
    assert guess.year is None
    assert guess.authors == ["A"]
