"""Tests for the Jev client. The Vercel AI Gateway is mocked with respx; no
real evaluation traffic happens in the suite."""

from __future__ import annotations

import dataclasses
import json

import httpx
import pytest
import respx

from treexiv import jev
from treexiv.exceptions import JevError

_EVAL_URL = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"


@pytest.fixture
def jev_settings(settings):
    return dataclasses.replace(settings, ai_gateway_api_key="vck-test", jev_batch_size=2)


def _echo_answers(request: httpx.Request) -> httpx.Response:
    """Answer every question: scores get their index, choices pick 'extends'."""
    body = json.loads(request.content)
    answers = {}
    for key, question in body["questions"].items():
        if question["type"] == "score":
            answers[key] = {"type": "score", "score": 3.0, "probabilities": {"3": 1.0}}
        elif question["type"] == "choice":
            answers[key] = {
                "type": "choice",
                "choice": "extends",
                "probabilities": {"extends": 0.8, "unrelated": 0.2},
            }
        else:
            answers[key] = {"type": "boolean", "probability": 0.9}
    return httpx.Response(200, json={"answers": answers, "usage": {"inputTokens": 10}})


def test_question_builders_match_the_wire_format() -> None:
    assert jev.boolean("Is it?") == {"type": "boolean", "instructions": "Is it?"}
    assert jev.choice("Which?", {"a": "first", "b": None}) == {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"a": "first", "b": None},
    }
    assert jev.score({"paper": {}, "question": "How?"}, ["low", "high"])["criteria"] == [
        "low",
        "high",
    ]


@respx.mock
def test_evaluate_posts_state_and_questions_with_gateway_headers(jev_settings) -> None:
    route = respx.post(_EVAL_URL).mock(side_effect=_echo_answers)
    answers = jev.evaluate(
        jev_settings,
        {"core_idea": "x"},
        {"q": jev.boolean("Is it?"), "c": jev.choice("Which?", {"extends": None})},
    )
    assert answers["q"] == jev.Answer(type="boolean", value=0.9)
    assert answers["c"].value == "extends"
    assert answers["c"].p("extends") == 0.8
    assert answers["c"].p("missing") == 0.0

    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer vck-test"
    assert request.headers["ai-model-id"] == "typesafe-ai/jev"
    assert request.headers["ai-evaluation-model-specification-version"] == "4"
    body = json.loads(request.content)
    assert body["state"] == {"core_idea": "x"}
    assert set(body["questions"]) == {"q", "c"}


def test_evaluate_requires_an_api_key(settings) -> None:
    with pytest.raises(JevError, match="AI_GATEWAY_API_KEY"):
        jev.evaluate(settings, "state", {"q": jev.boolean("?")})


@respx.mock
def test_evaluate_many_batches_and_merges(jev_settings) -> None:
    route = respx.post(_EVAL_URL).mock(side_effect=_echo_answers)
    questions = {f"p{i}": jev.score("How?", ["a", "b", "c", "d"]) for i in range(5)}
    answers = jev.evaluate_many(jev_settings, "state", questions)
    assert set(answers) == set(questions)
    assert route.call_count == 3  # batches of 2, 2, 1


@respx.mock
def test_typesafe_noul_answers_are_read_as_boolean(jev_settings) -> None:
    respx.post(_EVAL_URL).mock(
        return_value=httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.25}}})
    )
    assert jev.evaluate(jev_settings, "s", {"q": jev.boolean("?")})["q"].value == 0.25


@respx.mock
def test_unreadable_answers_raise(jev_settings) -> None:
    respx.post(_EVAL_URL).mock(
        return_value=httpx.Response(200, json={"answers": {"q": {"type": "score"}}})
    )
    with pytest.raises(JevError, match="no score"):
        jev.evaluate(jev_settings, "s", {"q": jev.score("?", ["a", "b"])})


@respx.mock
def test_client_errors_are_not_retried(jev_settings) -> None:
    route = respx.post(_EVAL_URL).mock(return_value=httpx.Response(401, text="bad key"))
    with pytest.raises(JevError, match="401"):
        jev.evaluate(dataclasses.replace(jev_settings, max_retries=3), "s", {"q": jev.boolean("?")})
    assert route.call_count == 1
