"""Jev client: typed questions in, scores and probabilities out.

Jev (TypeSafe AI, served as `typesafe-ai/jev` through Vercel AI Gateway) is an
*evaluation* model, not a chat model. A request is one `state` plus a map of
named, typed questions, and every answer comes back as a structure your code
can branch on, never as text to parse:

- **boolean**: `{"type": "boolean", "probability": 0.93}`
- **choice**: `{"type": "choice", "choice": "extends", "probabilities": {...}}`
- **score**: `{"type": "score", "score": 3.2, "probabilities": {"0": ..., ...}}`

Exhaustive mode (`exhaustive.py`) uses it for every keep/drop and relation
decision. The generative LLM only writes prose afterwards.

The wire format is the gateway's own evaluation-model protocol, the same one
`@ai-sdk/gateway`'s `GatewayEvaluationModel` speaks: POST
`{base}/evaluation-model` with `{state, questions}` and the model ID in a
header. It is hand-rolled on `httpx` like every other call in this package.

A question's `instructions` may be a string or a JSON object. Exhaustive mode
uses objects: the paper being judged goes in one field and the question text
refers to it by name in backticks. That is TypeSafe's documented way to put
per-question data next to a shared state, and it lets one request judge a
whole batch of papers against the same idea.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx

from treexiv.config import Settings
from treexiv.exceptions import JevError

_EVALUATE_PATH = "/evaluation-model"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_PROTOCOL_VERSION = "0.0.1"

Question = dict[str, Any]


def boolean(instructions: Any, criteria: Mapping[str, Any] | None = None) -> Question:
    """A yes/no question; the answer is the probability of yes."""
    question: Question = {"type": "boolean", "instructions": instructions}
    if criteria:
        question["criteria"] = dict(criteria)
    return question


def choice(instructions: Any, options: Mapping[str, Any]) -> Question:
    """Pick one of `options` (name -> description). Up to 255 options."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: Any, levels: list[Any]) -> Question:
    """Rate on ordered `levels`, lowest first (2-10 levels). The answer is a
    probability-weighted level index, so 0 .. len(levels) - 1."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


@dataclass(slots=True, frozen=True)
class Answer:
    """One typed answer.

    `value` is the probability of yes for a boolean, the chosen option for a
    choice, and the weighted level for a score. `probabilities` is the full
    distribution for choice and score answers (empty for boolean).
    """

    type: str
    value: Any
    probabilities: dict[str, float] = field(default_factory=dict)

    def p(self, option: str) -> float:
        """Probability of `option` for a choice answer (0.0 if absent)."""
        return float(self.probabilities.get(option, 0.0))


def evaluate(
    settings: Settings,
    state: Any,
    questions: Mapping[str, Question],
    *,
    http_client: httpx.Client | None = None,
) -> dict[str, Answer]:
    """Ask Jev every question in `questions` against `state` in one request.

    Raises `JevError` if the key is missing, the request keeps failing, or an
    answer can't be read. Answers come back under the same keys.
    """
    if not settings.ai_gateway_api_key:
        raise JevError("AI_GATEWAY_API_KEY is not set — needed for Jev. See .env.example.")
    if not questions:
        return {}

    payload = {"state": state, "questions": dict(questions)}
    data = _post(settings, payload, http_client)
    raw_answers = data.get("answers")
    if not isinstance(raw_answers, dict):
        raise JevError(f"Jev response had no 'answers' object: {str(data)[:300]}")
    return {key: _parse_answer(key, raw) for key, raw in raw_answers.items()}


def evaluate_many(
    settings: Settings,
    state: Any,
    questions: Mapping[str, Question],
    *,
    http_client: httpx.Client | None = None,
) -> dict[str, Answer]:
    """`evaluate`, split into `settings.jev_batch_size`-question requests run
    `settings.jev_concurrency` at a time.

    Every question is judged against the same `state`, so batching changes
    nothing about the answers, only how many round trips they take. One
    failing batch fails the whole call. Partial relevance scores would
    silently bias what gets kept.
    """
    keys = list(questions)
    size = max(1, settings.jev_batch_size)
    batches = [
        {k: questions[k] for k in keys[start : start + size]} for start in range(0, len(keys), size)
    ]
    if not batches:
        return {}

    owns_client = http_client is None
    client = http_client or _new_client(settings)
    try:
        workers = max(1, min(settings.jev_concurrency, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(
                lambda batch: evaluate(settings, state, batch, http_client=client), batches
            )
            merged: dict[str, Answer] = {}
            for result in results:
                merged.update(result)
    finally:
        if owns_client:
            client.close()
    return merged


def _new_client(settings: Settings) -> httpx.Client:
    return httpx.Client(base_url=settings.ai_gateway_base_url, timeout=settings.timeout_seconds)


def _headers(settings: Settings) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.ai_gateway_api_key}",
        "ai-gateway-protocol-version": _PROTOCOL_VERSION,
        "ai-gateway-auth-method": "api-key",
        "ai-evaluation-model-specification-version": "4",
        "ai-model-id": settings.jev_model,
    }


def _post(
    settings: Settings, payload: dict[str, Any], http_client: httpx.Client | None
) -> dict[str, Any]:
    owns_client = http_client is None
    client = http_client or _new_client(settings)
    last_error: Exception | None = None
    try:
        for attempt in range(settings.max_retries):
            try:
                response = client.post(_EVALUATE_PATH, json=payload, headers=_headers(settings))
            except httpx.TransportError as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue
            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_error = JevError(f"Jev returned retryable status {response.status_code}")
                time.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise JevError(f"Jev request failed: {response.status_code} {response.text[:300]}")
            try:
                data = response.json()
            except ValueError as exc:
                raise JevError(f"Jev response was not JSON: {response.text[:300]}") from exc
            if not isinstance(data, dict):
                raise JevError(f"Unexpected Jev response shape: {response.text[:300]}")
            return data
    finally:
        if owns_client:
            client.close()
    raise JevError(f"Jev request failed after {settings.max_retries} attempts: {last_error}")


def _probabilities(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = float(value)
    return out


def _parse_answer(key: str, raw: Any) -> Answer:
    """Read one answer. TypeSafe's own API says `noul` where the gateway says
    `boolean`; both are accepted so `AI_GATEWAY_BASE_URL` can point either way."""
    if not isinstance(raw, dict):
        raise JevError(f"Jev answer {key!r} is not an object: {raw!r}")
    kind = raw.get("type")
    if kind in ("boolean", "noul"):
        value = raw.get("probability", raw.get("noul"))
        if not isinstance(value, (int, float)):
            raise JevError(f"Jev boolean answer {key!r} has no probability: {raw!r}")
        return Answer(type="boolean", value=float(value))
    if kind == "choice":
        picked = raw.get("choice")
        if not isinstance(picked, str):
            raise JevError(f"Jev choice answer {key!r} has no choice: {raw!r}")
        return Answer(
            type="choice", value=picked, probabilities=_probabilities(raw.get("probabilities"))
        )
    if kind == "score":
        value = raw.get("score")
        if not isinstance(value, (int, float)):
            raise JevError(f"Jev score answer {key!r} has no score: {raw!r}")
        return Answer(
            type="score", value=float(value), probabilities=_probabilities(raw.get("probabilities"))
        )
    raise JevError(f"Jev answer {key!r} has unknown type {kind!r}")
