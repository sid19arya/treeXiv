"""Shared OpenRouter chat plumbing: POST, retry, and JSON-object extraction.

Two steps in the pipeline call an LLM — Step 0 (`seed_llm.py`, guess which
paper a description means) and Step 4 (`curate.py`, pick and cluster the
papers worth showing). Both want the same things: a chat completion from
OpenRouter, retried on transient failures, whose reply is parsed as a single
JSON object even when the model wraps it in code fences or prose.

Callers pass their own `error_cls` so failures surface as the exception type
that step's callers already catch (`SeedIdentificationError`, `CurationError`)
rather than a generic one. Everything here stays a raw `httpx` POST — no LLM
SDK, per the project's minimal-dependency convention.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from treexiv.config import Settings
from treexiv.exceptions import LLMError

_CHAT_PATH = "/chat/completions"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_REFERER = "https://github.com/sid19arya/treeXiv"


@dataclass(slots=True)
class ChatResult:
    """A parsed JSON reply plus the raw assistant message it came from.

    The raw message is kept because some callers need fields alongside the
    content — Step 0 reads `annotations` for the URLs the web plugin cited.
    """

    data: dict[str, Any]
    message: dict[str, Any] = field(default_factory=dict)
    model: str = ""


def chat_json(
    settings: Settings,
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    plugins: list[dict[str, Any]] | None = None,
    http_client: httpx.Client | None = None,
    error_cls: type[LLMError] = LLMError,
) -> ChatResult:
    """Send `messages` to OpenRouter and parse the reply as a JSON object.

    Raises `error_cls` if the API key is missing, the request keeps failing,
    or the reply can't be read as a JSON object.
    """
    if not settings.openrouter_api_key:
        raise error_cls(
            "OPENROUTER_API_KEY is not set — needed for this step. See .env.example."
        )

    resolved_model = model or settings.openrouter_model
    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if plugins:
        payload["plugins"] = plugins

    data = post_chat(settings, payload, http_client=http_client, error_cls=error_cls)
    message = _assistant_message(data, error_cls)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise error_cls("OpenRouter returned an empty completion.")
    return ChatResult(
        data=extract_json_object(content, error_cls), message=message, model=resolved_model
    )


def post_chat(
    settings: Settings,
    payload: dict[str, Any],
    *,
    http_client: httpx.Client | None,
    error_cls: type[LLMError] = LLMError,
) -> dict[str, Any]:
    """POST a chat-completion payload, retrying transient failures with backoff."""
    owns_client = http_client is None
    client = http_client or httpx.Client(
        base_url=settings.openrouter_base_url, timeout=settings.llm_timeout_seconds
    )
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": _REFERER,
        "X-Title": "TreeXiv",
    }
    last_error: Exception | None = None
    try:
        for attempt in range(settings.max_retries):
            try:
                response = client.post(_CHAT_PATH, json=payload, headers=headers)
            except httpx.TransportError as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue
            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_error = error_cls(
                    f"OpenRouter returned retryable status {response.status_code}"
                )
                time.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise error_cls(
                    f"OpenRouter request failed: {response.status_code} {response.text[:300]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise error_cls(
                    f"OpenRouter response was not JSON: {response.text[:300]}"
                ) from exc
    finally:
        if owns_client:
            client.close()
    raise error_cls(
        f"OpenRouter request failed after {settings.max_retries} attempts: {last_error}"
    )


def _assistant_message(data: dict[str, Any], error_cls: type[LLMError]) -> dict[str, Any]:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise error_cls(
            f"Unexpected OpenRouter response shape: {json.dumps(data)[:300]}"
        ) from exc
    if not isinstance(message, dict):
        raise error_cls(f"Unexpected OpenRouter response shape: {json.dumps(data)[:300]}")
    return message


def extract_json_object(content: str, error_cls: type[LLMError] = LLMError) -> dict[str, Any]:
    """Parse a JSON object out of a model reply, tolerating code fences or a
    stray sentence around it.

    Worth the leniency: the models these steps run on (a cheap, fast one by
    default) honour `response_format: json_object` most of the time, not all
    of the time.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise error_cls(f"Model reply was not JSON: {content[:300]}") from None
        try:
            obj = json.loads(text[start : end + 1])
        except ValueError as exc:
            raise error_cls(f"Model reply was not JSON: {content[:300]}") from exc
    if not isinstance(obj, dict):
        raise error_cls(f"Model reply was not a JSON object: {content[:300]}")
    return obj


def clean_str(value: Any) -> str | None:
    """Trim a model-supplied value to a non-empty string, or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce_int(value: Any) -> int | None:
    """Read an int a model may have sent as a string ("3") or a float (3.0)."""
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
