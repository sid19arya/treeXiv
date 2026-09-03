"""Exception hierarchy for the treexiv package."""

from __future__ import annotations


class TreeXivError(Exception):
    """Base class for all errors raised by treexiv."""


class OpenAlexAPIError(TreeXivError):
    """Raised when the OpenAlex API returns an unrecoverable error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SeedResolutionError(TreeXivError):
    """Raised when a seed paper query cannot be resolved to any candidate."""


class LLMError(TreeXivError):
    """Base class for failures in a step that calls an LLM (see `llm.py`)."""


class SeedIdentificationError(LLMError):
    """Raised when the Step 0 LLM seed-identification call fails or returns
    output that can't be parsed into a usable lead."""


class CurationError(LLMError):
    """Raised when the Step 4 LLM curation call fails or returns output that
    can't be parsed into a usable selection of papers."""


class GraphIOError(TreeXivError):
    """Raised when reading/writing expansion or filtered-graph JSON fails."""
