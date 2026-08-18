"""Step 3: build a BM25 corpus over collected nodes' title + abstract text.

Uses `rank_bm25` (pure-Python, in-process, no external service) per the PRD.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from treexiv.models import Node

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenizer. Deliberately simple — BM25
    is a relevance stand-in for this MVP, not a tuned search system."""
    return _TOKEN_PATTERN.findall(text.lower())


def document_text(node: Node) -> str:
    """The BM25 document for a node: title + abstract, per PRD Step 3."""
    return f"{node.title} {node.abstract}".strip()


class BM25Corpus:
    """A BM25 index over a fixed list of nodes, queryable by free text."""

    def __init__(self, nodes: list[Node]) -> None:
        self._nodes = nodes
        self._tokenized_docs = [tokenize(document_text(n)) for n in nodes]
        self._bm25 = BM25Okapi(self._tokenized_docs) if self._tokenized_docs else None

    def __len__(self) -> int:
        return len(self._nodes)

    def scores(self, query: str) -> dict[str, float]:
        """BM25 relevance score for every node against `query`, keyed by node ID."""
        if self._bm25 is None:
            return {}
        raw_scores = self._bm25.get_scores(tokenize(query))
        return {node.id: float(score) for node, score in zip(self._nodes, raw_scores, strict=True)}

    def top_k(self, query: str, k: int) -> list[tuple[Node, float]]:
        """The `k` highest-scoring nodes for `query`, descending by score."""
        scores = self.scores(query)
        ranked = sorted(self._nodes, key=lambda n: scores.get(n.id, 0.0), reverse=True)
        return [(n, scores.get(n.id, 0.0)) for n in ranked[:k]]
