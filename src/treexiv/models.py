"""Data model shared across the pipeline: OpenAlex works, graph nodes/edges,
and the two JSON-serializable payloads that pass between CLI stages
(`ExpansionResult` after Step 2, `FilteredGraph` after Step 4).

Kept as plain dataclasses with explicit to_dict/from_dict rather than a
validation library — the MVP has few enough shapes that hand-written
(de)serialization stays easy to read and test, and it keeps dependencies
minimal per the project's phase-0 conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from treexiv.abstract import reconstruct_abstract

OPENALEX_URL_PREFIX = "https://openalex.org/"


def normalize_work_id(raw_id: str) -> str:
    """Normalize an OpenAlex work identifier to its short form, e.g. "W123".

    Accepts either the short form or a full `https://openalex.org/W123` URL,
    and is tolerant of surrounding whitespace and casing on the "W" prefix.
    """
    stripped = raw_id.strip()
    if stripped.startswith(OPENALEX_URL_PREFIX):
        stripped = stripped[len(OPENALEX_URL_PREFIX) :]
    return stripped.upper() if stripped[:1].lower() == "w" else stripped


@dataclass(slots=True)
class Work:
    """A work record as returned by the OpenAlex `/works` endpoint (subset of fields)."""

    id: str
    title: str
    publication_year: int | None
    cited_by_count: int
    authors: list[str] = field(default_factory=list)
    venue: str | None = None
    doi: str | None = None
    referenced_works: list[str] = field(default_factory=list)
    abstract_inverted_index: dict[str, list[int]] | None = None

    @property
    def abstract(self) -> str:
        return reconstruct_abstract(self.abstract_inverted_index)

    @classmethod
    def from_api(cls, payload: dict) -> Work:
        """Parse one element of an OpenAlex `/works` response into a `Work`."""
        authorships = payload.get("authorships") or []
        authors = [
            a["author"]["display_name"]
            for a in authorships
            if a.get("author", {}).get("display_name")
        ]
        host_venue = payload.get("primary_location") or {}
        source = host_venue.get("source") or {}
        return cls(
            id=normalize_work_id(payload["id"]),
            title=payload.get("display_name") or payload.get("title") or "(untitled)",
            publication_year=payload.get("publication_year"),
            cited_by_count=payload.get("cited_by_count", 0) or 0,
            authors=authors,
            venue=source.get("display_name"),
            doi=payload.get("doi"),
            referenced_works=[normalize_work_id(w) for w in payload.get("referenced_works") or []],
            abstract_inverted_index=payload.get("abstract_inverted_index"),
        )


@dataclass(slots=True)
class Node:
    """A graph node: a `Work` reduced to what the corpus/render steps need,
    plus its hop distance from the seed."""

    id: str
    title: str
    publication_year: int | None
    cited_by_count: int
    authors: list[str]
    venue: str | None
    abstract: str
    hop: int
    doi: str | None = None

    @classmethod
    def from_work(cls, work: Work, hop: int) -> Node:
        return cls(
            id=work.id,
            title=work.title,
            publication_year=work.publication_year,
            cited_by_count=work.cited_by_count,
            authors=work.authors,
            venue=work.venue,
            abstract=work.abstract,
            hop=hop,
            doi=work.doi,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "publication_year": self.publication_year,
            "cited_by_count": self.cited_by_count,
            "authors": self.authors,
            "venue": self.venue,
            "abstract": self.abstract,
            "hop": self.hop,
            "doi": self.doi,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Node:
        return cls(
            id=payload["id"],
            title=payload["title"],
            publication_year=payload.get("publication_year"),
            cited_by_count=payload.get("cited_by_count", 0),
            authors=payload.get("authors", []),
            venue=payload.get("venue"),
            abstract=payload.get("abstract", ""),
            hop=payload.get("hop", 0),
            doi=payload.get("doi"),
        )


@dataclass(slots=True, frozen=True)
class Edge:
    """A citation edge: `source` cites `target` (source is the citing work)."""

    source: str
    target: str

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target}

    @classmethod
    def from_dict(cls, payload: dict) -> Edge:
        return cls(source=payload["source"], target=payload["target"])


@dataclass(slots=True)
class ExpansionResult:
    """Output of Step 2 (two-hop bidirectional expansion): the deduplicated
    node/edge set, before any relevance filtering."""

    seed_id: str
    nodes: dict[str, Node]
    edges: list[Edge]
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "seed_id": self.seed_id,
            "truncated": self.truncated,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ExpansionResult:
        nodes = {n["id"]: Node.from_dict(n) for n in payload["nodes"]}
        edges = [Edge.from_dict(e) for e in payload["edges"]]
        return cls(
            seed_id=payload["seed_id"],
            nodes=nodes,
            edges=edges,
            truncated=payload.get("truncated", False),
        )


@dataclass(slots=True)
class ScoredNode:
    """A `Node` annotated with its BM25 relevance score against the core-idea query."""

    node: Node
    score: float

    def to_dict(self) -> dict:
        return {**self.node.to_dict(), "bm25_score": self.score}

    @classmethod
    def from_dict(cls, payload: dict) -> ScoredNode:
        node_fields = {k: v for k, v in payload.items() if k != "bm25_score"}
        return cls(node=Node.from_dict(node_fields), score=payload.get("bm25_score", 0.0))


@dataclass(slots=True)
class FilteredGraph:
    """Output of Step 4 (BM25 relevance filter): the surviving nodes, scored,
    and only the edges whose endpoints both survived."""

    seed_id: str
    idea_text: str
    top_k: int
    nodes: list[ScoredNode]
    edges: list[Edge]

    def to_dict(self) -> dict:
        return {
            "seed_id": self.seed_id,
            "idea_text": self.idea_text,
            "top_k": self.top_k,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> FilteredGraph:
        return cls(
            seed_id=payload["seed_id"],
            idea_text=payload["idea_text"],
            top_k=payload["top_k"],
            nodes=[ScoredNode.from_dict(n) for n in payload["nodes"]],
            edges=[Edge.from_dict(e) for e in payload["edges"]],
        )
