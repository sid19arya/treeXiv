"""Step 4b: write the lineage story that goes with a curated graph.

A graph of papers is not an answer. Once curation has decided *which* papers
belong (`curate.py`), this second call says what they add up to: a headline,
a few paragraphs on where the idea came from and what it grew into, and an
ordered set of beats — the turns in the story, each pointing at the papers
that mark it.

Deliberately narrow: the model is given only the papers already selected, and
it names them by index, the same trick `curate.py` uses. It is never asked
*how two papers cite each other* — that stays with `narrative.py`, which reads
it off the actual edges. An LLM inventing a plausible-sounding citation
relationship is exactly the failure this tool can't afford, so the two kinds
of sidebar text have different authors: the story is written, the citation
path is derived.

Synthesis is best-effort. `synthesize_lineage` raises `SynthesisError` on any
problem and `filtering.build_graph` carries on with the graph unannotated.
"""

from __future__ import annotations

from typing import Any

import httpx

from treexiv.config import Settings
from treexiv.exceptions import SynthesisError
from treexiv.llm import chat_json, clean_str, coerce_int
from treexiv.models import Cluster, FilteredGraph, LineageNarrative, NarrativeBeat, ScoredNode

MAX_BEATS = 6

_SYSTEM_PROMPT = """\
You explain how a research idea developed, given a curated set of papers around
one seed paper.

You will get the seed paper, the idea the reader cares about, the concept
clusters the papers were grouped into, and a numbered list of the papers
themselves. Write the story those papers tell.

Respond with ONLY a JSON object, no prose or code fences:
{
  "headline": "one sentence naming the through-line of this lineage",
  "overview": "2-4 short paragraphs, separated by \\n\\n: where the idea came
    from, what the seed paper changed, and what it grew into. Name specific
    papers by title where it helps.",
  "beats": [
    {"title": "short name for this turn in the story (3-6 words)",
     "text": "2-3 sentences on what changed here and why it mattered",
     "papers": [<paper numbers that mark this beat>]}
  ]
}

Rules:
- Write for someone who knows the field but not this specific lineage. No
  throat-clearing, no "in recent years", no restating the task.
- 3 to %(max_beats)d beats, in chronological order of what they describe.
- "papers" must be numbers from the list you were given. Never invent one.
- Only claim a paper did something if the material you were given says so. If
  the connection between two papers is unclear, describe the shift in ideas
  rather than asserting who cited whom.
- Be specific about mechanisms and results, not about importance. "Replaced the
  learned router with a k-NN baseline and matched it" beats "was influential"."""


def _paper_lines(
    nodes: list[ScoredNode], clusters: dict[str, Cluster]
) -> tuple[str, dict[int, str]]:
    """Render the kept papers as a numbered list, plus the index→node-ID map."""
    lines: list[str] = []
    index_to_id: dict[int, str] = {}
    for i, scored in enumerate(nodes, start=1):
        node = scored.node
        index_to_id[i] = node.id
        cluster = clusters.get(scored.cluster_id or "")
        parts = [f"[{i}] {node.publication_year or 'n.d.'} — {node.title}"]
        if cluster:
            parts.append(f"cluster: {cluster.name}")
        if scored.why:
            parts.append(f"role: {scored.why}")
        lines.append(" | ".join(parts))
    return "\n".join(lines), index_to_id


def build_prompt(
    graph: FilteredGraph, seed_title: str
) -> tuple[list[dict[str, str]], dict[int, str]]:
    """The system + user messages for one synthesis call, and the index map."""
    clusters = {c.id: c for c in graph.clusters}
    non_seed = [sn for sn in graph.nodes if sn.node.id != graph.seed_id]
    paper_text, index_to_id = _paper_lines(non_seed, clusters)

    cluster_text = "\n".join(
        f"- {c.name} ({c.role}): {c.summary}" for c in graph.clusters
    ) or "(none)"

    user = (
        f"SEED PAPER: {seed_title}\n\n"
        f"THE IDEA THE READER CARES ABOUT: {graph.idea_text}\n\n"
        f"CONCEPT CLUSTERS:\n{cluster_text}\n\n"
        f"PAPERS ({len(non_seed)}):\n{paper_text}"
    )
    return (
        [
            {"role": "system", "content": _SYSTEM_PROMPT % {"max_beats": MAX_BEATS}},
            {"role": "user", "content": user},
        ],
        index_to_id,
    )


def synthesize_lineage(
    graph: FilteredGraph,
    settings: Settings,
    *,
    http_client: httpx.Client | None = None,
) -> LineageNarrative:
    """Write the lineage story for an already-curated `graph`.

    Raises `SynthesisError` if there is nothing to write about, the call fails,
    or the reply has no usable overview.
    """
    seed = next((sn.node for sn in graph.nodes if sn.node.id == graph.seed_id), None)
    if seed is None:
        raise SynthesisError(f"Graph has no seed node {graph.seed_id!r} to write about.")
    if len(graph.nodes) < 2:
        raise SynthesisError("Graph has only the seed paper — no lineage to describe.")

    messages, index_to_id = build_prompt(graph, seed.title)
    result = chat_json(
        settings,
        messages,
        model=settings.resolved_curation_model,
        temperature=0.3,
        http_client=http_client,
        error_cls=SynthesisError,
    )

    overview = clean_str(result.data.get("overview"))
    if not overview:
        raise SynthesisError(f"Synthesis reply had no 'overview': {str(result.data)[:300]}")

    return LineageNarrative(
        headline=clean_str(result.data.get("headline")) or "",
        overview=overview,
        beats=_parse_beats(result.data.get("beats"), index_to_id),
        model=result.model,
    )


def _parse_beats(raw: Any, index_to_id: dict[int, str]) -> list[NarrativeBeat]:
    beats: list[NarrativeBeat] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        text = clean_str(item.get("text"))
        if not text:
            continue
        node_ids: list[str] = []
        for entry in item.get("papers") or []:
            index = coerce_int(entry)
            node_id = index_to_id.get(index) if index is not None else None
            if node_id and node_id not in node_ids:
                node_ids.append(node_id)
        beats.append(
            NarrativeBeat(
                title=clean_str(item.get("title")) or "",
                text=text,
                node_ids=node_ids,
            )
        )
    return beats[:MAX_BEATS]
