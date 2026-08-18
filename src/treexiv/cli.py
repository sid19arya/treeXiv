"""Command-line interface for the treexiv MVP pipeline.

Each subcommand is one PRD step, callable in isolation and piping JSON to the
next: `search-seed` -> (caller resolves ambiguity) -> `expand` -> `filter` ->
`render`. `run` chains expand/filter/render for a seed ID that's already
resolved.

This CLI is designed to be driven by an agent (see
`.claude/skills/treexiv-lineage/SKILL.md`) rather than a human typing
commands directly: `search-seed` deliberately does *not* try to resolve
ambiguity itself (e.g. via a web-search cross-check) — Step 1 of the PRD asks
for exactly that kind of judgment call, which belongs to whatever LLM/agent
is orchestrating the run, not to this package. See README.md and CLAUDE.md
for the harness-integration note.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from treexiv.cache import WorkCache
from treexiv.config import Settings
from treexiv.exceptions import TreeXivError
from treexiv.expand import expand_two_hop
from treexiv.filtering import filter_by_idea
from treexiv.models import ExpansionResult, FilteredGraph
from treexiv.openalex import OpenAlexClient
from treexiv.render import render_html


def _write_json(path: Path, payload: dict) -> None:
    """Write `payload` as indented JSON to `path`, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _settings_from_options(
    total_cap: int | None,
    fanout_cap: int | None,
    sampling: str | None,
    top_k: int | None,
    cache_dir: str | None,
) -> Settings:
    base = Settings.from_env()
    return Settings(
        base_url=base.base_url,
        mailto=base.mailto,
        api_key=base.api_key,
        total_corpus_cap=total_cap if total_cap is not None else base.total_corpus_cap,
        per_node_fanout_cap=fanout_cap if fanout_cap is not None else base.per_node_fanout_cap,
        sampling_strategy=sampling if sampling is not None else base.sampling_strategy,  # type: ignore[arg-type]
        bm25_top_k=top_k if top_k is not None else base.bm25_top_k,
        timeout_seconds=base.timeout_seconds,
        max_retries=base.max_retries,
        cache_dir=cache_dir if cache_dir is not None else base.cache_dir,
    )


@click.group()
@click.version_option(package_name="treexiv")
def main() -> None:
    """treexiv: trace a paper's citation lineage as an interactive HTML graph."""


@main.command("search-seed")
@click.argument("query")
@click.option("--limit", default=5, show_default=True, help="Number of candidates to return.")
def search_seed(query: str, limit: int) -> None:
    """Search OpenAlex for candidate seed works matching QUERY.

    Prints a JSON array of candidates (id, title, year, authors, venue,
    cited_by_count) to stdout. Resolving ambiguity between candidates
    (including any external cross-check) is the caller's responsibility.
    """
    settings = Settings.from_env()
    with OpenAlexClient(settings) as client:
        candidates = client.search_works(query, limit=limit)
    click.echo(
        json.dumps(
            [
                {
                    "id": w.id,
                    "title": w.title,
                    "publication_year": w.publication_year,
                    "cited_by_count": w.cited_by_count,
                    "authors": w.authors,
                    "venue": w.venue,
                    "doi": w.doi,
                }
                for w in candidates
            ],
            indent=2,
        )
    )


@main.command("expand")
@click.argument("work_id")
@click.option("--total-cap", type=int, default=None, help="Global node cap (default 500).")
@click.option("--fanout-cap", type=int, default=None, help="Per-node fan-out cap (default 100).")
@click.option(
    "--sampling",
    type=click.Choice(["top_cited", "random"]),
    default=None,
    help="Fan-out prioritization strategy (default top_cited).",
)
@click.option("--sample-seed", type=int, default=None, help="RNG seed for --sampling random.")
@click.option("--cache-dir", default=None, help="Optional per-run JSON cache directory.")
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
def expand_cmd(
    work_id: str,
    total_cap: int | None,
    fanout_cap: int | None,
    sampling: str | None,
    sample_seed: int | None,
    cache_dir: str | None,
    out_path: Path,
) -> None:
    """Two-hop bidirectional expansion from WORK_ID (an OpenAlex work ID).

    Writes the resulting node/edge set as JSON to --out.
    """
    settings = _settings_from_options(total_cap, fanout_cap, sampling, None, cache_dir)
    with OpenAlexClient(settings) as client:
        seed_work = client.get_work(work_id)
        cache = WorkCache(settings.cache_dir, seed_id=seed_work.id)
        result = expand_two_hop(client, settings, seed_work, cache=cache, sample_seed=sample_seed)
    _write_json(out_path, result.to_dict())
    click.echo(
        f"Expanded {seed_work.title!r}: {len(result.nodes)} nodes, {len(result.edges)} edges"
        f"{' (truncated by cap)' if result.truncated else ''} -> {out_path}",
        err=True,
    )


@main.command("filter")
@click.argument("expansion_json", type=click.Path(exists=True, path_type=Path))
@click.option("--idea", required=True, help="Free-text description of the core idea to filter by.")
@click.option("--top-k", type=int, default=None, help="Number of nodes to keep (default 40).")
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
def filter_cmd(expansion_json: Path, idea: str, top_k: int | None, out_path: Path) -> None:
    """BM25-filter an expansion (from `expand`) against --idea.

    Writes the filtered node/edge set as JSON to --out.
    """
    settings = _settings_from_options(None, None, None, top_k, None)
    expansion = ExpansionResult.from_dict(json.loads(expansion_json.read_text(encoding="utf-8")))
    filtered = filter_by_idea(expansion, idea, top_k=settings.bm25_top_k)
    _write_json(out_path, filtered.to_dict())
    click.echo(
        f"Filtered to {len(filtered.nodes)} nodes, {len(filtered.edges)} edges -> {out_path}",
        err=True,
    )


@main.command("render")
@click.argument("filtered_json", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--title", default="TreeXiv Lineage", show_default=True)
def render_cmd(filtered_json: Path, out_path: Path, title: str) -> None:
    """Render a filtered graph (from `filter`) as an interactive HTML file."""
    graph = FilteredGraph.from_dict(json.loads(filtered_json.read_text(encoding="utf-8")))
    written = render_html(graph, out_path, title=title)
    click.echo(f"Rendered {len(graph.nodes)} nodes -> {written}", err=True)


@main.command("run")
@click.argument("work_id")
@click.option("--idea", required=True, help="Free-text description of the core idea to filter by.")
@click.option("--total-cap", type=int, default=None)
@click.option("--fanout-cap", type=int, default=None)
@click.option("--sampling", type=click.Choice(["top_cited", "random"]), default=None)
@click.option("--sample-seed", type=int, default=None)
@click.option("--top-k", type=int, default=None)
@click.option("--cache-dir", default=None)
@click.option(
    "--out-json", required=True, type=click.Path(path_type=Path), help="Filtered graph JSON output."
)
@click.option(
    "--out-expansion",
    "out_expansion",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Full pre-filter expansion JSON (every node/edge the traversal collected, "
        "not just the top-K survivors). Defaults to '<out-json stem>.expansion.json' "
        "next to --out-json."
    ),
)
@click.option(
    "--out-html", required=True, type=click.Path(path_type=Path), help="Rendered HTML output."
)
@click.option("--title", default="TreeXiv Lineage", show_default=True)
def run_cmd(
    work_id: str,
    idea: str,
    total_cap: int | None,
    fanout_cap: int | None,
    sampling: str | None,
    sample_seed: int | None,
    top_k: int | None,
    cache_dir: str | None,
    out_json: Path,
    out_expansion: Path | None,
    out_html: Path,
    title: str,
) -> None:
    """Chain expand -> filter -> render for an already-resolved WORK_ID.

    This assumes Step 1 (seed resolution / ambiguity confirmation) already
    happened upstream, e.g. via `search-seed` plus the calling agent's own
    judgment.
    """
    settings = _settings_from_options(total_cap, fanout_cap, sampling, top_k, cache_dir)
    with OpenAlexClient(settings) as client:
        seed_work = client.get_work(work_id)
        cache = WorkCache(settings.cache_dir, seed_id=seed_work.id)
        expansion = expand_two_hop(
            client, settings, seed_work, cache=cache, sample_seed=sample_seed
        )
    filtered = filter_by_idea(expansion, idea, top_k=settings.bm25_top_k)

    expansion_path = out_expansion or out_json.with_name(out_json.stem + ".expansion.json")
    _write_json(expansion_path, expansion.to_dict())
    _write_json(out_json, filtered.to_dict())
    written = render_html(filtered, out_html, title=title)
    click.echo(
        f"{seed_work.title!r}: {len(expansion.nodes)} expanded -> {len(filtered.nodes)} filtered"
        f"{' (expansion truncated by cap)' if expansion.truncated else ''}\n"
        f"Full expansion JSON: {expansion_path}\nFiltered JSON: {out_json}\nHTML: {written}",
        err=True,
    )


def entry_point() -> None:
    try:
        main()
    except TreeXivError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    entry_point()
