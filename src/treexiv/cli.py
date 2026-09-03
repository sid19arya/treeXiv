"""Command-line interface for the treexiv MVP pipeline.

Each subcommand is one PRD step, callable in isolation and piping JSON to the
next: `identify-seed` (optional Step 0) -> `search-seed` -> (caller resolves
ambiguity) -> `expand` -> `filter` -> `render`. `run` chains
expand/filter/render for a seed ID that's already resolved.

This CLI is designed to be driven by an agent (see
`.claude/skills/treexiv-lineage/SKILL.md`) but is usable standalone.
`search-seed` deliberately does *not* try to resolve ambiguity itself (that
judgment call belongs to the orchestrating agent). `identify-seed` is the one
step that reaches for an LLM: it turns a vague description into a title worth
searching by calling a web-search-grounded OpenRouter model, since plain
OpenAlex title search can't bridge that gap. It's still only a lead —
`search-seed` and the usual disambiguation run after it.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import click

from treexiv.cache import WorkCache
from treexiv.config import Settings
from treexiv.exceptions import TreeXivError
from treexiv.expand import expand_two_hop
from treexiv.filtering import build_graph
from treexiv.models import ExpansionResult, FilteredGraph
from treexiv.openalex import OpenAlexClient
from treexiv.render import render_html
from treexiv.seed_llm import identify_seed


def _write_json(path: Path, payload: dict) -> None:
    """Write `payload` as indented JSON to `path`, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _settings_from_options(
    total_cap: int | None = None,
    fanout_cap: int | None = None,
    sampling: str | None = None,
    top_k: int | None = None,
    cache_dir: str | None = None,
    curation: str | None = None,
    max_nodes: int | None = None,
) -> Settings:
    base = Settings.from_env()
    return dataclasses.replace(
        base,
        total_corpus_cap=total_cap if total_cap is not None else base.total_corpus_cap,
        per_node_fanout_cap=fanout_cap if fanout_cap is not None else base.per_node_fanout_cap,
        sampling_strategy=sampling if sampling is not None else base.sampling_strategy,  # type: ignore[arg-type]
        bm25_top_k=top_k if top_k is not None else base.bm25_top_k,
        cache_dir=cache_dir if cache_dir is not None else base.cache_dir,
        curation_mode=curation if curation is not None else base.curation_mode,  # type: ignore[arg-type]
        curation_max_nodes=max_nodes if max_nodes is not None else base.curation_max_nodes,
    )


def _warn(message: str) -> None:
    click.echo(f"warning: {message}", err=True)


def _describe_graph(graph: FilteredGraph) -> str:
    """One-line summary of what filtering produced, for stderr."""
    how = "LLM-curated" if graph.curation == "llm" else "BM25 top-K"
    clusters = f", {len(graph.clusters)} concept clusters" if graph.clusters else ""
    return f"{how}: {len(graph.nodes)} nodes, {len(graph.edges)} edges{clusters}"


_CURATION_OPTION = click.option(
    "--curation",
    type=click.Choice(["auto", "llm", "bm25"]),
    default=None,
    help=(
        "How to pick which papers survive: 'llm' curates and clusters them "
        "(needs OPENROUTER_API_KEY), 'bm25' uses the deterministic top-K filter, "
        "'auto' (default) curates when a key is available and falls back to BM25."
    ),
)
_MAX_NODES_OPTION = click.option(
    "--max-nodes",
    type=int,
    default=None,
    help="Cap on papers LLM curation may keep (default 35). Ignored by --curation bm25.",
)


@click.group()
@click.version_option(package_name="treexiv")
def main() -> None:
    """treexiv: trace a paper's citation lineage as an interactive HTML graph."""


@main.command("identify-seed")
@click.argument("description")
@click.option(
    "--model",
    default=None,
    help="OpenRouter model slug (overrides OPENROUTER_MODEL for this call).",
)
@click.option(
    "--web/--no-web",
    "web_search",
    default=None,
    help="Toggle web-search grounding (default: on, or TREEXIV_LLM_WEB_SEARCH).",
)
def identify_seed_cmd(description: str, model: str | None, web_search: bool | None) -> None:
    """Guess which paper DESCRIPTION refers to, via a web-searching LLM (Step 0).

    Prints a JSON object with a best-guess title / arXiv ID / DOI and a
    `search_query` string to feed straight into `search-seed`. This is a
    *lead*, not a resolution — run `search-seed` and confirm the match before
    `run`. Requires OPENROUTER_API_KEY (see .env.example).
    """
    settings = Settings.from_env()
    if model:
        settings = dataclasses.replace(settings, openrouter_model=model)
    guess = identify_seed(description, settings, web_search=web_search)
    click.echo(json.dumps(guess.to_dict(), indent=2))


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
@click.option("--top-k", type=int, default=None, help="BM25 nodes to keep (default 40).")
@_CURATION_OPTION
@_MAX_NODES_OPTION
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
def filter_cmd(
    expansion_json: Path,
    idea: str,
    top_k: int | None,
    curation: str | None,
    max_nodes: int | None,
    out_path: Path,
) -> None:
    """Filter an expansion (from `expand`) down to the papers worth showing.

    By default an LLM curates the set and groups it into concept clusters;
    `--curation bm25` keeps the older deterministic top-K behaviour. Writes the
    filtered node/edge set as JSON to --out.
    """
    settings = _settings_from_options(top_k=top_k, curation=curation, max_nodes=max_nodes)
    expansion = ExpansionResult.from_dict(json.loads(expansion_json.read_text(encoding="utf-8")))
    filtered = build_graph(expansion, idea, settings, on_warning=_warn)
    _write_json(out_path, filtered.to_dict())
    click.echo(f"{_describe_graph(filtered)} -> {out_path}", err=True)


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
@_CURATION_OPTION
@_MAX_NODES_OPTION
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
    curation: str | None,
    max_nodes: int | None,
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
    settings = _settings_from_options(
        total_cap, fanout_cap, sampling, top_k, cache_dir, curation, max_nodes
    )
    with OpenAlexClient(settings) as client:
        seed_work = client.get_work(work_id)
        cache = WorkCache(settings.cache_dir, seed_id=seed_work.id)
        expansion = expand_two_hop(
            client, settings, seed_work, cache=cache, sample_seed=sample_seed
        )
    filtered = build_graph(expansion, idea, settings, on_warning=_warn)

    expansion_path = out_expansion or out_json.with_name(out_json.stem + ".expansion.json")
    _write_json(expansion_path, expansion.to_dict())
    _write_json(out_json, filtered.to_dict())
    written = render_html(filtered, out_html, title=title)
    click.echo(
        f"{seed_work.title!r}: {len(expansion.nodes)} expanded -> {_describe_graph(filtered)}"
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
