---
name: treexiv-lineage
description: Build a citation-lineage tree for a seed paper and a stated "core idea" — two-hop OpenAlex expansion, BM25 relevance filtering, rendered as an interactive HTML graph. Use when the user wants to trace a paper's ancestry/descendants, map "where an idea came from and what it grew into," build a citation tree/lineage map, or explicitly asks for treexiv.
---

# TreeXiv lineage tree (Phase 0 MVP, OpenAlex)

This skill drives the `treexiv` Python package (`src/treexiv/`, managed with
`uv`) to build a query-time citation-lineage tree. Full spec:
`scratch/treexiv-mvp-openalex-prd.md`. The package does the mechanical work
(HTTP calls, graph expansion, BM25, rendering); **you** do the judgment
calls the PRD assigns to Step 1 — this skill is that missing piece. See
`README.md` for why the split is drawn this way.

## Prerequisites

- `uv sync` has been run in the repo root at least once.
- `OPENALEX_API_KEY` (and optionally `OPENALEX_MAILTO`) set in `.env` — see
  `README.md` Setup. Without a key, requests still work but are more likely
  to be rate-limited.
- Run all commands from the repo root as `uv run treexiv <subcommand> ...`.

## Step 1 — Get the seed paper and the core idea from the user

If either is missing, ask for it directly:
- The seed paper: a title, DOI, or arXiv ID is all fine.
- The "core idea": a short free-text description of what the user actually
  cares about tracing (this drives BM25 filtering in Step 4 below — it does
  not need to be the seed paper's own abstract).

## Step 2 — Resolve the seed to one OpenAlex work ID

1. Run `uv run treexiv search-seed "<title or ID>"` (add `--limit 5` if you
   want more candidates). This prints a JSON array of candidates with
   `id`, `title`, `publication_year`, `authors`, `venue`, `cited_by_count`,
   `doi` — deliberately unranked-by-confidence beyond OpenAlex's own search
   relevance.
2. Cross-check the top candidate against a real web search (use your
   `WebSearch` tool) on title + first author, to catch common-title
   collisions and preprint-vs-published duplicates. This is the step the
   PRD calls out explicitly in Step 1 and that the CLI deliberately does
   *not* attempt itself — it needs judgment and live search, both of which
   you have and a standalone script doesn't.
3. If, after that, more than one candidate still looks plausible, surface
   the top 2-3 to the user (title, year, venue, first author) and ask them
   to confirm rather than guessing.
4. Once resolved, note the winning OpenAlex work ID (e.g. `W2741809807`) —
   everything downstream keys off it.

## Step 3 — Run the pipeline

For a standard run, chain everything in one call:

```
uv run treexiv run <WORK_ID> --idea "<core idea text>" \
  --out-json output/filtered.json --out-html output/tree.html
```

Useful overrides (all optional, defaults from
`scratch/treexiv-mvp-openalex-prd.md` Section 4):
- `--total-cap N` — global node cap (default 500).
- `--fanout-cap N` — per-node fan-out cap (default 100).
- `--sampling top_cited|random` — `random` trades prominence for a chance at
  catching divergent, less-cited branches; use it if the user explicitly
  wants breadth over "core lineage."
- `--top-k N` — how many BM25-relevant nodes survive filtering (default 40).
- `--cache-dir PATH` — reuse fetched OpenAlex records across repeat runs on
  the same seed.
- `--out-expansion PATH` — where the *full* pre-filter expansion JSON goes
  (every node/edge the two-hop traversal collected, not just the top-K
  survivors). Defaults to `<out-json stem>.expansion.json` next to
  `--out-json`, and is always written — this is the "hold onto everything
  the API surfaced" artifact, useful for auditing what BM25 filtered out.

If you want to inspect or report intermediate stats (node counts per hop,
whether the expansion got truncated by a cap) before filtering/rendering,
run the stages separately instead: `expand` -> `filter` -> `render` (see
`uv run treexiv --help` and each subcommand's `--help`).

## Step 4 — Report back and hand over the artifacts

- Tell the user: how many nodes were expanded vs. kept after filtering,
  whether the expansion hit a cap (truncated), and the seed paper actually
  resolved to (title/year), so they can catch a bad Step-2 resolution.
- Send the rendered HTML file to the user as a file attachment/render — it's
  a self-contained interactive graph, nothing else needs to be shared.
- Mention the full-expansion JSON path if the user might want to dig into
  what got filtered out (e.g. `--out-expansion`'s default location); it's
  not usually worth sending as its own attachment unless they ask.

## Explicitly out of scope here

Per `CLAUDE.md` phase discipline and the PRD's own non-goals: no persistent
store beyond the optional per-run cache, no citation-intent classification,
no embeddings. If a request needs those, say so rather than building them
into this skill.
