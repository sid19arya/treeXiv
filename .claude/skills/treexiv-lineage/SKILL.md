---
name: treexiv-lineage
description: Build a citation-lineage tree for a seed paper and a stated "core idea" — two-hop OpenAlex expansion, LLM curation into concept strands, a written lineage story, rendered as an interactive HTML graph. Use when the user wants to trace a paper's ancestry/descendants, map "where an idea came from and what it grew into," build a citation tree/lineage map, or explicitly asks for treexiv.
---

# TreeXiv lineage tree

This skill drives the `treexiv` Python package (`src/treexiv/`, managed with
`uv`) to build a query-time citation-lineage tree. Full spec:
`scratch/treexiv-mvp-openalex-prd.md`. The package does the mechanical work
(HTTP calls, graph expansion, curation, rendering); **you** do the judgment
calls the PRD assigns to Step 1 — this skill is that missing piece. See
`README.md` for why the split is drawn this way.

## Prerequisites

- `uv sync` has been run in the repo root at least once.
- `OPENALEX_API_KEY` (and optionally `OPENALEX_MAILTO`) set in `.env` — see
  `README.md` Setup. Without a key, requests still work but are more likely
  to be rate-limited.
- `OPENROUTER_API_KEY` for LLM curation and the lineage story. Without it the
  run still completes, but falls back to the old keyword-only filter — which
  is a much worse map. If it's missing, say so rather than handing over a
  degraded result without comment.
- Semantic Scholar needs no key at all. `S2_API_KEY` is optional and only
  lifts its rate limit.
- Run all commands from the repo root as `uv run treexiv <subcommand> ...`.

## Step 0 — (optional) Identify the seed from a vague description

Only when the user *doesn't* have a specific paper in mind — they describe an
idea/finding/era instead of naming a paper. If they already gave a title, DOI,
or arXiv ID, skip straight to Step 1.

Run `uv run treexiv identify-seed "<the user's description>"`. This calls a
web-search-grounded OpenRouter model and prints a JSON object: `search_query`
(feed this into Step 2), plus `title`, `arxiv_id`, `doi`, `year`,
`confidence`, `reasoning`, `alternatives`, and `sources` (URLs it consulted).

Treat it as a *lead, not an answer*: still run Step 2's `search-seed` on the
`search_query`, still do the Step 2 cross-check, and if `confidence` is
`low`/`medium` or `alternatives` is non-empty, show the user what it guessed
(and the alternatives) before proceeding. If the guess looks wrong, ask the
user to describe the paper differently or name it directly.

## Step 1 — Get the seed paper and the core idea from the user

If either is missing, ask for it directly:
- The seed paper: a title, DOI, or arXiv ID is all fine. (Or run Step 0 if the
  user can only describe it.)
- The "core idea": a short free-text description of what the user actually
  cares about tracing. This is the single biggest lever on output quality —
  curation keeps or drops each paper by how load-bearing it is *for this
  idea*, so "efficient attention" and "why quadratic attention was replaced in
  long-context models" produce genuinely different trees. If the user gives
  you something very broad, it's worth one question to sharpen it.

## Step 2 — Resolve the seed to one OpenAlex work ID

1. Run `uv run treexiv search-seed "<title or ID>"` (add `--limit 5` for more
   candidates). This prints a JSON array of candidates with `id`, `title`,
   `publication_year`, `authors`, `venue`, `cited_by_count`, `doi`, and
   `matched_by`.
2. **Read `matched_by`.** `semantic_scholar` means S2's title matcher found
   that exact paper — it is usually right, and much more reliable than the
   `openalex_search` entries below it, which are relevance hits and can be a
   different paper entirely. It is still a match, not a confirmation.
3. Cross-check the top candidate against a real web search (use your
   `WebSearch` tool) on title + first author, to catch common-title collisions
   and preprint-vs-published duplicates. This is the step the PRD calls out
   explicitly and that the CLI deliberately does *not* attempt itself — it
   needs judgment and live search, both of which you have and a standalone
   script doesn't.
4. If more than one candidate still looks plausible, surface the top 2-3 to
   the user (title, year, venue, first author) and ask them to confirm rather
   than guessing.
5. Once resolved, note the winning OpenAlex work ID (e.g. `W2741809807`) —
   everything downstream keys off it.

## Step 3 — Run the pipeline

For a standard run, chain everything in one call:

```
uv run treexiv run <WORK_ID> --idea "<core idea text>" \
  --out-json output/filtered.json --out-html output/tree.html
```

**Warn the user this takes a few minutes before you start it.** The curation
call reads a shortlist of abstracts and is the slow part — several minutes is
normal on the default model. If they want a fast, rough answer instead, use
`--curation bm25`, and tell them that's what they're getting.

Useful overrides (all optional):
- `--curation auto|llm|bm25` — `auto` (default) curates when a key is present
  and falls back to keyword filtering otherwise; `bm25` is the fast path.
- `--max-nodes N` — cap on papers curation may keep (default 35).
- `--no-narrative` — skip the lineage story (saves one LLM call).
- `--source auto|openalex` — `openalex` skips Semantic Scholar entirely.
- `--total-cap N` / `--fanout-cap N` — how wide the traversal goes (500 / 100).
- `--top-k N` — how many papers survive the BM25 fallback filter (default 40).
- `--sampling top_cited|random` — `random` trades prominence for a chance at
  catching divergent, less-cited branches.
- `--cache-dir PATH` — reuse fetched records across repeat runs on the same seed.
- `--out-expansion PATH` — where the *full* pre-filter expansion JSON goes
  (every node/edge the traversal collected, not just the survivors). Defaults
  to `<out-json stem>.expansion.json` and is always written.

**Watch stderr.** The run reports what actually happened, and some of it
changes what the output means:
- `LLM-curated: N nodes, ... concept clusters, narrative in N beats` — the
  good path.
- `warning: ... falling back to the BM25 top-K filter` — curation failed or
  no key. The user got the weaker map; tell them.
- `Semantic Scholar: N edges labelled with citation intent` — how many
  citations came back with intent data. Zero is common and fine for recent
  arXiv preprints; it isn't an error.

To iterate on the idea text without re-crawling, keep the expansion JSON and
re-run only `filter` + `render` against it.

## Step 4 — Report back and hand over the artifacts

- Read the filtered JSON and tell the user **what the tree actually says**:
  the `narrative.headline`, the concept strands (`clusters`, with names and
  roles), and how many papers were expanded vs. kept. Do not just report file
  paths — the story is the output.
- Name the seed paper that was actually resolved (title/year) so they can
  catch a bad Step-2 resolution.
- Send the rendered HTML to the user as a file attachment — it's a
  self-contained interactive graph. Mention that it opens on the concept
  strands and that clicking one expands it into its papers, since that isn't
  obvious from a static preview.
- Mention the full-expansion JSON path if they might want to dig into what got
  filtered out; it's not usually worth sending as its own attachment.

## Explicitly out of scope here

Per `CLAUDE.md` phase discipline: no persistent store beyond the optional
per-run cache, no embeddings, no ANN/semantic search for papers with no
citation edge to the seed. If a request needs those, say so rather than
building them into this skill.
