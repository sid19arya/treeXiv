# TreeXiv

**Trace a paper's citation lineage as an interactive graph, right from Claude Code.**

Give it a seed paper and the idea you actually care about, and TreeXiv maps where that idea came from and what it grew into — a navigable graph instead of a flat reference list, generated fresh from live citation data in one conversation turn.

## The problem

Understanding how a body of work developed usually means manually pulling references, opening citing papers, and building a mental map by hand — tedious, easy to lose track of, and gone the moment you close the tab. Flat citation lists don't show *shape*: what's foundational, what's a recent offshoot, and how any given paper actually connects back to the one you started from.

TreeXiv turns that into a single request: point it at a paper and a stated interest, and it hands back an interactive HTML graph you can open, click through, and keep.

## What's inside

| Component | What it does |
|---|---|
| `/treexiv-lineage` skill | Drives the whole flow inside Claude Code: resolves your seed paper (cross-checking ambiguous matches with a live web search), runs the pipeline, and hands you the result. |
| `treexiv` CLI | The engine underneath — OpenAlex traversal, relevance filtering, and rendering, exposed as composable commands for direct/scripted use. |

## Installation

```bash
git clone https://github.com/sid19arya/treeXiv.git
cd treeXiv
uv sync
cp .env.example .env   # add your OpenAlex API key (see Configuration below)
```

That's it — open the repo in Claude Code and the skill is available automatically. No database, no deployment, nothing else to stand up.

## Usage

Just describe what you want to trace:

```
Trace the lineage of "Attention Is All You Need" — I'm interested in how
self-attention led to modern large language model architectures.
```

```
Build a citation tree for the GPT-3 paper, focused on in-context learning
and few-shot prompting specifically (not the scaling-laws side of it).
```

If your seed paper's title is ambiguous (a lot of papers share near-identical
titles), the skill will cross-check candidates with a web search and ask you
to confirm before running anything.

You'll get back an interactive HTML file — open it in any browser.

### Prefer the CLI directly?

The skill is a thin orchestration layer over the `treexiv` command. If you
already know the exact OpenAlex work ID for your seed paper, you can skip
straight to it:

```bash
uv run treexiv search-seed "attention is all you need"   # find the work ID
uv run treexiv run W2626778328 \
  --idea "self-attention mechanisms for sequence modeling" \
  --out-json output/filtered.json --out-html output/tree.html
```

Run `uv run treexiv --help` for the full command list (`search-seed`,
`expand`, `filter`, `render`, `run`).

## How it works

1. **Resolve** — your seed reference (title, DOI, or arXiv ID) is matched to
   one OpenAlex work.
2. **Expand** — a two-hop traversal in both directions: papers your seed
   cites, papers that cite your seed, and one hop further out from each of
   those, capped so it stays fast and the graph stays readable.
3. **Filter** — every collected paper is scored with BM25 against the idea
   you described, and only the most relevant ones make the cut (your seed
   paper is always kept, regardless of score).
4. **Render** — the result becomes a single self-contained HTML file: no
   server, nothing to host, just open it.

## What you get

- **A sidebar** with the seed paper's details by default; click any node to
  see its title, authors, abstract, and a plain-English description of how
  it connects back to the seed (direct citation, or a two-hop path through
  an intermediate paper).
- **A timeline layout** — older papers sit toward the top-left, newer papers
  toward the bottom-right, with the seed paper pinned at the center, so
  position alone tells you roughly when something happened relative to your
  seed.
- **Directional arrows** that read as "led to": an arrow from A to B means A
  came first and B builds on it (i.e. B cites A) — not just "a citation
  exists."
- Alongside the HTML, you also get the full pre-filter dataset as JSON —
  every paper and citation edge the traversal actually collected, not just
  the ones that made it into the rendered graph — in case you want to dig
  into what got filtered out or rerun filtering with a different framing
  without hitting the API again.

## Configuration

All optional except the API key, which you'll want for anything beyond a
handful of requests:

| Variable | Purpose |
|---|---|
| `OPENALEX_API_KEY` | Recommended — avoids aggressive rate limiting |
| `OPENALEX_MAILTO` | Email for OpenAlex's polite-pool header |
| `TREEXIV_TOTAL_CORPUS_CAP` | Global cap on papers collected (default 500) |
| `TREEXIV_FANOUT_CAP` | Per-paper cap on references/citations followed (default 100) |
| `TREEXIV_BM25_TOP_K` | How many papers survive relevance filtering (default 40) |
| `TREEXIV_SAMPLING_STRATEGY` | `top_cited` (default, favors established papers) or `random` (favors catching less-cited, divergent branches) |
| `TREEXIV_CACHE_DIR` | If set, caches fetched papers per seed so repeat runs don't re-hit the API |

## Design notes

TreeXiv is deliberately simple: no database, no embeddings, no background
jobs. Every run is a fresh, disposable query against live OpenAlex data,
filtered with BM25 rather than a semantic model. That's a real tradeoff —
it won't catch a relevant paper that shares no vocabulary with your stated
idea, and citation coverage varies by field and publisher — but it means
there's nothing to maintain, nothing to go stale, and a result in seconds
rather than a pipeline to operate.

## License

MIT
