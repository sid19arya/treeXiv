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
| `treexiv identify-seed` | Optional Step 0: don't know the exact paper? Describe it and a web-searching LLM (OpenRouter) guesses the title / arXiv ID and a query to feed the pipeline. |

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

Don't have a specific paper in mind? Describe the idea and let TreeXiv guess
the paper first:

```
I'm thinking of a mid-2010s paper that framed dropout as approximate
Bayesian inference in deep networks — find it and trace its lineage.
```

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

Only have a fuzzy description? Step 0 turns it into a search:

```bash
uv run treexiv identify-seed "the 2017 paper that introduced the transformer" \
  # -> JSON with a "search_query" field; pass that to search-seed
```

`identify-seed` needs `OPENROUTER_API_KEY` (see Configuration); every other
command only talks to OpenAlex. Web-search grounding is on by default
(`--no-web` to disable, or `--model` to override `OPENROUTER_MODEL`).

Curation runs automatically when `OPENROUTER_API_KEY` is set. To pin the
behaviour explicitly, pass `--curation llm|bm25|auto` (and `--max-nodes N` to
change how many papers curation may keep) to `run` or `filter`.

Run `uv run treexiv --help` for the full command list (`identify-seed`,
`search-seed`, `expand`, `filter`, `render`, `run`).

## Deploy as a private web app (optional)

There's a single-page web front-end (`treexiv.web`, a small FastAPI app) that
wraps the same pipeline: search a seed (or describe it and let Step 0 guess),
pick the right match, state the idea, get the HTML back in the browser. It's
built to run on [Render](https://render.com)'s free tier and is **private** —
every route except `/health` is behind HTTP Basic Auth, so without your
credentials a request gets a `401` and nothing runs.

Run it locally:

```bash
uv sync --extra web
TREEXIV_WEB_USER=me TREEXIV_WEB_PASSWORD=secret \
OPENROUTER_API_KEY=sk-or-...  `# optional — enables the "describe it" box` \
  uv run uvicorn treexiv.web:app --reload
# open http://127.0.0.1:8000
```

Deploy to Render: the repo ships a `render.yaml` Blueprint. In the Render
dashboard, **New → Blueprint**, point it at your fork, and provide the
prompted secrets:

| Secret | Value |
|---|---|
| `OPENALEX_API_KEY` | your OpenAlex key |
| `OPENALEX_MAILTO` | your email (OpenAlex polite-pool header) |
| `TREEXIV_WEB_USER` | any username |
| `TREEXIV_WEB_PASSWORD` | a long random string — `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `OPENROUTER_API_KEY` | optional — enables the Step 0 "describe the paper" box; leave blank and that one feature returns `501` while everything else works |

First load prompts for the username/password once; the browser caches it for
the session. Free-tier services spin down after 15 minutes idle and take
~1 minute to wake on the next request.

## How it works

0. **Identify** *(optional)* — if you only have a description, not a paper, a
   web-searching LLM (OpenRouter) guesses which paper you mean and hands back
   a title / arXiv ID and a query string. It's a lead, not a resolution —
   step 1 still runs.
1. **Resolve** — your seed reference (title, DOI, or arXiv ID) is matched to
   one OpenAlex work.
2. **Expand** — a two-hop traversal in both directions: papers your seed
   cites, papers that cite your seed, and one hop further out from each of
   those, capped so it stays fast and the graph stays readable.
3. **Curate** — BM25 narrows the collected papers to a shortlist, then an
   LLM reads that shortlist and decides which papers are actually load-bearing
   for the idea you described, grouping the survivors into a handful of named
   concept clusters and saying in one line why each one earned its place.
   Your seed paper is always kept. Without `OPENROUTER_API_KEY` — or with
   `--curation bm25` — this falls back to the older behaviour: keep the top-K
   papers by BM25 score and nothing else.
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
| `OPENROUTER_API_KEY` | Required for `identify-seed` (Step 0) and for LLM curation; without it, curation falls back to BM25 |
| `OPENROUTER_MODEL` | Model slug for the LLM steps (default `z-ai/glm-5.3-flash`) |
| `TREEXIV_CURATION` | `auto` (default: curate when a key exists, else BM25), `llm` (require curation), or `bm25` (never call an LLM) |
| `TREEXIV_CURATION_MAX_NODES` | Cap on papers curation may keep (default 35) |
| `TREEXIV_CURATION_PREFILTER` | How many BM25-ranked papers curation reads (default 120) |
| `TREEXIV_CURATION_MODEL` | Model slug for curation only, if it should differ from `OPENROUTER_MODEL` |
| `TREEXIV_LLM_WEB_SEARCH` | `true` (default) grounds Step 0 in a web search; `false` disables it |
| `TREEXIV_TOTAL_CORPUS_CAP` | Global cap on papers collected (default 500) |
| `TREEXIV_FANOUT_CAP` | Per-paper cap on references/citations followed (default 100) |
| `TREEXIV_BM25_TOP_K` | How many papers survive the BM25 fallback filter (default 40) |
| `TREEXIV_SAMPLING_STRATEGY` | `top_cited` (default, favors established papers) or `random` (favors catching less-cited, divergent branches) |
| `TREEXIV_CACHE_DIR` | If set, caches fetched papers per seed so repeat runs don't re-hit the API |

## Design notes

TreeXiv is deliberately simple: no database, no embeddings, no background
jobs. Every run is a fresh, disposable query against live OpenAlex data.

Retrieval is still lexical — BM25 over titles and abstracts — so a paper that
shares no vocabulary with your stated idea won't be collected in the first
place, and citation coverage varies by field and publisher. What BM25 no
longer does is decide what you see: it hands a shortlist to an LLM, which
judges lineage rather than word overlap. That's where the graph goes from
"forty papers that mention your keywords" to "the dozen or so that tell the
story." Every LLM step degrades to a deterministic fallback if the key is
missing or the call fails, so a run never depends on it.

## License

MIT
