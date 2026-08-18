# CLAUDE.md

Guidance for working in this repo. Full design context: `README.md`, `scratch/treexiv-product-design.md` (phase-by-phase scope, technical tasks, exit criteria), `scratch/treexiv-project-management.md` (process, tooling, cost).

## What this is

TreeXiv builds a citation- and semantic-lineage tree from a seed paper or concept: ancestors (where the idea came from) and descendants (what it grew into), with edges labeled by *why* two papers are connected, not just that they are. Everything in the system is either **creation-time** (offline ingestion: papers, citation graph, embeddings, idea-stem extraction) or **inference-time** (per-query: traverse, expand, prune, render).

## Phase discipline — the single most important rule

The project is split into three phases (0: MVP query-time lineage pipeline, 1: semantic discovery + pruning + frontend, 2: multi-user product), each with an explicit exit criterion in `scratch/treexiv-product-design.md`. **Do not build Phase N+1 scope while Phase N's exit criteria are unmet.** Concretely, until Phase 0 is validated:

- No persistent store, no `pgvector`, no ANN/embedding search. Query-time BM25 over the corpus collected for one run *is* in scope for Phase 0 — see Phase 0 specifics below — but a persistent/indexed search layer (Postgres `tsvector`, a dedicated search service, anything that outlives one run) is not.
- No weekly batch ingestion job — Phase 0 is a one-shot script/skill run against a single seed.
- No embeddings, no SPECTER2, no citation-intent classification or triangulation-based edge typing.
- No real frontend — a self-contained HTML artifact generated per run is sufficient.
- No auth, no multi-user anything.

If a task looks like it belongs to a later phase, say so and confirm before building it rather than quietly scope-creeping.

## Phase 0 specifics (current phase)

**Data source note:** the original Phase 0 plan assumed the Semantic Scholar Graph API and native citation-intent fields (`intents`, `isInfluential`) as the primary relevance signal. S2 tightened API key issuance with approval timing unknown, so Phase 0 now runs on **OpenAlex** instead (free, no approval gate), with intent-based edge typing deferred to whenever the Phase 0.5 data-source decision point in `scratch/treexiv-product-design.md` is reached. Full spec for what's actually being built: `scratch/treexiv-mvp-openalex-prd.md`.

Goal: given one seed paper and a stated "core idea," prove that a capped two-hop OpenAlex expansion + BM25 relevance filter produces a lineage tree worth having, before investing in citation-intent classification or embeddings.

Process, in order (detail in the PRD):
1. Resolve the seed reference to one OpenAlex work ID (title search + disambiguation — see "How Phase 0 is driven" below for who does this).
2. Two-hop bidirectional expansion (backward via `referenced_works`, forward via `filter=cites:{id}`), capped per-node (fan-out) and globally (total corpus).
3. Build a BM25 corpus over title + reconstructed abstract for every collected node.
4. Filter to the top-K nodes by BM25 score against the user's core-idea text (seed always retained).
5. Render the filtered graph as a standalone interactive HTML file.

No `papers`/`edges` database tables in this phase — the `ExpansionResult` node/edge JSON (see `src/treexiv/models.py`) is the same shape a later persistent store would hold, but nothing here writes it anywhere durable beyond an optional per-run cache.

When Phase 0 work lands, track against its exit criteria explicitly (see `scratch/treexiv-product-design.md`): does the pipeline beat the manual RLM map, is a full run hands-off beyond seed/idea input and ambiguity confirmation, does call volume stay within OpenAlex's free allowance.

## How Phase 0 is driven

Phase 0 is built as **a Claude Code skill (`.claude/skills/treexiv-lineage/SKILL.md`) backed by a plain Python package (`src/treexiv/`)**, not a standalone human-facing tool. The package handles everything mechanical and is fully unit-tested in isolation; the PRD's Step 1 (confirming a seed-paper match, including a live web-search cross-check) is deliberately left to whatever LLM/agent invokes the skill, since that step needs judgment and tool use a deterministic script doesn't have. Don't try to fold that disambiguation logic into the package itself — see the skill file and README for the reasoning. Making this robust to harnesses other than Claude Code, and any further harness/tool integration, is explicitly deferred past this first cut.

## Conventions

- This is a solo prototyping project; Linear is the source of truth for tickets, GitHub for code. Ticket-sized units of work are things completable and testable in isolation (see README).
- Phase 0 code lives in `src/treexiv/`, managed with `uv` (`uv sync`, `uv run pytest`, `uv run treexiv ...`). Dependencies are kept minimal: an OpenAlex HTTP client, `rank_bm25`, `click` for the CLI. Rendering (`render.py`) writes its own HTML/CSS/JS template around a vendored copy of vis-network (`src/treexiv/assets/vis-network/`, see that dir's `NOTICE.md`) rather than depending on a wrapper library — see `render.py`'s module docstring for why. No Postgres, no Anthropic SDK, no GCP infra, no frontend framework until the phase that calls for them.
- Secrets (`OPENALEX_API_KEY`, optionally `OPENALEX_MAILTO`) belong in a local `.env`, never committed — see `.env.example` and `.gitignore`. `ANTHROPIC_API_KEY`/S2 API key/`DATABASE_URL` are not needed until later phases.
