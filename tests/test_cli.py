"""CLI integration tests. Network calls are mocked with respx; no real
OpenAlex traffic happens in the test suite."""

from __future__ import annotations

import json

import httpx
import respx
from click.testing import CliRunner

from tests.conftest import make_work_payload
from treexiv.cli import main
from treexiv.exceptions import SeedIdentificationError
from treexiv.models import Edge, ExpansionResult, Node


@respx.mock
def test_identify_seed_command_prints_guess_json(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
    reply = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"search_query": "Attention Is All You Need", "confidence": "high"}
                    ),
                }
            }
        ]
    }
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=reply)
    )
    runner = CliRunner()
    result = runner.invoke(main, ["identify-seed", "the transformer paper", "--no-web"])
    assert result.exit_code == 0, result.output
    assert route.called
    guess = json.loads(result.output)
    assert guess["search_query"] == "Attention Is All You Need"


def test_identify_seed_command_errors_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    runner = CliRunner()
    result = runner.invoke(main, ["identify-seed", "the transformer paper"])
    assert result.exit_code != 0
    assert isinstance(result.exception, SeedIdentificationError)
    assert "OPENROUTER_API_KEY" in str(result.exception)


@respx.mock
def test_search_seed_prints_json_candidates() -> None:
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200, json={"results": [make_work_payload("W1", "Recursive LMs", cited_by_count=10)]}
        )
    )
    runner = CliRunner()
    result = runner.invoke(main, ["search-seed", "recursive language models"])
    assert result.exit_code == 0
    candidates = json.loads(result.output)
    assert candidates[0]["id"] == "W1"
    assert candidates[0]["title"] == "Recursive LMs"


@respx.mock
def test_expand_command_writes_expansion_json(tmp_path) -> None:
    seed_payload = make_work_payload("W1", "Seed", referenced_works=["R1"])
    respx.get("https://api.openalex.org/works/W1").mock(
        return_value=httpx.Response(200, json=seed_payload)
    )
    ref_payload = make_work_payload("R1", "Reference", cited_by_count=2)
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [ref_payload]})
    )
    out_path = tmp_path / "expansion.json"
    runner = CliRunner()
    result = runner.invoke(main, ["expand", "W1", "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["seed_id"] == "W1"
    assert any(n["id"] == "R1" for n in payload["nodes"])


def test_filter_command_creates_missing_output_directory(tmp_path) -> None:
    seed = Node("SEED", "Seed", 2020, 10, [], None, "abstract", 0)
    expansion = ExpansionResult(seed_id="SEED", nodes={"SEED": seed}, edges=[])
    expansion_path = tmp_path / "expansion.json"
    expansion_path.write_text(json.dumps(expansion.to_dict()), encoding="utf-8")

    out_path = tmp_path / "nested" / "does" / "not" / "exist" / "filtered.json"
    runner = CliRunner()
    result = runner.invoke(
        main, ["filter", str(expansion_path), "--idea", "seed", "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()


def test_filter_command_writes_filtered_json(tmp_path) -> None:
    seed = Node("SEED", "Recursive language models", 2020, 10, [], None, "core idea text", 0)
    other = Node("W1", "Gardening tips", 2020, 1, [], None, "tomatoes and soil", 1)
    expansion = ExpansionResult(
        seed_id="SEED", nodes={"SEED": seed, "W1": other}, edges=[Edge("SEED", "W1")]
    )
    expansion_path = tmp_path / "expansion.json"
    expansion_path.write_text(json.dumps(expansion.to_dict()), encoding="utf-8")

    out_path = tmp_path / "filtered.json"
    runner = CliRunner()
    args = [
        "filter",
        str(expansion_path),
        "--idea",
        "recursive language models",
        "--top-k",
        "1",
        "--out",
        str(out_path),
    ]
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    ids = {n["id"] for n in payload["nodes"]}
    assert "SEED" in ids


def test_render_command_writes_html(tmp_path) -> None:
    seed = Node("SEED", "Seed Paper", 2020, 10, [], None, "abstract", 0)
    from treexiv.models import FilteredGraph, ScoredNode

    graph = FilteredGraph(
        seed_id="SEED", idea_text="idea", top_k=10, nodes=[ScoredNode(seed, 1.0)], edges=[]
    )
    filtered_path = tmp_path / "filtered.json"
    filtered_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")

    out_path = tmp_path / "tree.html"
    runner = CliRunner()
    result = runner.invoke(main, ["render", str(filtered_path), "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    assert out_path.exists()


@respx.mock
def test_run_command_chains_expand_filter_render(tmp_path) -> None:
    respx.get("https://api.openalex.org/works/W1").mock(
        return_value=httpx.Response(
            200, json=make_work_payload("W1", "Recursive language models", referenced_works=[])
        )
    )
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    out_json = tmp_path / "filtered.json"
    out_html = tmp_path / "tree.html"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "W1",
            "--idea",
            "recursive language models",
            "--out-json",
            str(out_json),
            "--out-html",
            str(out_html),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_json.exists()
    assert out_html.exists()


@respx.mock
def test_run_command_writes_full_expansion_json_by_default(tmp_path) -> None:
    seed_payload = make_work_payload("W1", "Seed", referenced_works=["R1"])
    respx.get("https://api.openalex.org/works/W1").mock(
        return_value=httpx.Response(200, json=seed_payload)
    )
    ref_payload = make_work_payload("R1", "Reference", cited_by_count=2)
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [ref_payload]})
    )
    out_json = tmp_path / "filtered.json"
    out_html = tmp_path / "tree.html"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "W1",
            "--idea",
            "reference",
            "--out-json",
            str(out_json),
            "--out-html",
            str(out_html),
        ],
    )
    assert result.exit_code == 0, result.output
    expansion_path = tmp_path / "filtered.expansion.json"
    assert expansion_path.exists()
    payload = json.loads(expansion_path.read_text(encoding="utf-8"))
    assert payload["seed_id"] == "W1"
    # The full expansion keeps every collected node, unlike the filtered file.
    assert any(n["id"] == "R1" for n in payload["nodes"])


@respx.mock
def test_run_command_respects_explicit_out_expansion(tmp_path) -> None:
    seed_payload = make_work_payload("W1", "Seed", referenced_works=[])
    respx.get("https://api.openalex.org/works/W1").mock(
        return_value=httpx.Response(200, json=seed_payload)
    )
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    custom_path = tmp_path / "custom_expansion.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "W1",
            "--idea",
            "idea",
            "--out-json",
            str(tmp_path / "filtered.json"),
            "--out-html",
            str(tmp_path / "tree.html"),
            "--out-expansion",
            str(custom_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert custom_path.exists()
    assert not (tmp_path / "filtered.expansion.json").exists()
