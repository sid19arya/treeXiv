from treexiv.config import Settings


def test_defaults_match_prd() -> None:
    settings = Settings()
    assert settings.total_corpus_cap == 500
    assert settings.per_node_fanout_cap == 100
    assert settings.bm25_top_k == 40
    assert settings.sampling_strategy == "top_cited"


def test_from_env_reads_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-key")
    monkeypatch.setenv("OPENALEX_MAILTO", "me@example.com")
    monkeypatch.setenv("TREEXIV_TOTAL_CORPUS_CAP", "10")
    monkeypatch.setenv("TREEXIV_FANOUT_CAP", "5")
    monkeypatch.setenv("TREEXIV_BM25_TOP_K", "7")
    monkeypatch.setenv("TREEXIV_SAMPLING_STRATEGY", "random")

    settings = Settings.from_env()

    assert settings.api_key == "secret-key"
    assert settings.mailto == "me@example.com"
    assert settings.total_corpus_cap == 10
    assert settings.per_node_fanout_cap == 5
    assert settings.bm25_top_k == 7
    assert settings.sampling_strategy == "random"


def test_from_env_reads_openrouter_settings(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-abc")
    monkeypatch.setenv("OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
    monkeypatch.setenv("TREEXIV_LLM_WEB_SEARCH", "false")

    settings = Settings.from_env()

    assert settings.openrouter_api_key == "sk-or-abc"
    assert settings.openrouter_model == "z-ai/glm-5.3-flash"
    assert settings.llm_web_search is False
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_openrouter_model_defaults_when_unset(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_MODEL", "")
    monkeypatch.delenv("TREEXIV_LLM_WEB_SEARCH", raising=False)

    settings = Settings.from_env()

    assert settings.openrouter_api_key is None
    assert settings.openrouter_model == "z-ai/glm-5.3-flash"
    assert settings.llm_web_search is True


def test_from_env_defaults_when_unset(monkeypatch) -> None:
    # Set (not delete) each var to "": `Settings.from_env()` calls
    # `load_dotenv()`, which would otherwise repopulate these from the
    # repo's real `.env` (load_dotenv doesn't override already-set vars, so
    # an explicit "" wins where a plain delenv would not).
    for var in (
        "OPENALEX_API_KEY",
        "OPENALEX_MAILTO",
        "TREEXIV_CACHE_DIR",
    ):
        monkeypatch.setenv(var, "")
    for var in ("TREEXIV_TOTAL_CORPUS_CAP", "TREEXIV_FANOUT_CAP", "TREEXIV_BM25_TOP_K"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TREEXIV_SAMPLING_STRATEGY", "top_cited")

    settings = Settings.from_env()

    assert settings.api_key is None
    assert settings.mailto is None
    assert settings.cache_dir is None
    assert settings.total_corpus_cap == 500
