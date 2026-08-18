import json

from tests.conftest import make_work_payload
from treexiv.cache import WorkCache
from treexiv.models import Work


def test_disabled_cache_returns_none(tmp_path) -> None:
    cache = WorkCache(cache_dir=None, seed_id="W1")
    assert cache.enabled is False
    assert cache.get("W2") is None


def test_put_and_get_round_trips(tmp_path) -> None:
    cache = WorkCache(cache_dir=tmp_path, seed_id="W1")
    cache.put("W2", make_work_payload("W2", "Cached Paper", cited_by_count=3))
    work = cache.get("W2")
    assert work is not None
    assert work.id == "W2"
    assert work.title == "Cached Paper"
    assert work.cited_by_count == 3


def test_put_work_reconstructs_retrievable_payload(tmp_path) -> None:
    original = Work.from_api(make_work_payload("W3", "Roundtrip", authors=["Ana"], venue="ICML"))
    cache = WorkCache(cache_dir=tmp_path, seed_id="W1")
    cache.put_work(original)
    fetched = cache.get("W3")
    assert fetched is not None
    assert fetched.title == "Roundtrip"
    assert fetched.authors == ["Ana"]
    assert fetched.venue == "ICML"


def test_get_many_returns_only_present_entries(tmp_path) -> None:
    cache = WorkCache(cache_dir=tmp_path, seed_id="W1")
    cache.put("W2", make_work_payload("W2", "P2"))
    result = cache.get_many(["W2", "W3"])
    assert set(result.keys()) == {"W2"}


def test_save_persists_to_disk_and_reloads(tmp_path) -> None:
    cache = WorkCache(cache_dir=tmp_path, seed_id="W1")
    cache.put("W2", make_work_payload("W2", "Persisted"))
    cache.save()

    cache_file = tmp_path / "W1.json"
    assert cache_file.exists()
    on_disk = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "W2" in on_disk

    reloaded = WorkCache(cache_dir=tmp_path, seed_id="W1")
    assert reloaded.get("W2").title == "Persisted"


def test_save_is_noop_when_disabled(tmp_path) -> None:
    cache = WorkCache(cache_dir=None, seed_id="W1")
    cache.put("W2", make_work_payload("W2", "P2"))
    cache.save()
    assert list(tmp_path.iterdir()) == []
