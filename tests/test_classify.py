import pytest

from airouter import config
from airouter.classify import classify

RULES = config.load()["rules"]


@pytest.mark.parametrize("prompt, tier", [
    ("apa itu decorator di python?", "light"),
    ("translate 'retry on transport errors' ke bahasa indonesia", "light"),
    ("perbaiki error di fungsi parse_period", "medium"),
    ("tambah test untuk PERIOD_API", "medium"),
    ("kenapa mic ISO kehilangan frame tiap 160 detik? cari root cause", "heavy"),
    ("rancang arsitektur router multi-model dengan fallback", "heavy"),
    ("rancang arsitektur cache yang tahan restart, cukup 3 poin singkat", "heavy"),
])
def test_tier(prompt, tier):
    assert classify(prompt, RULES).tier == tier


def test_no_keyword_is_ambiguous():
    assert classify("joypad dongle hari ini", RULES).ambiguous


def test_keyword_is_not_ambiguous():
    assert not classify("apa itu LC3", RULES).ambiguous


def test_repo_session_raises_tier():
    assert classify("apa itu LC3", RULES).score < classify("apa itu LC3", RULES, repo=True).score


def test_traceback_counts_as_code():
    prompt = "Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError"
    assert "code or log" in classify(prompt, RULES).reasons


def test_config_with_bom_loads(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_bytes(b"\xef\xbb\xbf" + config.PACKAGE_CONFIG.read_bytes())
    monkeypatch.setenv("AI_ROUTER_CONFIG", str(path))
    assert "claude" in config.load()["providers"]


def test_keyword_needs_word_start():
    assert not classify("cdebugger", RULES).reasons
