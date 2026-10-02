import pytest


@pytest.fixture(autouse=True)
def _log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ROUTER_LOG", str(tmp_path / "routes.jsonl"))
    monkeypatch.setenv("AI_ROUTER_LABELS", str(tmp_path / "labels.jsonl"))
    monkeypatch.setenv("AI_ROUTER_ATTACH", str(tmp_path / "attachments"))
