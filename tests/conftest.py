import pytest

from airouter import learn


@pytest.fixture(autouse=True)
def _log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ROUTER_LOG", str(tmp_path / "routes.jsonl"))
    monkeypatch.setenv("AI_ROUTER_LABELS", str(tmp_path / "labels.jsonl"))
    monkeypatch.setenv("AI_ROUTER_ATTACH", str(tmp_path / "attachments"))
    monkeypatch.setenv("AI_ROUTER_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(learn, "warm_encoder", lambda info=None: None)
