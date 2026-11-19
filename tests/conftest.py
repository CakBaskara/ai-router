import os

import pytest

from airouter import config, dispatch, headroom, learn, paths

os.environ["AI_ROUTER_CONFIG"] = str(config.PACKAGE_CONFIG)


@pytest.fixture(autouse=True)
def _log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_ROUTER_CONFIG", str(config.PACKAGE_CONFIG))
    monkeypatch.setenv("AI_ROUTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(paths, "config_dir", lambda: tmp_path / "config")
    monkeypatch.setenv("AI_ROUTER_LOG", str(tmp_path / "routes.jsonl"))
    monkeypatch.setenv("AI_ROUTER_LABELS", str(tmp_path / "labels.jsonl"))
    monkeypatch.setenv("AI_ROUTER_ATTACH", str(tmp_path / "attachments"))
    monkeypatch.setenv("AI_ROUTER_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(dispatch, "read_limits", lambda timeout=20: {})
    monkeypatch.setattr(dispatch, "read_claude_usage", lambda timeout=15: {})
    monkeypatch.setattr(learn, "teach_due", lambda cfg: False)
    monkeypatch.setattr(learn, "warm_encoder", lambda info=None: None)
    monkeypatch.setattr(headroom, "settings", lambda: {})
    monkeypatch.setattr(headroom, "_routed", set())
    monkeypatch.delenv(headroom.CODEX_ENV, raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
