from airouter import config, headroom
from airouter.providers import codex
from airouter.security import SecurityPolicy


CFG = {"enabled": True, "providers": ["claude", "codex"], "url": "http://127.0.0.1:8787",
       "binary": "headroom", "home": "unused"}


def test_disabled_changes_nothing(monkeypatch):
    monkeypatch.setattr(headroom, "settings", lambda: {"enabled": False})
    monkeypatch.setattr(headroom, "start", lambda cfg: (_ for _ in ()).throw(AssertionError("started")))
    headroom.apply()
    assert "ANTHROPIC_BASE_URL" not in headroom.os.environ
    assert headroom.CODEX_ENV not in headroom.os.environ
    assert headroom.codex_args() == []


def test_routes_both_providers_when_proxy_is_up(monkeypatch):
    monkeypatch.setattr(headroom, "settings", lambda: CFG)
    monkeypatch.setattr(headroom, "alive", lambda url: True)
    headroom.apply()
    assert headroom.os.environ["ANTHROPIC_BASE_URL"] == CFG["url"]
    assert headroom.os.environ[headroom.CODEX_ENV] == CFG["url"]
    override = ["-c", 'openai_base_url="http://127.0.0.1:8787/v1"']
    policy = SecurityPolicy.from_config(config.load())
    cmd, _ = codex.build("m", "low", "x", False, policy)
    assert cmd[cmd.index("exec"):].count(override[1]) == 1
    chat = codex.chat_command("m", "low", None, [], policy, [])
    assert override[1] in chat[chat.index("exec"):]
    server = codex.server_command(policy, [])
    assert override[1] in server[server.index("app-server"):]
    monkeypatch.delenv("ANTHROPIC_BASE_URL")
    monkeypatch.delenv(headroom.CODEX_ENV)


def test_falls_back_to_direct_when_proxy_cannot_start(monkeypatch):
    monkeypatch.setattr(headroom, "settings", lambda: CFG)
    monkeypatch.setattr(headroom, "alive", lambda url: False)
    monkeypatch.setattr(headroom, "start", lambda cfg: None)
    headroom.apply()
    assert "ANTHROPIC_BASE_URL" not in headroom.os.environ
    assert headroom.CODEX_ENV not in headroom.os.environ


def test_respects_existing_claude_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://example.test")
    monkeypatch.setattr(headroom, "settings", lambda: {**CFG, "providers": ["claude"]})
    monkeypatch.setattr(headroom, "alive", lambda url: (_ for _ in ()).throw(AssertionError("probed")))
    headroom.apply()
    assert headroom.os.environ["ANTHROPIC_BASE_URL"] == "http://example.test"


def test_codex_only(monkeypatch):
    monkeypatch.setattr(headroom, "settings", lambda: {**CFG, "providers": ["codex"]})
    monkeypatch.setattr(headroom, "alive", lambda url: True)
    headroom.apply()
    assert "ANTHROPIC_BASE_URL" not in headroom.os.environ
    assert headroom.os.environ[headroom.CODEX_ENV] == CFG["url"]
    monkeypatch.delenv(headroom.CODEX_ENV)


def test_proxy_runs_lossy_and_never_points_at_itself(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(headroom, "_routed", {"claude"})
    monkeypatch.setenv("ANTHROPIC_BASE_URL", CFG["url"])
    monkeypatch.setattr(headroom.subprocess, "Popen", lambda args, **kwargs: seen.append((args, kwargs)) or FakeProc())
    monkeypatch.setattr(headroom, "alive", lambda url: True)
    assert headroom.start({**CFG, "home": str(tmp_path / "home")})
    args, kwargs = seen[0]
    assert "--lossless" not in args
    assert "ANTHROPIC_BASE_URL" not in kwargs["env"]


class FakeProc:
    pid = 4242


def test_ensure_restarts_a_dead_proxy(monkeypatch):
    monkeypatch.setattr(headroom, "_routed", {"claude"})
    monkeypatch.setenv("ANTHROPIC_BASE_URL", CFG["url"])
    monkeypatch.setattr(headroom, "settings", lambda: CFG)
    monkeypatch.setattr(headroom, "alive", lambda url: False)
    monkeypatch.setattr(headroom, "start", lambda cfg: True)
    assert headroom.ensure(lambda text: None) is False
    assert headroom.os.environ["ANTHROPIC_BASE_URL"] == CFG["url"]


def test_ensure_moves_to_direct_calls_when_restart_fails(monkeypatch):
    monkeypatch.setattr(headroom, "_routed", {"claude", "codex"})
    monkeypatch.setenv("ANTHROPIC_BASE_URL", CFG["url"])
    monkeypatch.setenv(headroom.CODEX_ENV, CFG["url"])
    monkeypatch.setattr(headroom, "settings", lambda: CFG)
    monkeypatch.setattr(headroom, "alive", lambda url: False)
    monkeypatch.setattr(headroom, "start", lambda cfg: False)
    assert headroom.ensure(lambda text: None) is True
    assert "ANTHROPIC_BASE_URL" not in headroom.os.environ
    assert headroom.codex_args() == []
    assert headroom.ensure(lambda text: None) is False


def test_ensure_ignores_a_session_without_headroom(monkeypatch):
    monkeypatch.setattr(headroom, "alive", lambda url: (_ for _ in ()).throw(AssertionError("probed")))
    assert headroom.ensure(lambda text: None) is False


def _fake_install(version, venv, say):
    binary = headroom._venv_binary(venv)
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("")
    (venv / ".sela-ok").write_text("")


def _installed_state(version):
    _fake_install(version, headroom.root() / "versions" / version, None)
    headroom._write_json(headroom.root() / "state.json", {"current": version})


def test_update_switches_only_after_the_check_passes(monkeypatch):
    monkeypatch.setattr(headroom, "settings", lambda: {**CFG, "binary": ""})
    _installed_state("0.43.0")
    _fake_install("0.42.0", headroom.root() / "versions" / "0.42.0", None)
    monkeypatch.setattr(headroom, "latest_version", lambda: "0.45.0")
    monkeypatch.setattr(headroom, "_install", _fake_install)
    monkeypatch.setattr(headroom, "verify", lambda cfg, exe, say: None)
    assert headroom.update(say=lambda text: None)
    state = headroom.state()
    assert (state["current"], state["previous"], state["result"]["ok"]) == ("0.45.0", "0.43.0", True)
    assert sorted(p.name for p in (headroom.root() / "versions").iterdir()) == ["0.43.0", "0.45.0"]
    assert not (headroom.root() / "update.lock").exists()


def test_failed_check_keeps_the_old_version_and_is_not_retried(monkeypatch):
    monkeypatch.setattr(headroom, "settings", lambda: {**CFG, "binary": ""})
    _installed_state("0.44.0")
    installs = []
    monkeypatch.setattr(headroom, "latest_version", lambda: "0.45.0")
    monkeypatch.setattr(headroom, "_install", lambda *args: installs.append(args) or _fake_install(*args))
    monkeypatch.setattr(headroom, "verify", lambda cfg, exe, say: "codex tidak menjawab lewat proxy")
    assert not headroom.update(say=lambda text: None)
    state = headroom.state()
    assert state["current"] == "0.44.0" and state["failed"] == ["0.45.0"]
    assert state["result"] == {**state["result"], "ok": False, "version": "0.45.0"}
    assert not (headroom.root() / "versions" / "0.45.0").exists()
    assert not headroom.update(say=lambda text: None)
    assert len(installs) == 1


def test_update_result_is_announced_once(monkeypatch):
    headroom._write_json(headroom.root() / "state.json",
                         {"current": "0.44.0", "result": {"version": "0.45.0", "ok": False,
                                                          "reason": "x", "at": "t1"}})
    monkeypatch.setattr(headroom, "_notices", [])
    headroom._notify_update(lambda text: None)
    headroom._notify_update(lambda text: None)
    notes = headroom.take_notices()
    assert len(notes) == 1 and "0.45.0" in notes[0] and "0.44.0" in notes[0]


def test_missing_install_runs_direct_and_installs_in_background(monkeypatch):
    launched = []
    monkeypatch.setattr(headroom, "settings", lambda: {**CFG, "binary": ""})
    monkeypatch.setattr(headroom.subprocess, "Popen", lambda args, **kwargs: launched.append(args))
    monkeypatch.setattr(headroom, "alive", lambda url: (_ for _ in ()).throw(AssertionError("probed")))
    headroom.apply(say=lambda text: None)
    assert launched and launched[0][-1] == "--headroom-auto"
    assert "ANTHROPIC_BASE_URL" not in headroom.os.environ


def test_outdated_proxy_is_stopped(monkeypatch):
    _installed_state("0.45.0")
    headroom._write_json(headroom.root() / "running.json", {"pid": 4242, "version": "0.44.0"})
    killed = []
    monkeypatch.setattr(headroom, "_is_headroom", lambda pid: True)
    monkeypatch.setattr(headroom, "_kill_tree", killed.append)
    monkeypatch.setattr(headroom, "alive", lambda url: False)
    headroom._stop_outdated({**CFG, "binary": ""})
    assert killed == [4242]
