"""Per-device Headroom proxy for Claude and Codex; a new release is used only after a test proxy answers both."""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from . import config, paths

PRIVACY_ENV = {"DO_NOT_TRACK": "1", "HEADROOM_BEACON": "off", "HEADROOM_TELEMETRY": "off",
               "HEADROOM_UPDATE_CHECK": "off", "PYTHONIOENCODING": "utf-8"}
PROXY_FLAGS = ["--no-subscription-tracking", "--no-telemetry", "--no-cache", "--no-rate-limit"]
START_WAIT_SECONDS = 60
CODEX_ENV = "SELA_HEADROOM_CODEX_URL"
PROVIDERS = ("claude", "codex")
DEFAULT_URL = "http://127.0.0.1:8787"
PACKAGE = "headroom-ai[proxy]"
RELEASES = "https://api.github.com/repos/headroomlabs-ai/headroom/releases/latest"
CHECK_EVERY_SECONDS = 24 * 3600
LOCK_STALE_SECONDS = 2 * 3600
PROBE = "Balas hanya dengan kata OK."
# DETACHED_PROCESS leaves Headroom's own python child without a console, and Windows opens a window for it.
_DETACHED = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

_routed: set[str] = set()
_url = DEFAULT_URL
_notices: list[str] = []


def settings() -> dict:
    return config.load().get("headroom", {})


def root() -> Path:
    return paths.data_dir() / "headroom"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def state() -> dict:
    return _read_json(root() / "state.json")


def _venv_binary(venv: Path) -> Path:
    return venv / ("Scripts/headroom.exe" if os.name == "nt" else "bin/headroom")


def _installed(venv: Path) -> bool:
    return (venv / ".sela-ok").exists() and _venv_binary(venv).exists()


def binary(cfg: dict) -> Path | None:
    if cfg.get("binary"):
        return Path(cfg["binary"])
    current = state().get("current")
    venv = root() / "versions" / str(current)
    return _venv_binary(venv) if current and _installed(venv) else None


def home(cfg: dict) -> Path:
    return Path(cfg["home"]) if cfg.get("home") else root() / "home"


def alive(url: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url}/livez", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def _wait(url: str, seconds: int = START_WAIT_SECONDS) -> bool:
    for _ in range(seconds * 2):
        if alive(url):
            return True
        time.sleep(0.5)
    return False


def _spawn(exe: Path, workdir: Path, port: str) -> subprocess.Popen:
    workdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **PRIVACY_ENV, "HEADROOM_WORKSPACE_DIR": str(workdir),
           "HF_HOME": str(root() / "hf-cache")}
    # The proxy must reach the real APIs, not the base URL sela points the providers at.
    if "claude" in _routed:
        env.pop("ANTHROPIC_BASE_URL", None)
    env.pop(CODEX_ENV, None)
    with open(workdir / "proxy.out", "ab") as log:
        options = {"creationflags": _DETACHED} if os.name == "nt" else {"start_new_session": True}
        return subprocess.Popen([str(exe), "proxy", "--port", port, *PROXY_FLAGS,
                                 "--log-file", str(workdir / "proxy.jsonl")],
                                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options)


def _kill_tree(pid: int):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(pid, 15)
        except OSError:
            pass


def _is_headroom(pid: int) -> bool:
    if os.name != "nt":
        return Path(f"/proc/{pid}").exists()
    result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True)
    return "headroom" in result.stdout.lower()


def start(cfg: dict) -> bool:
    exe = binary(cfg)
    if exe is None:
        return False
    url = cfg.get("url", DEFAULT_URL)
    proc = _spawn(exe, home(cfg), url.rsplit(":", 1)[-1])
    _write_json(root() / "running.json", {"pid": proc.pid, "version": state().get("current"), "at": _now()})
    return _wait(url)


def _stop_outdated(cfg: dict) -> None:
    running = _read_json(root() / "running.json")
    current = state().get("current")
    pid = running.get("pid")
    if cfg.get("binary") or not isinstance(pid, int) or not current or running.get("version") == current:
        return
    if _is_headroom(pid):
        _kill_tree(pid)
        for _ in range(20):
            if not alive(cfg.get("url", DEFAULT_URL)):
                break
            time.sleep(0.5)


def _say(say, text: str):
    _notices.append(text)
    say(text)


def take_notices() -> list[str]:
    taken = list(_notices)
    _notices.clear()
    return taken


def _notify_update(say):
    result = state().get("result")
    if not isinstance(result, dict):
        return
    seen = root() / "seen.json"
    if _read_json(seen).get("at") == result.get("at"):
        return
    if not result.get("ok"):
        _say(say, f"Headroom {result.get('version')} gagal diperiksa ({result.get('reason')}); "
                  f"tetap memakai {state().get('current') or 'koneksi langsung'}")
    _write_json(seen, {"at": result.get("at")})


def _update_due() -> bool:
    checked = state().get("checked_at")
    if not checked:
        return True
    try:
        age = time.time() - datetime.fromisoformat(checked).timestamp()
    except ValueError:
        return True
    return age >= CHECK_EVERY_SECONDS


def _schedule_update(cfg: dict):
    if cfg.get("binary") or not cfg.get("auto_update", True):
        return
    if binary(cfg) is not None and not _update_due():
        return
    if _lock_held():
        return
    log_path = root() / "update.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    options = {"creationflags": _DETACHED} if os.name == "nt" else {"start_new_session": True}
    try:
        with open(log_path, "ab") as log:
            subprocess.Popen([sys.executable, "-m", "airouter", "--headroom-auto"], stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, **options)
    except OSError:
        pass


def apply(say=config.say) -> None:
    global _url
    cfg = settings()
    if not cfg.get("enabled"):
        return
    _notify_update(say)
    _schedule_update(cfg)
    wanted = [name for name in cfg.get("providers", PROVIDERS) if name in PROVIDERS]
    if "claude" in wanted and "ANTHROPIC_BASE_URL" in os.environ:
        wanted.remove("claude")
    if not wanted:
        return
    if binary(cfg) is None:
        how = "sedang dipasang di latar belakang" if cfg.get("auto_update", True) else \
            "jalankan sela --headroom-update untuk memasang"
        _say(say, f"Headroom belum terpasang di device ini ({how}); sementara Claude dan Codex dipanggil langsung")
        return
    _url = cfg.get("url", DEFAULT_URL)
    if alive(_url):
        _stop_outdated(cfg)
    if not alive(_url):
        say("Menyalakan Headroom, butuh sekitar 40 detik...")
        try:
            start(cfg)
        except OSError:
            pass
    if not alive(_url):
        _say(say, "Headroom tidak jalan; Claude dan Codex dipanggil langsung")
        return
    _route(wanted)


def _route(names):
    for name in names:
        _routed.add(name)
        os.environ["ANTHROPIC_BASE_URL" if name == "claude" else CODEX_ENV] = _url


def _unroute():
    for name in _routed:
        os.environ.pop("ANTHROPIC_BASE_URL" if name == "claude" else CODEX_ENV, None)
    _routed.clear()


def ensure(say) -> bool:
    """Restart a proxy that died mid-session; True when providers were moved to direct calls and must restart."""
    if not _routed or alive(_url):
        return False
    say("Headroom mati; menyalakan ulang, butuh sekitar 40 detik...")
    try:
        if start(settings()):
            return False
    except OSError:
        pass
    _unroute()
    say("Headroom tidak bisa dinyalakan; Claude dan Codex sekarang dipanggil langsung")
    return True


def codex_args() -> list[str]:
    url = os.environ.get(CODEX_ENV)
    return ["-c", f'openai_base_url="{url}/v1"'] if url else []


def _lock_held() -> bool:
    lock = root() / "update.lock"
    try:
        return time.time() - lock.stat().st_mtime < LOCK_STALE_SECONDS
    except OSError:
        return False


def latest_version() -> str:
    request = urllib.request.Request(RELEASES, headers={"User-Agent": "sela", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        tag = str(json.loads(response.read().decode("utf-8")).get("tag_name", ""))
    version = tag.lstrip("v")
    if not re.fullmatch(r"\d+(\.\d+)*", version):
        raise ValueError(f"tag rilis tidak dikenal: {tag!r}")
    return version


def _install(version: str, venv: Path, say):
    shutil.rmtree(venv, ignore_errors=True)
    venv.parent.mkdir(parents=True, exist_ok=True)
    python = getattr(sys, "_base_executable", sys.executable)
    say(f"Memasang Headroom {version} ke {venv}")
    subprocess.run([python, "-m", "venv", str(venv)], check=True, capture_output=True, timeout=600)
    inner = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(inner), "-m", "pip", "install", "--disable-pip-version-check", "-q",
                    f"{PACKAGE}=={version}"], check=True, capture_output=True, timeout=3600)
    (venv / ".sela-ok").write_text(_now(), encoding="utf-8")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _light_model(provider: str, fallback: str) -> str:
    route = config.load().get("tiers", {}).get("light", {}).get(provider, {})
    return route.get("model", fallback) if isinstance(route, dict) else fallback


def _probe(provider: str, url: str) -> str | None:
    from .providers import claude, codex
    env = {key: value for key, value in os.environ.items() if key not in ("ANTHROPIC_BASE_URL", CODEX_ENV)}
    if provider == "claude":
        env["ANTHROPIC_BASE_URL"] = url
        cmd = claude.executable() + ["-p", "--model", _light_model("claude", "haiku"), "--tools", "",
                                     "--no-session-persistence", "--strict-mcp-config", "--output-format", "json"]
    else:
        cmd = codex.executable() + ["exec", "-m", _light_model("codex", "gpt-5.6-luna"),
                                    "-c", "model_reasoning_effort=low", "-c", f'openai_base_url="{url}/v1"',
                                    "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-"]
    try:
        result = subprocess.run(cmd, input=PROBE.encode("utf-8"), capture_output=True, timeout=180,
                                env=env, cwd=str(root()))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{provider}: {exc}"
    out = result.stdout.decode("utf-8", "replace")
    if provider == "claude":
        try:
            data = json.loads(out)
        except ValueError:
            data = {"is_error": True, "result": out[-200:]}
        answer = "" if data.get("is_error") else str(data.get("result", ""))
    else:
        answer = out if result.returncode == 0 else ""
    if "OK" not in answer.upper():
        tail = result.stderr.decode("utf-8", "replace").strip()[-200:] or out.strip()[-200:]
        return f"{provider} tidak menjawab lewat proxy: {tail or 'tanpa output'}"
    return None


def verify(cfg: dict, exe: Path, say) -> str | None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    workdir = root() / "verify"
    shutil.rmtree(workdir, ignore_errors=True)
    proc = _spawn(exe, workdir, str(port))
    try:
        if not _wait(url, START_WAIT_SECONDS * 3):
            return "proxy tes tidak menyala"
        for provider in [name for name in cfg.get("providers", PROVIDERS) if name in PROVIDERS]:
            say(f"Memeriksa {provider} lewat proxy tes")
            reason = _probe(provider, url)
            if reason:
                return reason
        return None
    finally:
        _kill_tree(proc.pid)
        for _ in range(20):
            shutil.rmtree(workdir, ignore_errors=True)
            if not workdir.exists():
                break
            time.sleep(0.5)


def _prune(keep: set[str]):
    versions = root() / "versions"
    if not versions.exists():
        return
    for child in versions.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child, ignore_errors=True)


def update(say=print, force: bool = False) -> bool:
    cfg = settings()
    if cfg.get("binary"):
        say("[headroom] binary diatur manual; update otomatis dilewati")
        return False
    lock = root() / "update.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if _lock_held():
        say("Update Headroom lain sedang berjalan")
        return False
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return _update(cfg, say, force)
    finally:
        lock.unlink(missing_ok=True)


def _update(cfg: dict, say, force: bool) -> bool:
    current_state = state()
    current_state["checked_at"] = _now()
    try:
        latest = latest_version()
    except (OSError, ValueError) as exc:
        say(f"Gagal cek versi Headroom: {exc}")
        _write_json(root() / "state.json", current_state)
        return False
    current = current_state.get("current")
    say(f"Headroom terbaru {latest}, terpasang {current or 'belum ada'}")
    if latest == current and binary(cfg) is not None:
        _write_json(root() / "state.json", current_state)
        return True
    failed = list(current_state.get("failed", []))
    if latest in failed and not force:
        _write_json(root() / "state.json", current_state)
        return False
    venv = root() / "versions" / latest
    try:
        _install(latest, venv, say)
        reason = verify(cfg, _venv_binary(venv), say)
    except (OSError, subprocess.SubprocessError) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        reason = f"pemasangan gagal: {stderr.decode('utf-8', 'replace').strip()[-200:] or exc}"
    current_state = state()
    current_state["checked_at"] = _now()
    if reason:
        shutil.rmtree(venv, ignore_errors=True)
        current_state["failed"] = sorted(set(failed) | {latest})
        current_state["result"] = {"version": latest, "ok": False, "reason": reason, "at": _now()}
        _write_json(root() / "state.json", current_state)
        say(f"Headroom {latest} gagal diperiksa: {reason}")
        return False
    current_state.update(current=latest, previous=current,
                         failed=[version for version in failed if version != latest],
                         result={"version": latest, "ok": True, "at": _now()})
    _write_json(root() / "state.json", current_state)
    _prune({latest, str(current)})
    say(f"Headroom {latest} lolos pemeriksaan dan sekarang dipakai")
    return True
