import subprocess
import sys
import threading
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="ConPTY hanya ada di Windows")

from airouter import cli, session

KEY = b"k" * 32
ECHO = "import sys\nfor line in sys.stdin:\n    print('ECHO:' + line.strip(), flush=True)\n"


def start_host(tmp_path, script=ECHO):
    child = tmp_path / "child.py"
    child.write_text(script, encoding="utf-8")
    addr = f"{session.PIPE_PREFIX}sela-test-{uuid.uuid4().hex}"
    host = session.Host(subprocess.list2cmdline([sys.executable, "-u", str(child)]), 80, 24, str(tmp_path), addr, KEY)
    thread = threading.Thread(target=host.run, daemon=True)
    thread.start()
    return host, thread, addr


def read_until(conn, marker: bytes, timeout=10.0):
    seen = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if conn.poll(0.1):
            kind, payload = conn.recv()
            if kind == "ver":
                continue
            if kind == "bye":
                return seen, payload
            seen += payload
            if marker in seen:
                return seen, None
    raise AssertionError(f"{marker!r} tidak muncul; diterima {seen[-300:]!r}")


def test_client_types_into_the_session_and_sees_output(tmp_path):
    host, thread, addr = start_host(tmp_path)
    conn = session.connect(addr, KEY)
    conn.send(("size", 80, 24))
    conn.send(("in", b"halo\r"))
    seen, bye = read_until(conn, b"ECHO:halo")
    assert bye is None
    conn.close()
    host.pty.close()


def test_session_survives_detach_and_redraws_for_the_next_client(tmp_path):
    host, thread, addr = start_host(tmp_path)
    first = session.connect(addr, KEY)
    first.send(("size", 80, 24))
    first.send(("in", b"satu\r"))
    read_until(first, b"ECHO:satu")
    first.close()
    time.sleep(0.5)
    assert host.pty.alive()
    second = session.connect(addr, KEY)
    second.send(("size", 60, 20))
    seen, _ = read_until(second, b"ECHO:satu")
    second.send(("in", b"dua\r"))
    read_until(second, b"ECHO:dua")
    second.close()
    host.pty.close()


def test_new_client_takes_over_and_old_one_is_told(tmp_path):
    host, thread, addr = start_host(tmp_path)
    phone = session.connect(addr, KEY)
    phone.send(("size", 80, 24))
    laptop = session.connect(addr, KEY)
    laptop.send(("size", 120, 40))
    _, bye = read_until(phone, b"\x00never", timeout=5)
    assert bye == session.MOVED
    laptop.send(("in", b"laptop\r"))
    read_until(laptop, b"ECHO:laptop")
    laptop.close()
    host.pty.close()


def test_wrong_key_is_refused_and_host_keeps_serving(tmp_path):
    host, thread, addr = start_host(tmp_path)
    assert session.connect(addr, b"x" * 32) is None
    conn = session.connect(addr, KEY)
    assert conn is not None
    conn.send(("size", 80, 24))
    conn.send(("in", b"ok\r"))
    read_until(conn, b"ECHO:ok")
    conn.close()
    host.pty.close()


def test_client_is_released_when_the_app_exits(tmp_path):
    host, thread, addr = start_host(tmp_path, "import sys\nsys.stdin.readline()\n")
    conn = session.connect(addr, KEY)
    conn.send(("size", 80, 24))
    conn.send(("in", b"keluar\r"))
    _, bye = read_until(conn, b"\x00never")
    assert bye == ""
    thread.join(5)
    assert not thread.is_alive()


def test_connect_returns_none_without_a_host():
    assert session.connect(f"{session.PIPE_PREFIX}sela-test-{uuid.uuid4().hex}", KEY) is None


@pytest.fixture
def tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)


@pytest.mark.parametrize("ssh, alive, expected", [
    (False, False, ["no-host"]),
    (False, True, ["attach"]),
    (True, True, ["attach"]),
    (True, False, ["start", "attach"]),
])
def test_enter_picks_attach_start_or_plain_run(monkeypatch, tty, ssh, alive, expected):
    calls = []
    state = {"alive": alive}
    monkeypatch.setattr(session, "remote_shell", lambda: ssh)
    monkeypatch.setattr(session, "connect", lambda *a: object() if state["alive"] else None)

    def start(argv, cols, rows):
        calls.append("start")
        state["alive"] = True

    monkeypatch.setattr(session, "start_host", start)
    monkeypatch.setattr(session, "attach", lambda conn: calls.append("attach") or 0)
    result = session.enter(["-p", "codex"])
    if result is None:
        calls.append("no-host")
    assert calls == expected


def test_cli_chat_goes_through_session_unless_no_host(monkeypatch, tty):
    seen = []
    from airouter import tui
    monkeypatch.setattr(session, "enter", lambda argv: seen.append(("enter", argv)) or None)
    monkeypatch.setattr(tui, "run", lambda *a, **k: seen.append(("tui",)) or 0)
    assert cli.main(["-p", "codex"]) == 0
    assert seen == [("enter", ["-p", "codex"]), ("tui",)]
    seen.clear()
    assert cli.main(["--no-host"]) == 0
    assert seen == [("tui",)]


def test_host_tells_its_code_version_first(tmp_path):
    host, thread, addr = start_host(tmp_path)
    conn = session.connect(addr, KEY)
    assert conn.recv() == ("ver", session.code_version())
    conn.close()
    host.pty.close()


@pytest.mark.parametrize("first, warned", [(("ver", None), False), (("ver", "lama"), True), (("out", b"x"), True)])
def test_attach_warns_when_the_host_runs_other_code(monkeypatch, capsys, first, warned):
    first = (first[0], session.code_version()) if first == ("ver", None) else first
    monkeypatch.setattr("builtins.input", lambda: "")

    class Conn:
        def poll(self, timeout):
            return True

        def recv(self):
            return first

    early = session._check_version(Conn())
    assert (session.STALE in capsys.readouterr().out) is warned
    assert early == ([] if first[0] == "ver" else [first])
