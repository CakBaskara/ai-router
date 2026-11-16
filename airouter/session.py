import codecs
import getpass
import hashlib
import json
import os
import secrets
import struct
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
from pathlib import Path

from . import paths

PIPE_PREFIX = "\\\\.\\pipe\\"
MOVED = "Sesi dipindah ke perangkat lain. Ketik sela untuk masuk lagi."
STOPPED = "Sesi latar berhenti."
DROP = (b"\x1b[?9001h", b"\x1b[?9001l")
RESET = "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l\x1b[?2004l\x1b[?1004l\x1b[?25h\x1b[?1049l"
BREAKAWAY_FROM_JOB = 0x01000000
NEW_PROCESS_GROUP = 0x00000200
NO_WINDOW = 0x08000000
KEY_EVENT = 0x0001


def remote_shell() -> bool:
    return any(os.environ.get(name) for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))


def non_interactive() -> bool:
    # Session 0 has no desktop, and Codex's Windows sandbox cannot start USER32 programs there.
    if sys.platform != "win32":
        return False
    import ctypes
    sid = ctypes.c_ulong()
    if not ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(sid)):
        return False
    return sid.value == 0


def address() -> str:
    tag = hashlib.sha256(str(paths.data_dir()).lower().encode()).hexdigest()[:8]
    user = "".join(c for c in getpass.getuser() if c.isalnum()) or "user"
    return f"{PIPE_PREFIX}sela-{user}-{tag}"


def key_path() -> Path:
    return paths.data_dir() / "session.key"


def load_key(create: bool = False) -> bytes | None:
    path = key_path()
    try:
        return path.read_bytes()
    except FileNotFoundError:
        if not create:
            return None
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    try:
        with open(path, "xb") as handle:
            handle.write(key)
    except FileExistsError:
        return path.read_bytes()
    return key


def connect(addr: str | None = None, key: bytes | None = None):
    key = key if key is not None else load_key()
    if not key:
        return None
    try:
        return Client(addr or address(), family="AF_PIPE", authkey=key)
    except (AuthenticationError, EOFError, OSError):
        return None


def _log(text: str):
    try:
        path = paths.log_dir() / "session.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {text}\n")
    except OSError:
        pass


class Host:
    def __init__(self, cmdline: str, cols: int = 100, rows: int = 30, cwd: str | None = None,
                 addr: str | None = None, key: bytes | None = None):
        from .conpty import Pty
        self.key = key if key is not None else load_key(create=True)
        self.listener = Listener(addr or address(), family="AF_PIPE", authkey=self.key)
        self.lock = threading.Lock()
        self.conn = None
        self.pty = Pty(cmdline, cols, rows, self._output, cwd)

    def _send(self, conn, message) -> bool:
        try:
            conn.send(message)
            return True
        except (OSError, ValueError):
            return False

    def _output(self, data: bytes):
        with self.lock:
            if self.conn is not None and not self._send(self.conn, ("out", data)):
                self.conn = None

    def _attach(self, conn):
        with self.lock:
            old, self.conn = self.conn, conn
            if old is not None:
                self._send(old, ("bye", MOVED))
        if old is not None:
            old.close()
        threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        first = True
        try:
            while True:
                kind, *rest = conn.recv()
                if kind == "in":
                    self.pty.write(rest[0])
                elif kind == "size":
                    cols, rows = rest
                    if first:
                        # A resize makes the app inside redraw everything, which is how a
                        # newly attached screen gets the current picture instead of a blank one.
                        self.pty.resize(cols, rows - 1)
                        first = False
                    self.pty.resize(cols, rows)
        except (EOFError, OSError, ValueError, TypeError):
            pass
        finally:
            with self.lock:
                if self.conn is conn:
                    self.conn = None
            conn.close()

    def _accept(self):
        while True:
            try:
                conn = self.listener.accept()
            except (AuthenticationError, EOFError, ConnectionError):
                continue
            except OSError:
                return
            self._attach(conn)

    def run(self) -> int:
        threading.Thread(target=self._accept, daemon=True).start()
        self.pty.wait()
        time.sleep(0.2)
        with self.lock:
            conn, self.conn = self.conn, None
            if conn is not None:
                self._send(conn, ("bye", ""))
        if conn is not None:
            conn.close()
        self.pty.close()
        return 0


def host_main(spec_path: str) -> int:
    spec_file = Path(spec_path)
    try:
        spec = json.loads(spec_file.read_text(encoding="utf-8"))
    finally:
        spec_file.unlink(missing_ok=True)
    os.environ.clear()
    os.environ.update(spec["env"])
    _log(f"host {os.getpid()} mulai di {spec['cwd']}")
    try:
        host = Host(subprocess.list2cmdline(spec["cmd"]), spec["cols"], spec["rows"], spec["cwd"])
    except Exception:
        _log("host gagal dibuat:\n" + traceback.format_exc())
        return 1
    code = host.run()
    _log(f"host {os.getpid()} selesai")
    return code


def _start_with_wmi(cmd: list[str]):
    line = subprocess.list2cmdline(cmd).replace("'", "''")
    cwd = os.getcwd().replace("'", "''")
    script = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
              f"@{{CommandLine='{line}'; CurrentDirectory='{cwd}'}}; exit $r.ReturnValue")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, timeout=60)
    if result.returncode != 0:
        raise OSError(f"Win32_Process.Create gagal ({result.returncode})")


def start_host(argv: list[str], cols: int, rows: int):
    load_key(create=True)
    spec = paths.data_dir() / f"session-start-{secrets.token_hex(4)}.json"
    spec.write_text(json.dumps({"cmd": [sys.executable, "-P", "-m", "airouter", "--no-host", *argv],
                                "cwd": os.getcwd(), "env": dict(os.environ), "cols": cols, "rows": rows}),
                    encoding="utf-8")
    runner = Path(sys.executable).with_name("pythonw.exe")
    cmd = [str(runner if runner.exists() else sys.executable), "-P", "-m", "airouter.session", "host", str(spec)]
    try:
        # The SSH session's processes are killed when it ends, so the host has to leave that job.
        subprocess.Popen(cmd, cwd=os.getcwd(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True,
                         creationflags=BREAKAWAY_FROM_JOB | NEW_PROCESS_GROUP | NO_WINDOW)
        _log("host dijalankan lewat breakaway")
    except OSError:
        _start_with_wmi(cmd)
        _log("host dijalankan lewat WMI")


class Console:
    def __init__(self):
        import ctypes
        from textual.drivers import win32
        self.ctypes, self.win32 = ctypes, win32
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32.WriteConsoleW.argtypes = [win32.HANDLE, ctypes.c_wchar_p, win32.DWORD,
                                           ctypes.POINTER(win32.DWORD), ctypes.c_void_p]
        self.hin = win32.HANDLE(win32.GetStdHandle(win32.STD_INPUT_HANDLE))
        self.hout = win32.HANDLE(win32.GetStdHandle(win32.STD_OUTPUT_HANDLE))
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.modes = None
        self.pending = []
        self.closed = threading.Event()

    def _mode(self, handle) -> int:
        mode = self.win32.DWORD()
        self.win32.KERNEL32.GetConsoleMode(handle, self.ctypes.byref(mode))
        return mode.value

    def enter(self):
        w = self.win32
        self.modes = (self._mode(self.hin), self._mode(self.hout))
        w.KERNEL32.SetConsoleMode(self.hin, w.ENABLE_VIRTUAL_TERMINAL_INPUT | w.ENABLE_WINDOW_INPUT)
        w.KERNEL32.SetConsoleMode(self.hout, self.modes[1] | w.ENABLE_VIRTUAL_TERMINAL_PROCESSING)

    def leave(self):
        self.closed.set()
        self.write_text(RESET)
        if self.modes:
            self.win32.KERNEL32.SetConsoleMode(self.hin, self.modes[0])
            self.win32.KERNEL32.SetConsoleMode(self.hout, self.modes[1])

    @staticmethod
    def size() -> tuple[int, int]:
        try:
            size = os.get_terminal_size(sys.stdout.fileno())
            return size.columns, size.lines
        except (OSError, ValueError):
            return 100, 30

    def write_text(self, text: str):
        written = self.win32.DWORD()
        for start in range(0, len(text), 8192):
            part = text[start:start + 8192]
            self.k32.WriteConsoleW(self.hout, part, len(part), self.ctypes.byref(written), None)

    def write(self, data: bytes):
        for sequence in DROP:
            data = data.replace(sequence, b"")
        self.write_text(self.decoder.decode(data))

    def _text(self) -> bytes:
        units = self.pending
        keep = []
        if units and 0xD800 <= units[-1] <= 0xDBFF:
            keep = [units.pop()]
        text = struct.pack(f"<{len(units)}H", *units).decode("utf-16-le", "replace")
        self.pending = keep
        return text.encode("utf-8")

    def pump(self, send):
        w = self.win32
        records = (w.INPUT_RECORD * 64)()
        count = w.DWORD()
        last = self.size()
        while not self.closed.is_set():
            now = self.size()
            if now != last:
                last = now
                send(("size", *now))
            if w.wait_for_handles([self.hin], 100) is None:
                continue
            w.KERNEL32.ReadConsoleInputW(self.hin, self.ctypes.byref(records), 64, self.ctypes.byref(count))
            for record in records[:count.value]:
                if record.EventType == KEY_EVENT:
                    key = record.Event.KeyEvent
                    unit = ord(key.uChar.UnicodeChar) if key.uChar.UnicodeChar else 0
                    if key.bKeyDown and unit:
                        self.pending.extend([unit] * max(key.wRepeatCount, 1))
            data = self._text()
            if data:
                send(("in", data))


def attach(conn) -> int:
    console = Console()
    lock = threading.Lock()

    def send(message):
        with lock:
            conn.send(message)

    reason = STOPPED
    console.enter()
    try:
        send(("size", *console.size()))
        threading.Thread(target=_pump, args=(console, send), daemon=True).start()
        while True:
            kind, payload = conn.recv()
            if kind == "out":
                console.write(payload)
            elif kind == "bye":
                reason = payload
                break
    except (EOFError, OSError):
        pass
    finally:
        console.leave()
        conn.close()
    if reason:
        print(reason)
    return 0


def _pump(console, send):
    try:
        console.pump(send)
    except (EOFError, OSError, ValueError):
        pass


def enter(argv: list[str]) -> int | None:
    if sys.platform != "win32" or not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    conn = connect()
    if conn is None and remote_shell():
        cols, rows = Console.size()
        try:
            start_host(argv, cols, rows)
        except (OSError, subprocess.SubprocessError) as exc:
            _log(f"host gagal dijalankan: {exc}")
            return None
        deadline = time.monotonic() + 20
        while conn is None and time.monotonic() < deadline:
            time.sleep(0.1)
            conn = connect()
        if conn is None:
            _log("host tidak menjawab")
            return None
    if conn is None:
        return None
    return attach(conn)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "host":
    sys.exit(host_main(sys.argv[2]))
