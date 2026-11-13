import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import config

MAX_ITEMS = 100
_lock = threading.Lock()


def path() -> Path:
    return config.log_path().with_name("history.json")


def new_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def _process_lock(target: Path):
    lock_path = target.with_suffix(target.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _read() -> list[dict]:
    try:
        rows = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)]


def title(transcript: list) -> str:
    for item in transcript:
        if (isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "User"
                and isinstance(item[1], str) and item[1].strip()):
            return " ".join(item[1].split())[:80]
    return "Percakapan baru"


def entries() -> list[dict]:
    return sorted(_read(), key=lambda row: row.get("updated_at", ""), reverse=True)


def load(history_id: str) -> dict | None:
    for row in entries():
        if row.get("id") == history_id and isinstance(row.get("state"), dict):
            return row["state"]
    return None


def save(state: dict) -> bool:
    transcript = state.get("transcript") or []
    history_id = state.get("history_id")
    if not transcript or not isinstance(history_id, str) or not history_id:
        return False
    try:
        encoded_state = json.loads(json.dumps(state, ensure_ascii=False))
    except (TypeError, ValueError):
        return False
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    target = path()
    temporary = target.with_suffix(target.suffix + ".tmp")
    with _lock:
        try:
            with _process_lock(target):
                rows = _read()
                prior = next((row for row in rows if row.get("id") == history_id), {})
                row = {
                    "id": history_id,
                    "created_at": prior.get("created_at", now),
                    "updated_at": now,
                    "title": title(transcript),
                    "state": encoded_state,
                }
                rows = [row] + [item for item in rows if item.get("id") != history_id]
                _write(rows[:MAX_ITEMS], target, temporary)
            return True
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False


def _write(rows: list[dict], target: Path, temporary: Path):
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def delete(history_ids) -> int:
    wanted = set(history_ids)
    target = path()
    temporary = target.with_suffix(target.suffix + ".tmp")
    with _lock:
        try:
            with _process_lock(target):
                rows = _read()
                kept = [row for row in rows if row["id"] not in wanted]
                if len(kept) != len(rows):
                    _write(kept, target, temporary)
                return len(rows) - len(kept)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return 0
