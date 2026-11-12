import hashlib
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Phase(str, Enum):
    NOT_DISPATCHED = "not_dispatched"
    DISPATCHED = "dispatched"
    EFFECTS_OBSERVED = "effects_observed"
    FINISHED = "finished"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorkspaceSnapshot:
    root: Path
    status_paths: tuple[str, ...]
    files: tuple[tuple[str, str], ...]
    staged: str = ""
    unstaged: str = ""
    complete: bool = True


@dataclass(frozen=True)
class AttemptResult:
    exit_code: int
    phase: Phase
    cancelled: bool = False
    file_changes_detected: bool = False
    external_effects_possible: bool = True
    changed_files: tuple[str, ...] = ()


def auto_retry_allowed(result: AttemptResult) -> bool:
    return (
        not result.cancelled
        and result.phase is Phase.NOT_DISPATCHED
        and not result.file_changes_detected
        and not result.external_effects_possible
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path, max_bytes: int) -> str:
    stat = path.stat()
    if stat.st_size > max_bytes:
        return f"large:{stat.st_size}:{stat.st_mtime_ns}"
    return _digest(path.read_bytes())


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=20, check=False)


def _porcelain_paths(raw: bytes) -> tuple[str, ...]:
    paths = []
    entries = raw.decode("utf-8", "surrogateescape").split("\0")
    index = 0
    while index < len(entries):
        item = entries[index]
        index += 1
        if not item:
            continue
        code, name = item[:2], item[3:]
        if name:
            paths.append(name)
        if code and ("R" in code or "C" in code) and index < len(entries):
            old = entries[index]
            index += 1
            if old:
                paths.append(old)
    return tuple(sorted(set(paths)))


def _git_snapshot(root: Path, max_file_bytes: int) -> WorkspaceSnapshot | None:
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status.returncode:
        return None
    paths = _porcelain_paths(status.stdout)
    files = []
    complete = True
    for name in paths:
        candidate = root / name
        try:
            if candidate.is_file() and candidate.resolve().is_relative_to(root):
                files.append((name, _file_digest(candidate, max_file_bytes)))
            else:
                files.append((name, "missing"))
        except OSError:
            complete = False
            files.append((name, "unreadable"))
    return WorkspaceSnapshot(root, paths, tuple(files), _digest(status.stdout), complete=complete)


def _tree_snapshot(root: Path, max_files: int, max_file_bytes: int) -> WorkspaceSnapshot:
    files = []
    complete = True
    for current, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in names:
            candidate = Path(current) / name
            relative = candidate.relative_to(root).as_posix()
            if len(files) >= max_files:
                complete = False
                return WorkspaceSnapshot(root, tuple(path for path, _ in files), tuple(files), complete=complete)
            try:
                if candidate.is_file() and candidate.resolve().is_relative_to(root):
                    files.append((relative, _file_digest(candidate, max_file_bytes)))
            except OSError:
                complete = False
                files.append((relative, "unreadable"))
    return WorkspaceSnapshot(root, tuple(path for path, _ in files), tuple(files), complete=complete)


def workspace_snapshot(root: Path, max_files: int = 20_000,
                       max_file_bytes: int = 16 * 1024 * 1024) -> WorkspaceSnapshot:
    root = root.resolve()
    try:
        return _git_snapshot(root, max_file_bytes) or _tree_snapshot(root, max_files, max_file_bytes)
    except (OSError, subprocess.SubprocessError):
        return WorkspaceSnapshot(root, (), (), complete=False)


def changed_files(before: WorkspaceSnapshot, after: WorkspaceSnapshot) -> tuple[str, ...]:
    before_files = dict(before.files)
    after_files = dict(after.files)
    changed = {path for path in before_files.keys() | after_files.keys()
               if before_files.get(path) != after_files.get(path)}
    if before.staged != after.staged or before.unstaged != after.unstaged:
        changed.update(before.status_paths)
        changed.update(after.status_paths)
    return tuple(sorted(changed))


def classify_attempt(exit_code: int, dispatched: bool, cancelled: bool,
                     before: WorkspaceSnapshot, after: WorkspaceSnapshot) -> AttemptResult:
    changed = changed_files(before, after)
    if changed:
        phase = Phase.EFFECTS_OBSERVED
    elif dispatched:
        phase = Phase.UNKNOWN
    else:
        phase = Phase.NOT_DISPATCHED
    return AttemptResult(
        exit_code=exit_code,
        phase=phase,
        cancelled=cancelled,
        file_changes_detected=bool(changed),
        external_effects_possible=dispatched,
        changed_files=changed,
    )
