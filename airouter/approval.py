import difflib
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .security import Mode, SecurityPolicy


SKIP_DIRS = {".git", ".codex_work", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
MAX_DIFF_CHARS = 100_000
MAX_TEXT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class FileState:
    kind: str
    digest: str
    size: int


@dataclass(frozen=True)
class PatchResult:
    decision: str
    changed: tuple[str, ...]
    conflicts: tuple[str, ...] = ()
    preview: str = ""


class IsolatedWorkspace:
    def __init__(self, source: Path, read_roots: tuple[Path, ...] = ()):
        self.source = source.resolve()
        self.read_roots = read_roots
        self.temp_root = Path(tempfile.mkdtemp(prefix="ai-router-approval-"))
        self.workspace = self.temp_root / "workspace"
        self.worktree = False
        self.preserve = False
        try:
            self._create()
            self.baseline = self._snapshot(self.workspace)
        except Exception:
            self.close()
            raise

    @property
    def policy(self) -> SecurityPolicy:
        return SecurityPolicy(Mode.WORKSPACE_WRITE, self.workspace, network_allowed=False,
                              read_roots=self.read_roots, confirm_destructive=False)

    @staticmethod
    def _run(root: Path, *args: str, input_data: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *args], input=input_data, capture_output=True,
                              timeout=60, check=False)

    def _is_git_root(self) -> bool:
        result = self._run(self.source, "rev-parse", "--show-toplevel")
        if result.returncode:
            return False
        try:
            return Path(result.stdout.decode("utf-8", "replace").strip()).resolve() == self.source
        except OSError:
            return False

    def _create(self):
        if self._is_git_root():
            added = self._run(self.source, "worktree", "add", "--detach", str(self.workspace), "HEAD")
            if added.returncode == 0:
                self.worktree = True
                diff = self._run(self.source, "diff", "--binary", "HEAD")
                if diff.returncode:
                    raise OSError(diff.stderr.decode("utf-8", "replace").strip() or "git diff gagal")
                if diff.stdout:
                    applied = self._run(self.workspace, "apply", "--binary", "--whitespace=nowarn", "-",
                                        input_data=diff.stdout)
                    if applied.returncode:
                        raise OSError(applied.stderr.decode("utf-8", "replace").strip() or "git apply gagal")
                untracked = self._run(self.source, "ls-files", "-z", "--others", "--exclude-standard")
                if untracked.returncode:
                    raise OSError(untracked.stderr.decode("utf-8", "replace").strip() or "git ls-files gagal")
                for raw in untracked.stdout.split(b"\0"):
                    if raw:
                        self._copy_relative(self.source, self.workspace, os.fsdecode(raw))
                return
        shutil.copytree(self.source, self.workspace, symlinks=True,
                        ignore=shutil.ignore_patterns(*SKIP_DIRS))

    def _listed_paths(self, root: Path) -> set[str]:
        if (root / ".git").exists():
            result = self._run(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
            if result.returncode == 0:
                return {Path(os.fsdecode(raw)).as_posix() for raw in result.stdout.split(b"\0") if raw}
        paths = set()
        for current, dirs, names in os.walk(root):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
            for name in names:
                paths.add((Path(current) / name).relative_to(root).as_posix())
        return paths

    @staticmethod
    def _state(path: Path) -> FileState | None:
        try:
            if path.is_symlink():
                raw = os.readlink(path).encode("utf-8", "surrogateescape")
                return FileState("symlink", hashlib.sha256(raw).hexdigest(), len(raw))
            if not path.exists():
                return None
            if not path.is_file():
                return FileState("other", "", 0)
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            return FileState("file", digest.hexdigest(), size)
        except OSError:
            return FileState("unreadable", "", 0)

    def _snapshot(self, root: Path) -> dict[str, FileState]:
        return {name: state for name in self._listed_paths(root)
                if (state := self._state(root / Path(name))) is not None}

    @staticmethod
    def _safe_path(root: Path, name: str) -> Path:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"path tidak aman: {name}")
        target = root / relative
        if not target.parent.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"path keluar workspace: {name}")
        return target

    @classmethod
    def _copy_relative(cls, source_root: Path, target_root: Path, name: str):
        source = cls._safe_path(source_root, name)
        target = cls._safe_path(target_root, name)
        if source.is_symlink() or not source.is_file():
            raise OSError(f"hanya file biasa yang dapat dipindahkan: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.close(descriptor)
        temporary = Path(raw_temporary)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def _remove_relative(cls, root: Path, name: str):
        target = cls._safe_path(root, name)
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            raise OSError(f"bukan file biasa: {name}")

    def refresh(self):
        staged = self._snapshot(self.workspace)
        pending = self._changed(self.baseline, staged)
        if pending:
            raise OSError(f"preview sebelumnya belum selesai: {', '.join(pending[:8])}")
        active = self._snapshot(self.source)
        for name in self._changed(self.baseline, active):
            if self._state(self.source / Path(name)) is None:
                self._remove_relative(self.workspace, name)
            else:
                self._copy_relative(self.source, self.workspace, name)
        self.baseline = self._snapshot(self.workspace)

    @staticmethod
    def _changed(before: dict[str, FileState], after: dict[str, FileState]) -> tuple[str, ...]:
        return tuple(sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name)))

    @staticmethod
    def _text(path: Path, state: FileState | None) -> list[str] | None:
        if not state or state.kind != "file" or state.size > MAX_TEXT_BYTES:
            return [] if state is None else None
        try:
            raw = path.read_bytes()
            if b"\0" in raw:
                return None
            return raw.decode("utf-8").splitlines(keepends=True)
        except (OSError, UnicodeDecodeError):
            return None

    def _preview(self, changed: tuple[str, ...], staged: dict[str, FileState]) -> str:
        chunks = []
        for name in changed:
            old = self._text(self.source / Path(name), self.baseline.get(name))
            new = self._text(self.workspace / Path(name), staged.get(name))
            if old is None or new is None:
                chunks.append(f"binary/large: {name}\n")
                continue
            chunks.extend(difflib.unified_diff(old, new, fromfile=f"a/{name}", tofile=f"b/{name}"))
            if sum(map(len, chunks)) >= MAX_DIFF_CHARS:
                chunks.append("\n... diff dipotong ...\n")
                break
        return "".join(chunks)[:MAX_DIFF_CHARS]

    def review(self, approve, cancelled) -> PatchResult:
        staged = self._snapshot(self.workspace)
        changed = self._changed(self.baseline, staged)
        if not changed:
            return PatchResult("none", ())
        preview = self._preview(changed, staged)
        params = {"requestMethod": "isolatedPatch/requestApproval", "workspace": str(self.source),
                  "paths": list(changed), "diff": preview}
        allowed = False if cancelled() else bool(approve and approve(params, cancelled))
        if not allowed or cancelled():
            self._reset(changed)
            return PatchResult("rejected", changed, preview=preview)
        conflicts = tuple(name for name in changed
                          if self._state(self.source / Path(name)) != self.baseline.get(name))
        unsupported = tuple(name for name in changed
                            if (state := staged.get(name)) and state.kind != "file")
        conflicts = tuple(sorted(set(conflicts + unsupported)))
        if conflicts:
            self.preserve = True
            return PatchResult("conflict", changed, conflicts, preview)
        for name in changed:
            if staged.get(name) is None:
                self._remove_relative(self.source, name)
            else:
                self._copy_relative(self.workspace, self.source, name)
        self.baseline = staged
        return PatchResult("applied", changed, preview=preview)

    def _reset(self, changed: tuple[str, ...]):
        for name in changed:
            if self._state(self.source / Path(name)) is None:
                self._remove_relative(self.workspace, name)
            else:
                self._copy_relative(self.source, self.workspace, name)
        self.baseline = self._snapshot(self.workspace)

    def close(self):
        if self.preserve:
            return
        prune = False
        if self.worktree and self.workspace.exists():
            removed = self._run(self.source, "worktree", "remove", "--force", str(self.workspace))
            if removed.returncode:
                prune = True
        try:
            root = self.temp_root.resolve()
            if root.parent == Path(tempfile.gettempdir()).resolve() and root.name.startswith("ai-router-approval-"):
                shutil.rmtree(root, ignore_errors=True)
        except OSError:
            pass
        if prune:
            self._run(self.source, "worktree", "prune")
