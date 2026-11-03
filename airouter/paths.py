import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "AI-Router"


def config_dir() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False, roaming=True))


def data_dir() -> Path:
    return Path(os.environ.get("AI_ROUTER_DATA_DIR", user_data_dir(APP_NAME, appauthor=False)))


def log_dir() -> Path:
    return data_dir() / "logs"


def attachment_dir() -> Path:
    return Path(os.environ.get("AI_ROUTER_ATTACH", data_dir() / "attachments"))


def model_dir() -> Path:
    return Path(os.environ.get("AI_ROUTER_MODELS", data_dir() / "models"))


@dataclass(frozen=True)
class MigrationEntry:
    source: Path
    target: Path
    status: str
    detail: str = ""
    digest: str = ""


def _reparse(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _check_target(target: Path):
    for path in (target, *target.parents):
        if path.is_symlink() or (path.exists() and _reparse(path)):
            raise ValueError(f"symlink/reparse point: {path}")
        if path != target and path.exists() and not path.is_dir():
            raise ValueError(f"bukan direktori: {path}")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish(temporary: Path, target: Path):
    if os.name == "nt":
        os.rename(temporary, target)
    else:
        os.link(temporary, target)


def _inspect(source: Path, target: Path, legacy: Path) -> MigrationEntry:
    entry = MigrationEntry(source, target, "copy")
    try:
        if _reparse(source) or not source.is_file():
            return replace(entry, status="skipped", detail="bukan file biasa")
        _check_target(target)
        if source.resolve() == target.resolve():
            return replace(entry, status="unchanged", detail="lokasi sama")
        if target.resolve().is_relative_to(legacy.resolve()):
            raise ValueError("tujuan berada di direktori sumber migrasi")
        entry = replace(entry, digest=_digest(source))
        if target.exists():
            if target.is_file() and entry.digest == _digest(target):
                return replace(entry, status="unchanged", detail="isi sama")
            return replace(entry, status="conflict", detail="tujuan sudah ada dengan isi berbeda")
        return entry
    except (OSError, ValueError) as exc:
        return replace(entry, status="error", detail=str(exc))


def migration_plan(source_dir: Path | None = None) -> list[MigrationEntry]:
    from . import config, learn

    legacy = Path(source_dir) if source_dir is not None else config.REPO_ROOT / "logs"
    entries = []

    def destination(relative):
        first, *rest = relative.parts
        if first == "attachments":
            return attachment_dir().joinpath(*rest)
        if first == "models":
            return model_dir().joinpath(*rest)
        if len(relative.parts) == 1:
            if first == "routes.jsonl":
                return config.log_path()
            if first == "labels.jsonl":
                return learn.labels_path()
            if first.startswith("ml_"):
                return learn.labels_path().with_name(first)
            if first in {"quota.json", "replies.jsonl", "style.md", "style_state.json"}:
                return config.log_path().with_name(first)
        return log_dir() / relative

    def visit(directory):
        try:
            if _reparse(directory):
                raise ValueError(f"direktori sumber adalah symlink/reparse point: {directory}")
            for source in sorted(directory.iterdir()):
                target = destination(source.relative_to(legacy))
                if source.suffix.lower() in {".lock", ".part"} or _reparse(source):
                    entries.append(MigrationEntry(source, target, "skipped", "lock, file parsial, atau tautan"))
                elif source.is_dir():
                    visit(source)
                else:
                    entries.append(_inspect(source, target, legacy))
        except (OSError, ValueError) as exc:
            entries.append(MigrationEntry(directory, log_dir(), "error", str(exc)))

    if legacy.exists():
        visit(legacy)
    else:
        entries.append(MigrationEntry(legacy, log_dir(), "error" if source_dir is not None else "skipped",
                                      "direktori sumber tidak ada"))
    old_config = legacy.parent / "airouter" / "config.toml"
    if old_config.exists() and "AI_ROUTER_CONFIG" not in os.environ:
        entries.append(_inspect(old_config, config_dir() / "config.toml", legacy))
    destinations = {}
    for i, entry in enumerate(entries):
        if entry.status not in {"copy", "unchanged"}:
            continue
        key = entry.target.resolve()
        if key in destinations and destinations[key] != entry.source:
            entries[i] = replace(entry, status="conflict", detail="beberapa sumber memakai tujuan yang sama")
        destinations[key] = entry.source
    for i, entry in enumerate(entries):
        if entry.status in {"copy", "unchanged"} and any(
                parent in destinations for parent in entry.target.resolve().parents):
            entries[i] = replace(entry, status="conflict", detail="tujuan file juga dipakai sebagai direktori")
    return entries


def migrate_data(source_dir: Path | None = None, *, apply: bool = False) -> list[MigrationEntry]:
    from . import config

    legacy = Path(source_dir) if source_dir is not None else config.REPO_ROOT / "logs"
    entries = migration_plan(source_dir)
    if not apply or any(e.status in {"conflict", "error"} for e in entries):
        return entries
    for i, entry in enumerate(entries):
        if entry.status != "copy":
            continue
        temporary = None
        try:
            _check_target(entry.target)
            if _reparse(entry.source) or not entry.source.is_file():
                raise ValueError("sumber bukan file biasa")
            entry.target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=entry.target.parent, prefix=".ai-router-",
                                             suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                with entry.source.open("rb") as source:
                    shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            if _digest(temporary) != entry.digest or _digest(entry.source) != entry.digest:
                raise ValueError("sumber berubah selama migrasi; hentikan ai lalu ulangi")
            shutil.copystat(entry.source, temporary)
            _check_target(entry.target)
            try:
                _publish(temporary, entry.target)
                entries[i] = replace(entry, status="copied")
            except FileExistsError:
                entries[i] = _inspect(entry.source, entry.target, legacy)
        except (OSError, ValueError) as exc:
            entries[i] = replace(entry, status="error", detail=str(exc))
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass
        if entries[i].status in {"conflict", "error"}:
            break
    return entries
