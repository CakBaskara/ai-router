from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Mode(str, Enum):
    FULL_ACCESS = "full-access"
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    APPROVAL_REQUIRED = "approval-required"


@dataclass(frozen=True)
class SecurityPolicy:
    mode: Mode
    workspace: Path
    network_allowed: bool = False
    extra_write_roots: tuple[Path, ...] = ()
    read_roots: tuple[Path, ...] = ()
    confirm_destructive: bool = True

    def __post_init__(self):
        workspace = self.workspace.expanduser().resolve()
        if not workspace.is_dir():
            raise ValueError(f"workspace tidak ada atau bukan direktori: {workspace}")
        object.__setattr__(self, "workspace", workspace)
        if self.mode is Mode.FULL_ACCESS:
            object.__setattr__(self, "network_allowed", True)
        object.__setattr__(self, "extra_write_roots", self._roots(self.extra_write_roots, "write root"))
        object.__setattr__(self, "read_roots", self._roots(self.read_roots, "read root"))

    @staticmethod
    def _roots(roots, label: str) -> tuple[Path, ...]:
        resolved = []
        for root in roots:
            path = Path(root).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f"{label} tidak ada atau bukan direktori: {path}")
            if path not in resolved:
                resolved.append(path)
        return tuple(resolved)

    @classmethod
    def from_config(cls, cfg: dict, workspace: Path | None = None) -> "SecurityPolicy":
        raw = cfg.get("security", {})
        try:
            mode = Mode(raw.get("mode", Mode.FULL_ACCESS.value))
        except ValueError as exc:
            choices = ", ".join(item.value for item in Mode)
            raise ValueError(f"security.mode harus salah satu dari: {choices}") from exc
        return cls(
            mode=mode,
            workspace=workspace or Path.cwd(),
            network_allowed=bool(raw.get("allow_network", False)),
            extra_write_roots=tuple(Path(item) for item in raw.get("extra_write_roots", [])),
            read_roots=tuple(Path(item) for item in cfg.get("chat", {}).get("dirs", [])),
            confirm_destructive=bool(raw.get("confirm_destructive", True)),
        )

    def inside_workspace(self, path: Path) -> bool:
        return path.expanduser().resolve().is_relative_to(self.workspace)

    @property
    def sandbox(self) -> str:
        if self.mode is Mode.FULL_ACCESS:
            return "danger-full-access"
        return "read-only" if self.mode is Mode.READ_ONLY else "workspace-write"

    @property
    def approval_policy(self) -> str:
        if self.mode is Mode.APPROVAL_REQUIRED:
            return "on-request"
        if self.mode is Mode.WORKSPACE_WRITE and self.confirm_destructive:
            return "on-request"
        return "never"

    @property
    def key(self) -> tuple:
        return (self.mode.value, str(self.workspace), self.network_allowed, self.confirm_destructive,
                tuple(map(str, self.extra_write_roots)), tuple(map(str, self.read_roots)))
