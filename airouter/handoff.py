"""Local, provider-independent context for a chat handoff."""

import subprocess
from datetime import datetime
from pathlib import Path


def _git(workspace: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", *args], cwd=workspace, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=3, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.rstrip("\n") if result.returncode == 0 else None


def workspace_state(workspace: Path) -> dict:
    return {
        "branch": _git(workspace, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git(workspace, "rev-parse", "HEAD"),
        "status": _git(workspace, "-c", "core.quotePath=false", "status", "--short", "--untracked-files=normal"),
    }


def record(checkpoint: dict, workspace: Path, provider: str, request: str, reply: str, outcome: str) -> dict:
    turn = {"provider": provider, "request": request[:2000], "outcome": outcome}
    return {
        **checkpoint,
        "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": workspace_state(workspace),
        "latest": turn,
        "completed": {**turn, "reply": reply[:3000]} if outcome == "completed" else checkpoint.get("completed"),
    }


def render(checkpoint: dict, workspace: Path, *, refresh: bool = False) -> str:
    saved = checkpoint.get("git") or {}
    current = workspace_state(workspace) if refresh else saved
    status = current.get("status")
    if status is not None and len(status) > 2000:
        status = status[:2000] + "\n…"
    lines = [
        f"Workspace: {workspace}",
        f"Checkpoint updated: {checkpoint.get('updated', 'unknown')}",
        f"Git branch: {current.get('branch') or 'unknown'}",
        f"Git commit: {current.get('commit') or 'unknown'}",
        "Git status:",
        status if status else ("clean" if status == "" else "unavailable"),
    ]
    if refresh and saved and saved != current:
        lines.append("Git branch, commit, or status differs from the saved checkpoint.")
    if note := checkpoint.get("note"):
        lines.extend(["User handoff note:", note])
    if completed := checkpoint.get("completed"):
        lines.extend([f"Last completed request ({completed['provider']}):", completed["request"],
                      "Previous provider's report (verify against project evidence):", completed["reply"]])
    if (latest := checkpoint.get("latest")) and latest["outcome"] != "completed":
        lines.extend([f"Latest request ({latest['provider']}, {latest['outcome']}):", latest["request"],
                      "The prior turn may have had side effects. Verify what happened before retrying."])
    return "\n".join(lines)
