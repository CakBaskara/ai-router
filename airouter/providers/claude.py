import json
import shutil
import subprocess
from pathlib import Path

from ..security import Mode, SecurityPolicy
from .common import shim_dir


def executable() -> list[str]:
    directory = shim_dir("claude")
    if directory:
        binary = directory / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if binary.exists():
            return [str(binary)]
    return [shutil.which("claude") or "claude"]


def read_rule(root: Path) -> str:
    path = root.as_posix()
    if root.drive:
        path = "/" + root.drive[0].lower() + path[len(root.drive):]
    return f"Read(/{path.rstrip('/')}/**)"


def security_args(policy: SecurityPolicy) -> list[str]:
    if policy.mode is Mode.READ_ONLY:
        args = ["--permission-mode", "plan", "--permission-prompts", "none", "--restricted",
                "--strict-mcp-config", "--tools", "Read,Grep,Glob"]
        return args + [arg for root in policy.read_roots for arg in ("--add-dir", str(root))]
    mode = {Mode.FULL_ACCESS: "bypassPermissions", Mode.APPROVAL_REQUIRED: "manual"}.get(policy.mode, "auto")
    prompts = "none" if policy.mode is Mode.WORKSPACE_WRITE and not policy.confirm_destructive else "host"
    args = ["--permission-mode", mode, "--permission-prompts", prompts]
    if not policy.network_allowed:
        args += ["--disallowedTools", "WebFetch,WebSearch"]
    if policy.mode is Mode.FULL_ACCESS:
        roots = dict.fromkeys(policy.extra_write_roots + policy.read_roots)
        return args + [arg for root in roots for arg in ("--add-dir", str(root))]
    args += ["--restricted"]
    if policy.read_roots:
        args += ["--allowedTools", ",".join(read_rule(root) for root in policy.read_roots)]
    return args + [arg for root in policy.extra_write_roots for arg in ("--add-dir", str(root))]


def build(model: str, effort: str, prompt: str, interactive: bool,
          policy: SecurityPolicy) -> tuple[list[str], str | None]:
    base = executable() + ["--model", model, "--effort", effort] + security_args(policy)
    if interactive:
        return base + ([prompt] if prompt else []), None
    return base + ["-p", "--no-session-persistence"], prompt


def chat_command(model: str, effort: str, session: str | None, policy: SecurityPolicy) -> list[str]:
    command = executable() + [
        "-p", "--model", model, "--effort", effort, *security_args(policy),
        "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--include-partial-messages", "--replay-user-messages"]
    return command + (["--resume", session] if session else [])


def ask(system: str, text: str, model: str, effort: str, timeout: int) -> str | None:
    command = executable() + [
        "-p", "--model", model, "--effort", effort, "--tools", "",
        "--no-session-persistence", "--strict-mcp-config",
        "--output-format", "json", "--system-prompt", system,
    ]
    try:
        result = subprocess.run(command, input=text.encode("utf-8"), capture_output=True, timeout=timeout)
        data = json.loads(result.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if data.get("is_error"):
        return None
    return str(data.get("result", ""))
