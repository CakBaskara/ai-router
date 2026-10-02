import json
import shutil
import subprocess
import sys
from pathlib import Path

CLASSIFIER_PROMPT = (
    "Classify the user's request by how capable a model must be to answer it well. "
    "light: lookup, definition, short syntax or command question. "
    "medium: write, fix or explain ordinary code. "
    "heavy: design, deep debugging, root cause, multi-step reasoning. "
    "Reply with exactly one word: light, medium, or heavy."
)


def _shim_dir(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found).parent if found else None


def claude_cmd() -> list[str]:
    d = _shim_dir("claude")
    if d:
        exe = d / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if exe.exists():
            return [str(exe)]
    return [shutil.which("claude") or "claude"]


def codex_cmd() -> list[str]:
    d = _shim_dir("codex")
    if d:
        js = d / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        node = d / "node.exe"
        if js.exists():
            return [str(node) if node.exists() else (shutil.which("node") or "node"), str(js)]
    return [shutil.which("codex") or "codex"]


def build(provider: str, model: str, effort: str, prompt: str, interactive: bool) -> tuple[list[str], str | None]:
    if provider == "claude":
        base = claude_cmd() + ["--model", model, "--effort", effort]
        if interactive:
            return base + ([prompt] if prompt else []), None
        return base + ["-p", "--no-session-persistence"], prompt
    if provider == "codex":
        effort_cfg = ["-c", f"model_reasoning_effort={effort}"]
        if interactive:
            return codex_cmd() + ["-m", model] + effort_cfg + ([prompt] if prompt else []), None
        return codex_cmd() + ["exec", "-m", model] + effort_cfg + [
            "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-"], prompt
    raise ValueError(f"unknown provider: {provider}")


def chat_cmd(provider: str, model: str, effort: str, session: str | None) -> list[str]:
    if provider == "claude":
        cmd = claude_cmd() + [
            "-p", "--model", model, "--effort", effort, "--permission-mode", "auto",
            "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--include-partial-messages"]
        return cmd + (["--resume", session] if session else [])
    if provider == "codex":
        common = ["-m", model, "-c", f"model_reasoning_effort={effort}", "--json", "--skip-git-repo-check"]
        if session:
            return codex_cmd() + ["exec", "resume"] + common + ["-c", 'sandbox_mode="workspace-write"', session, "-"]
        return codex_cmd() + ["exec"] + common + ["-s", "workspace-write", "-"]
    raise ValueError(f"unknown provider: {provider}")


def run(cmd: list[str], stdin_text: str | None, quiet_stderr: bool = False) -> int:
    if stdin_text is None:
        return subprocess.run(cmd).returncode
    if not quiet_stderr:
        return subprocess.run(cmd, input=stdin_text.encode("utf-8")).returncode
    done = subprocess.run(cmd, input=stdin_text.encode("utf-8"), stderr=subprocess.PIPE)
    if done.returncode != 0:
        sys.stderr.write(done.stderr.decode("utf-8", "replace"))
    return done.returncode


def llm_classify(prompt: str, model: str) -> str | None:
    cmd = claude_cmd() + [
        "-p", "--model", model, "--effort", "low", "--tools", "",
        "--no-session-persistence", "--strict-mcp-config",
        "--output-format", "json", "--system-prompt", CLASSIFIER_PROMPT,
    ]
    try:
        out = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True, timeout=60)
        data = json.loads(out.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    if data.get("is_error"):
        return None
    answer = str(data.get("result", "")).strip().lower()
    for tier in ("light", "medium", "heavy"):
        if tier in answer:
            return tier
    return None


def codex_models() -> list[str]:
    try:
        out = subprocess.run(codex_cmd() + ["debug", "models"], capture_output=True, timeout=30)
        data = json.loads(out.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    return [m["slug"] for m in data.get("models", []) if m.get("visibility") == "list" and m.get("slug")]
