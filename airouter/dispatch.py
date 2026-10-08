import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import attach, config

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


def _node_script(name: str, *rel: str) -> list[str]:
    d = _shim_dir(name)
    if d:
        js = d.joinpath("node_modules", *rel)
        node = d / "node.exe"
        if js.exists():
            return [str(node) if node.exists() else (shutil.which("node") or "node"), str(js)]
    return [shutil.which(name) or name]


def codex_cmd() -> list[str]:
    return _node_script("codex", "@openai", "codex", "bin", "codex.js")


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


def chat_roots(attach_dir=None):
    roots = [str(config.REPO_ROOT.resolve())]
    roots += [str(Path(d).expanduser().resolve()) for d in config.load().get("chat", {}).get("dirs", [])]
    if attach_dir:
        roots.append(str(Path(attach_dir).resolve()))
    return list(dict.fromkeys(roots))


def codex_server_cmd():
    roots = json.dumps(chat_roots(str(attach.folder())))
    return codex_cmd() + ["app-server", "--listen", "stdio://", "-c", 'sandbox_mode="workspace-write"',
                          "-c", f"sandbox_workspace_write.writable_roots={roots}"]


def chat_cmd(provider: str, model: str, effort: str, session: str | None, files=(),
             attach_dir: str | None = None) -> list[str]:
    images = [str(f) for f in files if attach.is_image(Path(f))]
    if provider == "claude":
        cmd = claude_cmd() + [
            "-p", "--model", model, "--effort", effort, "--permission-mode", "auto",
            "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--include-partial-messages", "--replay-user-messages"]
        cmd += [arg for root in chat_roots(attach_dir) for arg in ("--add-dir", root)]
        return cmd + (["--resume", session] if session else [])
    if provider == "codex":
        common = ["-m", model, "-c", f"model_reasoning_effort={effort}", "--json", "--skip-git-repo-check"]
        common += [arg for img in images for arg in ("-i", img)]
        roots = json.dumps(chat_roots(attach_dir))
        common += ["-c", f"sandbox_workspace_write.writable_roots={roots}"]
        if session:
            return codex_cmd() + ["exec", "resume"] + common + ["-c", 'sandbox_mode="workspace-write"', session, "-"]
        return codex_cmd() + ["exec"] + common + ["-s", "workspace-write", "-"]
    raise ValueError(f"unknown provider: {provider}")


REVIEW_TOOLS = "Read,Grep,Glob,Bash(git diff:*),Bash(git status:*),Bash(git log:*),Bash(git show:*)"


def review_cmd(provider: str, model: str, effort: str) -> list[str]:
    if provider == "claude":
        cmd = claude_cmd() + [
            "-p", "--model", model, "--effort", effort, "--no-session-persistence", "--allowedTools", REVIEW_TOOLS,
            "--output-format", "stream-json", "--verbose", "--include-partial-messages"]
        return cmd + [arg for root in chat_roots() for arg in ("--add-dir", root)]
    if provider == "codex":
        return codex_cmd() + ["exec", "-m", model, "-c", f"model_reasoning_effort={effort}", "--json",
                              "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-"]
    raise ValueError(f"no reviewer for provider: {provider}")


def run(cmd: list[str], stdin_text: str | None, quiet_stderr: bool = False) -> int:
    if stdin_text is None:
        return subprocess.run(cmd).returncode
    if not quiet_stderr:
        return subprocess.run(cmd, input=stdin_text.encode("utf-8")).returncode
    done = subprocess.run(cmd, input=stdin_text.encode("utf-8"), stderr=subprocess.PIPE)
    if done.returncode != 0:
        sys.stderr.write(done.stderr.decode("utf-8", "replace"))
    return done.returncode


BATCH_PROMPT = (
    CLASSIFIER_PROMPT.rsplit(" Reply", 1)[0]
    + " You receive a JSON array of requests, each judged on its own. "
    "Reply with only a JSON array of the same length holding light, medium or heavy for each."
)


def _ask_claude(system: str, text: str, model: str, effort: str, timeout: int) -> str | None:
    cmd = claude_cmd() + [
        "-p", "--model", model, "--effort", effort, "--tools", "",
        "--no-session-persistence", "--strict-mcp-config",
        "--output-format", "json", "--system-prompt", system,
    ]
    try:
        out = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, timeout=timeout)
        data = json.loads(out.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if data.get("is_error"):
        return None
    return str(data.get("result", ""))


def llm_classify_batch(prompts: list[str], model: str) -> list[str | None]:
    text = _ask_claude(BATCH_PROMPT, json.dumps(prompts, ensure_ascii=False), model, "low", 300)
    try:
        tiers = json.loads(text[text.index("["):text.rindex("]") + 1]) if text else []
    except ValueError:
        tiers = []
    if len(tiers) != len(prompts):
        return [None] * len(prompts)
    return [t.strip().lower() if isinstance(t, str) else None for t in tiers]


def llm_classify(prompt: str, model: str) -> str | None:
    answer = (_ask_claude(CLASSIFIER_PROMPT, prompt, model, "low", 60) or "").strip().lower()
    for tier in ("light", "medium", "heavy"):
        if tier in answer:
            return tier
    return None


def llm_text(system: str, text: str, model: str, timeout: int = 300) -> str | None:
    return (_ask_claude(system, text, model, "medium", timeout) or "").strip() or None


def codex_models() -> list[str]:
    try:
        out = subprocess.run(codex_cmd() + ["debug", "models"], capture_output=True, timeout=30)
        data = json.loads(out.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    return [m["slug"] for m in data.get("models", []) if m.get("visibility") == "list" and m.get("slug")]


DISCOVER = {"codex": codex_models}
