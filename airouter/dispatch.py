import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import attach

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


def gemini_cmd() -> list[str]:
    return _node_script("gemini", "@google", "gemini-cli", "bundle", "gemini.js")


def copilot_cmd() -> list[str]:
    return _node_script("copilot", "@github", "copilot", "npm-loader.js")


COPILOT_DENY = [
    "shell(git push)", "shell(git commit)", "shell(git reset)", "shell(git clean)", "shell(git checkout)",
    "shell(gh repo)", "shell(gh pr)", "shell(npm publish)", "shell(rm)", "shell(rmdir)", "shell(del)",
    "shell(rd)", "shell(Remove-Item)", "shell(format)",
]


def _deny(rules: list[str]) -> list[str]:
    return [arg for rule in rules for arg in ("--deny-tool", rule)]


def _copilot_model(model: str, effort: str) -> list[str]:
    if model == "auto":
        return ["--model", "auto"]
    return ["--model", model, "--reasoning-effort", effort]


def load_user_env(*names: str):
    if sys.platform != "win32":
        return
    import winreg
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
    except OSError:
        return
    with key:
        for name in names:
            if os.environ.get(name):
                continue
            try:
                os.environ[name] = str(winreg.QueryValueEx(key, name)[0])
            except OSError:
                pass


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
    if provider == "gemini":
        if interactive:
            return gemini_cmd() + ["-m", model] + (["-i", prompt] if prompt else []), None
        return gemini_cmd() + ["-p", " ", "-m", model, "--approval-mode", "plan", "--skip-trust"], prompt
    if provider == "copilot":
        base = copilot_cmd() + _copilot_model(model, effort)
        if interactive:
            return base + (["-i", prompt] if prompt else []), None
        return base + ["-s", "--allow-all-tools", "--no-ask-user"] + _deny(["shell", "write"]), prompt
    raise ValueError(f"unknown provider: {provider}")


def chat_cmd(provider: str, model: str, effort: str, session: str | None, files=(),
             attach_dir: str | None = None) -> list[str]:
    images = [str(f) for f in files if attach.is_image(Path(f))]
    if provider == "claude":
        cmd = claude_cmd() + [
            "-p", "--model", model, "--effort", effort, "--permission-mode", "auto",
            "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--include-partial-messages"]
        cmd += ["--add-dir", attach_dir] if attach_dir else []
        return cmd + (["--resume", session] if session else [])
    if provider == "codex":
        common = ["-m", model, "-c", f"model_reasoning_effort={effort}", "--json", "--skip-git-repo-check"]
        common += [arg for img in images for arg in ("-i", img)]
        if session:
            return codex_cmd() + ["exec", "resume"] + common + ["-c", 'sandbox_mode="workspace-write"', session, "-"]
        return codex_cmd() + ["exec"] + common + ["-s", "workspace-write", "-"]
    if provider == "gemini":
        cmd = gemini_cmd() + ["-p", " ", "-m", model, "-o", "stream-json", "--approval-mode", "auto_edit",
                              "--skip-trust"]
        cmd += ["--include-directories", attach_dir] if files and attach_dir else []
        return cmd + (["--resume", session] if session else [])
    if provider == "copilot":
        cmd = copilot_cmd() + _copilot_model(model, effort) + [
            "--output-format", "json", "--allow-all-tools", "--no-ask-user"] + _deny(COPILOT_DENY)
        cmd += [arg for f in files for arg in ("--attachment", str(f))
                if attach.is_image(Path(f)) or Path(f).suffix.lower() == ".pdf"]
        cmd += ["--add-dir", attach_dir] if files and attach_dir else []
        return cmd + (["--session-id", session] if session else [])
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


GEMINI_SKIP = ("tts", "image", "robotics", "computer-use", "transcribe", "customtools", "embedding", "-pro")


def gemini_models() -> list[str]:
    import urllib.request
    load_user_env("GEMINI_API_KEY")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return []
    req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                 headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return []
    names = []
    for m in data.get("models", []):
        name = m.get("name", "").removeprefix("models/")
        if (name.startswith("gemini-") and "generateContent" in m.get("supportedGenerationMethods", [])
                and not any(s in name for s in GEMINI_SKIP)):
            names.append(name)
    return names


DISCOVER = {"codex": lambda: codex_models(), "gemini": lambda: gemini_models()}


def codex_models() -> list[str]:
    try:
        out = subprocess.run(codex_cmd() + ["debug", "models"], capture_output=True, timeout=30)
        data = json.loads(out.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    return [m["slug"] for m in data.get("models", []) if m.get("visibility") == "list" and m.get("slug")]
