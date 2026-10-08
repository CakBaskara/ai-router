import json
import os
import queue
import shutil
import subprocess
import sys
import threading
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


class CodexProc:
    def __init__(self, model, effort, session, approve=None):
        self.key = (model, effort)
        self.approve = approve
        self.session = session
        self.proc = subprocess.Popen(codex_server_cmd(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.lock = threading.RLock()
        self.events = queue.Queue()
        self.requests = {}
        self.serial = 0
        self.active = False
        self.turn_id = None
        self.pending = {}
        self.deferred = []
        self.cancelled = False
        self.errors = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        try:
            self._rpc("initialize", {"clientInfo": {"name": "ai-router", "title": "ai-router",
                                                   "version": "0.1.0"}})
            self._write({"method": "initialized", "params": {}})
            params = {"model": model, "cwd": os.getcwd(), "sandbox": "workspace-write",
                      "config": {"model_reasoning_effort": effort}}
            if session:
                params["threadId"] = session
            result = self._rpc("thread/resume" if session else "thread/start", params)
            self.session = result["thread"]["id"]
        except Exception:
            self.close()
            raise

    def _drain(self):
        for raw in self.proc.stderr:
            self.errors = (self.errors + [raw.decode("utf-8", "replace").strip()])[-5:]

    def _write(self, message):
        with self.lock:
            self.proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()

    def _request(self, method, params, wait=True):
        with self.lock:
            self.serial += 1
            request_id = self.serial
            response = {"ready": threading.Event()} if wait else None
            self.requests[request_id] = response
            try:
                self._write({"id": request_id, "method": method, "params": params})
            except (OSError, ValueError):
                self.requests.pop(request_id, None)
                raise
        return request_id, response

    def _rpc(self, method, params):
        request_id, response = self._request(method, params)
        if not response["ready"].wait(30):
            with self.lock:
                self.requests.pop(request_id, None)
            raise OSError(f"Codex {method} timed out")
        message = response["message"]
        if "error" in message:
            raise OSError(message["error"].get("message", "Codex request failed"))
        return message["result"]

    def _read(self):
        try:
            for raw in self.proc.stdout:
                try:
                    message = json.loads(raw.decode("utf-8", "replace"))
                except (json.JSONDecodeError, UnicodeError):
                    continue
                if "id" in message and "method" not in message:
                    with self.lock:
                        response = self.requests.pop(message["id"], None)
                        if response is not None:
                            response["message"] = message
                            response["ready"].set()
                        else:
                            self.events.put(message)
                elif "id" in message:
                    if message.get("method") == "item/commandExecution/requestApproval":
                        self.events.put(message)
                    else:
                        self._write({"id": message["id"], "error": {
                            "code": -32601, "message": "ai-router does not support this client request"}})
                else:
                    if message.get("method") == "account/rateLimits/updated":
                        limits = (message.get("params") or {}).get("rateLimits") or {}
                        config.save_quota("codex", config.codex_windows(limits))
                    self.events.put(message)
        finally:
            with self.lock:
                for response in self.requests.values():
                    if response is not None:
                        response["message"] = {"error": {"message": "Codex app-server disconnected"}}
                        response["ready"].set()
                self.requests.clear()
            self.events.put(None)

    def alive(self):
        return self.proc.poll() is None

    @staticmethod
    def _input(prompt, files):
        inputs = [{"type": "text", "text": prompt}]
        inputs += [{"type": "localImage", "path": str(file)} for file in files if attach.is_image(file)]
        return inputs

    def _steer(self, inputs):
        request_id, _ = self._request("turn/steer", {"threadId": self.session,
                                                   "expectedTurnId": self.turn_id,
                                                   "input": inputs}, wait=False)
        self.pending[request_id] = inputs

    def _start(self, inputs):
        result = self._rpc("turn/start", {"threadId": self.session, "input": inputs,
                                          "model": self.key[0], "effort": self.key[1]})
        with self.lock:
            self.turn_id = result["turn"]["id"]
            messages, self.deferred = self.deferred, []
            for message in messages:
                self._steer(message)
            if self.cancelled:
                self._request("turn/interrupt", {"threadId": self.session,
                                                 "turnId": self.turn_id}, wait=False)

    def ask(self, prompt, turn, show, files=()):
        with self.lock:
            self.active = True
        turn.session = self.session
        completed = None
        try:
            self._start(self._input(prompt, files))
            while True:
                message = self.events.get()
                if message is None:
                    raise OSError(self.errors[-1] if self.errors else "Codex app-server disconnected")
                if "id" in message:
                    if message.get("method") == "item/commandExecution/requestApproval":
                        params = message.get("params", {})
                        allowed = (params.get("threadId") == self.session
                                   and params.get("turnId") == self.turn_id
                                   and not self.cancelled and self.approve
                                   and self.approve(params, lambda: self.cancelled or not self.alive()))
                        self._write({"id": message["id"], "result": {
                            "decision": "accept" if allowed and not self.cancelled else "decline"}})
                        continue
                    with self.lock:
                        inputs = self.pending.pop(message["id"], None)
                        if inputs and "error" in message:
                            error = message["error"].get("message", "Codex steering failed")
                            if "active turn" in error.lower() or "turn id" in error.lower():
                                self.deferred.append(inputs)
                            else:
                                raise OSError(error)
                else:
                    method, params = message.get("method"), message.get("params", {})
                    if params.get("threadId") != self.session:
                        continue
                    if params.get("turnId") and params["turnId"] != self.turn_id:
                        continue
                    if method == "turn/completed":
                        if params["turn"]["id"] != self.turn_id:
                            continue
                        with self.lock:
                            completed = params["turn"]
                            self.turn_id = None
                    else:
                        shown = turn.feed_app(method, params)
                        if shown:
                            show(*shown)
                with self.lock:
                    if completed is None or self.pending:
                        continue
                    if self.deferred and completed["status"] == "completed" and not self.cancelled:
                        inputs = [item for message_input in self.deferred for item in message_input]
                        self.deferred = []
                    else:
                        self.active = False
                        inputs = None
                if inputs is not None:
                    completed = None
                    self._start(inputs)
                    continue
                status = completed["status"]
                if status == "failed":
                    turn.error = (completed.get("error") or {}).get("message", "Codex turn failed")
                return 130 if status == "interrupted" else 1 if status == "failed" else 0
        except (OSError, ValueError, KeyError) as exc:
            turn.error = str(exc)
            self.close()
            return 1
        finally:
            with self.lock:
                self.active = False
                self.turn_id = None
                self.pending.clear()
                self.deferred = []

    def inject(self, prompt, files=()):
        with self.lock:
            if not self.active or self.cancelled or not self.alive():
                return False
            try:
                inputs = self._input(prompt, files)
                if self.turn_id is None:
                    self.deferred.append(inputs)
                else:
                    self._steer(inputs)
            except (OSError, ValueError):
                return False
            return True

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.active and self.turn_id:
                self._request("turn/interrupt", {"threadId": self.session,
                                                 "turnId": self.turn_id}, wait=False)

    def close(self):
        if self.alive():
            self.proc.kill()


def read_limits(timeout: float = 20) -> dict:
    try:
        proc = subprocess.Popen(codex_server_cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return {}
    result = {}

    def send(message):
        proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        proc.stdin.flush()

    def reply(request_id):
        for raw in proc.stdout:
            try:
                message = json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            if message.get("id") == request_id and "method" not in message:
                return message
        return {}

    # Closing stdin early makes the app-server exit before it answers, so wait for each reply.
    def talk():
        try:
            send({"id": 1, "method": "initialize", "params": {"clientInfo": {
                "name": "ai-router", "title": "ai-router", "version": "0.1.0"}}})
            reply(1)
            send({"method": "initialized", "params": {}})
            send({"id": 2, "method": "account/rateLimits/read", "params": {}})
            result.update(reply(2).get("result") or {})
        except (OSError, ValueError):
            pass

    worker = threading.Thread(target=talk, daemon=True)
    worker.start()
    worker.join(timeout)
    proc.kill()
    return result.get("rateLimits") or {}
