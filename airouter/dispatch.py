import json
import queue
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

from . import __version__, attach, config
from .providers import claude as claude_provider
from .providers import codex as codex_provider
from .security import Mode, SecurityPolicy

CLASSIFIER_PROMPT = (
    "Classify the user's request by how capable a model must be to answer it well. "
    "light: lookup, definition, short syntax or command question. "
    "medium: write, fix or explain ordinary code. "
    "heavy: design, deep debugging, root cause, multi-step reasoning. "
    "Reply with exactly one word: light, medium, or heavy."
)


def claude_cmd() -> list[str]:
    return claude_provider.executable()


def codex_cmd() -> list[str]:
    return codex_provider.executable()


def build(provider: str, model: str, effort: str, prompt: str, interactive: bool,
          policy: SecurityPolicy | None = None) -> tuple[list[str], str | None]:
    policy = policy or SecurityPolicy.from_config(config.load())
    if not interactive and policy.mode is Mode.APPROVAL_REQUIRED:
        policy = SecurityPolicy(Mode.READ_ONLY, policy.workspace, False, (), policy.read_roots, True)
    if provider == "claude":
        return claude_provider.build(model, effort, prompt, interactive, policy)
    if provider == "codex":
        return codex_provider.build(model, effort, prompt, interactive, policy)
    raise ValueError(f"unknown provider: {provider}")


def _claude_security(policy: SecurityPolicy) -> list[str]:
    return claude_provider.security_args(policy)


def _read_rule(root: Path) -> str:
    return claude_provider.read_rule(root)


def chat_roots(policy: SecurityPolicy | None = None):
    policy = policy or SecurityPolicy.from_config(config.load())
    return [str(root) for root in policy.extra_write_roots]


def codex_server_cmd(policy: SecurityPolicy | None = None):
    policy = policy or SecurityPolicy.from_config(config.load())
    return codex_provider.server_command(policy, chat_roots(policy))


def chat_cmd(provider: str, model: str, effort: str, session: str | None, files=(),
             attach_dir: str | None = None, policy: SecurityPolicy | None = None) -> list[str]:
    policy = policy or SecurityPolicy.from_config(config.load())
    images = [str(f) for f in files if attach.is_image(Path(f))]
    if provider == "claude":
        return claude_provider.chat_command(model, effort, session, policy)
    if provider == "codex":
        return codex_provider.chat_command(model, effort, session, images, policy, chat_roots(policy))
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


BATCH_PROMPT = (
    CLASSIFIER_PROMPT.rsplit(" Reply", 1)[0]
    + " You receive a JSON array of requests, each judged on its own. "
    "Reply with only a JSON array of the same length holding light, medium or heavy for each."
)


def _ask_claude(system: str, text: str, model: str, effort: str, timeout: int) -> str | None:
    return claude_provider.ask(system, text, model, effort, timeout)


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


EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
CODEX_DEFAULT_EFFORTS = ("low", "medium", "high", "xhigh")
CODEX_EFFORTS: dict[str, list[str]] = {}


def efforts(provider: str, model: str | None) -> tuple[str, ...]:
    if provider == "claude":
        return CLAUDE_EFFORTS
    return tuple(CODEX_EFFORTS.get(model or "") or CODEX_DEFAULT_EFFORTS)


def fit_effort(level: str, supported) -> str:
    supported = list(supported)
    if level in supported or not supported:
        return level
    rank = EFFORT_ORDER.index(level) if level in EFFORT_ORDER else 2
    return min(supported, key=lambda s: (abs((EFFORT_ORDER.index(s) if s in EFFORT_ORDER else 2) - rank),
                                         EFFORT_ORDER.index(s) if s in EFFORT_ORDER else 2))


def codex_models() -> list[str]:
    try:
        out = subprocess.run(codex_cmd() + ["debug", "models"], capture_output=True, timeout=30)
        data = json.loads(out.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    listed = [m for m in data.get("models", []) if m.get("visibility") == "list" and m.get("slug")]
    for m in listed:
        levels = [lv.get("effort") for lv in m.get("supported_reasoning_levels") or [] if lv.get("effort")]
        if levels:
            CODEX_EFFORTS[m["slug"]] = levels
    return [m["slug"] for m in listed]


DISCOVER = {"codex": codex_models}


def read_claude_usage(timeout: float = 15) -> dict:
    # Same endpoint Claude Code's /usage reads; the token is only sent to Anthropic, and an expired one is
    # left for Claude Code to refresh.
    try:
        cred = json.loads((Path.home() / ".claude" / ".credentials.json").read_text(encoding="utf-8"))
        token = cred["claudeAiOauth"]["accessToken"]
        request = urllib.request.Request("https://api.anthropic.com/api/oauth/usage", headers={
            "Authorization": f"Bearer {token}", "anthropic-beta": "oauth-2025-04-20"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, KeyError, TypeError, ValueError):
        return {}


APPROVAL_METHODS = {
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "item/permissions/requestApproval",
}


class CodexProc:
    def __init__(self, model, effort, session, approve=None, policy: SecurityPolicy | None = None):
        self.policy = policy or SecurityPolicy.from_config(config.load())
        self.key = (model, effort, self.policy.key)
        self.approve = approve
        self.session = session
        self.proc = subprocess.Popen(codex_server_cmd(self.policy), stdin=subprocess.PIPE,
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
                                                   "version": __version__}})
            self._write({"method": "initialized", "params": {}})
            params = {"model": model, "cwd": str(self.policy.workspace), "sandbox": self.policy.sandbox,
                      "approvalPolicy": self.policy.approval_policy,
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
                    if message.get("method") in APPROVAL_METHODS:
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
                    if message.get("method") in APPROVAL_METHODS:
                        params = message.get("params", {})
                        shown = {**params, "requestMethod": message.get("method")}
                        allowed = (params.get("threadId") == self.session
                                   and params.get("turnId") == self.turn_id
                                   and not self.cancelled and self.approve
                                   and self.approve(shown, lambda: self.cancelled or not self.alive()))
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
                "name": "ai-router", "title": "ai-router", "version": __version__}}})
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
