import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from . import attach, dispatch, learn
from .codex import CodexProc
from .config import log, say, usage_today
from .learn import TIERS, classify

HELP = (
    "/new                      percakapan baru\n"
    "/model                    tampilkan tier, provider, model\n"
    "/model light|medium|heavy kunci tier\n"
    "/model auto               kembali ke routing otomatis\n"
    "/models                   daftar model yang bisa dipilih\n"
    "/model 3  /model sol      pilih model dari daftar (nomor atau nama)\n"
    "/claude /codex /gemini /copilot  kunci provider\n"
    "/codex gpt-5.6-sol        kunci provider dan model persis\n"
    "pakai opus  ganti ke codex  pakai auto   sama seperti /model dan /<provider>\n"
    "ganti model yang lebih ringan punya codex  turun/naik satu tier, boleh sebut provider\n"
    "/exit                   keluar\n"
    "Baris diakhiri \\ untuk lanjut ke baris berikutnya; teks yang di-paste ikut terkirim utuh."
)
CHAT_NOTE = (
    "[Context from the `ai` chat app, not from the user] You are answering inside `ai`, the user's terminal chat "
    "that routes each message to Claude Code, Codex, Gemini or Copilot CLI and picks the model automatically. "
    "Other assistants may have answered earlier turns; treat their answers as part of this conversation. You cannot switch "
    "models yourself. If the user wants another model, tell them to type /models for the list, /model <number or "
    "name> to pick one, or /model auto to go back to automatic routing; a short message such as \"pakai opus\" or "
    "\"ganti ke codex\" does the same. /new starts a new conversation. "
    "Answer naturally, like a helpful colleague, not curtly.\n\n"
)
STATE = ("tier", "provider", "model", "sessions", "synced", "transcript",
         "pinned_provider", "pinned_tier", "pinned_model")
RECAP_ENTRIES = 12
RECAP_CHARS = 2000
SWITCH = re.compile(
    r"(?:tolong |coba |please )?(?:ganti|ubah|pindah|pakai|pake|gunakan|switch|use|change)"
    r"(?: (?:model|provider|tier))?(?: (?:ke|jadi|to))? (\S+)(?: (?:aja|saja|dong|ya|deh|lagi))?[.!]?"
)
SWITCH_VERB = re.compile(r"\b(?:ganti|ubah|pindah|pakai|pake|gunakan|switch|use|change|naik\w*|turun\w*)\b")
NOT_SWITCH = re.compile(r"^(?:kenapa|mengapa|why|apa|apakah|bagaimana|gimana|how|what)\b|\b(?:untuk|for)\b")
LIGHTEST = ("paling ringan", "paling murah", "termurah", "paling hemat", "lightest", "cheapest")
STRONGEST = ("paling kuat", "paling pintar", "terkuat", "terpintar", "strongest", "smartest")
LIGHTER = ("ringan", "murah", "hemat", "turun", "lighter", "cheaper", "smaller")
STRONGER = ("kuat", "pintar", "berat", "naik", "stronger", "smarter", "bigger")
BARE_STEP = re.compile(r"(?:(?:oke|ok|coba|tolong|please) )*(?:turun(?:kan|in)?|naik(?:kan|in)?)"
                       r"(?: (?:aja|saja|dong|ya|deh))?[.!]?")
TOOL_KEYS = ("file_path", "command", "pattern", "path", "url", "query", "description")


def higher(a: str, b: str) -> str:
    return a if TIERS.index(a) >= TIERS.index(b) else b


def recap(entries: list[tuple[str, str]]) -> str:
    parts = []
    for role, text in entries[-RECAP_ENTRIES:]:
        if len(text) > RECAP_CHARS:
            text = text[:RECAP_CHARS] + " [...]"
        parts.append(f"{role}: {text}")
    return (
        "Earlier turns of this conversation that you have not seen, for context:\n\n"
        + "\n\n".join(parts)
        + "\n\n---\nNew message:\n"
    )


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _tool(name: str, args: dict) -> str:
    detail = str(next((args[k] for k in TOOL_KEYS if args.get(k)), ""))
    if os.path.isabs(detail) and detail.lower().startswith(os.getcwd().lower()):
        detail = os.path.relpath(detail)
    return _short(f"{name} {detail}")


class ClaudeTurn:
    def __init__(self):
        self.session = None
        self.reply = []
        self.usage = ""
        self.error = None

    def feed(self, ev: dict):
        if ev.get("session_id"):
            self.session = ev["session_id"]
        kind = ev.get("type")
        if ev.get("parent_tool_use_id"):
            return None
        if kind == "stream_event":
            e = ev.get("event", {})
            delta = e.get("delta", {})
            if e.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                self.reply.append(delta["text"])
                return "text", delta["text"]
            if e.get("type") == "message_start" and self.reply:
                self.reply.append("\n\n")
                return "break", ""
        elif kind == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    return "note", _tool(block.get("name"), block.get("input", {}))
        elif kind == "result":
            u = ev.get("usage", {})
            cached = u.get("cache_read_input_tokens", 0)
            total = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + cached
            self.usage = f"in {_k(total)} (cache {_k(cached)}) · out {_k(u.get('output_tokens', 0))}"
            if ev.get("is_error"):
                self.error = _short(ev.get("result") or ev.get("subtype") or "error", 300)
        return None

    def text(self) -> str:
        return "".join(self.reply).strip()


class CodexTurn:
    def __init__(self):
        self.session = None
        self.reply = []
        self.usage = ""
        self.error = None
        self.streamed = set()

    def feed(self, ev: dict):
        kind = ev.get("type")
        item = ev.get("item", {})
        itype = item.get("type")
        if kind == "thread.started":
            self.session = ev.get("thread_id")
        elif kind == "item.delta" and itype == "agent_message":
            prefix = "\n\n" if item["id"] not in self.streamed and self.reply else ""
            self.streamed.add(item["id"])
            self.reply.append(prefix + item.get("text", ""))
            return "text", prefix + item.get("text", "")
        elif kind == "item.started" and itype == "command_execution":
            return "note", _short(f"$ {item.get('command', '')}")
        elif kind == "item.started" and itype == "mcp_tool_call":
            return "note", _tool(f"{item.get('server', '')}.{item.get('tool', '')}", item.get("arguments") or {})
        elif kind == "item.started" and itype == "web_search":
            return "note", _short(f"search {item.get('query', '')}")
        elif kind == "item.completed" and itype == "agent_message":
            if item.get("id") in self.streamed:
                return None
            prefix = "\n\n" if self.reply else ""
            self.reply.append(prefix + item.get("text", ""))
            return "text", prefix + item.get("text", "")
        elif kind == "item.completed" and itype == "file_change":
            paths = ", ".join(os.path.basename(c.get("path", "")) for c in item.get("changes", []))
            return "note", _short(f"edit {paths}")
        elif kind == "item.completed" and itype == "error":
            return "note", _short(item.get("message", ""), 200)
        elif kind == "turn.completed":
            u = ev.get("usage", {})
            self.usage = (f"in {_k(u.get('input_tokens', 0))} (cache {_k(u.get('cached_input_tokens', 0))})"
                          f" · out {_k(u.get('output_tokens', 0))}")
        elif kind == "turn.failed":
            self.error = _short(ev.get("error", {}).get("message", "turn failed"), 300)
        elif kind == "error":
            self.error = _short(ev.get("message", "error"), 300)
        return None

    def text(self) -> str:
        return "".join(self.reply).strip()


    def feed_app(self, method, params):
        if method == "item/agentMessage/delta":
            return self.feed({"type": "item.delta", "item": {"type": "agent_message",
                              "id": params["itemId"], "text": params["delta"]}})
        if method == "thread/tokenUsage/updated":
            usage = params["tokenUsage"]["last"]
            return self.feed({"type": "turn.completed", "usage": {
                "input_tokens": usage["inputTokens"], "cached_input_tokens": usage["cachedInputTokens"],
                "output_tokens": usage["outputTokens"]}})
        if method in ("item/started", "item/completed"):
            item = dict(params["item"])
            item["type"] = {"agentMessage": "agent_message", "commandExecution": "command_execution",
                            "mcpToolCall": "mcp_tool_call", "webSearch": "web_search",
                            "fileChange": "file_change"}.get(item["type"], item["type"])
            return self.feed({"type": method.replace("/", "."), "item": item})
        if method == "error":
            return "note", _short((params.get("error") or {}).get("message", "Codex error"), 200)
        return None


class GeminiTurn:
    def __init__(self):
        self.session = None
        self.reply = []
        self.usage = ""
        self.error = None
        self.after_tool = False

    def feed(self, ev: dict):
        kind = ev.get("type")
        if kind == "init":
            self.session = ev.get("session_id")
        elif kind == "message" and ev.get("role") == "assistant":
            text = ev.get("content", "")
            if self.after_tool and self.reply:
                text = "\n\n" + text
            self.after_tool = False
            self.reply.append(text)
            return "text", text
        elif kind == "tool_use":
            self.after_tool = True
            return "note", _tool(ev.get("tool_name", ""), ev.get("parameters") or {})
        elif kind == "tool_result" and ev.get("status") != "success":
            return "note", _short(f"gagal: {ev.get('error', {}).get('message') or ev.get('status')}", 200)
        elif kind == "result":
            s = ev.get("stats", {})
            self.usage = (f"in {_k(s.get('input_tokens', 0))} (cache {_k(s.get('cached', 0))})"
                          f" · out {_k(s.get('output_tokens', 0))}")
            if ev.get("status") != "success":
                self.error = _short((ev.get("error") or {}).get("message") or ev.get("status") or "error", 300)
        elif kind == "error":
            self.error = _short(ev.get("message") or "error", 300)
        return None

    def text(self) -> str:
        return "".join(self.reply).strip()


class CopilotTurn:
    def __init__(self):
        self.session = None
        self.reply = []
        self.usage = ""
        self.error = None

    def feed(self, ev: dict):
        kind = ev.get("type")
        data = ev.get("data", {})
        if kind == "assistant.message_start" and self.reply:
            self.reply.append("\n\n")
            return "break", ""
        if kind == "assistant.message_delta":
            text = data.get("deltaContent", "")
            self.reply.append(text)
            return "text", text
        if kind == "tool.execution_start":
            return "note", _tool(data.get("toolName", ""), data.get("arguments") or {})
        if kind == "tool.execution_complete" and not data.get("success"):
            return "note", _short(f"gagal: {(data.get('error') or {}).get('message', '')}", 200)
        if kind == "session.error":
            self.error = _short(data.get("message") or "error", 300)
        elif kind == "result":
            self.session = ev.get("sessionId")
            premium = ev.get("usage", {}).get("premiumRequests")
            self.usage = f"premium request {premium}" if premium is not None else ""
            if ev.get("exitCode"):
                self.error = self.error or f"exit {ev['exitCode']}"
        return None

    def text(self) -> str:
        return "".join(self.reply).strip()


def with_files(provider: str, prompt: str, files) -> str:
    if not files:
        return prompt
    if provider == "gemini":
        return prompt + "\n\n" + " ".join("@" + str(f).replace("\\", "/") for f in files)
    listed = "\n".join(f"- {f}" for f in files)
    return f"{prompt}\n\nAttached files (read them with your tools if they are not shown inline):\n{listed}"


def image_blocks(files) -> list[dict]:
    blocks = []
    for f in files:
        if attach.is_image(f):
            data = base64.b64encode(Path(f).read_bytes()).decode("ascii")
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": attach.media_type(f),
                                                       "data": data}})
    return blocks


TURNS = {"claude": ClaudeTurn, "codex": CodexTurn, "gemini": GeminiTurn, "copilot": CopilotTurn}


def lineup(cfg: dict, tier: str, sticky: str | None = None, usage: dict | None = None) -> list[str]:
    routing = cfg.get("routing", {})
    providers = list(cfg["providers"])
    free = [p for p in providers if p in routing.get("free", [])]
    paid = [p for p in providers if p not in free]
    if usage:
        paid.sort(key=lambda p: usage.get(p, 0))
    free_first = tier in routing.get("free_tiers", [])
    if sticky in paid:
        paid.remove(sticky)
        paid.insert(0, sticky)
    elif sticky in free and free_first:
        free.remove(sticky)
        free.insert(0, sticky)
    return free + paid if free_first else paid + free


class Printer:
    def __init__(self):
        self.fresh = True

    def show(self, kind: str, text: str):
        if kind == "text":
            sys.stdout.write(text)
            sys.stdout.flush()
            self.fresh = text.endswith("\n")
        elif kind == "note":
            self.end()
            say(f"  · {text}")

    def end(self):
        if not self.fresh:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.fresh = True


class ConsoleUI:
    def __init__(self):
        self.out = Printer()

    def info(self, text: str):
        say(f"· {text}")

    def start(self, tier: str, provider: str, route: dict, reasons: list[str]):
        say(f"→ {tier} · {provider} {route['model']} ({route['effort']}) · {'; '.join(reasons) or 'default'}")

    def show(self, kind: str, text: str):
        self.out.show(kind, text)

    def done(self, seconds: float, usage: str):
        self.out.end()
        say(f"· {seconds}s · {usage}" if usage else f"· {seconds}s")

    def failed(self, provider: str, code: int, error: str, fallback: str | None):
        self.out.end()
        if code == 130:
            say("· dibatalkan")
            return
        say(f"· {provider} gagal (exit {code}): {error or 'tanpa pesan'}")
        if fallback:
            say(f"· pindah ke {fallback}")


def stream(cmd: list[str], prompt: str, turn, show, started=None) -> int:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if started:
        started(proc)
    errors = []
    reader = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
    reader.start()
    try:
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
        for raw in proc.stdout:
            try:
                ev = json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            shown = turn.feed(ev)
            if shown:
                show(*shown)
        code = proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        return 130
    reader.join(timeout=2)
    if code != 0 and not turn.error and errors and errors[0]:
        turn.error = _short(errors[0].decode("utf-8", "replace").strip().splitlines()[-1], 300)
    return code


class ClaudeProc:
    def __init__(self, model: str, effort: str, session: str | None):
        self.key = (model, effort)
        self.session = session
        cmd = dispatch.chat_cmd("claude", model, effort, session, attach_dir=str(attach.folder()))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.errors = []
        self.lock = threading.Lock()
        self.active = False
        self.sent = self.acked = 0
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for raw in self.proc.stderr:
            self.errors = (self.errors + [raw.decode("utf-8", "replace").strip()])[-5:]

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _write(self, prompt: str, files=()):
        content = [{"type": "text", "text": prompt}] + image_blocks(files)
        msg = {"type": "user", "message": {"role": "user", "content": content}}
        self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def ask(self, prompt: str, turn, show, files=()) -> int:
        try:
            with self.lock:
                self.active, self.sent, self.acked = True, 1, 0
                self._write(prompt, files)
            for raw in self.proc.stdout:
                try:
                    ev = json.loads(raw.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                if ev.get("isReplay"):
                    self.acked += 1
                    continue
                shown = turn.feed(ev)
                if shown:
                    show(*shown)
                if ev.get("type") == "result":
                    # A message replayed only after this result opens a turn of its own; keep reading until
                    # every message written during this ask has been taken in.
                    with self.lock:
                        if self.acked < self.sent:
                            continue
                        self.active = False
                    self.session = turn.session or self.session
                    return 1 if ev.get("is_error") else 0
        except (OSError, ValueError):
            pass
        self.active = False
        code = self.proc.wait()
        if not turn.error and self.errors:
            turn.error = _short(self.errors[-1], 300)
        return code or 1

    def inject(self, prompt: str, files=()) -> bool:
        with self.lock:
            if not self.active or not self.alive():
                return False
            try:
                self._write(prompt, files)
            except (OSError, ValueError):
                return False
            self.sent += 1
            return True

    def close(self):
        if self.alive():
            self.proc.kill()


class Chat:
    def __init__(self, cfg: dict, provider: str | None = None, tier: str | None = None, use_llm: bool = True,
                 ui=None):
        self.cfg = cfg
        self.ui = ui or ConsoleUI()
        self.pinned_provider = provider
        self.pinned_tier = tier
        self.pinned_model = None
        self._catalog = None
        self._catalog_lock = threading.Lock()
        self._proc = None
        self._claude = None
        self._codex = None
        self._live = None
        self._input_lock = threading.RLock()
        self._cancelled = False
        self.injected = []
        self.carry = {}
        self.use_llm = use_llm and cfg["classifier"].get("llm_fallback", False)
        self.reset()

    def reset(self):
        self.close()
        self.tier = None
        self.provider = None
        self.model = None
        self.sessions = {}
        self.synced = {}
        self.transcript = []

    def route(self, msg: str) -> tuple[str, list[str], bool]:
        if self.pinned_tier:
            return self.pinned_tier, ["dikunci"], False
        v = classify(msg, self.cfg.get("rules", {}))
        tier, reasons, llm = v.tier, list(v.reasons), False
        if v.ambiguous:
            ask = self.use_llm and self.tier is None
            guess, why, llm = learn.judge(msg, self.cfg, self.ui.info, ask, wait=False)
            if guess:
                tier = guess
                reasons.append(why)
            if guess and not llm and ask and learn.audit_due(self.cfg):
                threading.Thread(target=learn.audit, args=(msg, guess, self.cfg), daemon=True).start()
        floor = self.cfg.get("chat", {}).get("min_tier")
        if floor and higher(floor, tier) != tier:
            reasons.append(f"minimum chat {floor}")
            tier = floor
        if self.tier and higher(self.tier, tier) != tier:
            reasons.append(f"tetap {self.tier}")
            tier = self.tier
        return tier, reasons, llm

    def order(self, tier: str) -> list[str]:
        if self.pinned_provider:
            return [self.pinned_provider]
        return lineup(self.cfg, tier, self.provider, usage_today())

    def _started(self, proc):
        self._proc = proc
        if self._cancelled:
            proc.kill()

    def cancel(self):
        self._cancelled = True
        if isinstance(self._live, CodexProc):
            self._live.cancel()
            return
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def close(self):
        if self._claude:
            self._claude.close()
            self._claude = None
        if self._codex:
            self._codex.close()
            self._codex = None

    def claude_route(self, tier: str) -> dict:
        route = dict(self.cfg["tiers"][tier]["claude"])
        if self.pinned_model and self.pinned_provider == "claude":
            route["model"] = self.pinned_model
        return route

    def _claude_proc(self, route: dict) -> ClaudeProc:
        key, session = (route["model"], route["effort"]), self.sessions.get("claude")
        p = self._claude
        if p and p.alive() and p.key == key and p.session == session:
            return p
        self.close()
        self._claude = ClaudeProc(route["model"], route["effort"], session)
        return self._claude

    def prewarm(self):
        tier = self.pinned_tier or self.tier or self.cfg.get("chat", {}).get("min_tier") or "medium"
        if self.order(tier)[0] != "claude":
            return
        self._claude_proc(self.claude_route(tier))

    def _run(self, provider: str, route: dict, prompt: str, turn, runner, files=()) -> int:
        prompt = with_files(provider, prompt, files)
        if runner or provider not in ("claude", "codex"):
            cmd = dispatch.chat_cmd(provider, route["model"], route["effort"], self.sessions.get(provider),
                                    files, str(attach.folder()) if files else None)
            return (runner or stream)(cmd, prompt, turn, self.ui.show, self._started)
        if provider == "codex":
            key, session = (route["model"], route["effort"]), self.sessions.get("codex")
            p = self._codex
            if not p or not p.alive() or p.key != key or p.session != session:
                if p:
                    p.close()
                self._codex = p = CodexProc(*key, session)
            with p.lock:
                p.cancelled = False
            self._live = p
            self._started(p.proc)
            return p.ask(prompt, turn, self.ui.show, files)
        p = self._claude_proc(route)
        self._live = p
        self._started(p.proc)
        return p.ask(prompt, turn, self.ui.show, files)

    def send(self, msg: str, runner=None, attachments=()) -> int:
        self._cancelled = False
        self.injected = []
        tier, reasons, llm = self.route(msg)
        providers = self.order(tier)
        code = 1
        for i, provider in enumerate(providers):
            if self.injected:
                msg, self.injected = "\n\n".join([msg] + self.injected), []
            route = dict(self.cfg["tiers"][tier][provider])
            if self.pinned_model and provider == self.pinned_provider:
                route["model"] = self.pinned_model
            self.ui.start(tier, provider, route, reasons)
            unseen = self.transcript[self.synced.get(provider, 0):]
            prompt = recap(unseen) + msg if unseen else msg
            if provider not in self.sessions:
                prompt = CHAT_NOTE + prompt
            turn = TURNS[provider]()
            started = time.time()
            try:
                code = self._run(provider, route, prompt, turn, runner, attachments)
            except OSError as exc:
                turn.error, code = _short(str(exc), 300), 1
            self._proc = None
            if self._cancelled:
                code = 130
            seconds = round(time.time() - started, 1)
            if code == 0 and not turn.error and not turn.text():
                turn.error = "tidak ada jawaban"
            ok = code == 0 and not turn.error
            self._log(msg, tier, reasons, llm, provider, route, code, seconds, turn.session)
            if ok or code == 130:
                if turn.session:
                    self.sessions[provider] = turn.session
                reply = turn.text() + (" [dibatalkan]" if code == 130 else "")
                with self._input_lock:
                    said = "\n\n".join([msg] + self.injected)
                    self.injected = []
                said += f"\n[lampiran: {', '.join(f.name for f in attachments)}]" if attachments else ""
                self.transcript += [("User", said), ("Assistant", reply)]
                self.synced[provider] = len(self.transcript)
                self.provider, self.tier, self.model = provider, tier, route["model"]
                if ok:
                    self.ui.done(seconds, turn.usage)
                else:
                    self.ui.failed(provider, code, "", None)
                return code
            fallback = providers[i + 1] if i < len(providers) - 1 else None
            self.ui.failed(provider, code, turn.error, fallback)
        with self._input_lock:
            said = "\n\n".join([msg] + self.injected)
            self.injected = []
        said += f"\n[lampiran: {', '.join(f.name for f in attachments)}]" if attachments else ""
        self.transcript += [("User", said), ("Assistant", f"[gagal: {turn.error or code}]")]
        return code

    def inject(self, msg: str, files=()) -> bool:
        with self._input_lock:
            p = self._live
            content = with_files("codex" if isinstance(p, CodexProc) else "claude", msg, files)
            if not p or self._proc is not p.proc or not p.inject(content, files):
                return False
            self.injected.append(content)
            return True

    def _log(self, msg, tier, reasons, llm, provider, route, code, seconds, session):
        log({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cwd": os.getcwd(),
            "mode": "chat",
            "session": session,
            "tier": tier,
            "score": None,
            "reasons": reasons,
            "llm_classified": llm,
            "provider": provider,
            "model": route["model"],
            "effort": route["effort"],
            "exit": code,
            "seconds": seconds,
            "prompt": msg[:500],
        })

    def label(self) -> str:
        return self.pinned_model or self.model or "auto"

    def snapshot(self) -> dict:
        return {**{k: getattr(self, k) for k in STATE}, "carry": self.carry}

    def restore(self, state: dict):
        for k in STATE:
            if k in state:
                setattr(self, k, state[k])

    def catalog_ready(self) -> bool:
        return self._catalog is not None

    def catalog(self) -> list[tuple[str, str]]:
        with self._catalog_lock:
            return self._load_catalog()

    def _load_catalog(self) -> list[tuple[str, str]]:
        if self._catalog is None:
            models = self.cfg.get("models", {})
            catalog = []
            for provider in self.cfg["providers"]:
                found = dispatch.DISCOVER[provider]() if provider in dispatch.DISCOVER else []
                listed = list(dict.fromkeys(models.get(provider, []) + found))
                catalog += [(provider, m) for m in listed]
            self._catalog = catalog
        return self._catalog

    def pin(self, provider: str, model: str | None):
        self.pinned_provider, self.pinned_model = provider, model
        self.ui.info(f"provider dikunci: {provider}" + (f" · model {model}" if model else ""))
        if model:
            self.correct(learn.tier_of(self.cfg, provider, model))

    def correct(self, tier: str | None):
        users = [text for role, text in self.transcript if role == "User"]
        if not tier or not users or tier == self.tier:
            return
        learn.Learner.load().add(users[-1], tier, "user")
        self.ui.info(f"dicatat: pesan terakhir seharusnya {tier}, router belajar dari ini")

    def pick(self, arg: str):
        models = self.catalog()
        if arg.isdigit() and 1 <= int(arg) <= len(models):
            self.pin(*models[int(arg) - 1])
            return
        match = [pm for pm in models if pm[1] == arg] or [pm for pm in models if arg.lower() in pm[1].lower()]
        if len(match) == 1:
            self.pin(*match[0])
        elif match:
            self.ui.info("lebih dari satu cocok: " + ", ".join(m for _, m in match))
        else:
            self.ui.info(f"model '{arg}' tidak ada di /models; pakai /<provider> {arg} untuk memaksa, "
                         f"misalnya /codex {arg}")

    def status(self) -> str:
        return (f"tier {self.pinned_tier or self.tier or '-'}{' (dikunci)' if self.pinned_tier else ''}"
                f" · provider {self.pinned_provider or self.provider or '-'}"
                f"{' (dikunci)' if self.pinned_provider else ''} · model {self.label()}")

    def switch_command(self, msg: str) -> str | None:
        text = " ".join(msg.lower().split())
        match = SWITCH.fullmatch(text)
        if not match:
            return self._switch_sentence(text)
        target = match.group(1)
        if target in ("auto", "otomatis"):
            return "/model auto"
        if target in TIERS:
            return f"/model {target}"
        if target in self.cfg["providers"]:
            return f"/{target}"
        listed = self._catalog or [(p, m) for p, models in self.cfg.get("models", {}).items() for m in models]
        names = [m.lower() for _, m in listed]
        if target in names or (len(target) >= 3 and any(target in m for m in names)):
            return f"/model {target}"
        return self._switch_sentence(text)

    def _switch_sentence(self, text: str) -> str | None:
        if len(text.split()) > 14 or NOT_SWITCH.search(text):
            return None
        if not (SWITCH_VERB.search(text) or self._says(text, LIGHTER + STRONGER)):
            return None
        providers = [p for p in self.cfg["providers"] if re.search(rf"\b{p}\b", text)]
        listed = self._catalog or [(p, m) for p, models in self.cfg.get("models", {}).items() for m in models]
        named = [m for _, m in listed
                 if len(m) >= 4 and re.search(rf"(?<![\w.-]){re.escape(m.lower())}(?![\w.-])", text)]
        bare_step = (self.pinned_provider or self.provider) and BARE_STEP.fullmatch(text)
        if not (providers or named or re.search(r"\bmodel", text) or bare_step):
            return None
        if named:
            return f"/model {named[0]}"
        current = self.pinned_tier or self.tier or "medium"
        if self.pinned_provider and self.pinned_model:
            current = learn.tier_of(self.cfg, self.pinned_provider, self.pinned_model) or current
        step = TIERS.index(current)
        if self._says(text, LIGHTEST):
            tier = TIERS[0]
        elif self._says(text, STRONGEST):
            tier = TIERS[-1]
        elif self._says(text, LIGHTER):
            tier = TIERS[max(step - 1, 0)]
        elif self._says(text, STRONGER):
            tier = TIERS[min(step + 1, len(TIERS) - 1)]
        else:
            tier = next((t for t in TIERS if re.search(rf"\b{t}\b", text)), None)
        provider = providers[0] if providers else self.pinned_provider or self.provider
        if tier and provider:
            return f"/{provider} {self.cfg['tiers'][tier][provider]['model']}"
        if tier:
            return f"/model {tier}"
        if providers:
            return f"/{providers[0]}"
        return None

    @staticmethod
    def _says(text: str, words) -> bool:
        return any(re.search(r"(?<!\w)" + re.escape(w), text) for w in words)

    def list_models(self):
        current = self.label()
        rows = [f"{'*' if m == current else ' '} {i:>2}  {p:<7} {m}" for i, (p, m) in enumerate(self.catalog(), 1)]
        self.ui.info("\n".join(rows) + "\npilih: /model <nomor atau nama>")

    def command(self, line: str) -> bool:
        name, _, arg = line.partition(" ")
        arg = arg.strip()
        if name in ("/exit", "/quit"):
            return False
        if name == "/new":
            self.reset()
            self.ui.info("percakapan baru")
        elif name == "/model" and arg in TIERS:
            self.pinned_tier = arg
            self.ui.info(f"tier dikunci: {arg}")
            self.correct(arg)
        elif name == "/model" and arg == "auto":
            self.pinned_tier = self.pinned_provider = self.pinned_model = None
            self.ui.info("routing otomatis")
        elif name == "/model" and not arg:
            self.ui.info(self.status())
        elif name == "/model":
            self.pick(arg)
        elif name == "/models":
            self.list_models()
        elif name[1:] in self.cfg["providers"]:
            self.pin(name[1:], arg or None)
        else:
            self.ui.info(HELP)
        return True


def _pending() -> bool:
    try:
        import msvcrt
    except ImportError:
        import select
        return bool(select.select([sys.stdin], [], [], 0)[0])
    return msvcrt.kbhit()


def read_message(label: str) -> str | None:
    lines = [input(f"\033[1m{label} ›\033[0m ")]
    while True:
        if lines[-1].endswith("\\"):
            lines[-1] = lines[-1][:-1]
            lines.append(input("… "))
        elif _pending():
            lines.append(input())
        else:
            break
    return "\n".join(lines).strip()


def run(cfg: dict, provider: str | None = None, tier: str | None = None, use_llm: bool = True) -> int:
    chat = Chat(cfg, provider, tier, use_llm)
    say(f"ai chat · {os.getcwd()} · edit dan perintah dijalankan otomatis · /help")
    try:
        return _loop(chat)
    finally:
        chat.close()


def _loop(chat: Chat) -> int:
    chat.prewarm()
    while True:
        try:
            msg = read_message(chat.label())
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            say("· /exit untuk keluar")
            continue
        if not msg:
            continue
        msg = chat.switch_command(msg) or msg
        if msg.startswith("/"):
            if not chat.command(msg):
                return 0
            continue
        chat.send(msg)
        print()
