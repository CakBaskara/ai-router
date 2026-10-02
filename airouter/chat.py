import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

from . import dispatch, learn
from .classify import TIERS, classify
from .journal import log, say

HELP = (
    "/new                      percakapan baru\n"
    "/model                    tampilkan tier, provider, model\n"
    "/model light|medium|heavy kunci tier\n"
    "/model auto               kembali ke routing otomatis\n"
    "/models                   daftar model yang bisa dipilih\n"
    "/model 3  /model sol      pilih model dari daftar (nomor atau nama)\n"
    "/claude  /codex           kunci provider\n"
    "/codex gpt-5.6-sol        kunci provider dan model persis\n"
    "/exit                    keluar\n"
    "Baris diakhiri \\ untuk lanjut ke baris berikutnya; teks yang di-paste ikut terkirim utuh."
)
CHAT_NOTE = (
    "[Context from the `ai` chat app, not from the user] You are answering inside `ai`, the user's terminal chat "
    "that routes each message to Claude Code or Codex CLI and picks the model automatically. You cannot switch "
    "models yourself. If the user wants another model, tell them to type /models for the list, /model <number or "
    "name> to pick one, or /model auto to go back to automatic routing; /new starts a new conversation. "
    "Answer naturally, like a helpful colleague, not curtly.\n\n"
)
STATE = ("tier", "provider", "model", "sessions", "synced", "transcript",
         "pinned_provider", "pinned_tier", "pinned_model")
RECAP_ENTRIES = 12
RECAP_CHARS = 2000
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
                    args = block.get("input", {})
                    detail = str(next((args[k] for k in TOOL_KEYS if args.get(k)), ""))
                    if os.path.isabs(detail) and detail.lower().startswith(os.getcwd().lower()):
                        detail = os.path.relpath(detail)
                    return "note", _short(f"{block.get('name')} {detail}")
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

    def feed(self, ev: dict):
        kind = ev.get("type")
        item = ev.get("item", {})
        itype = item.get("type")
        if kind == "thread.started":
            self.session = ev.get("thread_id")
        elif kind == "item.started" and itype == "command_execution":
            return "note", _short(f"$ {item.get('command', '')}")
        elif kind == "item.completed" and itype == "agent_message":
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
        cmd = dispatch.chat_cmd("claude", model, effort, session)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.errors = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for raw in self.proc.stderr:
            self.errors = (self.errors + [raw.decode("utf-8", "replace").strip()])[-5:]

    def alive(self) -> bool:
        return self.proc.poll() is None

    def ask(self, prompt: str, turn, show) -> int:
        msg = {"type": "user", "message": {"role": "user", "content": prompt}}
        try:
            self.proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
            for raw in self.proc.stdout:
                try:
                    ev = json.loads(raw.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                shown = turn.feed(ev)
                if shown:
                    show(*shown)
                if ev.get("type") == "result":
                    self.session = turn.session or self.session
                    return 1 if ev.get("is_error") else 0
        except (OSError, ValueError):
            pass
        code = self.proc.wait()
        if not turn.error and self.errors:
            turn.error = _short(self.errors[-1], 300)
        return code or 1

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
        self._proc = None
        self._claude = None
        self._cancelled = False
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
            guess, why, llm = learn.judge(msg, self.cfg, self.ui.info, self.use_llm and self.tier is None)
            if guess:
                tier = guess
                reasons.append(why)
        floor = self.cfg.get("chat", {}).get("min_tier")
        if floor and higher(floor, tier) != tier:
            reasons.append(f"minimum chat {floor}")
            tier = floor
        if self.tier and higher(self.tier, tier) != tier:
            reasons.append(f"tetap {self.tier}")
            tier = self.tier
        return tier, reasons, llm

    def order(self) -> list[str]:
        if self.pinned_provider:
            return [self.pinned_provider]
        providers = list(self.cfg["providers"])
        if self.provider in providers:
            providers.remove(self.provider)
            providers.insert(0, self.provider)
        return providers

    def _started(self, proc):
        self._proc = proc
        if self._cancelled:
            proc.kill()

    def cancel(self):
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def close(self):
        if self._claude:
            self._claude.close()
            self._claude = None

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
        if self.order()[0] != "claude":
            return
        tier = self.pinned_tier or self.tier or self.cfg.get("chat", {}).get("min_tier") or "medium"
        self._claude_proc(self.claude_route(tier))

    def _run(self, provider: str, route: dict, prompt: str, turn, runner) -> int:
        if runner:
            cmd = dispatch.chat_cmd(provider, route["model"], route["effort"], self.sessions.get(provider))
            return runner(cmd, prompt, turn, self.ui.show, self._started)
        if provider != "claude":
            cmd = dispatch.chat_cmd(provider, route["model"], route["effort"], self.sessions.get(provider))
            return stream(cmd, prompt, turn, self.ui.show, self._started)
        p = self._claude_proc(route)
        self._started(p.proc)
        return p.ask(prompt, turn, self.ui.show)

    def send(self, msg: str, runner=None) -> int:
        self._cancelled = False
        tier, reasons, llm = self.route(msg)
        providers = self.order()
        code = 1
        for i, provider in enumerate(providers):
            route = dict(self.cfg["tiers"][tier][provider])
            if self.pinned_model and provider == self.pinned_provider:
                route["model"] = self.pinned_model
            self.ui.start(tier, provider, route, reasons)
            unseen = self.transcript[self.synced.get(provider, 0):]
            prompt = recap(unseen) + msg if unseen else msg
            if provider not in self.sessions:
                prompt = CHAT_NOTE + prompt
            turn = ClaudeTurn() if provider == "claude" else CodexTurn()
            started = time.time()
            code = self._run(provider, route, prompt, turn, runner)
            self._proc = None
            if self._cancelled:
                code = 130
            seconds = round(time.time() - started, 1)
            ok = code == 0 and not turn.error
            self._log(msg, tier, reasons, llm, provider, route, code, seconds, turn.session)
            if ok or code == 130:
                if turn.session:
                    self.sessions[provider] = turn.session
                reply = turn.text() + (" [dibatalkan]" if code == 130 else "")
                self.transcript += [("User", msg), ("Assistant", reply)]
                self.synced[provider] = len(self.transcript)
                self.provider, self.tier, self.model = provider, tier, route["model"]
                if ok:
                    self.ui.done(seconds, turn.usage)
                else:
                    self.ui.failed(provider, code, "", None)
                return code
            fallback = providers[i + 1] if i < len(providers) - 1 else None
            self.ui.failed(provider, code, turn.error, fallback)
        return code

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
        return {k: getattr(self, k) for k in STATE}

    def restore(self, state: dict):
        for k in STATE:
            if k in state:
                setattr(self, k, state[k])

    def catalog(self) -> list[tuple[str, str]]:
        if self._catalog is None:
            models = self.cfg.get("models", {})
            codex = models.get("codex") or dispatch.codex_models()
            self._catalog = [("claude", m) for m in models.get("claude", [])] + [("codex", m) for m in codex]
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
            self.ui.info(f"model '{arg}' tidak ada di /models; pakai /claude {arg} atau /codex {arg} untuk memaksa")

    def status(self) -> str:
        return (f"tier {self.pinned_tier or self.tier or '-'}{' (dikunci)' if self.pinned_tier else ''}"
                f" · provider {self.pinned_provider or self.provider or '-'}"
                f"{' (dikunci)' if self.pinned_provider else ''} · model {self.label()}")

    def list_models(self):
        current = self.label()
        rows = [f"{'*' if m == current else ' '} {i:>2}  {p:<6} {m}" for i, (p, m) in enumerate(self.catalog(), 1)]
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
        elif name in ("/claude", "/codex"):
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
        if msg.startswith("/"):
            if not chat.command(msg):
                return 0
            continue
        chat.send(msg)
        print()
