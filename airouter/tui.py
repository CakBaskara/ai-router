import importlib
import json
import os
import sys
import threading
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, Markdown, Static, TextArea, Tree

from . import attach, codex, config, learn
from .chat import HELP, Chat

PICKER_ORDER = ("codex", "claude")
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
PACKAGE = Path(__file__).parent
MODULES = ("config", "attach", "dispatch", "learn", "codex", "chat", "tui")


def _watched() -> list[Path]:
    return sorted(PACKAGE.glob("*.py")) + [Path(os.environ.get("AI_ROUTER_CONFIG", config.PACKAGE_CONFIG))]


def _stamp() -> tuple:
    return tuple((p.name, p.stat().st_mtime_ns) for p in _watched() if p.exists())


def _broken() -> str | None:
    for path in _watched():
        if path.suffix != ".py":
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            return f"{path.name}:{exc.lineno}: {exc.msg}"
    try:
        config.load()
    except Exception as exc:
        return f"config: {exc}"
    return None

DARK_MODERN = Theme(
    name="vscode-dark-modern",
    dark=True,
    background="#1F1F1F",
    surface="#181818",
    panel="#202020",
    foreground="#CCCCCC",
    primary="#9CDCFE",
    secondary="#4EC9B0",
    accent="#569CD6",
    warning="#DCDCAA",
    success="#B5CEA8",
    error="#F85149",
    variables={
        **{f"markdown-h{i}-color": "#E7E7E7" for i in range(1, 7)},
        **{f"markdown-h{i}-background": "transparent" for i in range(1, 7)},
        "block-cursor-background": "#0078D4",
        "input-selection-background": "#264F78",
        "scrollbar": "#424242",
        "scrollbar-hover": "#4F4F4F",
        "scrollbar-active": "#5A5A5A",
        "scrollbar-background": "ansi_default",
    },
)

CSS = """
Screen { background: ansi_default; }
Screen .screen--selection { background: #264F78; }
#top { height: 1; background: ansi_default; padding: 0 1; }
#brand { width: auto; color: #CCCCCC; text-style: bold; }
#cwd { width: 1fr; color: #9D9D9D; padding: 0 2; }
#quota { width: auto; color: #9D9D9D; padding: 0 1; }
#model { width: auto; color: #CCCCCC; padding: 0 1; }
#model:hover { background: #2B2B2B; }
#body { height: 1fr; background: ansi_default; }
#sticky { visibility: hidden; height: 1; padding: 0 0 0 2; background: ansi_default; }
#sticky .who { width: auto; color: #CCCCCC; text-style: bold; }
#sticky .why { width: auto; color: #6E7681; }
#log { height: 1fr; padding: 0 0 0 2; background: ansi_default; scrollbar-size-vertical: 1; }
#log > * { margin-right: 2; }
.arrows { height: 1; align: right top; background: ansi_default; }
ScrollButton { width: 1; height: 1; color: #9D9D9D; background: ansi_default; }
ScrollButton:hover { color: #FFFFFF; background: #2B2B2B; }
UserBubble { margin: 1 0 0 0; padding: 0 1; background: #262626; }
UserBubble.queued { color: #6E7681; }
Reply { height: auto; margin: 1 0 0 0; }
Reply .who { color: #CCCCCC; text-style: bold; }
Reply .why { color: #6E7681; }
Reply .tools { color: #9D9D9D; padding-left: 2; display: none; }
Reply Markdown { margin: 0; padding: 0 0 0 2; background: transparent; }
Reply .stats { color: #6E7681; padding-left: 2; }
Reply.failed .who { color: #F85149; }
Info { color: #6E7681; text-style: italic; margin: 1 0 0 0; }
#bottom { dock: bottom; height: auto; padding: 0 1 1 1; background: ansi_default; border-top: solid #2B2B2B; }
#prompt { width: 2; color: #6E7681; background: ansi_default; }
#inputrow { height: auto; background: ansi_default; }
#clip { width: 4; padding: 0 1; color: #9D9D9D; background: ansi_default; }
#clip:hover { background: #2B2B2B; }
#stop { width: 4; padding: 0 1; color: #3C3C3C; background: ansi_default; }
#stop.busy { color: #F85149; }
#stop:hover { background: #2B2B2B; }
#files { height: auto; color: #9D9D9D; background: ansi_default; display: none; }
#files.has { display: block; }
Composer { width: 1fr; height: auto; min-height: 1; max-height: 10; background: ansi_default; color: #CCCCCC; border: none; padding: 0; }
Composer:focus { border: none; }
Composer .text-area--selection { background: #264F78; }
Composer .text-area--cursor { color: #1F1F1F; background: #CCCCCC; text-style: none; }
ModelPicker { align: center middle; background: #000000 50%; }
CommandApproval { align: center middle; background: #000000 50%; }
#approval { width: 80%; height: 80%; border: round #454545; background: #202020; padding: 1 2; }
#approval-details { height: 1fr; }
#picker { width: 64; height: 90%; border: round #454545; background: #202020; padding: 0 2; }
#picker-title { color: #CCCCCC; text-style: bold; margin: 1 0; }
#picker-hint { color: #6E7681; margin: 1 0; }
#picker-tree { height: 1fr; border: none; background: #202020; color: #CCCCCC; padding: 0; }
#picker-tree > .tree--cursor { background: #04395E; color: #FFFFFF; }
#picker-tree > .tree--highlight { background: #2A2D2E; }
#picker-tree > .tree--guides { color: #3C3C3C; }
#picker-tree > .tree--guides-selected { color: #6E7681; }
"""
KEY_EVENT = 0x0001
SHIFT_PRESSED = 0x0010


def _shift_enter_as_newline():
    if sys.platform != "win32":
        return
    from textual.drivers import win32
    real = win32.KERNEL32.ReadConsoleInputW
    if getattr(real, "_shift_enter", False):
        return

    def read(handle, records, size, count):
        ok = real(handle, records, size, count)
        for record in records._obj[: count._obj.value]:
            key = record.Event.KeyEvent
            if (record.EventType == KEY_EVENT and key.bKeyDown and key.uChar.UnicodeChar == "\r"
                    and key.dwControlKeyState & SHIFT_PRESSED):
                key.uChar.UnicodeChar = "\n"
        return ok

    read._shift_enter = True
    win32.KERNEL32.ReadConsoleInputW = read


class UserBubble(Static):
    pass


class Info(Static):
    pass


class Reply(Vertical):
    def __init__(self, model: str, why: str, text: str = "", done: bool = False):
        super().__init__()
        self.model = model
        self.why = why
        self.text = text
        self.done = done
        self.tools = []
        self.frame = 0
        self.timer = None
        self.ready = False
        self.end = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="head"):
            yield Static(f"{'●' if self.done else SPINNER[0]} {self.model}", classes="who", markup=False)
            yield Static(f"  {self.why}", classes="why", markup=False)
        yield Static("", classes="tools", markup=False)
        yield Markdown(self.text)
        yield Static("", classes="stats", markup=False)

    def on_mount(self):
        self.query_one(".head").styles.height = 1
        self.query_one(".who").styles.width = "auto"
        self.ready = True
        if self.tools:
            self.show_tools()
        if self.end:
            self.show_end(*self.end)
        elif not self.done:
            self.timer = self.set_interval(0.08, self.spin)

    def spin(self):
        self.frame += 1
        self.query_one(".who", Static).update(f"{SPINNER[self.frame % len(SPINNER)]} {self.model}")

    def write(self, text: str):
        shown = self.query(Markdown)
        if shown:
            return shown.first().append(text)
        self.text += text
        return None

    def add_tool(self, text: str):
        self.tools.append(text)
        if self.ready:
            self.show_tools()

    def show_tools(self):
        tools = self.query_one(".tools", Static)
        tools.update("\n".join(f"⎿ {t}" for t in self.tools))
        tools.display = True

    def finish(self, mark: str, stats: str):
        # The worker thread can stream into and finish a reply before Textual has mounted it.
        self.done = True
        self.end = (mark, stats)
        if self.timer:
            self.timer.stop()
        if self.ready:
            self.show_end(mark, stats)

    def show_end(self, mark: str, stats: str):
        self.query_one(".who", Static).update(f"{mark} {self.model}")
        self.query_one(".stats", Static).update(stats)


class Composer(TextArea):
    BINDINGS = [
        Binding("ctrl+a", "select_everything", "Pilih semua", show=False),
        Binding("ctrl+c", "copy", "Salin", show=False),
        Binding("alt+v", "paste_image", "Tempel gambar", show=False),
        Binding("ctrl+v", "paste_any", "Tempel", show=False),
    ]

    class Submitted(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    def on_mount(self):
        self.cursor_blink = False

    def action_select_everything(self):
        if self.text:
            self.select_all()
        else:
            self.screen._select_all_in_widget(self.app.query_one("#log"))

    def action_copy(self):
        if self.selected_text:
            self.app.copy_to_clipboard(self.selected_text)
        else:
            self.screen.action_copy_text()

    def action_paste_image(self):
        self.app.attach_from_clipboard()

    def action_paste_any(self):
        if not self.app.attach_from_clipboard(quiet=True):
            self.action_paste()

    async def _on_paste(self, event):
        if not event.text.strip():
            event.prevent_default()
            event.stop()
            self.app.attach_from_clipboard()

    async def _on_key(self, event):
        if event.key == "backspace" and not self.text and self.app.attachments:
            event.prevent_default()
            event.stop()
            self.app.drop_attachment()
        elif event.key == "enter" and self.text.endswith("\\"):
            event.prevent_default()
            event.stop()
            self.text = self.text[:-1] + "\n"
            self.move_cursor(self.document.end)
        elif event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self.text))
        elif event.key in ("shift+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")


class ModelChip(Static):
    def on_click(self):
        self.app.action_models()


class AttachButton(Static):
    def on_click(self):
        self.app.pick_files()


class StopButton(Static):
    def on_click(self):
        self.app.action_cancel()


class ScrollButton(Static):
    STEP = 3

    def __init__(self, label: str, direction: int, **kwargs):
        super().__init__(label, **kwargs)
        self.direction = direction
        self.timers = []

    def on_mouse_down(self, event):
        event.stop()
        self.capture_mouse()
        self.step()
        self.timers.append(self.set_timer(0.35, self.repeat))

    def repeat(self):
        self.timers.append(self.set_interval(0.05, self.step))

    def on_mouse_up(self, event):
        event.stop()
        self.release_mouse()
        self.halt()

    def halt(self):
        for timer in self.timers:
            timer.stop()
        self.timers = []

    def step(self):
        self.app.query_one("#log", VerticalScroll).scroll_relative(y=self.direction * self.STEP, animate=False)


class ModelPicker(ModalScreen):
    BINDINGS = [
        Binding("escape", "dismiss", "Tutup"),
        Binding("home", "first", show=False, priority=True),
        Binding("end", "last", show=False, priority=True),
    ]

    def __init__(self, catalog: list[tuple[str, str]], current: str, auto: bool, provider: str | None = None):
        super().__init__()
        self.catalog = catalog
        self.current = current
        self.auto = auto
        self.provider = provider

    def compose(self) -> ComposeResult:
        tree = Tree("model", id="picker-tree")
        tree.show_root = False
        tree.guide_depth = 3
        tree.root.add_leaf(Text.assemble(("● " if self.auto else "  "), ("Otomatis", "bold"),
                                         ("  router memilih per pesan", "dim")), data="auto")
        present = list(dict.fromkeys(p for p, _ in self.catalog))
        for provider in [p for p in PICKER_ORDER if p in present] + [p for p in present if p not in PICKER_ORDER]:
            models = [(i, m) for i, (p, m) in enumerate(self.catalog) if p == provider]
            mine = not self.auto and provider == self.provider
            folder = tree.root.add(Text.assemble((provider.capitalize(), "bold"),
                                                 (f"  {len(models)} model", "dim")),
                                   expand=mine)
            folder.add_leaf(Text.assemble(("● " if mine and self.current == provider else "  "),
                                          ("otomatis", "italic"), ("  model dipilih router", "dim")),
                            data=f"prov:{provider}")
            for i, model in models:
                mark = "● " if mine and model == self.current else "  "
                folder.add_leaf(Text.assemble(mark, model), data=str(i + 1))
        with Vertical(id="picker"):
            yield Static("Pilih model", id="picker-title")
            yield tree
            yield Static("↑↓ pilih · Enter buka / pakai · Esc tutup", id="picker-hint")

    def on_mount(self):
        tree = self.query_one(Tree)
        tree.focus()
        tree.cursor_line = 0

    def action_first(self):
        self.query_one(Tree).move_cursor_to_line(0)

    def action_last(self):
        tree = self.query_one(Tree)
        tree.move_cursor_to_line(tree.last_line)

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        if event.node.data:
            self.dismiss(event.node.data)


class CommandApproval(ModalScreen):
    BINDINGS = [Binding("escape", "reject", "Tolak")]

    def __init__(self, params):
        super().__init__()
        self.params = params

    def compose(self) -> ComposeResult:
        with Vertical(id="approval"):
            yield Static("Codex meminta izin menjalankan perintah di luar sandbox")
            with VerticalScroll(id="approval-details"):
                yield Static(json.dumps(self.params, ensure_ascii=False, indent=2), markup=False)
            with Horizontal():
                yield Button("Tolak", id="reject")
                yield Button("Izinkan sekali", id="accept")

    def action_reject(self):
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed):
        self.dismiss(event.button.id == "accept")


class TuiUI:
    def __init__(self, app: "ChatApp"):
        self.app = app

    def _call(self, fn, *args):
        if threading.current_thread() is threading.main_thread():
            return fn(*args)
        return self.app.call_from_thread(fn, *args)

    def info(self, text: str):
        self._call(self.app.add_info, text)

    def approve(self, params, cancelled):
        ready = threading.Event()
        decision = []
        screen = CommandApproval(params)

        def answered(value):
            decision.append(bool(value))
            ready.set()

        self._call(self.app.push_screen, screen, answered)
        while not ready.wait(0.1):
            if cancelled():
                self._call(screen.dismiss, False)
                return False
        return decision[0]

    def start(self, tier: str, provider: str, route: dict, reasons: list[str]):
        self._call(self.app.start_reply, route["model"], f"{provider} · {tier} · {route['effort']}")

    def show(self, kind: str, text: str):
        self._call(self.app.reply_show, kind, text)

    def done(self, seconds: float, usage: str):
        self._call(self.app.end_reply, "●", f"✓ {seconds}s" + (f" · {usage}" if usage else ""), False)

    def failed(self, provider: str, code: int, error: str, fallback: str | None):
        if code == 130:
            self._call(self.app.end_reply, "■", "dibatalkan", False)
            return
        note = f"✗ gagal (exit {code}): {error or 'tanpa pesan'}" + (f" · pindah ke {fallback}" if fallback else "")
        self._call(self.app.end_reply, "✗", note, True)


class ChatApp(App):
    CSS = CSS
    TITLE = "ai"
    BINDINGS = [
        Binding("escape", "cancel", "Batal"),
        Binding("ctrl+o", "models", "Model"),
        Binding("ctrl+n", "new", "Baru"),
        Binding("ctrl+q", "quit", "Keluar"),
    ]

    def __init__(self, cfg: dict, provider: str | None, tier: str | None, use_llm: bool, state: dict | None = None):
        super().__init__(ansi_color=True)
        self.chat = Chat(cfg, provider, tier, use_llm, ui=TuiUI(self))
        self.restored = bool(state)
        if state:
            self.chat.restore(state)
        carry = (state or {}).get("carry") or {}
        self.draft = carry.get("draft", "")
        self.pending = list(carry.get("queue", []))
        self.reply = None
        self._busy = False
        self.queue = []
        self.attachments = list(carry.get("attachments", []))
        self.picking = False
        self.stamp = _stamp()

    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield Static("✻ ai", id="brand")
            yield Static(os.getcwd(), id="cwd", markup=False)
            yield Static("", id="quota", markup=False)
            yield ModelChip("", id="model", markup=False)
        with Vertical(id="body"):
            with Horizontal(classes="arrows"):
                yield ScrollButton("▲", -1, id="up")
            with Horizontal(id="sticky"):
                yield Static("", classes="who", id="sticky-who", markup=False)
                yield Static("", classes="why", id="sticky-why", markup=False)
            yield VerticalScroll(id="log")
            with Horizontal(classes="arrows"):
                yield ScrollButton("▼", 1, id="down")
        with Vertical(id="bottom"):
            yield Static("", id="files", markup=False)
            with Horizontal(id="inputrow"):
                yield Static("›", id="prompt")
                yield Composer(id="input", placeholder="Tanya apa saja…", highlight_cursor_line=False)
                yield StopButton("■", id="stop")
                yield AttachButton("📎", id="clip")

    def on_mount(self):
        self.register_theme(DARK_MODERN)
        self.theme = DARK_MODERN.name
        if self.restored:
            for role, text in self.chat.transcript:
                if role == "User":
                    self._bubble(text)
                else:
                    self._mount(Reply("sebelumnya", "", text, done=True))
            self.add_info("Kode diperbarui dan dimuat ulang; percakapan tetap berlanjut.")
        else:
            self.add_info("Edit dan perintah dijalankan otomatis (mode auto). "
                          "Ctrl+O ganti model · Esc hentikan jawaban · /help perintah")
        self.query_one("#log", VerticalScroll).anchor()
        self.refresh_status()
        composer = self.query_one(Composer)
        if self.draft:
            composer.text = self.draft
            composer.move_cursor(composer.document.end)
        if self.attachments:
            self.refresh_files()
        composer.focus()
        if self.pending:
            self.queue = [(text, self._bubble(text, files, queued=True), files) for text, files in self.pending]
            self.call_after_refresh(self.after_send)
        self.warm_catalog()
        self.warm_classifier()
        self.warm_quota()
        self.set_interval(30, self.refresh_quota)
        self.chat.prewarm()
        self.set_interval(2, self.check_code)
        self.set_interval(0.1, self.inject_queued)
        self.set_interval(0.1, self.refresh_sticky)

    def on_unmount(self):
        self.chat.close()

    @property
    def busy(self) -> bool:
        return self._busy

    @busy.setter
    def busy(self, value: bool):
        self._busy = value
        for stop in self.query("#stop"):
            stop.set_class(value, "busy")

    def code_changed(self) -> bool:
        if isinstance(self.screen, ModalScreen):
            return False
        stamp = _stamp()
        if stamp == self.stamp:
            return False
        self.stamp = stamp
        problem = _broken()
        if problem:
            self.notify(f"Kode berubah tapi belum valid, belum dimuat ulang: {problem}", severity="warning")
            return False
        return True

    def check_code(self):
        if not self.busy and self.code_changed():
            self.reload()

    def reload(self):
        self.busy = True
        self.chat.carry = {"draft": self.query_one(Composer).text, "attachments": list(self.attachments),
                           "queue": [(text, list(files)) for text, _, files in self.queue]}
        self.exit("reload")

    @work(thread=True)
    def warm_catalog(self):
        self.chat.catalog()

    @work(thread=True)
    def warm_quota(self):
        config.save_quota("codex", config.codex_windows(codex.read_limits()))
        self.call_from_thread(self.refresh_quota)

    def refresh_quota(self):
        for widget in self.query("#quota"):
            widget.update(config.quota_line())

    @work(thread=True)
    def warm_classifier(self):
        if self.chat.cfg["classifier"].get("embed", True):
            learn.warm_encoder(self.chat.ui.info)

    def refresh_status(self):
        queued = f"  ·  antre {len(self.queue)}" if self.queue else ""
        model = self.reply.model if self.busy and self.reply else self.chat.label()
        self.query_one("#model", Static).update(f"◆ {model} ▾{queued}")
        self.refresh_quota()

    def refresh_sticky(self):
        logs = self.query("#log")
        stickies = self.query("#sticky")
        if not logs or not stickies:
            return
        log = logs[0]
        top = log.content_region.y
        visible = next((reply for reply in reversed(list(log.query(Reply)))
                        if reply.region.y < top < reply.region.bottom), None)
        heads = {target: visible.query_one(source, Static).render()
                 for target, source in (("#sticky-who", ".who"), ("#sticky-why", ".why"))} if visible else {}
        shown = {target: str(content) for target, content in heads.items()}
        # Each update reflows the whole log, and this runs ten times a second; touching the
        # widgets only on a change keeps an idle window from burning a CPU core.
        if shown == getattr(self, "_sticky_shown", None):
            return
        self._sticky_shown = shown
        stickies[0].visible = visible is not None
        for target, content in heads.items():
            for widget in self.query(target):
                widget.update(content)

    def _mount(self, widget):
        self.query_one("#log", VerticalScroll).mount(widget)

    def add_info(self, text: str):
        self._mount(Info(text, markup=False))

    def start_reply(self, model: str, why: str):
        self.reply = Reply(model, why)
        self._mount(self.reply)
        self.refresh_status()

    def reply_show(self, kind: str, text: str):
        if not self.reply:
            return None
        if kind == "note":
            self.reply.add_tool(text)
            return None
        if kind == "break":
            text = "\n\n"
        return self.reply.write(text)

    def end_reply(self, mark: str, stats: str, failed: bool):
        if self.reply:
            self.reply.finish(mark, stats)
            self.reply.set_class(failed, "failed")

    @on(Composer.Submitted)
    def submitted(self, event: Composer.Submitted):
        text = event.text.strip()
        text = self.chat.switch_command(text) or text
        files = [] if text.startswith("/") else list(self.attachments)
        if not text and not files:
            return
        text = text or "Lihat lampiran."
        self.query_one(Composer).clear()
        if files:
            self.attachments = []
            self.refresh_files()
        if self.busy:
            if not text.startswith("/") and self.chat.inject(text, files):
                self.continue_reply(text, files)
                return
            bubble = None if text.startswith("/") else self._bubble(text, files, queued=True)
            self.queue.append((text, bubble, files))
            self.refresh_status()
            return
        self.dispatch_input(text, None, files)

    def continue_reply(self, text: str, files=(), bubble: UserBubble | None = None):
        old = self.reply
        if old:
            old.finish("●", "")
        if bubble:
            bubble.remove_class("queued")
        else:
            self._bubble(text, files)
        if old:
            self.start_reply(old.model, old.why)

    def inject_queued(self):
        if not self.busy:
            return
        while self.queue:
            text, bubble, files = self.queue[0]
            if text.startswith("/") or not self.chat.inject(text, files):
                return
            self.queue.pop(0)
            self.continue_reply(text, files, bubble)
            self.refresh_status()

    def _bubble(self, text: str, files=(), queued: bool = False) -> UserBubble:
        shown = text + ("\n" + attach.describe(files) if files else "")
        bubble = UserBubble(shown, markup=False, classes="queued" if queued else "")
        self._mount(bubble)
        return bubble

    def dispatch_input(self, text: str, bubble: UserBubble | None = None, files=()) -> bool:
        if text.startswith("/"):
            self.run_command(text)
            return False
        if self.code_changed():
            self.queue.insert(0, (text, bubble, files))
            self.reload()
            return True
        if bubble:
            bubble.remove_class("queued")
        else:
            self._bubble(text, files)
        self.busy = True
        self.send(text, list(files))
        return True

    def attach_files(self, paths):
        for path in paths:
            try:
                self.attachments.append(attach.store(path))
            except (OSError, ValueError) as exc:
                self.notify(f"Tidak bisa melampirkan {path.name}: {exc}", severity="warning")
        self.refresh_files()

    @work(thread=True)
    def pick_files(self):
        if self.picking:
            return
        self.picking = True
        try:
            paths = attach.pick_files()
        except Exception as exc:
            self.call_from_thread(self.notify, f"Dialog file gagal dibuka: {exc}", severity="warning")
            return
        finally:
            self.picking = False
        if paths:
            self.call_from_thread(self.attach_files, paths)

    def attach_from_clipboard(self, quiet: bool = False) -> bool:
        try:
            files = attach.from_clipboard()
        except Exception as exc:
            files, problem = [], str(exc)
        else:
            problem = "clipboard tidak berisi gambar atau file"
        if not files:
            if not quiet:
                self.notify(f"Tidak ada yang dilampirkan: {problem}", severity="warning")
            return False
        self.attachments += files
        self.refresh_files()
        return True

    def drop_attachment(self):
        self.attachments.pop()
        self.refresh_files()

    def refresh_files(self):
        files = self.query_one("#files", Static)
        files.update("  ".join(attach.describe([f]) for f in self.attachments) + "   (⌫ hapus)")
        files.set_class(bool(self.attachments), "has")

    def run_command(self, text: str):
        name = text.split()[0]
        if name in ("/exit", "/quit"):
            self.exit()
        elif name == "/models":
            self.action_models()
        elif name == "/new":
            self.action_new()
        elif name == "/help":
            self.add_info(HELP)
        elif name == "/reload":
            self.reload()
        else:
            self.chat.command(text)
            self.refresh_status()
            self.chat.prewarm()

    @work(thread=True, exclusive=True)
    def send(self, text: str, files=()):
        try:
            self.chat.send(text, attachments=files)
        except Exception as exc:
            self.call_from_thread(self.add_info, f"error: {exc}")
        finally:
            self.call_from_thread(self.after_send)

    def after_send(self):
        self.busy = False
        self.reply = None
        while self.queue:
            text, bubble, files = self.queue.pop(0)
            if self.dispatch_input(text, bubble, files):
                break
        self.refresh_status()
        if not self.busy:
            self.chat.prewarm()

    def action_cancel(self):
        if self.busy:
            self.chat.cancel()

    def action_new(self):
        self.queue.clear()
        if self.busy:
            self.chat.cancel()
        self.chat.reset()
        self.query_one("#log", VerticalScroll).remove_children()
        self.add_info("Percakapan baru.")
        self.refresh_status()
        self.chat.prewarm()

    def action_models(self):
        if isinstance(self.screen, ModelPicker):
            return
        if self.chat.catalog_ready():
            self.open_picker()
        else:
            self.notify("Memuat daftar model…")
            self.load_then_open()

    @work(thread=True, exclusive=True, group="catalog")
    def load_then_open(self):
        self.chat.catalog()
        self.call_from_thread(self.open_picker)

    def open_picker(self):
        if isinstance(self.screen, ModelPicker):
            return
        c = self.chat
        auto = not (c.pinned_model or c.pinned_provider)
        provider = c.pinned_provider or c.provider
        current = c.pinned_model or (provider if c.pinned_provider and not c.pinned_model else c.label())
        self.push_screen(ModelPicker(c.catalog(), current, auto, provider), self.picked)

    def picked(self, choice):
        if choice == "auto":
            self.chat.command("/model auto")
            self.notify("Model: otomatis")
        elif choice and choice.startswith("prov:"):
            self.chat.command("/" + choice[5:])
            self.notify(f"Provider: {choice[5:]} (model dipilih router)")
        elif choice:
            self.chat.pick(choice)
            self.notify(f"Model: {self.chat.label()} ({self.chat.pinned_provider})")
        self.refresh_status()
        self.chat.prewarm()
        self.query_one(Composer).focus()


def run(cfg: dict, provider: str | None = None, tier: str | None = None, use_llm: bool = True) -> int:
    state = None
    while True:
        module = sys.modules[__name__]
        module._shift_enter_as_newline()
        app = module.ChatApp(cfg, provider, tier, use_llm, state)
        if app.run() != "reload":
            return 0
        state = app.chat.snapshot()
        for name in MODULES:
            loaded = sys.modules.get(f"{__package__}.{name}")
            if loaded and Path(loaded.__file__).exists():
                importlib.reload(loaded)
        cfg = sys.modules[f"{__package__}.config"].load()
