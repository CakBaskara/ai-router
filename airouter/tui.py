import importlib
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
from textual.widgets import Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from . import config
from .chat import HELP, Chat

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
PACKAGE = Path(__file__).parent
MODULES = ("config", "classify", "journal", "dispatch", "learn", "chat", "tui")


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
        "scrollbar-background": "#1F1F1F",
    },
)

CSS = """
Screen { background: #1F1F1F; }
Screen .screen--selection { background: #264F78; }
#top { height: 1; background: #181818; padding: 0 1; }
#brand { width: auto; color: #CCCCCC; text-style: bold; }
#cwd { width: 1fr; color: #9D9D9D; padding: 0 2; }
#model { width: auto; color: #CCCCCC; padding: 0 1; }
#model:hover { background: #2B2B2B; }
#log { padding: 0 2 1 2; background: #1F1F1F; }
UserBubble {
    margin: 1 0 0 0; padding: 0 1; background: #262626;
    border: round #3C3C3C; border-title-color: #9D9D9D;
}
Reply { height: auto; margin: 1 0 0 0; }
Reply .who { color: #CCCCCC; text-style: bold; }
Reply .why { color: #6E7681; }
Reply .tools { color: #9D9D9D; padding-left: 2; display: none; }
Reply Markdown { margin: 0; padding: 0 0 0 2; background: transparent; }
Reply .stats { color: #6E7681; padding-left: 2; }
Reply.failed .who { color: #F85149; }
Info { color: #6E7681; text-style: italic; margin: 1 0 0 0; }
#bottom { dock: bottom; height: auto; padding: 0 1; background: #1F1F1F; }
Composer {
    height: auto; min-height: 3; max-height: 12; background: #313131; color: #CCCCCC;
    border: round #3C3C3C;
}
Composer:focus { border: round #0078D4; }
Composer .text-area--selection { background: #264F78; }
#status { height: 1; background: #181818; color: #9D9D9D; padding: 0 1; }
ModelPicker { align: center middle; background: #000000 50%; }
#picker { width: 64; height: auto; max-height: 90%; border: round #454545; background: #202020; padding: 1 2; }
#picker-title { color: #CCCCCC; text-style: bold; margin-bottom: 1; }
#picker OptionList { height: auto; max-height: 26; border: none; background: #202020; color: #CCCCCC; }
#picker OptionList > .option-list--option-highlighted { background: #04395E; color: #FFFFFF; }
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
        self.timer = self.set_interval(0.08, self.spin, pause=self.done)

    def spin(self):
        self.frame += 1
        self.query_one(".who", Static).update(f"{SPINNER[self.frame % len(SPINNER)]} {self.model}")

    def add_tool(self, text: str):
        self.tools.append(text)
        tools = self.query_one(".tools", Static)
        tools.update("\n".join(f"⎿ {t}" for t in self.tools))
        tools.display = True

    def finish(self, mark: str, stats: str):
        self.timer.stop()
        self.query_one(".who", Static).update(f"{mark} {self.model}")
        self.query_one(".stats", Static).update(stats)


class Composer(TextArea):
    BINDINGS = [
        Binding("ctrl+a", "select_everything", "Pilih semua", show=False),
        Binding("ctrl+c", "copy", "Salin", show=False),
    ]

    class Submitted(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

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

    async def _on_key(self, event):
        if event.key == "enter" and self.text.endswith("\\"):
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


class ModelPicker(ModalScreen):
    BINDINGS = [Binding("escape", "dismiss", "Tutup")]

    def __init__(self, catalog: list[tuple[str, str]], current: str, auto: bool):
        super().__init__()
        self.catalog = catalog
        self.current = current
        self.auto = auto

    def compose(self) -> ComposeResult:
        options = [Option(Text.assemble(("● " if self.auto else "  "), ("Otomatis", "bold"),
                                        ("  router memilih per pesan", "dim")), id="auto")]
        for provider in dict.fromkeys(p for p, _ in self.catalog):
            options.append(Option(Text(provider.capitalize(), style="bold dim"), disabled=True))
            for i, (p, model) in enumerate(self.catalog):
                if p == provider:
                    mark = "● " if model == self.current and not self.auto else "  "
                    options.append(Option(Text.assemble(mark, model), id=str(i + 1)))
        with Vertical(id="picker"):
            yield Static("Pilih model", id="picker-title")
            yield OptionList(*options)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        self.dismiss(event.option.id)


class TuiUI:
    def __init__(self, app: "ChatApp"):
        self.app = app

    def _call(self, fn, *args):
        if threading.current_thread() is threading.main_thread():
            return fn(*args)
        return self.app.call_from_thread(fn, *args)

    def info(self, text: str):
        self._call(self.app.add_info, text)

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
        super().__init__()
        self.chat = Chat(cfg, provider, tier, use_llm, ui=TuiUI(self))
        self.restored = bool(state)
        if state:
            self.chat.restore(state)
        self.reply = None
        self.busy = False
        self.queue = []
        self.stamp = _stamp()

    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield Static("✻ ai", id="brand")
            yield Static(os.getcwd(), id="cwd", markup=False)
            yield ModelChip("", id="model", markup=False)
        yield VerticalScroll(id="log")
        with Vertical(id="bottom"):
            yield Composer(id="input", placeholder="Tanya apa saja…   Enter kirim · Shift+Enter baris baru",
                           highlight_cursor_line=False)
            yield Static("", id="status", markup=False)

    def on_mount(self):
        self.register_theme(DARK_MODERN)
        self.theme = DARK_MODERN.name
        if self.restored:
            for role, text in self.chat.transcript:
                if role == "User":
                    bubble = UserBubble(text, markup=False)
                    bubble.border_title = "kamu"
                    self._mount(bubble)
                else:
                    self._mount(Reply("sebelumnya", "", text, done=True))
            self.add_info("Kode diperbarui dan dimuat ulang; percakapan tetap berlanjut.")
        else:
            self.add_info("Edit dan perintah dijalankan otomatis (mode auto). "
                          "Ctrl+O ganti model · Esc hentikan jawaban · /help perintah")
        self.refresh_status()
        self.query_one(Composer).focus()
        self.warm_catalog()
        self.chat.prewarm()
        self.set_interval(2, self.check_code)

    def on_unmount(self):
        self.chat.close()

    def check_code(self):
        if self.busy or isinstance(self.screen, ModalScreen):
            return
        stamp = _stamp()
        if stamp == self.stamp:
            return
        self.stamp = stamp
        problem = _broken()
        if problem:
            self.notify(f"Kode berubah tapi belum valid, belum dimuat ulang: {problem}", severity="warning")
            return
        self.exit("reload")

    @work(thread=True)
    def warm_catalog(self):
        self.chat.catalog()

    def refresh_status(self):
        c = self.chat
        mode = "dikunci" if (c.pinned_model or c.pinned_provider or c.pinned_tier) else "otomatis"
        self.query_one("#model", Static).update(f"◆ {c.label()} ▾")
        parts = [f"{c.pinned_provider or c.provider or '-'} · {c.pinned_tier or c.tier or '-'} · {mode}",
                 "Ctrl+O model · Esc batal · Ctrl+N baru · Ctrl+Q keluar"]
        if self.queue:
            parts.insert(1, f"antre {len(self.queue)}")
        self.query_one("#status", Static).update("   │   ".join(parts))

    def _mount(self, widget):
        log = self.query_one("#log", VerticalScroll)
        follow = log.max_scroll_y - log.scroll_y <= 3
        log.mount(widget)
        if follow:
            self.call_after_refresh(log.scroll_end, animate=False)

    def _follow(self):
        log = self.query_one("#log", VerticalScroll)
        if log.max_scroll_y - log.scroll_y <= 3:
            self.call_after_refresh(log.scroll_end, animate=False)

    def add_info(self, text: str):
        self._mount(Info(text, markup=False))

    def start_reply(self, model: str, why: str):
        self.reply = Reply(model, why)
        self._mount(self.reply)

    def reply_show(self, kind: str, text: str):
        if not self.reply:
            return None
        self._follow()
        if kind == "note":
            self.reply.add_tool(text)
            return None
        if kind == "break":
            text = "\n\n"
        return self.reply.query_one(Markdown).append(text)

    def end_reply(self, mark: str, stats: str, failed: bool):
        if self.reply:
            self.reply.finish(mark, stats)
            self.reply.set_class(failed, "failed")
        self._follow()

    @on(Composer.Submitted)
    def submitted(self, event: Composer.Submitted):
        text = event.text.strip()
        if not text:
            return
        self.query_one(Composer).clear()
        if self.busy:
            bubble = None if text.startswith("/") else self._bubble(text, "kamu · antre")
            self.queue.append((text, bubble))
            self.refresh_status()
            return
        self.dispatch_input(text)

    def _bubble(self, text: str, title: str) -> UserBubble:
        bubble = UserBubble(text, markup=False)
        bubble.border_title = title
        self._mount(bubble)
        return bubble

    def dispatch_input(self, text: str, bubble: UserBubble | None = None) -> bool:
        if text.startswith("/"):
            self.run_command(text)
            return False
        if bubble:
            bubble.border_title = "kamu"
        else:
            self._bubble(text, "kamu")
        self.busy = True
        self.send(text)
        return True

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
            self.exit("reload")
        else:
            self.chat.command(text)
            self.refresh_status()
            self.chat.prewarm()

    @work(thread=True, exclusive=True)
    def send(self, text: str):
        try:
            self.chat.send(text)
        except Exception as exc:
            self.call_from_thread(self.add_info, f"error: {exc}")
        finally:
            self.call_from_thread(self.after_send)

    def after_send(self):
        self.busy = False
        self.reply = None
        while self.queue:
            text, bubble = self.queue.pop(0)
            if self.dispatch_input(text, bubble):
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
        c = self.chat
        auto = not (c.pinned_model or c.pinned_provider)
        self.push_screen(ModelPicker(c.catalog(), c.label(), auto), self.picked)

    def picked(self, choice):
        if choice == "auto":
            self.chat.command("/model auto")
        elif choice:
            self.chat.pick(choice)
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
            importlib.reload(sys.modules[f"{__package__}.{name}"])
        cfg = sys.modules[f"{__package__}.config"].load()
