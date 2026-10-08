import asyncio

from textual.app import App
from textual.widgets import Markdown, Static

from airouter import config
from airouter.chat import Chat
from textual.containers import VerticalScroll

from airouter.tui import ChatApp, Composer, ModelPicker, Reply, ScrollButton, UserBubble

CFG = {**{k: v for k, v in config.load().items() if k not in ("chat", "routing", "loop")}, "providers": ["claude", "codex"]}

EVENTS = [
    {"type": "system", "subtype": "init", "session_id": "s1"},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "iso.c"}}]}},
    {"type": "stream_event", "event": {"type": "content_block_delta",
                                       "delta": {"type": "text_delta", "text": "Penyebabnya **wrap-around**.\n\n"}}},
    {"type": "stream_event", "event": {"type": "content_block_delta",
                                       "delta": {"type": "text_delta", "text": "| Sisi | Lebar |\n|---|---|\n| tx | 16 |\n"}}},
    {"type": "result", "is_error": False, "usage": {"input_tokens": 5, "cache_read_input_tokens": 29000,
                                                    "output_tokens": 240}},
]


def fake_runner(cmd, prompt, turn, show, started):
    for ev in EVENTS:
        shown = turn.feed(ev)
        if shown:
            show(*shown)
    return 0


def make_app():
    app = ChatApp(CFG, None, None, False)
    app.chat.prewarm = lambda: None
    app.chat._catalog = [("claude", "haiku"), ("claude", "sonnet"), ("codex", "gpt-6.1-sol")]
    app.chat.send = lambda text, attachments=(): Chat.send(app.chat, text, runner=fake_runner)
    return app


async def _type_and_send(pilot, text):
    composer = pilot.app.query_one(Composer)
    composer.insert(text)
    await pilot.press("enter")
    for _ in range(50):
        await pilot.pause(0.05)
        if not pilot.app.busy:
            break
    await pilot.pause(0.2)


def test_message_renders_bubble_reply_and_stats(tmp_path):
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            await _type_and_send(pilot, "kenapa frame ISO hilang?")
            assert len(app.query(UserBubble)) == 1
            reply = app.query_one(Reply)
            assert "wrap-around" in reply.query_one(Markdown).source
            assert "Read iso.c" in reply.tools[0]
            assert "out 240" in str(reply.query_one(".stats").render())
            assert app.query_one(Composer).text == ""
            app.save_screenshot(str(tmp_path / "chat.svg"))
    asyncio.run(go())


def test_slash_model_command_does_not_send():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            await _type_and_send(pilot, "/model sol")
            assert app.chat.pinned_model == "gpt-6.1-sol"
            assert not app.query(UserBubble)
    asyncio.run(go())


def test_ctrl_o_opens_picker_and_selects():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert isinstance(app.screen, ModelPicker)
            app.screen.dismiss("2")
            await pilot.pause()
            assert (app.chat.pinned_provider, app.chat.pinned_model) == ("claude", "sonnet")
    asyncio.run(go())


def test_restored_state_redraws_conversation():
    async def go():
        first = make_app()
        first.chat.transcript = [("User", "halo"), ("Assistant", "hai **juga**")]
        first.chat.sessions = {"claude": "s1"}
        app = ChatApp(CFG, None, None, False, state=first.chat.snapshot())
        app.chat.prewarm = lambda: None
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            assert app.chat.sessions == {"claude": "s1"}
            assert len(app.query(UserBubble)) == 1
            assert app.query_one(Reply).query_one(Markdown).source == "hai **juga**"
    asyncio.run(go())


def test_changed_code_exits_for_reload():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            app.stamp = ()
            app.check_code()
            await pilot.pause()
        assert app.return_value == "reload"
    asyncio.run(go())


def test_ctrl_j_adds_line_instead_of_sending():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            app.query_one(Composer).insert("baris satu")
            await pilot.press("ctrl+j")
            assert app.query_one(Composer).text == "baris satu\n"
            assert not app.query(UserBubble)
    asyncio.run(go())


def test_ctrl_a_selects_input():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            app.query_one(Composer).insert("halo dunia")
            await pilot.press("ctrl+a")
            assert app.query_one(Composer).selected_text == "halo dunia"
    asyncio.run(go())


def test_dark_modern_theme_is_active():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)):
            assert app.theme == "vscode-dark-modern"
    asyncio.run(go())


def test_prompt_sent_while_busy_is_queued_then_sent():
    async def go():
        app = make_app()
        sent = []
        release = asyncio.Event()
        loop = asyncio.get_running_loop()

        def slow(text, attachments=()):
            sent.append(text)
            if len(sent) == 1:
                asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
            return Chat.send(app.chat, text, runner=fake_runner)

        app.chat.send = slow
        async with app.run_test(size=(100, 32)) as pilot:
            app.query_one(Composer).insert("pertama")
            await pilot.press("enter")
            await pilot.pause(0.2)
            app.query_one(Composer).insert("kedua")
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert app.queue and app.query(UserBubble)[-1].has_class("queued")
            release.set()
            for _ in range(60):
                await pilot.pause(0.05)
                if len(sent) == 2 and not app.busy:
                    break
            assert sent == ["pertama", "kedua"] and not app.queue
            assert not app.query(UserBubble)[-1].has_class("queued")
    asyncio.run(go())


def test_picker_shows_provider_folders_in_order():
    from textual.widgets import Tree

    async def go():
        app = make_app()
        app.chat._catalog = [("claude", "opus"), ("codex", "gpt-6.1-sol"), ("gemini", "gemini-3.8-flash"),
                             ("copilot", "auto")]
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()
            tree = app.screen.query_one(Tree)
            names = [str(n.label).split()[0] for n in tree.root.children[1:]]
            assert names == ["Gemini", "Copilot", "Codex", "Claude"]
            assert not any(n.is_expanded for n in tree.root.children[1:])
    asyncio.run(go())


def test_picker_opens_folder_then_picks_model_with_keys():
    from textual.widgets import Tree

    async def go():
        app = make_app()
        app.chat._catalog = [("gemini", "gemini-3.8-flash"), ("codex", "gpt-6.1-sol"), ("codex", "gpt-5.5")]
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()
            tree = app.screen.query_one(Tree)
            await pilot.press("down", "down")
            assert "Codex" in str(tree.cursor_node.label)
            await pilot.press("enter")
            await pilot.pause()
            assert tree.cursor_node.is_expanded
            await pilot.press("down", "down", "down", "enter")
            await pilot.pause()
            assert (app.chat.pinned_provider, app.chat.pinned_model) == ("codex", "gpt-5.5")
    asyncio.run(go())


def test_picker_provider_only_choice_pins_provider():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()
            app.screen.dismiss("prov:codex")
            await pilot.pause()
            assert (app.chat.pinned_provider, app.chat.pinned_model) == ("codex", None)
    asyncio.run(go())


def test_picker_scrolls_to_last_model_on_short_terminal():
    from textual.widgets import Tree

    async def go():
        app = make_app()
        app.chat._catalog = [("claude", f"model-{i}") for i in range(40)]
        async with app.run_test(size=(100, 20)) as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()
            tree = app.screen.query_one(Tree)
            await pilot.press("down", "enter")
            await pilot.pause()
            assert tree.size.height < 42
            await pilot.press("end")
            await pilot.pause(0.3)
            assert tree.cursor_node.data == "40" and tree.scroll_y > 0
            await pilot.press("enter")
            await pilot.pause()
            assert app.chat.pinned_model == "model-39"
    asyncio.run(go())


def test_sentence_switch_pins_model_without_sending():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            await _type_and_send(pilot, "pakai sonnet")
            assert (app.chat.pinned_provider, app.chat.pinned_model) == ("claude", "sonnet")
            assert not app.query(UserBubble)
            assert not app.query(Reply)
    asyncio.run(go())


def test_composer_cursor_is_steady():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            assert app.query_one(Composer).cursor_blink is False
    asyncio.run(go())


def test_scroll_buttons_move_the_log():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 20)) as pilot:
            for i in range(40):
                app.add_info(f"baris {i}")
            await pilot.pause(0.2)
            log = app.query_one("#log", VerticalScroll)
            log.scroll_end(animate=False)
            await pilot.pause(0.1)
            bottom = log.scroll_y
            assert bottom > 0
            await pilot.click("#up")
            await pilot.pause(0.1)
            assert log.scroll_y == bottom - ScrollButton.STEP
            await pilot.click("#down")
            await pilot.pause(0.1)
            assert log.scroll_y == bottom
    asyncio.run(go())


def test_long_reply_keeps_running_and_finished_status_visible():
    async def go():
        app = make_app()
        async with app.run_test(size=(80, 20)) as pilot:
            app.start_reply("gpt-6-sol", "codex · medium · medium")
            await pilot.pause()
            app.reply_show("text", "\n\n".join(f"baris {i}" for i in range(40)))
            await pilot.pause(0.2)
            log = app.query_one("#log", VerticalScroll)
            log.scroll_end(animate=False)
            for _ in range(20):
                await pilot.pause(0.05)
                if app.query_one("#sticky").visible:
                    break
            sticky = app.query_one("#sticky")
            assert sticky.visible
            assert "gpt-6-sol" in str(app.query_one("#sticky-who").render())
            assert "codex · medium · medium" in str(app.query_one("#sticky-why").render())
            app.end_reply("●", "✓", False)
            await pilot.pause(0.2)
            assert str(app.query_one("#sticky-who").render()).startswith("●")
            log.scroll_home(animate=False)
            await pilot.pause(0.2)
            assert not sticky.visible

    asyncio.run(go())


def test_idle_sticky_header_is_not_redrawn():
    async def go():
        app = make_app()
        async with app.run_test(size=(80, 20)) as pilot:
            app.start_reply("gpt-6-sol", "codex · medium · medium")
            await pilot.pause()
            app.reply_show("text", "\n\n".join(f"baris {i}" for i in range(40)))
            app.end_reply("●", "✓", False)
            log = app.query_one("#log", VerticalScroll)
            log.scroll_end(animate=False)
            await pilot.pause(0.3)
            assert app.query_one("#sticky").visible
            who = app.query_one("#sticky-who", Static)
            calls = []
            original = who.update
            who.update = lambda *args, **kwargs: (calls.append(args), original(*args, **kwargs))
            await pilot.pause(0.5)
            assert calls == []

    asyncio.run(go())


def test_reload_keeps_unsent_draft():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            app.query_one(Composer).insert("prompt yang belum dikirim")
            app.stamp = ()
            app.check_code()
            await pilot.pause()
        assert app.return_value == "reload"
        state = app.chat.snapshot()
        again = ChatApp(CFG, None, None, False, state)
        again.chat.prewarm = lambda: None
        again.chat._catalog = app.chat._catalog
        async with again.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            assert again.query_one(Composer).text == "prompt yang belum dikirim"
    asyncio.run(go())


def test_prompt_sent_while_claude_answers_is_injected_not_queued():
    async def go():
        app = make_app()
        release = asyncio.Event()
        loop = asyncio.get_running_loop()
        injected = []

        def slow(text, attachments=()):
            app.call_from_thread(app.start_reply, "opus", "claude · heavy · high")
            asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
            app.call_from_thread(app.end_reply, "●", "✓", False)
            return 0

        app.chat.send = slow
        app.chat.inject = lambda text, files=(): injected.append(text) or True
        async with app.run_test(size=(100, 32)) as pilot:
            app.query_one(Composer).insert("pertama")
            await pilot.press("enter")
            await pilot.pause(0.2)
            app.query_one(Composer).insert("kedua")
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert injected == ["kedua"] and not app.queue
            assert not app.query(UserBubble)[-1].has_class("queued")
            assert len(app.query(Reply)) == 2
            release.set()
            for _ in range(40):
                await pilot.pause(0.05)
                if not app.busy:
                    break
            assert not app.busy
    asyncio.run(go())


def test_prompt_queued_during_startup_joins_active_reply():
    async def go():
        app = make_app()
        release = asyncio.Event()
        loop = asyncio.get_running_loop()
        ready = False
        injected = []

        def slow(text, attachments=()):
            app.call_from_thread(app.start_reply, "opus", "claude · heavy · high")
            asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
            app.call_from_thread(app.end_reply, "●", "✓", False)
            return 0

        app.chat.send = slow

        def inject(text, files=()):
            if not ready:
                return False
            injected.append(text)
            return True

        app.chat.inject = inject
        async with app.run_test(size=(100, 32)) as pilot:
            await _type_and_send(pilot, "pertama")
            await _type_and_send(pilot, "kedua")
            assert len(app.queue) == 1
            assert app.query(UserBubble)[-1].has_class("queued")
            ready = True
            for _ in range(20):
                await pilot.pause(0.05)
                if injected:
                    break
            assert injected == ["kedua"]
            assert not app.queue
            assert not app.query(UserBubble)[-1].has_class("queued")
            release.set()
            for _ in range(40):
                await pilot.pause(0.05)
                if not app.busy:
                    break
            assert not app.busy

    asyncio.run(go())


def test_changed_code_reloads_between_back_to_back_prompts_and_keeps_the_next_one():
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 32)) as pilot:
            app.stamp = ()
            app.query_one(Composer).insert("prompt berikutnya")
            await pilot.press("enter")
            await pilot.pause()
        assert app.return_value == "reload"
        state = app.chat.snapshot()
        assert state["carry"]["queue"] == [("prompt berikutnya", [])]
        again = ChatApp(CFG, None, None, False, state)
        again.chat.prewarm = lambda: None
        again.chat._catalog = app.chat._catalog
        again.chat.send = lambda text, attachments=(): Chat.send(again.chat, text, runner=fake_runner)
        async with again.run_test(size=(100, 32)) as pilot:
            for _ in range(50):
                await pilot.pause(0.05)
                if again.query(Reply) and not again.busy:
                    break
            assert again.chat.transcript[-2] == ("User", "prompt berikutnya")
            assert not again.queue
    asyncio.run(go())


def test_stop_button_shows_while_busy_and_cancels():
    async def go():
        app = make_app()
        cancelled = []
        app.chat.cancel = lambda: cancelled.append(True)
        async with app.run_test(size=(100, 20)) as pilot:
            await pilot.pause()
            stop = app.query_one("#stop")
            assert stop.display and not stop.has_class("busy")
            app.busy = True
            await pilot.pause()
            assert stop.has_class("busy")
            await pilot.click("#stop")
            assert cancelled == [True]
            app.busy = False
            await pilot.pause()
            assert not stop.has_class("busy")
    asyncio.run(go())


def test_streaming_reply_follows_until_user_scrolls_up():
    async def go():
        app = make_app()
        async with app.run_test(size=(80, 20)) as pilot:
            app.start_reply("sonnet", "claude · medium")
            await pilot.pause()
            log = app.query_one("#log", VerticalScroll)
            for i in range(40):
                await app.reply_show("text", f"baris {i}\n\n")
                await pilot.pause(0.02)
            await pilot.pause(0.2)
            assert log.max_scroll_y > 0 and log.scroll_y == log.max_scroll_y
            await pilot.click("#up")
            await pilot.pause(0.1)
            held = log.scroll_y
            await app.reply_show("text", "baris baru\n\n")
            await pilot.pause(0.2)
            assert log.scroll_y == held < log.max_scroll_y
            log.scroll_end(animate=False)
            await pilot.pause(0.1)
            for i in range(5):
                await app.reply_show("text", f"lagi {i}\n\n")
            await pilot.pause(0.2)
            assert log.scroll_y == log.max_scroll_y
    asyncio.run(go())



def test_bottom_of_log_holds_still_while_header_sticks():
    async def go():
        app = make_app()
        async with app.run_test(size=(80, 20)) as pilot:
            app.start_reply("sonnet", "claude · medium")
            await pilot.pause()
            await app.reply_show("text", "\n\n".join(f"b {i}" for i in range(30)))
            app.end_reply("●", "", False)
            for i in range(13):
                app.add_info(f"info {i}")
            await pilot.pause(0.4)
            log = app.query_one("#log", VerticalScroll)
            seen = set()
            for _ in range(12):
                await pilot.pause(0.1)
                seen.add((log.scroll_y, log.size.height))
            assert len(seen) == 1 and log.scroll_y == log.max_scroll_y
    asyncio.run(go())


class _Bare(App):
    def compose(self):
        return []


def test_reply_finished_before_mount_shows_end_without_spinner():
    async def go():
        app = _Bare()
        async with app.run_test() as pilot:
            reply = Reply("haiku", "claude")
            app.mount(reply)
            assert not reply.ready
            reply.finish("✗", "gagal")
            await pilot.pause()
            assert reply.timer is None
            assert str(reply.query_one(".who").render()) == "✗ haiku"
            assert str(reply.query_one(".stats").render()) == "gagal"
    asyncio.run(go())


def test_reply_streamed_before_mount_keeps_text_and_tools():
    async def go():
        app = _Bare()
        async with app.run_test() as pilot:
            reply = Reply("sonnet", "claude")
            app.mount(reply)
            assert not reply.ready
            reply.add_tool("Read iso.c")
            reply.write("jawaban ")
            reply.write("akhir")
            reply.finish("●", "✓ 1s")
            await pilot.pause()
            assert reply.query_one(Markdown).source == "jawaban akhir"
            assert "Read iso.c" in str(reply.query_one(".tools").render())
            assert str(reply.query_one(".stats").render()) == "✓ 1s"
    asyncio.run(go())
