import asyncio

from textual.widgets import Markdown

from airouter import config
from airouter.chat import Chat
from airouter.tui import ChatApp, Composer, ModelPicker, Reply, UserBubble

CFG = {**{k: v for k, v in config.load().items() if k not in ("chat", "routing")}, "providers": ["claude", "codex"]}

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
