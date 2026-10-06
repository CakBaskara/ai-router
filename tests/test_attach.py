import asyncio
import json
from pathlib import Path

from PIL import Image

from airouter import attach, config, dispatch
from airouter.chat import Chat, image_blocks, with_files
from airouter.tui import ChatApp, Composer, UserBubble

CFG = {**{k: v for k, v in config.load().items() if k not in ("chat", "routing", "loop")}, "providers": ["claude", "codex"]}


def png(path, size=(4, 4)):
    Image.new("RGB", size, "white").save(path)
    return path


def test_store_copies_into_attachment_folder(tmp_path):
    src = png(tmp_path / "foto saya.png")
    dest = attach.store(src)
    assert dest.parent == attach.folder() and dest.name.endswith("foto_saya.png") and dest.exists()


def test_store_converts_bmp_to_png(tmp_path):
    src = tmp_path / "x.bmp"
    Image.new("RGB", (2, 2)).save(src)
    assert attach.store(src).suffix == ".png"


def test_store_rejects_huge_file(tmp_path, monkeypatch):
    src = tmp_path / "big.bin"
    src.write_bytes(b"0" * 10)
    monkeypatch.setattr(attach, "MAX_BYTES", 5)
    try:
        attach.store(src)
    except ValueError as exc:
        assert "20 MB" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_claude_gets_image_block(tmp_path):
    blocks = image_blocks([png(tmp_path / "a.png"), tmp_path / "note.txt"])
    assert len(blocks) == 1 and blocks[0]["source"]["media_type"] == "image/png"


def test_gemini_gets_at_reference_with_forward_slashes(tmp_path):
    f = png(tmp_path / "a.png")
    assert with_files("gemini", "lihat", [f]).endswith("@" + str(f).replace("\\", "/"))


def test_other_providers_get_file_list(tmp_path):
    f = png(tmp_path / "a.png")
    assert str(f) in with_files("codex", "lihat", [f])


def test_cmd_flags_per_provider(tmp_path):
    img, txt = png(tmp_path / "a.png"), tmp_path / "b.txt"
    txt.write_text("x")
    codex = dispatch.chat_cmd("codex", "m", "low", None, [img, txt], "D:/att")
    assert codex[codex.index("-i") + 1] == str(img) and codex.count("-i") == 1
    copilot = dispatch.chat_cmd("copilot", "auto", "low", None, [img, txt], "D:/att")
    assert copilot.count("--attachment") == 1 and "--add-dir" in copilot
    gemini = dispatch.chat_cmd("gemini", "g", "low", None, [img], "D:/att")
    assert str(Path("D:/att").resolve()) in gemini
    claude = dispatch.chat_cmd("claude", "opus", "high", None, attach_dir="D:/att")
    assert str(Path("D:/att").resolve()) in claude


def test_every_provider_gets_the_same_roots(monkeypatch):
    monkeypatch.setattr(dispatch.config, "load", lambda: {"chat": {"dirs": ["D:/extra"]}})
    flags = {"claude": "--add-dir", "copilot": "--add-dir", "gemini": "--include-directories"}
    want = dispatch.chat_roots("D:/att")
    assert str(Path("D:/extra").resolve()) in want
    for provider, flag in flags.items():
        cmd = dispatch.chat_cmd(provider, "m", "low", None, attach_dir="D:/att")
        assert [cmd[i + 1] for i, a in enumerate(cmd) if a == flag] == want
    codex = dispatch.chat_cmd("codex", "m", "low", None, attach_dir="D:/att")
    assert json.dumps(want) in " ".join(codex)


def test_send_passes_attachments_and_notes_them_in_recap(tmp_path):
    f = png(tmp_path / "a.png")
    c = Chat(CFG, use_llm=False)
    seen = []

    def runner(cmd, prompt, turn, show, started):
        seen.append((cmd, prompt))
        turn.feed({"type": "thread.started", "thread_id": "t"})
        turn.feed({"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}})
        return 0

    c.command("/codex")
    c.send("lihat", runner=runner, attachments=[f])
    cmd, prompt = seen[0]
    assert str(f) in cmd and str(f) in prompt
    assert c.transcript[0][1].endswith("[lampiran: a.png]")


def make_app():
    app = ChatApp(CFG, None, None, False)
    app.chat.prewarm = lambda: None
    app.chat._catalog = [("claude", "opus")]
    return app


def test_pasting_a_path_types_it_as_text(tmp_path):
    from textual import events
    f = png(tmp_path / "foto.png")

    async def go():
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app.post_message(events.Paste(f'"{f}"'))
            await pilot.pause()
            assert app.query_one(Composer).text == f'"{f}"' and app.attachments == []
    asyncio.run(go())


def test_backspace_on_empty_input_drops_attachment(tmp_path):
    async def go():
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app.attach_files([png(tmp_path / "foto.png")])
            await pilot.pause()
            assert "foto.png" in str(app.query_one("#files").render())
            await pilot.press("backspace")
            await pilot.pause()
            assert app.attachments == []
    asyncio.run(go())


def test_sending_with_attachment_shows_it_and_clears(tmp_path):
    f = png(tmp_path / "foto.png")

    async def go():
        app = make_app()
        sent = []
        app.chat.send = lambda text, attachments=(): sent.append((text, list(attachments)))
        async with app.run_test(size=(100, 30)) as pilot:
            app.attach_files([f])
            app.query_one(Composer).insert("apa ini?")
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert sent and sent[0][0] == "apa ini?" and sent[0][1][0].name.endswith("foto.png")
            assert "📎 foto.png" in str(app.query(UserBubble)[-1].render())
            assert app.attachments == []
    asyncio.run(go())


def test_clipboard_image_is_attached(monkeypatch, tmp_path):
    async def go():
        app = make_app()
        monkeypatch.setattr(attach, "from_clipboard", lambda: [png(tmp_path / "clip.png")])
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("alt+v")
            await pilot.pause()
            assert len(app.attachments) == 1
    asyncio.run(go())


def test_clip_button_attaches_picked_files(monkeypatch, tmp_path):
    async def go():
        app = make_app()
        monkeypatch.setattr(attach, "pick_files", lambda: [png(tmp_path / "a.png"), png(tmp_path / "b.png")])
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#clip")
            for _ in range(40):
                await pilot.pause(0.05)
                if app.attachments:
                    break
            assert [f.name.split("-", 2)[-1] for f in app.attachments] == ["a.png", "b.png"]
            assert app.query_one("#clip").region.right <= 100
    asyncio.run(go())
