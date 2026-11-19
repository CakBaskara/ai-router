import asyncio
import json
from pathlib import Path

from PIL import Image

from airouter import attach, config, dispatch, upload
from airouter.chat import Chat, image_blocks, with_files
from airouter.security import Mode, SecurityPolicy
from airouter.tui import ChatApp, Composer, FilePicker, UploadPopup, UserBubble

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


def test_other_providers_get_file_list(tmp_path):
    f = png(tmp_path / "a.png")
    assert str(f) in with_files("codex", "lihat", [f])


def test_cmd_flags_per_provider(tmp_path):
    img, txt = png(tmp_path / "a.png"), tmp_path / "b.txt"
    txt.write_text("x")
    codex = dispatch.chat_cmd("codex", "m", "low", None, [img, txt], "D:/att")
    assert codex[codex.index("-i") + 1] == str(img) and codex.count("-i") == 1
    claude = dispatch.chat_cmd("claude", "opus", "high", None, attach_dir="D:/att")
    assert str(Path("D:/att").resolve()) not in claude


def test_every_provider_gets_only_explicit_write_roots(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    policy = SecurityPolicy(Mode.APPROVAL_REQUIRED, tmp_path, extra_write_roots=(extra,))
    want = dispatch.chat_roots(policy)
    assert want == [str(extra.resolve())]
    claude = dispatch.chat_cmd("claude", "m", "low", None, policy=policy)
    assert [claude[i + 1] for i, a in enumerate(claude) if a == "--add-dir"] == want
    codex = dispatch.chat_cmd("codex", "m", "low", None, policy=policy)
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
        for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(attach, "pick_files", lambda: [png(tmp_path / "a.png"), png(tmp_path / "b.png")])
        async with app.run_test(size=(100, 30)) as pilot:
            clip = app.query_one("#clip")
            assert str(clip.render()) == "📎" and clip.region.width == 6
            await pilot.click("#clip")
            for _ in range(40):
                await pilot.pause(0.05)
                if app.attachments:
                    break
            assert [f.name.split("-", 2)[-1] for f in app.attachments] == ["a.png", "b.png"]
            assert clip.region.right <= 100
    asyncio.run(go())


def test_clip_button_over_ssh_opens_in_terminal_picker_and_attaches(monkeypatch, tmp_path):
    async def go():
        app = make_app()
        monkeypatch.setenv("SSH_CONNECTION", "127.0.0.1 5000 127.0.0.1 22")
        monkeypatch.chdir(tmp_path)
        png(tmp_path / "a.png")
        monkeypatch.setattr(attach, "pick_files", lambda: (_ for _ in ()).throw(AssertionError("native dialog opened")))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#clip")
            await pilot.pause(0.3)
            assert isinstance(app.screen, UploadPopup)
            await pilot.click("#upload-pc")
            await pilot.pause(0.3)
            assert isinstance(app.screen, FilePicker)
            await pilot.press("down", "enter")
            for _ in range(40):
                await pilot.pause(0.05)
                if app.attachments:
                    break
            assert [f.name.split("-", 2)[-1] for f in app.attachments] == ["a.png"]
            assert not isinstance(app.screen, FilePicker)
    asyncio.run(go())


def test_in_terminal_picker_close_button_closes_without_attaching(monkeypatch, tmp_path):
    async def go():
        app = make_app()
        monkeypatch.setenv("SSH_CONNECTION", "127.0.0.1 5000 127.0.0.1 22")
        monkeypatch.chdir(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#clip")
            await pilot.pause(0.2)
            await pilot.click("#upload-pc")
            await pilot.pause(0.2)
            await pilot.click("#file-close")
            await pilot.pause(0.2)
            assert not isinstance(app.screen, FilePicker) and app.attachments == []
    asyncio.run(go())


def test_failed_native_dialog_falls_back_to_in_terminal_picker(monkeypatch, tmp_path):
    async def go():
        app = make_app()
        for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(attach, "pick_files", lambda: (_ for _ in ()).throw(RuntimeError("no display")))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#clip")
            for _ in range(40):
                await pilot.pause(0.05)
                if isinstance(app.screen, FilePicker):
                    break
            assert isinstance(app.screen, FilePicker)
    asyncio.run(go())


def test_upload_page_needs_the_token_and_stores_the_file(tmp_path):
    import urllib.error
    import urllib.request
    got = []
    server = upload.Server(got.append, host="127.0.0.1")
    try:
        page = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert "Pilih file" in page
        try:
            urllib.request.urlopen(server.url.rsplit("/", 1)[0] + "/salah", timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("token salah diterima")
        request = urllib.request.Request(server.url + "/file", data=b"isi", method="POST",
                                         headers={"X-Name": "catatan%20hp.txt"})
        assert urllib.request.urlopen(request, timeout=5).read() == b"ok"
        assert len(got) == 1 and got[0].read_bytes() == b"isi" and got[0].name.endswith("catatan_hp.txt")
        assert got[0].parent == attach.folder()
    finally:
        server.close()


def test_upload_rejects_file_over_the_limit(monkeypatch):
    import urllib.error
    import urllib.request
    monkeypatch.setattr(attach, "MAX_BYTES", 2)
    server = upload.Server(lambda path: None, host="127.0.0.1")
    try:
        request = urllib.request.Request(server.url + "/file", data=b"besar", method="POST")
        try:
            urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 413
        else:
            raise AssertionError("file besar diterima")
    finally:
        server.close()


def test_clip_over_ssh_shows_upload_link_and_attaches_phone_file(monkeypatch, tmp_path):
    import urllib.request

    async def go():
        app = make_app()
        monkeypatch.setenv("SSH_CONNECTION", "127.0.0.1 5000 127.0.0.1 22")
        app.chat.cfg = {**app.chat.cfg, "chat": {**app.chat.cfg.get("chat", {}), "upload_port": 0}}
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.click("#clip")
            await pilot.pause(0.3)
            assert isinstance(app.screen, UploadPopup)
            assert app.screen.url.startswith("http://127.0.0.1:")
            request = urllib.request.Request(app.screen.url + "/file", data=b"foto", method="POST",
                                             headers={"X-Name": "hp.jpg"})
            await asyncio.get_running_loop().run_in_executor(None, lambda: urllib.request.urlopen(request, timeout=5))
            for _ in range(40):
                await pilot.pause(0.05)
                if app.attachments:
                    break
            assert [f.name.split("-", 2)[-1] for f in app.attachments] == ["hp.jpg"]
            assert app.screen.count == 1
            sent = []
            monkeypatch.setattr(app, "dispatch_input", lambda text, bubble, files: sent.append((text, files)))
            done = urllib.request.Request(app.screen.url + "/done", data="ringkas foto ini".encode(), method="POST")
            await asyncio.get_running_loop().run_in_executor(None, lambda: urllib.request.urlopen(done, timeout=5))
            for _ in range(40):
                await pilot.pause(0.05)
                if sent:
                    break
            assert not isinstance(app.screen, UploadPopup)
            assert sent[0][0] == "ringkas foto ini" and [f.name.split("-", 2)[-1] for f in sent[0][1]] == ["hp.jpg"]
            assert app.attachments == []
    asyncio.run(go())
