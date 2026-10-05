import asyncio
import json
import queue
import threading
import time

import pytest
from textual.widgets import Static

from airouter import config, dispatch
from airouter.chat import Chat, CodexTurn
from airouter.codex import CodexProc
from airouter.tui import ChatApp, Composer, Reply, UserBubble


class FakeServer:
    def __init__(self):
        self.output = queue.Queue()
        self.stderr_queue = queue.Queue()
        self.stdin = self
        self.stdout = iter(self.output.get, None)
        self.stderr = iter(self.stderr_queue.get, None)
        self.written = []
        self.exit_code = None
        self.start_count = 0
        self.hold_start = False
        self.hold_steer = False
        self.race_steer = False

    def write(self, raw):
        request = json.loads(raw)
        self.written.append(request)
        method = request.get("method")
        if method == "initialize":
            self.emit({"id": request["id"], "result": {}})
        elif method in ("thread/start", "thread/resume"):
            self.emit({"id": request["id"], "result": {"thread": {"id": "thread-1"}}})
        elif method == "turn/start":
            self.start_count += 1
            self.start_request = request
            if not self.hold_start:
                self.accept_start()
        elif method == "turn/steer" and not self.hold_steer:
            if self.race_steer:
                self.finish()
                self.emit({"id": request["id"], "error": {"message": "No active turn"}})
                self.race_steer = False
            else:
                self.emit({"id": request["id"], "result": {"turnId": "turn-1"}})
        elif method == "turn/interrupt":
            self.emit({"id": request["id"], "result": {}})
            self.finish("interrupted")

    def flush(self):
        pass

    def poll(self):
        return self.exit_code

    def kill(self):
        self.exit_code = 1
        self.output.put(None)
        self.stderr_queue.put(None)

    def emit(self, message):
        self.output.put((json.dumps(message) + "\n").encode())

    def accept_start(self):
        self.emit({"id": self.start_request["id"], "result": {
            "turn": {"id": f"turn-{self.start_count}", "status": "inProgress"}}})

    def notify(self, method, **params):
        self.emit({"method": method, "params": {"threadId": "thread-1", **params}})

    def finish(self, status="completed", error=None):
        self.notify("turn/completed", turn={"id": f"turn-{self.start_count}", "status": status,
                                            "error": {"message": error} if error else None})


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr("airouter.codex.subprocess.Popen", lambda *a, **kw: fake)
    yield fake
    if fake.poll() is None:
        fake.kill()


def wait_until(predicate):
    limit = time.monotonic() + 3
    while not predicate() and time.monotonic() < limit:
        time.sleep(0.005)
    assert predicate()


def ask_in_thread(proc):
    turn, result, shown = CodexTurn(), {}, []
    worker = threading.Thread(target=lambda: result.update(code=proc.ask(
        "first", turn, lambda *args: shown.append(args))))
    worker.start()
    return worker, turn, result, shown


def test_live_codex_steers_multiple_messages_without_killing_or_new_turn(server, tmp_path):
    proc = CodexProc("model-a", "low", None)
    worker, turn, result, shown = ask_in_thread(proc)
    wait_until(lambda: proc.turn_id == "turn-1")
    server.notify("item/agentMessage/delta", itemId="answer-1", turnId="turn-1", delta="ALPHA ")
    image = tmp_path / "image.png"
    assert proc.inject("second", [image])
    assert proc.inject("third")
    server.notify("item/agentMessage/delta", itemId="answer-1", turnId="turn-1", delta="BRAVO")
    server.notify("item/completed", turnId="turn-1", item={
        "id": "answer-1", "type": "agentMessage", "text": "ALPHA BRAVO"})
    server.finish()
    worker.join(3)
    assert not worker.is_alive() and result["code"] == 0
    assert turn.text() == "ALPHA BRAVO" and shown == [("text", "ALPHA "), ("text", "BRAVO")]
    steers = [r for r in server.written if r.get("method") == "turn/steer"]
    assert [r["params"]["input"][0]["text"] for r in steers] == ["second", "third"]
    assert steers[0]["params"]["input"][1] == {"type": "localImage", "path": str(image)}
    assert all(r["params"]["expectedTurnId"] == "turn-1" for r in steers)
    assert server.start_count == 1 and server.poll() is None
    assert not proc.inject("too late")


def test_input_during_start_is_delivered_after_turn_id_arrives(server):
    proc = CodexProc("model-a", "low", None)
    server.hold_start = True
    worker, turn, result, _ = ask_in_thread(proc)
    wait_until(lambda: server.start_count == 1)
    assert proc.inject("during startup")
    server.accept_start()
    wait_until(lambda: any(r.get("method") == "turn/steer" for r in server.written))
    server.finish()
    worker.join(3)
    assert result["code"] == 0


def test_completion_waits_for_steering_acknowledgment(server):
    proc = CodexProc("model-a", "low", None)
    worker, _, result, _ = ask_in_thread(proc)
    wait_until(lambda: proc.turn_id == "turn-1")
    server.hold_steer = True
    assert proc.inject("late message")
    server.finish()
    wait_until(lambda: proc.turn_id is None)
    assert worker.is_alive()
    request = next(r for r in server.written if r.get("method") == "turn/steer")
    server.emit({"id": request["id"], "result": {"turnId": "turn-1"}})
    worker.join(3)
    assert result["code"] == 0


def test_steer_rejected_at_completion_starts_followup_without_losing_input(server):
    proc = CodexProc("model-a", "low", None)
    worker, _, result, _ = ask_in_thread(proc)
    wait_until(lambda: proc.turn_id == "turn-1")
    server.race_steer = True
    assert proc.inject("arrived at completion")
    wait_until(lambda: proc.turn_id == "turn-2")
    request = [r for r in server.written if r.get("method") == "turn/start"][-1]
    assert request["params"]["input"] == [{"type": "text", "text": "arrived at completion"}]
    assert request["params"]["threadId"] == "thread-1"
    server.finish()
    worker.join(3)
    assert result["code"] == 0 and server.poll() is None


def test_cancel_uses_turn_interrupt_and_keeps_server_alive(server):
    proc = CodexProc("model-a", "low", None)
    worker, _, result, _ = ask_in_thread(proc)
    wait_until(lambda: proc.turn_id == "turn-1")
    proc.cancel()
    assert not proc.inject("after cancellation")
    worker.join(3)
    assert result["code"] == 130 and server.poll() is None
    assert any(r.get("method") == "turn/interrupt" for r in server.written)


def test_failed_turn_and_disconnection_propagate(server):
    proc = CodexProc("model-a", "low", None)
    worker, turn, result, _ = ask_in_thread(proc)
    wait_until(lambda: proc.turn_id == "turn-1")
    server.finish("failed", "usage limit")
    worker.join(3)
    assert result["code"] == 1 and turn.error == "usage limit"
    worker, turn, result, _ = ask_in_thread(proc)
    wait_until(lambda: proc.turn_id == "turn-2")
    server.kill()
    worker.join(3)
    assert result["code"] == 1 and "disconnected" in turn.error


def test_live_codex_resumes_saved_thread(server):
    proc = CodexProc("model-a", "medium", "saved-thread")
    resume = next(r for r in server.written if r.get("method") == "thread/resume")
    assert resume["params"]["threadId"] == "saved-thread"
    assert resume["params"]["sandbox"] == "workspace-write" and proc.session == "thread-1"


def test_chat_records_injected_text_and_reuses_live_codex(server):
    chat = Chat(config.load(), provider="codex", tier="heavy", use_llm=False)
    worker = threading.Thread(target=lambda: chat.send("first"))
    worker.start()
    wait_until(lambda: chat._codex and chat._codex.turn_id == "turn-1")
    assert chat.inject("second")
    server.notify("item/completed", turnId="turn-1", item={
        "id": "answer-1", "type": "agentMessage", "text": "both handled"})
    server.finish()
    worker.join(3)
    assert chat.transcript[-2:] == [("User", "first\n\nsecond"), ("Assistant", "both handled")]
    proc = chat._codex
    worker = threading.Thread(target=lambda: chat.send("third"))
    worker.start()
    wait_until(lambda: proc.turn_id == "turn-2")
    assert chat._codex is proc
    server.notify("item/completed", turnId="turn-2", item={
        "id": "answer-2", "type": "agentMessage", "text": "third handled"})
    server.finish()
    worker.join(3)
    assert chat.transcript[-1] == ("Assistant", "third handled")


def test_codex_ui_injects_without_queue_and_updates_model_label(server):
    async def go():
        app = ChatApp(config.load(), "codex", "heavy", False)
        app.chat.prewarm = lambda: None
        app.chat._catalog = [("codex", "gpt-6.1-sol")]
        app.chat.model = "opus"
        async with app.run_test(size=(100, 32)) as pilot:
            app.query_one(Composer).insert("first")
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause(0.02)
                if app.chat._codex and app.chat._codex.turn_id:
                    break
            assert app.chat._codex.turn_id == "turn-1"
            assert "gpt-6.1-sol" in str(app.query_one("#model", Static).render())
            app.query_one(Composer).insert("second")
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert not app.queue and not app.query(UserBubble)[-1].has_class("queued")
            assert len(app.query(Reply)) == 2
            server.notify("item/completed", turnId="turn-1", item={
                "id": "answer-1", "type": "agentMessage", "text": "both handled"})
            server.finish()
            for _ in range(60):
                await pilot.pause(0.02)
                if not app.busy:
                    break
            assert not app.busy and app.chat.transcript[-2][1] == "first\n\nsecond"
    asyncio.run(go())


def test_codex_and_claude_receive_router_root(monkeypatch):
    monkeypatch.setattr(dispatch.attach, "folder", lambda: config.REPO_ROOT / "attachments")
    root = str(config.REPO_ROOT.resolve())
    for provider in ("codex", "claude"):
        for session in (None, "saved"):
            command = dispatch.chat_cmd(provider, "model-a", "low", session)
            if provider == "claude":
                assert root in command
            else:
                setting = next(arg for arg in command if arg.startswith("sandbox_workspace_write.writable_roots="))
                assert root in json.loads(setting.split("=", 1)[1])
    setting = next(arg for arg in dispatch.codex_server_cmd()
                   if arg.startswith("sandbox_workspace_write.writable_roots="))
    assert root in json.loads(setting.split("=", 1)[1])


def test_short_tier_adjustment_uses_current_provider_only():
    chat = Chat(config.load(), use_llm=False)
    assert chat.switch_command("oke coba turunkan.") is None
    chat.tier, chat.provider = "heavy", "codex"
    medium = chat.cfg["tiers"]["medium"]["codex"]["model"]
    assert chat.switch_command("oke coba turunkan.") == f"/codex {medium}"
    for prompt in ("turunkan harga saham", "naikkan volume", "tolong turunkan suhu", "ganti warna"):
        assert chat.switch_command(prompt) is None


def test_failed_codex_preserves_injected_prompt_and_attachment(server, tmp_path):
    chat = Chat(config.load(), provider="codex", tier="heavy", use_llm=False)
    worker = threading.Thread(target=lambda: chat.send("first"))
    worker.start()
    wait_until(lambda: chat._codex and chat._codex.turn_id == "turn-1")
    file = tmp_path / "instructions.txt"
    assert chat.inject("second", [file])
    server.finish("failed", "usage limit")
    worker.join(3)
    assert not worker.is_alive()
    assert "first\n\nsecond" in chat.transcript[-2][1] and str(file) in chat.transcript[-2][1]
    assert chat.transcript[-1][1] == "[gagal: usage limit]"


def test_foreign_thread_events_and_server_requests_do_not_stall_reply(server):
    proc = CodexProc("model-a", "low", None)
    worker, turn, result, _ = ask_in_thread(proc)
    wait_until(lambda: proc.turn_id == "turn-1")
    server.emit({"method": "item/agentMessage/delta", "params": {
        "threadId": "other-thread", "turnId": "turn-1", "itemId": "foreign", "delta": "foreign"}})
    server.emit({"id": "server-question", "method": "item/tool/requestUserInput", "params": {}})
    server.notify("item/completed", turnId="turn-1", item={
        "id": "answer-1", "type": "agentMessage", "text": "own reply"})
    server.finish()
    worker.join(3)
    assert result["code"] == 0 and turn.text() == "own reply"
    response = next(r for r in server.written if r.get("id") == "server-question")
    assert response["error"]["code"] == -32601
