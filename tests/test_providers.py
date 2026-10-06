from datetime import datetime

from airouter import config, dispatch
from airouter.chat import Chat, CopilotTurn, GeminiTurn, lineup

CFG = {**{k: v for k, v in config.load().items() if k not in ("chat", "loop")},
       "providers": ["gemini", "copilot", "claude", "codex"],
       "routing": {"free": ["gemini", "copilot"], "free_tiers": ["light", "medium"]}}
CFG["models"] = {"gemini": ["gemini-3.8-flash"], "copilot": ["auto", "mai-code-1.1-flash", "gpt-6-luna"],
                 **CFG["models"]}


def test_gemini_turn_parses_stream():
    turn = GeminiTurn()
    events = [
        {"type": "init", "session_id": "g1", "model": "gemini-3.8-flash"},
        {"type": "message", "role": "user", "content": "x"},
        {"type": "message", "role": "assistant", "content": "File", "delta": True},
        {"type": "tool_use", "tool_name": "read_file", "parameters": {"file_path": "hello.txt"}},
        {"type": "message", "role": "assistant", "content": "tiga baris", "delta": True},
        {"type": "result", "status": "success", "stats": {"input_tokens": 18340, "output_tokens": 32, "cached": 0}},
    ]
    shown = [turn.feed(ev) for ev in events]
    assert ("note", "read_file hello.txt") in shown
    assert turn.session == "g1" and turn.text() == "File\n\ntiga baris" and "in 18.3k" in turn.usage


def test_gemini_failed_result_is_error():
    turn = GeminiTurn()
    turn.feed({"type": "result", "status": "error", "error": {"message": "quota exceeded"}})
    assert turn.error == "quota exceeded"


def test_copilot_turn_parses_stream():
    turn = CopilotTurn()
    events = [
        {"type": "tool.execution_start", "data": {"toolName": "view", "arguments": {"path": "hello.txt"}}},
        {"type": "assistant.message_start", "data": {}},
        {"type": "assistant.message_delta", "data": {"deltaContent": "3 "}},
        {"type": "assistant.message_delta", "data": {"deltaContent": "baris"}},
        {"type": "result", "sessionId": "c1", "exitCode": 0, "usage": {"premiumRequests": 1}},
    ]
    shown = [turn.feed(ev) for ev in events]
    assert ("note", "view hello.txt") in shown
    assert turn.session == "c1" and turn.text() == "3 baris" and turn.usage == "premium request 1"
    assert not turn.error


def test_copilot_denied_tool_is_shown():
    turn = CopilotTurn()
    shown = turn.feed({"type": "tool.execution_complete",
                       "data": {"success": False, "error": {"message": "Permission denied"}}})
    assert shown == ("note", "gagal: Permission denied")


def test_free_first_for_light_paid_first_for_heavy():
    assert lineup(CFG, "light")[:2] == ["gemini", "copilot"]
    assert lineup(CFG, "heavy")[:2] == ["claude", "codex"]
    assert lineup(CFG, "heavy")[2:] == ["gemini", "copilot"]


def test_paid_take_turns_by_todays_use():
    assert lineup(CFG, "heavy", usage={"claude": 5, "codex": 1})[0] == "codex"
    assert lineup(CFG, "heavy", usage={"claude": 1, "codex": 5})[0] == "claude"


def test_conversation_sticks_to_its_provider():
    assert lineup(CFG, "light", sticky="copilot")[0] == "copilot"
    assert lineup(CFG, "heavy", sticky="codex", usage={"codex": 9})[0] == "codex"


def test_free_provider_does_not_keep_heavy_message():
    assert lineup(CFG, "heavy", sticky="gemini")[0] == "claude"


def test_usage_today_counts_successful_runs():
    config.log({"ts": datetime.now().isoformat(timespec="seconds"), "provider": "codex", "exit": 0})
    config.log({"ts": datetime.now().isoformat(timespec="seconds"), "provider": "codex", "exit": 1})
    config.log({"ts": "2000-01-01T00:00:00", "provider": "claude", "exit": 0})
    assert config.usage_today() == {"codex": 1}


def test_chat_cmds_for_free_providers():
    gemini = dispatch.chat_cmd("gemini", "gemini-3.8-flash", "low", "g1")
    assert gemini[gemini.index("--approval-mode") + 1] == "auto_edit" and gemini[-2:] == ["--resume", "g1"]
    copilot = dispatch.chat_cmd("copilot", "auto", "low", None)
    assert "--allow-all-tools" in copilot and "shell(git push)" in copilot and "--session-id" not in copilot


def test_provider_command_pins_any_provider():
    c = Chat(CFG, use_llm=False)
    c.command("/gemini")
    assert c.order("heavy") == ["gemini"]


def test_catalog_lists_every_provider():
    c = Chat(CFG, use_llm=False)
    providers = {p for p, _ in c.catalog()}
    assert {"gemini", "copilot", "claude", "codex"} <= providers


def test_copilot_auto_model_sends_no_effort():
    cmd = dispatch.chat_cmd("copilot", "auto", "low", None)
    assert "--reasoning-effort" not in cmd
    pinned = dispatch.chat_cmd("copilot", "gpt-6-luna", "high", None)
    assert pinned[pinned.index("--reasoning-effort") + 1] == "high"


def test_empty_answer_falls_back_to_next_provider():
    c = Chat(CFG, use_llm=False)
    calls = []

    def runner(cmd, prompt, turn, show, started):
        calls.append(cmd)
        if len(calls) == 1:
            return 0
        turn.feed({"type": "assistant.message_delta", "data": {"deltaContent": "ok"}})
        turn.feed({"type": "result", "sessionId": "c1", "exitCode": 0, "usage": {}})
        return 0

    c.command("/model light")
    c.pinned_tier = "light"
    assert c.send("halo", runner=runner) == 0
    assert len(calls) == 2 and c.provider == "copilot" and c.transcript[-1] == ("Assistant", "ok")


def test_sync_copies_claude_rules_to_other_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "CLAUDE.md").write_text("# rules\n- talk in Bahasa Indonesia\n", encoding="utf-8")
    changed = config.sync_rules()
    assert len(changed) == 2
    for target in config.rules_targets():
        text = target.read_text(encoding="utf-8")
        assert text.startswith(config.RULES_HEADER) and "Bahasa Indonesia" in text
    assert config.sync_rules() == []


def test_sync_without_source_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.sync_rules() == []
