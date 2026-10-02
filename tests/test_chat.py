from airouter import config, dispatch
from airouter.chat import CHAT_NOTE, Chat, ClaudeTurn, CodexTurn, recap

CFG = {k: v for k, v in config.load().items() if k != "chat"}


def fake_runner(replies):
    calls = []

    def run(cmd, prompt, turn, show, started):
        calls.append((cmd, prompt))
        code, events = replies.pop(0)
        for ev in events:
            turn.feed(ev)
        return code

    return run, calls


def claude_reply(text, session="s-claude"):
    return [
        {"type": "system", "subtype": "init", "session_id": session},
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}},
        {"type": "result", "subtype": "success", "is_error": False, "usage": {"input_tokens": 3, "output_tokens": 2}},
    ]


def codex_reply(text, thread="t-codex"):
    return [
        {"type": "thread.started", "thread_id": thread},
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 1}},
    ]


def chat(**kw):
    return Chat(CFG, use_llm=False, **kw)


def test_claude_turn_streams_text_and_session():
    turn = ClaudeTurn()
    shown = [turn.feed(ev) for ev in claude_reply("Paris")]
    assert ("text", "Paris") in shown
    assert turn.session == "s-claude" and turn.text() == "Paris" and not turn.error


def test_claude_turn_ignores_subagent_text():
    turn = ClaudeTurn()
    ev = {"type": "stream_event", "parent_tool_use_id": "x",
          "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "inner"}}}
    assert turn.feed(ev) is None and turn.text() == ""


def test_claude_turn_notes_tool_use():
    turn = ClaudeTurn()
    ev = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.py"}}]}}
    assert turn.feed(ev) == ("note", "Edit a.py")


def test_claude_turn_error_result():
    turn = ClaudeTurn()
    turn.feed({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "limit"})
    assert turn.error == "limit"


def test_codex_turn():
    turn = CodexTurn()
    for ev in codex_reply("Tokyo"):
        turn.feed(ev)
    assert turn.session == "t-codex" and turn.text() == "Tokyo" and "in 10" in turn.usage


def test_codex_model_switch_warning_is_not_failure():
    turn = CodexTurn()
    shown = turn.feed({"type": "item.completed", "item": {"type": "error", "message": "recorded with model x"}})
    assert shown[0] == "note" and not turn.error


def test_codex_turn_failed():
    turn = CodexTurn()
    turn.feed({"type": "turn.failed", "error": {"message": "usage limit"}})
    assert turn.error == "usage limit"


def test_tier_only_goes_up():
    c = chat()
    run, _ = fake_runner([(0, claude_reply("a")), (0, claude_reply("b")), (0, claude_reply("c"))])
    c.send("apa itu LC3", runner=run)
    assert c.tier == "light"
    c.send("cari root cause frame hilang", runner=run)
    assert c.tier == "heavy"
    c.send("apa itu ISO", runner=run)
    assert c.tier == "heavy"


def test_new_resets_tier_and_sessions():
    c = chat()
    run, _ = fake_runner([(0, claude_reply("a"))])
    c.send("cari root cause", runner=run)
    c.command("/new")
    assert c.tier is None and c.sessions == {} and c.transcript == []


def test_pinned_tier_wins():
    c = chat(tier="light")
    run, _ = fake_runner([(0, claude_reply("a"))])
    c.send("rancang arsitektur router", runner=run)
    assert c.tier == "light"


def test_same_provider_resumes_without_recap():
    c = chat()
    run, calls = fake_runner([(0, claude_reply("a")), (0, claude_reply("b"))])
    c.send("apa itu LC3", runner=run)
    c.send("apa itu ISO", runner=run)
    cmd, prompt = calls[1]
    assert cmd[cmd.index("--resume") + 1] == "s-claude"
    assert prompt == "apa itu ISO"


def test_fallback_to_codex_sends_recap_and_sticks():
    c = chat()
    run, calls = fake_runner([(0, claude_reply("jawab satu")), (1, []), (0, codex_reply("jawab dua")),
                              (0, codex_reply("jawab tiga"))])
    c.send("apa itu LC3", runner=run)
    c.send("apa itu ISO", runner=run)
    assert c.provider == "codex"
    prompt = calls[2][1]
    assert "User: apa itu LC3" in prompt and "Assistant: jawab satu" in prompt and prompt.endswith("apa itu ISO")
    c.send("apa itu USB", runner=run)
    cmd, prompt = calls[3]
    assert "resume" in cmd and prompt == "apa itu USB"


def test_switch_back_recaps_only_unseen_turns():
    c = chat()
    run, calls = fake_runner([(0, claude_reply("satu")), (0, codex_reply("dua")), (0, claude_reply("tiga"))])
    c.send("apa itu LC3", runner=run)
    c.command("/codex")
    c.send("apa itu ISO", runner=run)
    c.command("/claude")
    c.send("apa itu USB", runner=run)
    prompt = calls[2][1]
    assert "apa itu ISO" in prompt and "dua" in prompt and "LC3" not in prompt


def test_interrupted_turn_is_kept():
    c = chat()
    run, _ = fake_runner([(130, claude_reply("sebagian"))])
    assert c.send("apa itu LC3", runner=run) == 130
    assert c.transcript[-1] == ("Assistant", "sebagian [dibatalkan]")


def test_recap_truncates_long_entries():
    text = recap([("User", "x" * 5000)])
    assert "[...]" in text and len(text) < 2500


def test_chat_cmd_codex_resume_keeps_write_sandbox():
    cmd = dispatch.chat_cmd("codex", "m", "low", "t1")
    assert cmd[-2:] == ["t1", "-"] and 'sandbox_mode="workspace-write"' in cmd


def test_chat_cmd_claude_runs_in_auto_mode():
    cmd = dispatch.chat_cmd("claude", "sonnet", "medium", None)
    assert cmd[cmd.index("--permission-mode") + 1] == "auto" and "--resume" not in cmd


def test_codex_with_model_pins_exact_model():
    c = chat()
    run, calls = fake_runner([(0, codex_reply("ok"))])
    c.command("/codex gpt-6.1-sol")
    c.send("apa itu LC3", runner=run)
    cmd = calls[0][0]
    assert cmd[cmd.index("-m") + 1] == "gpt-6.1-sol" and c.label() == "gpt-6.1-sol"


def test_model_auto_clears_pinned_model():
    c = chat()
    c.command("/codex gpt-6.1-sol")
    c.command("/model auto")
    assert c.pinned_model is None and c.pinned_provider is None


def catalog_chat():
    c = chat()
    c._catalog = [("claude", "haiku"), ("claude", "opus"), ("codex", "gpt-6.1-sol"), ("codex", "gpt-6-sol"),
                  ("codex", "gpt-5.6-luna")]
    return c


def test_model_by_number():
    c = catalog_chat()
    c.command("/model 2")
    assert (c.pinned_provider, c.pinned_model) == ("claude", "opus")


def test_model_by_exact_or_partial_name():
    c = catalog_chat()
    c.command("/model gpt-6-sol")
    assert c.pinned_model == "gpt-6-sol"
    c.command("/model luna")
    assert (c.pinned_provider, c.pinned_model) == ("codex", "gpt-5.6-luna")


def test_ambiguous_or_unknown_name_changes_nothing():
    c = catalog_chat()
    c.command("/model sol")
    c.command("/model gemini")
    assert c.pinned_model is None


def test_chat_floor_raises_light_message():
    c = Chat({**CFG, "chat": {"min_tier": "medium"}}, use_llm=False)
    run, _ = fake_runner([(0, claude_reply("a"))])
    c.send("apa itu LC3", runner=run)
    assert c.tier == "medium"


def test_context_note_only_on_first_message_per_provider():
    c = chat()
    run, calls = fake_runner([(0, claude_reply("a")), (0, claude_reply("b")), (0, codex_reply("c"))])
    c.send("apa itu LC3", runner=run)
    c.send("apa itu ISO", runner=run)
    c.command("/codex")
    c.send("apa itu USB", runner=run)
    assert calls[0][1].startswith(CHAT_NOTE)
    assert not calls[1][1].startswith(CHAT_NOTE)
    assert calls[2][1].startswith(CHAT_NOTE)


def test_cancel_marks_turn_interrupted():
    c = chat()

    def run(cmd, prompt, turn, show, started):
        for ev in claude_reply("sebagian"):
            turn.feed(ev)
        c.cancel()
        return 1

    assert c.send("apa itu LC3", runner=run) == 130
    assert c.transcript[-1][1].endswith("[dibatalkan]")
