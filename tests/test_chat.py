import json
import threading
import time
from datetime import datetime

from airouter import cli, config, dispatch, learn
from airouter.chat import CHAT_NOTE, Chat, ClaudeTurn, CodexTurn, lineup, parse_review, recap, review_due

CFG = {**{k: v for k, v in config.load().items() if k not in ("chat", "routing", "loop")}, "providers": ["claude", "codex"]}


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


def test_failed_chat_logs_the_error():
    run, _ = fake_runner([(1, [{"type": "result", "subtype": "error_during_execution",
                                "is_error": True, "result": "session hilang"}])])
    c = Chat({**CFG, "providers": ["claude"]}, use_llm=False, ui=RecordUI())
    assert c.send("coba lagi", runner=run) == 1
    row = json.loads(config.log_path().read_text(encoding="utf-8").strip())
    assert row["mode"] == "chat" and row["exit"] == 1 and row["error"] == "session hilang"


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
    assert prompt == learn.style_note()[1] + "apa itu ISO"


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
    assert "resume" in cmd and prompt == learn.style_note()[1] + "apa itu USB"


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
    c.command("/model mistral")
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


def test_short_sentence_switches_model():
    c = catalog_chat()
    assert c.switch_command("pakai opus") == "/model opus"
    assert c.switch_command("Ganti ke Codex aja") == "/codex"
    assert c.switch_command("tolong ganti model ke luna") == "/model luna"
    assert c.switch_command("pakai auto") == "/model auto"
    assert c.switch_command("ganti tier heavy") == "/model heavy"


def test_ordinary_prompts_are_not_switches():
    c = catalog_chat()
    for msg in ("pakai pandas", "ganti nama variabel ini", "pakai opus untuk review kode ini",
                "use it", "kenapa pakai opus?"):
        assert c.switch_command(msg) is None, msg


def test_switch_falls_back_to_configured_models_before_catalog_loads():
    c = chat()
    assert c._catalog is None
    assert c.switch_command("pakai sonnet") == "/model sonnet"
    assert c._catalog is None


class FakeClaude:
    def __init__(self):
        import queue
        self.events = queue.Queue()
        self.written = []
        self.stdin = self
        self.stdout = iter(self.events.get, None)

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass

    def poll(self):
        return None

    def emit(self, *events):
        for ev in events:
            self.events.put((json.dumps(ev) + "\n").encode("utf-8"))


def fake_claude_proc():
    from airouter.chat import ClaudeProc
    p = ClaudeProc.__new__(ClaudeProc)
    p.key, p.session, p.errors = ("haiku", "low"), None, []
    p.proc = FakeClaude()
    p.lock, p.active, p.sent, p.acked = threading.Lock(), False, 0, 0
    return p


def text_event(text):
    return {"type": "stream_event", "event": {"type": "content_block_delta",
                                              "delta": {"type": "text_delta", "text": text}}}


REPLAY = {"type": "user", "isReplay": True, "message": {"role": "user", "content": []}}
RESULT = {"type": "result", "is_error": False, "usage": {}}


def _ask_in_thread(p):
    turn, out = ClaudeTurn(), {}
    worker = threading.Thread(target=lambda: out.update(code=p.ask("pertama", turn, lambda *a: None)))
    worker.start()
    return turn, out, worker


def test_message_injected_mid_turn_joins_the_same_answer():
    p = fake_claude_proc()
    turn, out, worker = _ask_in_thread(p)
    p.proc.emit(REPLAY, text_event("ALPHA "))
    time.sleep(0.1)
    assert p.inject("kedua")
    p.proc.emit(REPLAY, text_event("BRAVO"), RESULT)
    worker.join(2)
    assert out["code"] == 0 and turn.text() == "ALPHA BRAVO" and len(p.proc.written) == 2
    assert not p.inject("terlambat")


def test_message_taken_in_after_result_is_read_as_its_own_turn():
    p = fake_claude_proc()
    turn, out, worker = _ask_in_thread(p)
    p.proc.emit(REPLAY, text_event("ALPHA"))
    time.sleep(0.1)
    assert p.inject("kedua")
    p.proc.emit(RESULT)
    time.sleep(0.1)
    assert worker.is_alive()
    p.proc.emit(REPLAY, text_event(" BRAVO"), RESULT)
    worker.join(2)
    assert out["code"] == 0 and turn.text() == "ALPHA BRAVO"


def test_inject_needs_a_running_claude_turn():
    c = chat()
    assert not c.inject("halo")
    p = fake_claude_proc()
    c._claude = p
    assert not c.inject("halo")


def test_sentence_moves_tier_down_or_up_on_the_named_provider():
    c = catalog_chat()
    c.tier, c.provider = "heavy", "claude"
    light, medium = CFG["tiers"]["light"]["codex"]["model"], CFG["tiers"]["medium"]["codex"]["model"]
    assert c.switch_command("aku mau ganti model yg lebih ringan, tp punya codex/") == f"/codex {medium}"
    assert c.switch_command("pakai model codex yang paling ringan") == f"/codex {light}"
    assert c.switch_command("turunin modelnya dong") == f"/claude {CFG['tiers']['medium']['claude']['model']}"
    c.tier = "light"
    assert c.switch_command("naikin model") == f"/claude {CFG['tiers']['medium']['claude']['model']}"


def test_sentence_naming_a_model_or_provider_pins_it():
    c = catalog_chat()
    assert c.switch_command("coba ganti ke model gpt-5.6-luna ya") == "/model gpt-5.6-luna"
    assert c.switch_command("aku mau pindah ke codex sekarang") == "/codex"


def test_sentences_about_other_things_are_not_switches():
    c = catalog_chat()
    for msg in ("kenapa kamu memindah kan modelnya skrg ke opus claude?",
                "pakai model yang lebih ringan untuk embedding classifier",
                "ganti warna kursornya jadi lebih terang",
                "apa bedanya model codex dan claude?",
                "tolong ubah fungsi ini supaya lebih ringan"):
        assert c.switch_command(msg) is None, msg


def test_claude_rate_limit_event_saves_quota():
    turn = ClaudeTurn()
    turn.feed({"type": "rate_limit_event", "rate_limit_info": {"unifiedWindows": {
        "five_hour": {"utilization": 0.28, "resetsAt": 4102444800},
        "seven_day": {"utilization": 0.02, "resetsAt": 4102444800}}}})
    assert config.quota_line() == "claude 5j 72% · mgg 98%"


def test_codex_limits_map_by_window_length():
    windows = config.codex_windows({"primary": {"usedPercent": 40, "windowDurationMins": 300, "resetsAt": 10},
                                    "secondary": {"usedPercent": 5, "windowDurationMins": 10080, "resetsAt": 99}})
    assert windows == {"5h": {"used": 40, "resets": 10}, "week": {"used": 5, "resets": 99}}
    assert config.quota_line({"codex": windows}, now=50) == "codex 5j 100% · mgg 95%"


def test_empty_limits_keep_previous_quota():
    config.save_quota("codex", {"5h": {"used": 10, "resets": None}})
    config.save_quota("codex", config.codex_windows({}))
    assert config.quota_line() == "codex 5j 90%"


def test_quota_report_lists_both_providers():
    from airouter.cli import quota_report
    text = quota_report({"codex": {"at": 0, "5h": {"used": 25, "resets": 4102444800}}})
    assert "codex   5 jam   sisa  75%" in text and "claude  belum ada data" in text


LOOP_CFG = {**CFG, "tiers": {**CFG["tiers"], "heavy": {**CFG["tiers"]["heavy"],
                                                       "claude": {"model": "claude-fable-5-1", "effort": "high"}}},
            "loop": {"models": {"claude": "fable 5", "codex": "astra 6"}, "min_quota": 20}}


class RecordUI:
    def __init__(self):
        self.shown = []
        self.infos = []

    def info(self, text):
        self.infos.append(text)

    def start(self, *args):
        pass

    def show(self, kind, text):
        self.shown.append((kind, text))

    def done(self, *args):
        pass

    def failed(self, *args):
        pass


def showing_runner(replies):
    calls = []

    def run(cmd, prompt, turn, show, started):
        calls.append((cmd, prompt))
        code, events = replies.pop(0)
        for ev in events:
            shown = turn.feed(ev)
            if shown:
                show(*shown)
        return code

    return run, calls


def test_heavy_answer_is_self_reviewed_and_only_final_is_shown():
    run, calls = showing_runner([(0, claude_reply("versi 1")), (0, claude_reply("Cek angka.\nREVISI\n- angka salah")),
                                 (0, claude_reply("versi 2")), (0, claude_reply("Sudah benar.\n**LOLOS**"))])
    ui = RecordUI()
    c = Chat(LOOP_CFG, provider="claude", tier="heavy", use_llm=False, ui=ui)
    assert c.send("hitung ulang anggaran", runner=run) == 0
    assert c.transcript[-1] == ("Assistant", "versi 2")
    assert [t for k, t in ui.shown if k == "text"] == ["versi 2"]
    reviews = [i for i, (cmd, _) in enumerate(calls) if "--no-session-persistence" in cmd]
    assert reviews == [1, 3] and all("claude-fable-5-1" in calls[i][0] for i in reviews)
    assert "versi 1" in calls[1][1] and "hitung ulang anggaran" in calls[1][1]
    assert "- angka salah" in calls[2][1]
    rows = [json.loads(line) for line in config.log_path().read_text(encoding="utf-8").splitlines()]
    assert [r["mode"] for r in rows] == ["chat", "review", "revise", "review"]
    assert all("prompt" not in r for r in rows[1:])


def test_failed_revision_is_retried_once():
    run, calls = fake_runner([(0, claude_reply("versi 1")), (0, claude_reply("REVISI\n- kurang")),
                              (1, []), (0, claude_reply("versi 2")), (0, claude_reply("LOLOS"))])
    c = Chat(LOOP_CFG, provider="claude", tier="heavy", use_llm=False, ui=RecordUI())
    assert c.send("rancang arsitektur", runner=run) == 0
    assert c.transcript[-1] == ("Assistant", "versi 2") and len(calls) == 5
    rows = [json.loads(line) for line in config.log_path().read_text(encoding="utf-8").splitlines()]
    assert [r["mode"] for r in rows] == ["chat", "review", "revise", "revise", "review"]
    assert rows[2]["error"] and "error" not in rows[3]


def test_loop_stops_after_max_rounds_with_last_version():
    replies = [(0, claude_reply("versi 1"))]
    for n in (2, 3):
        replies += [(0, claude_reply("REVISI\n- lagi")), (0, claude_reply(f"versi {n}"))]
    run, calls = fake_runner(replies)
    ui = RecordUI()
    c = Chat({**LOOP_CFG, "loop": {**LOOP_CFG["loop"], "max_rounds": 2}},
             provider="claude", tier="heavy", use_llm=False, ui=ui)
    assert c.send("rancang arsitektur", runner=run) == 0
    assert len(calls) == 5 and c.transcript[-1] == ("Assistant", "versi 3")
    assert any("belum lolos setelah 2 review" in i for i in ui.infos)


def test_loop_stops_when_quota_is_low():
    config.save_quota("claude", {"5h": {"used": 90, "resets": None}})
    run, calls = fake_runner([(0, claude_reply("jawaban"))])
    c = Chat(LOOP_CFG, provider="claude", tier="heavy", use_llm=False)
    assert c.send("rancang arsitektur", runner=run) == 0
    assert len(calls) == 1 and c.transcript[-1] == ("Assistant", "jawaban")


def test_loop_skips_models_not_listed():
    run, calls = fake_runner([(0, claude_reply("ok"))])
    c = Chat(LOOP_CFG, provider="claude", tier="medium", use_llm=False)
    assert c.send("fix bug ini", runner=run) == 0 and len(calls) == 1


def test_review_only_for_listed_families_at_or_above_version():
    due = [m for m in ("claude-fable-5-1", "claude-fable-5", "fable", "claude-opus-5-5", "opus", "sonnet")
           if review_due(LOOP_CFG, "claude", m)]
    assert due == ["claude-fable-5-1", "claude-fable-5", "fable"]
    due = [m for m in ("gpt-6-astra", "gpt-6.1-astra", "gpt-7-astra", "gpt-5.6-astra", "gpt-6.1-sol", "gpt-6-sol")
           if review_due(LOOP_CFG, "codex", m)]
    assert due == ["gpt-6-astra", "gpt-6.1-astra", "gpt-7-astra"]
    assert not review_due(LOOP_CFG, "other", "fable") and not review_due(CFG, "claude", "fable")


def test_pinned_high_model_is_reviewed_in_any_tier():
    run, calls = fake_runner([(0, claude_reply("jawaban")), (0, claude_reply("LOLOS"))])
    c = Chat(LOOP_CFG, provider="claude", tier="medium", use_llm=False, ui=RecordUI())
    c.pinned_provider, c.pinned_model = "claude", "claude-fable-5-1"
    assert c.send("fix bug ini", runner=run) == 0 and len(calls) == 2


def test_unreadable_review_keeps_first_answer():
    run, calls = fake_runner([(0, claude_reply("jawaban")), (0, codex_reply("kelihatannya bagus"))])
    c = Chat(LOOP_CFG, provider="claude", tier="heavy", use_llm=False)
    assert c.send("rancang arsitektur", runner=run) == 0
    assert len(calls) == 2 and c.transcript[-1] == ("Assistant", "jawaban")


def test_parse_review_takes_last_verdict():
    assert parse_review("REVISI\n- a") == ("REVISI", "- a")
    assert parse_review("Awalnya REVISI tapi\nsetelah cek:\nLOLOS") == ("LOLOS", "")
    assert parse_review("1. tambah test\n\nREVISI") == ("REVISI", "1. tambah test")
    assert parse_review("tidak ada vonis")[0] is None


def test_every_turn_carries_the_style_note():
    c = chat()
    run, calls = fake_runner([(0, claude_reply("a")), (0, claude_reply("b"))])
    c.send("apa itu LC3", runner=run)
    c.send("apa itu ISO", runner=run)
    assert all(learn.style_note()[1] in prompt for _, prompt in calls)
    assert "not curtly" not in CHAT_NOTE


def test_ramble_flags_padding_but_not_code():
    padded = ("Pertanyaan bagus! Anda bertanya soal ISO.\n\n## Ringkasan\nISO adalah standar.\n\n"
              "Semoga membantu, kalau ada pertanyaan lain silakan.")
    assert learn.ramble("apa itu ISO", padded)["flags"] == ["opener", "closer", "echo", "headings"]
    code = "Pakai ini:\n```python\n" + "x = 1\n" * 300 + "```"
    assert learn.ramble("contoh kode", code)["flags"] == []
    assert learn.ramble("apa itu ISO", "kata " * 250)["flags"] == ["long"]


def test_ramble_leaves_a_direct_answer_alone():
    assert learn.ramble("kenapa build zephyr gagal di windows",
                        "Build Zephyr gagal di Windows karena path terlalu panjang.")["flags"] == []


def test_reactions_read_short_follow_ups_only():
    assert learn.reactions("singkat aja") == ["too_long"]
    assert learn.reactions("maksudnya gimana?") == ["unclear"]
    assert learn.reactions("bukan itu yang saya tanya") == ["off"]
    assert learn.reactions("tolong ringkas " + "laporan ini " * 20) == []
    assert learn.reactions("ok lanjut") == []


def test_replies_and_reactions_are_logged():
    c = chat()
    run, _ = fake_runner([(0, claude_reply("Tentu! ISO itu standar.")), (0, claude_reply("ISO = standar."))])
    c.send("apa itu ISO", runner=run)
    c.send("kepanjangan, singkat aja", runner=run)
    rows = [json.loads(line) for line in learn.replies_path().read_text(encoding="utf-8").splitlines()]
    first, reaction, second = rows
    assert first["reply"] == "Tentu! ISO itu standar." and first["flags"] == ["opener"] and not first["cancelled"]
    assert reaction["ref"] == first["id"] and reaction["reaction"] == ["too_long"]
    assert second["prompt"] == "kepanjangan, singkat aja" and second["style"] == first["style"]
    report = learn.style_report()
    assert report["replies"] == 2 and report["groups"][0]["reactions"] == {"too_long": 1}
    assert "too_long 1" in cli.style_report(report)


def test_stopped_reply_is_logged_as_cancelled():
    c = chat()
    run, _ = fake_runner([(130, claude_reply("sebagian"))])
    c.send("apa itu LC3", runner=run)
    row = json.loads(learn.replies_path().read_text(encoding="utf-8"))
    assert row["cancelled"] and row["reply"] == "sebagian"


def test_style_report_without_data():
    assert cli.style_report(learn.style_report()) == "belum ada jawaban chat yang tercatat"


def test_providers_take_turns_by_todays_use():
    assert lineup(CFG, usage={"claude": 5, "codex": 1}) == ["codex", "claude"]
    assert lineup(CFG, usage={"claude": 1, "codex": 5}) == ["claude", "codex"]
    assert lineup(CFG) == ["claude", "codex"]


def test_conversation_sticks_to_its_provider():
    assert lineup(CFG, sticky="codex", usage={"codex": 9})[0] == "codex"
    assert lineup(CFG, sticky="claude", usage={"claude": 9})[0] == "claude"
    assert lineup(CFG, sticky="gone", usage={"claude": 9}) == ["codex", "claude"]


def test_usage_today_counts_successful_runs():
    config.log({"ts": datetime.now().isoformat(timespec="seconds"), "provider": "codex", "exit": 0})
    config.log({"ts": datetime.now().isoformat(timespec="seconds"), "provider": "codex", "exit": 1})
    config.log({"ts": "2000-01-01T00:00:00", "provider": "claude", "exit": 0})
    assert config.usage_today() == {"codex": 1}


def test_provider_command_pins_any_provider():
    c = Chat(CFG, use_llm=False)
    c.command("/codex")
    assert c.order("heavy") == ["codex"]


def test_catalog_lists_every_provider():
    c = Chat(CFG, use_llm=False)
    providers = {p for p, _ in c.catalog()}
    assert {"claude", "codex"} <= providers


def test_empty_answer_falls_back_to_next_provider():
    c = Chat(CFG, use_llm=False)
    calls = []

    def runner(cmd, prompt, turn, show, started):
        calls.append(cmd)
        if len(calls) == 1:
            return 0
        turn.feed({"type": "thread.started", "thread_id": "t1"})
        turn.feed({"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}})
        turn.feed({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
        return 0

    c.command("/model light")
    c.pinned_tier = "light"
    assert c.send("halo", runner=runner) == 0
    assert len(calls) == 2 and c.provider == "codex" and c.transcript[-1] == ("Assistant", "ok")
