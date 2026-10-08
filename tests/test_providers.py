from datetime import datetime

from airouter import config
from airouter.chat import Chat, lineup

CFG = {**{k: v for k, v in config.load().items() if k not in ("chat", "loop")}, "providers": ["claude", "codex"]}


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
