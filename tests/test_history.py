import json

from airouter import config, history
from airouter.chat import Chat


CFG = {**{k: v for k, v in config.load().items() if k not in ("chat", "routing", "loop")},
       "providers": ["claude", "codex"]}


def _state(history_id="one", prompt="halo", answer="hai"):
    return {
        "history_id": history_id,
        "tier": "light",
        "provider": "claude",
        "model": "haiku",
        "sessions": {"claude": "session-1"},
        "synced": {"claude": 2},
        "transcript": [("User", prompt), ("Assistant", answer)],
        "last_reply": None,
        "pinned_provider": None,
        "pinned_tier": None,
        "pinned_model": None,
        "pinned_effort": None,
    }


def test_history_save_load_and_update_are_atomic():
    assert history.save(_state())
    first = history.entries()
    assert len(first) == 1 and first[0]["title"] == "halo"
    created = first[0]["created_at"]

    assert history.save(_state(prompt="halo lagi", answer="siap"))
    rows = history.entries()
    assert len(rows) == 1 and rows[0]["created_at"] == created
    assert history.load("one")["transcript"][-1] == ["Assistant", "siap"]
    assert not history.path().with_suffix(".json.tmp").exists()


def test_history_ignores_corrupt_content_and_invalid_rows():
    target = history.path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("bukan json", encoding="utf-8")
    assert history.entries() == []
    assert history.save(_state())
    target.write_text(json.dumps([None, {"id": 2}, {"id": "ok", "state": []}]), encoding="utf-8")
    assert [row["id"] for row in history.entries()] == ["ok"]
    assert history.load("ok") is None


def test_history_keeps_only_the_newest_items(monkeypatch):
    monkeypatch.setattr(history, "MAX_ITEMS", 2)
    for number in range(3):
        assert history.save(_state(str(number), f"prompt {number}"))
    assert [row["id"] for row in history.entries()] == ["2", "1"]


def test_chat_can_resume_saved_conversation_and_new_keeps_the_old_one():
    chat = Chat(CFG, use_llm=False)
    chat.restore(_state())
    assert chat.transcript[-1] == ("Assistant", "hai")
    old_id = chat.history_id
    chat.reset()
    assert chat.history_id != old_id and chat.transcript == []
    assert history.load(old_id)["sessions"] == {"claude": "session-1"}

    assert chat.load_history(old_id)
    assert chat.history_id == old_id
    assert chat.sessions == {"claude": "session-1"}


def test_history_delete_removes_only_the_given_ids():
    for number in range(3):
        assert history.save(_state(str(number), f"prompt {number}"))
    assert history.delete({"1", "missing"}) == 1
    assert [row["id"] for row in history.entries()] == ["2", "0"]
    assert history.delete(set()) == 0
    assert not history.path().with_suffix(".json.tmp").exists()


def test_history_command_deletes_by_number_and_all():
    chat = Chat(CFG, use_llm=False)
    for number in range(3):
        assert history.save(_state(str(number), f"prompt {number}"))
    newest_first = [row["id"] for row in history.entries()]

    assert not chat.history_command("hapus 1 9")
    assert [row["id"] for row in history.entries()] == newest_first[1:]

    assert not chat.history_command("hapus semua")
    assert history.entries() == []


def test_deleting_the_open_conversation_clears_it_and_does_not_resave():
    chat = Chat(CFG, use_llm=False)
    chat.restore(_state())
    assert chat.remember()
    old_id = chat.history_id

    assert not chat.history_command("hapus 1")
    assert chat.history_id != old_id and chat.transcript == []
    assert history.load(old_id) is None
    chat.command("/history")
    assert history.entries() == []


def test_restore_rejects_invalid_persisted_fields():
    chat = Chat(CFG, use_llm=False)
    chat.restore({
        **_state(),
        "tier": "admin",
        "provider": "unknown",
        "pinned_effort": "infinite",
        "sessions": {"unknown": "bad", "claude": 7},
        "synced": {"claude": -4},
        "transcript": [["System", "bad"], ["User", "ok"], ["Assistant", 8]],
    })
    assert chat.tier is None and chat.provider is None and chat.pinned_effort is None
    assert chat.sessions == {} and chat.synced == {"claude": 0}
    assert chat.transcript == [("User", "ok")]


def test_empty_or_non_serializable_state_is_not_saved():
    assert not history.save({"history_id": "empty", "transcript": []})
    assert not history.save({"history_id": "bad", "transcript": [("User", "x")], "value": object()})
    assert not history.path().exists()
