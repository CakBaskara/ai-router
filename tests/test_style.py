import json

from airouter import cli, dispatch, learn

CFG = {"classifier": {"teacher": "sonnet"}}
PADDED = "Tentu! Anda bertanya soal ini. " + "isi " * 10 + "Semoga membantu."
CLEAN = "Jawabannya ini."


def fill(style: str, reply: str, n: int = learn.STYLE_MIN):
    for _ in range(n):
        learn.log_reply("apa itu ISO", reply, "claude", "sonnet", "light", style, False)


def teacher(monkeypatch, note="Answer first. Keep it short."):
    seen = []

    def fake(system, text, model, timeout=300):
        seen.append(json.loads(text))
        return note

    monkeypatch.setattr(dispatch, "llm_text", fake)
    return seen


def default_id():
    return learn.style_id(learn.DEFAULT_STYLE)


def test_not_due_before_enough_replies():
    fill(default_id(), PADDED, learn.STYLE_MIN - 1)
    assert not learn.style_due()
    assert "belajar setelah" in learn.learn_style(CFG)


def test_teacher_writes_a_trial_note_from_bad_replies(monkeypatch):
    seen = teacher(monkeypatch)
    fill(default_id(), PADDED)
    assert learn.style_due()
    assert "diuji" in learn.learn_style(CFG)
    assert learn.current_style() == "Answer first. Keep it short."
    assert seen[0]["current"] == learn.DEFAULT_STYLE and seen[0]["replies"][0]["flags"] == ["opener", "closer", "echo"]
    assert learn.style_report()["trial"] == {"against": default_id(), "replies": 0, "need": learn.STYLE_MIN}
    assert not learn.style_due()


def test_better_trial_is_kept(monkeypatch):
    teacher(monkeypatch)
    fill(default_id(), PADDED)
    learn.learn_style(CFG)
    new = learn.style_note()[0]
    fill(new, CLEAN)
    assert learn.style_due()
    assert "dipakai" in learn.learn_style(CFG)
    assert learn.style_note()[0] == new and learn.style_report()["trial"] is None
    assert not learn.style_due()


def test_worse_trial_is_rolled_back_and_remembered(monkeypatch):
    seen = teacher(monkeypatch)
    fill(default_id(), CLEAN)
    learn.log_reaction(learn.log_reply("x", PADDED, "claude", "sonnet", "light", default_id(), True), ["too_long"], "x")
    learn.learn_style(CFG)
    fill(learn.style_note()[0], PADDED)
    assert "dibuang" in learn.learn_style(CFG)
    assert learn.current_style() == learn.DEFAULT_STYLE
    assert not learn.style_due()
    fill(default_id(), PADDED)
    learn.learn_style(CFG)
    assert seen[-1]["tried_without_success"] == ["Answer first. Keep it short."]


def test_clean_style_is_left_alone(monkeypatch):
    seen = teacher(monkeypatch)
    fill(default_id(), CLEAN)
    assert "tidak punya masalah" in learn.learn_style(CFG)
    assert seen == [] and not learn.style_due()


def test_bad_teacher_output_changes_nothing(monkeypatch):
    teacher(monkeypatch, note="x" * (learn.STYLE_MAX_CHARS + 1))
    fill(default_id(), PADDED)
    assert "tidak memberi" in learn.learn_style(CFG)
    assert learn.current_style() == learn.DEFAULT_STYLE


def test_editing_the_note_by_hand_ends_the_trial(monkeypatch):
    teacher(monkeypatch)
    fill(default_id(), PADDED)
    learn.learn_style(CFG)
    learn.style_path().write_text("My own rules.\n", encoding="utf-8")
    assert learn.style_report()["trial"] is None
    assert "belajar setelah" in learn.learn_style(CFG)


def test_background_teacher_runs_the_style_step(monkeypatch):
    teacher(monkeypatch)
    fill(default_id(), PADDED)
    said = []
    assert learn.teach_if_due(CFG, said.append) is None
    assert "diuji" in said[0]


def test_learn_is_the_new_name_for_ml_train():
    assert cli._parse(["--learn"]).ml_train and cli._parse(["--ml-train"]).ml_train


def test_report_shows_the_trial(monkeypatch):
    teacher(monkeypatch)
    fill(default_id(), PADDED)
    learn.learn_style(CFG)
    fill(learn.style_note()[0], CLEAN, 3)
    assert f"diuji melawan {default_id()}: 3/{learn.STYLE_MIN}" in cli.style_report(learn.style_report())
