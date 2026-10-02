from airouter import config, learn
from airouter.chat import Chat
from airouter.learn import Learner

CFG = {k: v for k, v in config.load().items() if k != "chat"}


def rows(*pairs, source="seed"):
    return [{"text": t, "tier": tier, "source": source} for t, tier in pairs]


def test_learns_from_examples():
    m = Learner(rows(("halo apa kabar", "light"), ("halo selamat pagi", "light"),
                     ("rancang arsitektur sistem baru", "heavy"), ("rancang ulang arsitektur modul", "heavy")))
    assert m.predict("halo semua")[0] == "light"
    assert m.predict("tolong rancang arsitektur")[0] == "heavy"


def test_add_persists_and_reloads():
    m = Learner.load()
    before = len(m.rows)
    m.add("joypad dongle hari ini", "heavy", "user")
    assert learn.labels_path().exists()
    assert len(Learner.load().rows) == before + 1


def test_user_label_outweighs_seed():
    m = Learner(rows(("cek status dongle", "light")))
    m._fit({"text": "cek status dongle", "tier": "heavy", "source": "user"})
    assert m.predict("cek status dongle")[0] == "heavy"


def test_threshold_never_when_unreliable():
    m = Learner(rows(("a b", "light"), ("a b", "heavy")))
    assert m.threshold(0.9) == learn.NEVER


def test_seed_report_has_numbers():
    report = Learner.load().report(0.9)
    assert report["labels"] >= 30 and report["accuracy_all"] is not None


def test_judge_skips_llm_when_confident(monkeypatch):
    calls = []
    monkeypatch.setattr(learn.dispatch, "llm_classify", lambda *a: calls.append(a) or "medium")
    monkeypatch.setattr(Learner, "cached_threshold", lambda self, target: 0.5)
    tier, why, llm = learn.judge("halo", CFG, lambda t: None)
    assert tier == "light" and why.startswith("ml:") and not llm and not calls


def test_judge_asks_llm_and_learns_when_unsure(monkeypatch):
    monkeypatch.setattr(learn.dispatch, "llm_classify", lambda *a: "heavy")
    monkeypatch.setattr(Learner, "cached_threshold", lambda self, target: learn.NEVER)
    tier, why, llm = learn.judge("joypad dongle hari ini", CFG, lambda t: None)
    assert (tier, llm) == ("heavy", True)
    assert Learner.load().rows[-1] == {"text": "joypad dongle hari ini", "tier": "heavy", "source": "llm"}


def test_tier_change_after_answer_is_recorded_as_correction():
    c = Chat(CFG, use_llm=False)
    c.transcript = [("User", "cek dongle"), ("Assistant", "ok")]
    c.tier = "medium"
    c.command("/model heavy")
    assert Learner.load().rows[-1] == {"text": "cek dongle", "tier": "heavy", "source": "user"}


def test_model_pick_maps_to_tier_for_correction():
    c = Chat(CFG, use_llm=False)
    c._catalog = [("claude", "opus")]
    c.transcript = [("User", "cek dongle"), ("Assistant", "ok")]
    c.tier = "medium"
    c.command("/model opus")
    assert Learner.load().rows[-1]["tier"] == "heavy"
