import json
import random
import subprocess
import time
import zlib

import pytest

import numpy as np

from airouter import config, dispatch, learn
from airouter.chat import Chat
from airouter.learn import Learner, classify

REAL_TEACH_DUE = learn.teach_due
FREE = {"free_tiers": ["light", "medium"]}
CFG = {**{k: v for k, v in config.load().items() if k not in ("chat", "routing", "loop")}, "providers": ["claude", "codex"]}

RULES = config.load()["rules"]


def fake_encoder(texts):
    out = np.zeros((len(texts), 64))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            out[i, zlib.crc32(word.encode()) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def rows(*pairs, source="seed"):
    return [{"text": t, "tier": tier, "source": source} for t, tier in pairs]


TRAIN = rows(*[(f"halo apa kabar {i}", "light") for i in range(6)],
             *[(f"tolong fix bug fungsi {i}", "medium") for i in range(6)],
             *[(f"rancang arsitektur sistem {i}", "heavy") for i in range(6)])


def test_consolidate_keeps_highest_source_per_text():
    kept = learn.consolidate(rows(("Cek  Dongle", "light")) + rows(("cek dongle", "heavy"), source="user")
                             + rows(("cek dongle", "medium"), source="llm"))
    assert kept == [{"text": "cek dongle", "tier": "heavy", "source": "user"}]


def test_add_replaces_weaker_duplicate_but_not_stronger():
    m = Learner(rows(("cek dongle", "light")))
    m.add("cek dongle", "heavy", "llm")
    assert [r["tier"] for r in m.rows] == ["heavy"]
    m.add("cek dongle", "heavy", "user")
    m.add("cek dongle", "light", "audit")
    assert m.rows == [{"text": "cek dongle", "tier": "heavy", "source": "user"}]


def test_embedding_classifier_predicts_and_cross_validates():
    m = Learner(TRAIN, fake_encoder)
    assert m.backend == "granite"
    assert m.predict("rancang arsitektur baru")[0] == "heavy"
    assert m.predict("halo apa kabar")[0] == "light"
    cv = m.cross_val()
    assert len(cv) == len(TRAIN) and sum(max(p, key=p.get) == t for p, t in cv) / len(cv) >= 0.9


def test_logreg_respects_source_weights():
    X = fake_encoder(["cek dongle", "cek dongle"])
    clf = learn.fit_logreg(X, np.array([0, 2]), np.array([1.0, 3.0]))
    assert clf(X[:1])[0].argmax() == 2


def test_judge_uses_encoder_when_ready(monkeypatch):
    monkeypatch.setattr(learn, "get_encoder", lambda wait=True: fake_encoder)
    monkeypatch.setattr(learn, "_read", lambda path: TRAIN if path == learn.SEED else [])
    monkeypatch.setattr(Learner, "cached_gate", lambda self, *a: (0.1, 0.5))
    monkeypatch.setattr(dispatch, "llm_classify", lambda *a: None)
    cfg = {**CFG, "routing": FREE, "classifier": {**CFG["classifier"], "ml_min_labels": 10}}
    tier, why, llm = learn.judge("rancang arsitektur modul", cfg, lambda t: None)
    assert (tier, llm) == ("heavy", False) and why.startswith("ml:")


def test_audit_records_disagreement_and_learns(monkeypatch):
    monkeypatch.setattr(dispatch, "llm_classify", lambda *a: "heavy")
    learn.audit("cek dongle", "light", CFG)
    assert learn.audit_summary() == {"checked": 1, "agree": 0.0}
    assert Learner.load().rows[-1] == {"text": "cek dongle", "tier": "heavy", "source": "audit"}


def test_teach_labels_unlabeled_route_prompts(monkeypatch):
    log = config.log_path()
    log.write_text("\n".join(json.dumps({"prompt": p}) for p in
                             ["halo", "/model heavy", "kenapa iso drop", "kenapa iso drop", "rename var"]),
                   encoding="utf-8")
    seen = []
    monkeypatch.setattr(dispatch, "llm_classify_batch", lambda prompts, model: seen.extend(prompts) or
                        ["heavy" if "iso" in p else "light" for p in prompts])
    assert learn.teach(CFG, lambda t: None) == 2
    assert seen == ["kenapa iso drop", "rename var"]
    assert Learner.load().rows[-2] == {"text": "kenapa iso drop", "tier": "heavy", "source": "teacher"}


def test_batch_classify_parses_json_array(monkeypatch):
    out = json.dumps({"result": 'Here:\n["light", "Heavy"]', "is_error": False}).encode()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, b""))
    assert dispatch.llm_classify_batch(["a", "b"], "sonnet") == ["light", "heavy"]
    assert dispatch.llm_classify_batch(["a"], "sonnet") == [None]


def test_chat_audits_confident_ml_decision_in_background(monkeypatch):
    calls = []
    monkeypatch.setattr(learn, "judge", lambda *a, **k: ("light", "ml: light 0.97", False))
    monkeypatch.setattr(learn, "audit_due", lambda cfg: True)
    monkeypatch.setattr(learn, "audit", lambda *a: calls.append(a))
    c = Chat(CFG)
    c.use_llm = True
    c.route("halo semuanya")
    for _ in range(50):
        if calls:
            break
        time.sleep(0.01)
    assert calls == [("halo semuanya", "light", CFG)]


def test_audit_rate_from_config():
    random.seed(1)
    hits = sum(learn.audit_due({"classifier": {"audit_rate": 0.1}}) for _ in range(1000))
    assert 60 < hits < 140


@pytest.mark.parametrize("prompt, tier", [
    ("apa itu decorator di python?", "light"),
    ("translate 'retry on transport errors' ke bahasa indonesia", "light"),
    ("perbaiki error di fungsi parse_period", "medium"),
    ("tambah test untuk PERIOD_API", "medium"),
    ("kenapa mic ISO kehilangan frame tiap 160 detik? cari root cause", "heavy"),
    ("rancang arsitektur router multi-model dengan fallback", "heavy"),
    ("rancang arsitektur cache yang tahan restart, cukup 3 poin singkat", "heavy"),
])
def test_tier(prompt, tier):
    assert classify(prompt, RULES).tier == tier


def test_no_keyword_is_ambiguous():
    assert classify("joypad dongle hari ini", RULES).ambiguous


def test_keyword_is_not_ambiguous():
    assert not classify("apa itu LC3", RULES).ambiguous


def test_repo_session_raises_tier():
    assert classify("apa itu LC3", RULES).score < classify("apa itu LC3", RULES, repo=True).score


def test_traceback_counts_as_code():
    prompt = "Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError"
    assert "code or log" in classify(prompt, RULES).reasons


def test_config_with_bom_loads(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_bytes(b"\xef\xbb\xbf" + config.PACKAGE_CONFIG.read_bytes())
    monkeypatch.setenv("AI_ROUTER_CONFIG", str(path))
    assert "claude" in config.load()["providers"]


def test_keyword_needs_word_start():
    assert not classify("cdebugger", RULES).reasons


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


def test_gate_never_decides_when_unreliable():
    m = Learner(rows(("a b", "light"), ("a b", "heavy")))
    assert m.gate({"heavy"}, 0.05, 0.7) == (-1.0, learn.NEVER)


def test_gate_keeps_misses_and_precision_within_limits():
    cv = [({"light": 1 - s, "medium": 0.0, "heavy": s}, tier) for s, tier in [
        (0.02, "light"), (0.05, "medium"), (0.08, "light"), (0.1, "light"), (0.12, "medium"), (0.3, "heavy"),
        (0.4, "light"), (0.6, "heavy"), (0.7, "heavy"), (0.75, "medium"), (0.8, "heavy"), (0.9, "heavy")]]
    low, high = Learner().gate({"heavy"}, 0.05, 0.7, cv)
    assert (low, high) == (0.12, 0.3)


def test_decide_picks_within_the_trusted_side():
    probs = {"light": 0.3, "medium": 0.5, "heavy": 0.2}
    assert learn.decide(probs, 0.25, 0.8, {"heavy"}) == "medium"
    assert learn.decide(probs, 0.1, 0.8, {"heavy"}) is None
    assert learn.decide(probs, 0.1, 0.2, {"heavy"}) == "heavy"


def test_seed_report_has_numbers():
    report = Learner.load().report({**CFG, "routing": FREE})
    assert report["labels"] >= 30 and report["accuracy_all"] is not None
    assert {"gate", "ml_decides", "paid_caught_by_ml", "paid_sent_free_by_ml"} <= set(report)


def test_judge_skips_llm_when_confident(monkeypatch):
    calls = []
    monkeypatch.setattr(learn.dispatch, "llm_classify", lambda *a: calls.append(a) or "medium")
    monkeypatch.setattr(Learner, "cached_gate", lambda self, *a: (0.5, learn.NEVER))
    tier, why, llm = learn.judge("halo", {**CFG, "routing": FREE}, lambda t: None)
    assert tier == "light" and why.startswith("ml:") and not llm and not calls


def test_judge_asks_llm_and_learns_when_unsure(monkeypatch):
    monkeypatch.setattr(learn.dispatch, "llm_classify", lambda *a: "heavy")
    monkeypatch.setattr(Learner, "cached_gate", lambda self, *a: (-1.0, learn.NEVER))
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


def test_vector_cache_survives_restart():
    class Disk(learn.Encoder):
        def __init__(self):
            self.encoded = 0

        def __call__(self, texts):
            self.encoded += len(texts)
            return fake_encoder(texts)

    first, second = Disk(), Disk()
    learn.vectors(["halo apa kabar"], first)
    learn.vectors(["halo apa kabar"], second)
    assert (first.encoded, second.encoded) == (1, 0)


def test_teacher_runs_on_its_own_once_enough_prompts_wait(monkeypatch):
    monkeypatch.setattr(learn, "teach_due", REAL_TEACH_DUE)
    monkeypatch.setattr(dispatch, "llm_classify_batch", lambda prompts, model: ["heavy"] * len(prompts))
    cfg = {**CFG, "classifier": {**CFG["classifier"], "teach_every": 2, "embed": False}}
    config.log({"prompt": "rancang ulang modul pembayaran"})
    assert learn.teach_if_due(cfg) is None
    config.log({"prompt": "audit keamanan login"})
    assert learn.teach_if_due(cfg) == 2
    assert learn.teach_lock().read_text(encoding="utf-8") == "done"
    config.log({"prompt": "migrasi database lama"})
    config.log({"prompt": "analisa race condition worker"})
    assert learn.teach_if_due(cfg) is None


def test_teacher_waits_while_another_run_holds_the_lock(monkeypatch):
    monkeypatch.setattr(learn, "teach_due", REAL_TEACH_DUE)
    monkeypatch.setattr(dispatch, "llm_classify_batch", lambda prompts, model: ["heavy"] * len(prompts))
    cfg = {**CFG, "classifier": {**CFG["classifier"], "teach_every": 1, "embed": False}}
    learn.teach_lock().parent.mkdir(parents=True, exist_ok=True)
    learn.teach_lock().write_text("running", encoding="utf-8")
    config.log({"prompt": "rancang ulang modul pembayaran"})
    assert learn.teach_if_due(cfg) is None


def test_failed_teacher_batch_is_split_and_retried(monkeypatch):
    seen = []

    def batch(prompts, model):
        seen.append(len(prompts))
        return [None] * len(prompts) if len(prompts) > 2 else ["heavy"] * len(prompts)

    monkeypatch.setattr(dispatch, "llm_classify_batch", batch)
    for text in ("audit login", "migrasi db", "rancang modul", "analisa worker", "refactor parser"):
        config.log({"prompt": text})
    assert learn.teach(CFG, lambda text: None) == 5
    assert seen == [5, 2, 3, 1, 2]


def test_tuning_keeps_only_settings_that_score_better(monkeypatch):
    best = {"c": 16.0, "balanced": True}
    scores = {json.dumps(s, sort_keys=True): (0.1, 0.0) for s in learn.SETTINGS_GRID}
    scores[json.dumps(best, sort_keys=True)] = (0.5, 0.3)
    monkeypatch.setattr(Learner, "score", lambda self, cfg, cv=None: scores[json.dumps(self.settings, sort_keys=True)])
    first = Learner(TRAIN, fake_encoder).tune(CFG)
    assert first["changed"] and first["settings"] == best and learn.load_settings() == best
    again = Learner(TRAIN, fake_encoder).tune(CFG)
    assert not again["changed"] and learn.load_settings() == best


def test_balanced_settings_change_the_trained_model():
    skewed = TRAIN + rows(*[(f"halo lagi {i}", "light") for i in range(30)])
    plain = Learner(skewed, fake_encoder, {"c": 4.0, "balanced": False}).probs("rancang sistem")
    balanced = Learner(skewed, fake_encoder, {"c": 4.0, "balanced": True}).probs("rancang sistem")
    assert balanced["heavy"] > plain["heavy"]
