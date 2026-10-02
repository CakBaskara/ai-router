import json
import random
import subprocess
import time
import zlib

import numpy as np

from airouter import config, dispatch, embed, learn
from airouter.chat import Chat
from airouter.learn import Learner

CFG = {**{k: v for k, v in config.load().items() if k not in ("chat", "routing")}, "providers": ["claude", "codex"]}


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
    assert len(cv) == len(TRAIN) and sum(g == t for _, g, t in cv) / len(cv) >= 0.9


def test_logreg_respects_source_weights():
    X = fake_encoder(["cek dongle", "cek dongle"])
    clf = learn.fit_logreg(X, np.array([0, 2]), np.array([1.0, 3.0]))
    assert clf(X[:1])[0].argmax() == 2


def test_judge_uses_encoder_when_ready(monkeypatch):
    monkeypatch.setattr(embed, "get", lambda wait=True: fake_encoder)
    monkeypatch.setattr(learn, "_read", lambda path: TRAIN if path == learn.SEED else [])
    monkeypatch.setattr(Learner, "cached_threshold", lambda self, target: 0.5)
    monkeypatch.setattr(dispatch, "llm_classify", lambda *a: None)
    cfg = {**CFG, "classifier": {**CFG["classifier"], "ml_min_labels": 10}}
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
