import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import config, dispatch
from .classify import TIERS, _looks_like_code

SEED = Path(__file__).with_name("seed_labels.jsonl")
WEIGHTS = {"seed": 1.0, "llm": 1.0, "user": 3.0}


def labels_path() -> Path:
    return Path(os.environ.get("AI_ROUTER_LABELS", config.REPO_ROOT / "logs" / "labels.jsonl"))


def features(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    n = len(words)
    feats.append("len:short" if n < 8 else "len:mid" if n < 40 else "len:long")
    if _looks_like_code(text):
        feats.append("has:code")
    return feats


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("tier") in TIERS and row.get("text"):
            rows.append(row)
    return rows


class Learner:
    def __init__(self, rows: list[dict] = ()):
        self.rows = []
        self.prior = Counter()
        self.counts = defaultdict(Counter)
        self.total = Counter()
        self.vocab = set()
        for row in rows:
            self._fit(row)

    @classmethod
    def load(cls) -> "Learner":
        return cls(_read(SEED) + _read(labels_path()))

    def _fit(self, row: dict, sign: int = 1):
        w = WEIGHTS.get(row.get("source"), 1.0) * sign
        tier = row["tier"]
        self.prior[tier] += w
        for f in features(row["text"]):
            self.counts[tier][f] += w
            self.total[tier] += w
            self.vocab.add(f)
        if sign > 0:
            self.rows.append(row)

    def predict(self, text: str) -> tuple[str | None, float]:
        classes = [t for t in TIERS if self.prior[t] > 0]
        if not classes:
            return None, 0.0
        fs = features(text)
        v = len(self.vocab) + 1
        n = sum(self.prior[t] for t in classes)
        scores = {}
        for t in classes:
            s = math.log(self.prior[t] / n)
            for f in fs:
                s += math.log((self.counts[t][f] + 1) / (self.total[t] + v))
            scores[t] = s
        top = max(scores.values())
        exp = {t: math.exp(s - top) for t, s in scores.items()}
        best = max(exp, key=exp.get)
        return best, exp[best] / sum(exp.values())

    def add(self, text: str, tier: str, source: str):
        row = {"text": text[:2000], "tier": tier, "source": source}
        path = labels_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fit(row)

    def held_out(self) -> list[tuple[float, bool]]:
        out = []
        for row in list(self.rows):
            self._fit(row, -1)
            tier, prob = self.predict(row["text"])
            self._fit(row, 1)
            self.rows.pop()
            out.append((prob, tier == row["tier"]))
        return out

    def threshold(self, target: float, preds: list[tuple[float, bool]] | None = None) -> float:
        preds = self.held_out() if preds is None else preds
        for t in [x / 100 for x in range(60, 100, 2)]:
            sure = [ok for prob, ok in preds if prob >= t]
            if len(sure) >= MIN_SURE and sum(sure) / len(sure) >= target:
                return t
        return NEVER

    def cached_threshold(self, target: float) -> float:
        path = labels_path().with_name("ml_threshold.json")
        key = {"rows": len(self.rows), "target": target}
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if {k: cached.get(k) for k in key} == key:
                return cached["threshold"]
        except (OSError, ValueError, KeyError):
            pass
        t = self.threshold(target)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**key, "threshold": t}), encoding="utf-8")
        except OSError:
            pass
        return t

    def report(self, target: float) -> dict:
        preds = self.held_out()
        t = self.threshold(target, preds)
        sure = [ok for prob, ok in preds if prob >= t]
        n = len(preds)
        return {
            "labels": n,
            "by_source": dict(Counter(r.get("source", "?") for r in self.rows)),
            "accuracy_all": round(sum(ok for _, ok in preds) / n, 3) if n else None,
            "threshold": t,
            "ml_decides": round(len(sure) / n, 3) if n else None,
            "accuracy_when_ml_decides": round(sum(sure) / len(sure), 3) if sure else None,
        }


MIN_SURE = 5
NEVER = 1.01


def judge(prompt: str, cfg: dict, info, use_llm: bool = True) -> tuple[str | None, str | None, bool]:
    c = cfg["classifier"]
    model = Learner.load()
    tier, prob = model.predict(prompt)
    if tier and len(model.rows) >= c.get("ml_min_labels", 30) \
            and prob >= model.cached_threshold(c.get("ml_target_accuracy", 0.9)):
        return tier, f"ml: {tier} {prob:.2f}", False
    if not (use_llm and c.get("llm_fallback")):
        return None, None, False
    info("menilai prompt...")
    guess = dispatch.llm_classify(prompt, c["model"])
    if not guess:
        return None, None, False
    model.add(prompt, guess, "llm")
    return guess, f"llm: {guess}", True


def tier_of(cfg: dict, provider: str, model: str) -> str | None:
    for tier in TIERS:
        if cfg["tiers"][tier][provider]["model"] == model:
            return tier
    return None
