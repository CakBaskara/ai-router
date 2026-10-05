import hashlib
import json
import math
import os
import random
import re
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import config, dispatch

TIERS = ("light", "medium", "heavy")


@dataclass
class Verdict:
    tier: str
    score: int
    reasons: list[str] = field(default_factory=list)
    ambiguous: bool = False


def _hits(text: str, words: list[str]) -> list[str]:
    return [w for w in words if re.search(r"(?<!\w)" + re.escape(w.lower()), text)]


def _looks_like_code(prompt: str) -> bool:
    if "```" in prompt or "Traceback" in prompt:
        return True
    if re.search(r"^\s+at \S+", prompt, re.M):
        return True
    return prompt.count("\n") >= 10


def classify(prompt: str, rules: dict, repo: bool = False) -> Verdict:
    text = prompt.lower()
    score = 0
    reasons = []

    words = len(prompt.split())
    if words >= 150:
        score += 2
        reasons.append(f"{words} words")
    elif words >= 50:
        score += 1
        reasons.append(f"{words} words")

    if _looks_like_code(prompt):
        score += 1
        reasons.append("code or log")

    heavy = _hits(text, rules.get("heavy", []))
    medium = _hits(text, rules.get("medium", []))
    light = _hits(text, rules.get("light", []))
    if heavy:
        score += 3
        reasons.append("heavy: " + ", ".join(heavy))
    if medium:
        score += 1
        reasons.append("medium: " + ", ".join(medium))
    if light and not heavy:
        score -= 1
        reasons.append("light: " + ", ".join(light))

    if repo:
        score += 1
        reasons.append("repo session")

    tier = "light" if score <= 0 else "medium" if score <= 2 else "heavy"
    ambiguous = not (heavy or medium or light)
    return Verdict(tier, score, reasons, ambiguous)


MODEL_REPO = "ibm-granite/granite-embedding-97m-multilingual-r2"
MODEL_REVISION = "835ad14087e140460703cf0fae09f97d469d65c2"
MODEL_FILES = {"model.onnx": "onnx/model_quint8_avx2.onnx", "tokenizer.json": "tokenizer.json"}
MAX_TOKENS = 512

_lock = threading.Lock()
_encoder = None
_failed = None
_vectors = {}
_CACHE_SAVE_BATCH = 64


def model_folder() -> Path:
    return Path(os.environ.get("AI_ROUTER_MODELS", config.REPO_ROOT / "logs" / "models")) / MODEL_REPO.split("/")[1]


def model_present() -> bool:
    return all((model_folder() / name).exists() for name in MODEL_FILES)


def download_model(info=lambda text: None):
    dest = model_folder()
    dest.mkdir(parents=True, exist_ok=True)
    for name, remote in MODEL_FILES.items():
        target = dest / name
        if target.exists():
            continue
        info(f"mengunduh {MODEL_REPO} {remote}...")
        part = target.with_suffix(".part")
        urllib.request.urlretrieve(f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{remote}", part)
        part.replace(target)


class Encoder:
    def __init__(self, path: Path):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def __call__(self, texts: list[str]):
        np = self.np
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.zeros_like(ids)}
        cls = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][:, 0]
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)


def get_encoder(wait: bool = True):
    global _encoder, _failed
    if _encoder or _failed:
        return _encoder
    if not _lock.acquire(blocking=wait):
        return None
    try:
        if not (_encoder or _failed) and model_present():
            try:
                _encoder = Encoder(model_folder())
            except Exception as exc:
                _failed = str(exc)
    finally:
        _lock.release()
    return _encoder


def warm_encoder(info=lambda text: None):
    try:
        if not model_present():
            download_model(info)
    except OSError as exc:
        info(f"model klasifikasi tidak bisa diunduh, pakai Naive Bayes: {exc}")
        return None
    return get_encoder()


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def vectors_path() -> Path:
    return labels_path().with_name("ml_vectors.npz")


def _save_vectors(cache: dict[str, object], path: Path):
    import numpy as np
    if not cache:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=np.array(list(cache.keys())), vecs=np.stack(list(cache.values())))
    except (OSError, ValueError):
        pass


def vectors(texts: list[str], encoder):
    import numpy as np
    disk = isinstance(encoder, Encoder)
    cache = _vectors.setdefault(id(encoder), {})
    if disk and not cache and vectors_path().exists():
        try:
            with np.load(vectors_path()) as saved:
                cache.update(zip(saved["keys"].tolist(), saved["vecs"]))
        except (OSError, ValueError, KeyError):
            pass
    keys = [_key(t) for t in texts]
    missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in cache))
    before = len(cache)
    for i in range(0, len(missing), 16):
        batch = missing[i:i + 16]
        cache.update(zip(map(_key, batch), encoder(batch)))
    if missing and disk and (len(cache) - before >= _CACHE_SAVE_BATCH or not vectors_path().exists()):
        _save_vectors(cache, vectors_path())
    return np.stack([cache[k] for k in keys])


SEED = Path(__file__).with_name("seed_labels.jsonl")
WEIGHTS = {"seed": 1.0, "llm": 1.0, "audit": 1.0, "teacher": 2.0, "user": 3.0}
RANK = {"seed": 0, "llm": 1, "audit": 1, "teacher": 2, "user": 3}
FOLDS = 5
DEFAULT_SETTINGS = {"c": 4.0, "balanced": False}
SETTINGS_GRID = [{"c": c, "balanced": b} for c in (1.0, 4.0, 16.0) for b in (False, True)]
_trained = {}


def labels_path() -> Path:
    return Path(os.environ.get("AI_ROUTER_LABELS", config.REPO_ROOT / "logs" / "labels.jsonl"))


def settings_path() -> Path:
    return labels_path().with_name("ml_settings.json")


def load_settings() -> dict:
    try:
        saved = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULT_SETTINGS)
    return {k: saved.get(k, v) for k, v in DEFAULT_SETTINGS.items()}


def features(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    n = len(words)
    feats.append("len:short" if n < 8 else "len:mid" if n < 40 else "len:long")
    if _looks_like_code(text):
        feats.append("has:code")
    return feats


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _rank(row: dict) -> int:
    return RANK.get(row.get("source"), 1)


def consolidate(rows: list[dict]) -> list[dict]:
    best = {}
    for row in rows:
        key = _norm(row["text"])
        if key not in best or _rank(row) >= _rank(best[key]):
            best[key] = row
    return list(best.values())


def _softmax(logits):
    import numpy as np
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_logreg(X, y, w, c: float = 4.0, steps: int = 150):
    import numpy as np
    mu = X.mean(0)
    Z = X - mu
    scale = float(np.sqrt((Z ** 2).sum(1).mean())) or 1.0
    Z = Z / scale
    Y = np.eye(len(TIERS))[y]
    sw = w / w.sum()
    l2 = 1.0 / (c * w.sum())
    W = np.zeros((X.shape[1], len(TIERS)))
    b = np.zeros(len(TIERS))
    vW, vb = W.copy(), b.copy()
    for _ in range(steps):
        aW, ab = W + 0.9 * vW, b + 0.9 * vb
        G = (_softmax(Z @ aW + ab) - Y) * sw[:, None]
        vW = 0.9 * vW - (Z.T @ G + l2 * aW)
        vb = 0.9 * vb - G.sum(0)
        W, b = W + vW, b + vb
    return lambda V: _softmax((V - mu) / scale @ W + b)


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
    def __init__(self, rows: list[dict] = (), encoder=None, settings: dict | None = None):
        self.encoder = encoder
        self.settings = settings or load_settings()
        self.rows = []
        self.prior = Counter()
        self.counts = defaultdict(Counter)
        self.total = Counter()
        self.vocab = set()
        for row in rows:
            self._fit(row)

    @classmethod
    def load(cls, encoder=None) -> "Learner":
        return cls(consolidate(_read(SEED) + _read(labels_path())), encoder)

    @property
    def backend(self) -> str:
        return "granite" if self.encoder else "naive-bayes"

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

    def probs(self, text: str) -> dict[str, float]:
        if self.encoder and len({r["tier"] for r in self.rows}) > 1:
            return dict(zip(TIERS, map(float, self._classifier(self.rows)(vectors([text], self.encoder))[0])))
        return self._bayes(text)

    def predict(self, text: str) -> tuple[str | None, float]:
        probs = self.probs(text)
        if not probs:
            return None, 0.0
        best = max(probs, key=probs.get)
        return best, probs[best]

    def _classifier(self, rows: list[dict]):
        import numpy as np
        key = (id(self.encoder), json.dumps(self.settings, sort_keys=True), hashlib.sha1(json.dumps(
            [(r["text"], r["tier"], r.get("source")) for r in rows]).encode("utf-8")).hexdigest())
        if key not in _trained:
            X = vectors([r["text"] for r in rows], self.encoder)
            y = np.array([TIERS.index(r["tier"]) for r in rows])
            w = np.array([WEIGHTS.get(r.get("source"), 1.0) for r in rows])
            if self.settings.get("balanced"):
                counts = np.bincount(y, minlength=len(TIERS)).astype(float)
                w = w * (len(y) / (len(TIERS) * np.maximum(counts, 1)))[y]
            if len(_trained) > 8:
                _trained.clear()
            _trained[key] = fit_logreg(X, y, w, c=self.settings.get("c", 4.0))
        return _trained[key]

    def _bayes(self, text: str) -> dict[str, float]:
        classes = [t for t in TIERS if self.prior[t] > 0]
        if not classes:
            return {}
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
        return {t: exp.get(t, 0.0) / sum(exp.values()) for t in TIERS}

    def add(self, text: str, tier: str, source: str):
        row = {"text": text[:2000], "tier": tier, "source": source}
        path = labels_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        old = next((r for r in self.rows if _norm(r["text"]) == _norm(row["text"])), None)
        if old and _rank(row) < _rank(old):
            return
        if old:
            self._fit(old, -1)
            self.rows.remove(old)
        self._fit(row)

    def cross_val(self) -> list[tuple[dict[str, float], str]]:
        out = []
        if not self.encoder:
            for row in list(self.rows):
                self._fit(row, -1)
                probs = self._bayes(row["text"])
                self._fit(row, 1)
                self.rows.pop()
                out.append((probs, row["tier"]))
            return out
        for k in range(FOLDS):
            train = [r for i, r in enumerate(self.rows) if i % FOLDS != k]
            test = [r for i, r in enumerate(self.rows) if i % FOLDS == k]
            if not test or len({r["tier"] for r in train}) < 2:
                continue
            probs = self._classifier(train)(vectors([r["text"] for r in test], self.encoder))
            out += [(dict(zip(TIERS, map(float, p))), r["tier"]) for p, r in zip(probs, test)]
        return out

    def gate(self, paid: set[str], max_miss: float, min_precision: float,
             cv: list[tuple[dict[str, float], str]] | None = None) -> tuple[float, float]:
        if not paid:
            return 1.0, NEVER
        if paid >= set(TIERS):
            return -1.0, 0.0
        cv = self.cross_val() if cv is None else cv
        scored = [(sum(p.get(t, 0.0) for t in paid), truth in paid) for p, truth in cv if p]
        cuts = sorted({s for s, _ in scored})
        all_paid = max(sum(is_paid for _, is_paid in scored), 1)
        low, high = -1.0, NEVER
        for t in cuts:
            below = [is_paid for s, is_paid in scored if s <= t]
            if len(below) >= MIN_SURE and sum(below) / all_paid <= max_miss:
                low = t
        for t in reversed(cuts):
            above = [is_paid for s, is_paid in scored if s >= t]
            if len(above) >= MIN_SURE and sum(above) / len(above) >= min_precision:
                high = t
        return low, high

    def cached_gate(self, paid: set[str], max_miss: float, min_precision: float) -> tuple[float, float]:
        path = labels_path().with_name("ml_threshold.json")
        key = {"rows": len(self.rows), "backend": self.backend, "paid": sorted(paid), "max_miss": max_miss,
               "min_precision": min_precision, "settings": self.settings}
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if {k: cached.get(k) for k in key} == key:
                return tuple(cached["gate"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        gate = self.gate(paid, max_miss, min_precision)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**key, "gate": gate}), encoding="utf-8")
        except OSError:
            pass
        return gate

    def score(self, cfg: dict, cv: list[tuple[dict[str, float], str]] | None = None) -> tuple[float, float]:
        paid, max_miss, min_precision = gate_limits(cfg)
        cv = self.cross_val() if cv is None else cv
        low, high = self.gate(paid, max_miss, min_precision, cv)
        decided = [(decide(p, low, high, paid), truth) for p, truth in cv]
        decided = [(tier, truth) for tier, truth in decided if tier]
        all_paid = sum(truth in paid for _, truth in cv)
        coverage = len(decided) / len(cv) if cv else 0.0
        caught = sum(tier in paid and truth in paid for tier, truth in decided) / all_paid if all_paid else 0.0
        return round(coverage, 4), round(caught, 4)

    def tune(self, cfg: dict) -> dict:
        if not self.encoder:
            return {"settings": self.settings, "changed": False}
        current = self.score(cfg)
        tried = {json.dumps(s, sort_keys=True): Learner(self.rows, self.encoder, s).score(cfg) for s in SETTINGS_GRID}
        best = max(tried, key=tried.get)
        changed = tried[best] > current
        if changed:
            self.settings = json.loads(best)
            try:
                settings_path().parent.mkdir(parents=True, exist_ok=True)
                settings_path().write_text(best, encoding="utf-8")
            except OSError:
                pass
        return {"settings": self.settings, "changed": changed, "before": current,
                "after": tried[best] if changed else current}

    def report(self, cfg: dict) -> dict:
        paid, max_miss, min_precision = gate_limits(cfg)
        cv = self.cross_val()
        low, high = self.gate(paid, max_miss, min_precision, cv)
        n = len(cv)
        guesses = [(max(p, key=p.get) if p else None, truth) for p, truth in cv]
        decided = [(decide(p, low, high, paid), truth) for p, truth in cv]
        decided = [(tier, truth) for tier, truth in decided if tier]
        truly_paid = [tier for tier, truth in decided if truth in paid]
        all_paid = sum(truth in paid for _, truth in cv)
        result = {
            "backend": self.backend,
            "settings": self.settings,
            "labels": len(self.rows),
            "by_source": dict(Counter(r.get("source", "?") for r in self.rows)),
            "accuracy_all": round(sum(g == t for g, t in guesses) / n, 3) if n else None,
            "gate": [round(low, 3), round(high, 3)],
            "ml_decides": round(len(decided) / n, 3) if n else None,
            "route_accuracy_when_ml_decides": round(
                sum((tier in paid) == (truth in paid) for tier, truth in decided) / len(decided), 3) if decided else None,
            "paid_caught_by_ml": round(sum(t in paid for t in truly_paid) / all_paid, 3) if all_paid else None,
            "paid_sent_free_by_ml": round(sum(t not in paid for t in truly_paid) / all_paid, 3) if all_paid else None,
            "audit": audit_summary(),
        }
        try:
            with labels_path().with_name("ml_history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **result}) + "\n")
        except OSError:
            pass
        return result


MIN_SURE = 5
NEVER = 1.01


def gate_limits(cfg: dict) -> tuple[set[str], float, float]:
    c = cfg["classifier"]
    paid = set(TIERS) - set(cfg.get("routing", {}).get("free_tiers", []))
    return paid, c.get("ml_max_miss", 0.05), c.get("ml_min_precision", 0.7)


def decide(probs: dict[str, float], low: float, high: float, paid: set[str]) -> str | None:
    if not probs:
        return None
    score = sum(probs.get(t, 0.0) for t in paid)
    group = paid if score >= high else set(TIERS) - paid if score <= low else None
    return max(group, key=lambda t: probs.get(t, 0.0)) if group else None


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def audit_path() -> Path:
    return labels_path().with_name("ml_audit.jsonl")


def audit_due(cfg: dict) -> bool:
    return random.random() < cfg["classifier"].get("audit_rate", 0.1)


def audit(prompt: str, tier: str, cfg: dict):
    guess = dispatch.llm_classify(prompt, cfg["classifier"]["model"])
    if not guess:
        return
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "text": prompt[:200], "ml": tier, "teacher": guess,
           "agree": guess == tier}
    audit_path().parent.mkdir(parents=True, exist_ok=True)
    with audit_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    Learner.load().add(prompt, guess, "audit")


def audit_summary(last: int = 100) -> dict | None:
    rows = [r for r in _jsonl(audit_path()) if "agree" in r][-last:]
    if not rows:
        return None
    return {"checked": len(rows), "agree": round(sum(r["agree"] for r in rows) / len(rows), 3)}


def untaught(model: "Learner") -> list[str]:
    known = {_norm(r["text"]) for r in model.rows}
    prompts = []
    for row in _jsonl(config.log_path()):
        text = str(row.get("prompt") or "").strip()
        if text and not text.startswith("/") and _norm(text) not in known:
            known.add(_norm(text))
            prompts.append(text)
    return prompts


def _labels(prompts: list[str], teacher: str) -> list[str | None]:
    tiers = dispatch.llm_classify_batch(prompts, teacher)
    if len(prompts) > 1 and not any(t in TIERS for t in tiers):
        half = len(prompts) // 2
        return _labels(prompts[:half], teacher) + _labels(prompts[half:], teacher)
    return tiers


def teach(cfg: dict, info, batch: int = 25) -> int:
    model = Learner.load()
    prompts = untaught(model)
    teacher = cfg["classifier"].get("teacher", "sonnet")
    added = 0
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        info(f"guru {teacher} melabeli {i + len(chunk)}/{len(prompts)} prompt...")
        for text, tier in zip(chunk, _labels(chunk, teacher)):
            if tier in TIERS:
                model.add(text, tier, "teacher")
                added += 1
    return added


TEACH_RETRY = 600
TEACH_STALE = 1800


def teach_lock() -> Path:
    return labels_path().with_name("ml_teach.lock")


def _claim(lock: Path) -> bool:
    try:
        if lock.exists():
            age = time.time() - lock.stat().st_mtime
            busy = lock.read_text(encoding="utf-8").strip() == "running"
            if age < (TEACH_STALE if busy else TEACH_RETRY):
                return False
            lock.unlink()
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("running")
    return True


def teach_due(cfg: dict) -> bool:
    every = cfg["classifier"].get("teach_every", 0)
    return every > 0 and len(untaught(Learner.load())) >= every


def teach_if_due(cfg: dict, info=lambda text: None) -> int | None:
    c = cfg["classifier"]
    if not teach_due(cfg):
        return None
    lock = teach_lock()
    lock.parent.mkdir(parents=True, exist_ok=True)
    if not _claim(lock):
        return None
    try:
        added = teach(cfg, info)
        if added:
            retrain(cfg)
        return added
    finally:
        try:
            lock.write_text("done", encoding="utf-8")
        except OSError:
            pass


def retrain(cfg: dict, encoder=None) -> dict:
    if encoder is None and cfg["classifier"].get("embed", True):
        encoder = get_encoder()
    model = Learner.load(encoder)
    tuning = model.tune(cfg)
    return {**model.report(cfg), "tuning": tuning}


def judge(prompt: str, cfg: dict, info, use_llm: bool = True, wait: bool = True) -> tuple[str | None, str | None, bool]:
    c = cfg["classifier"]
    model = Learner.load(get_encoder(wait) if c.get("embed", True) else None)
    if len(model.rows) >= c.get("ml_min_labels", 30):
        probs = model.probs(prompt)
        paid, max_miss, min_precision = gate_limits(cfg)
        tier = decide(probs, *model.cached_gate(paid, max_miss, min_precision), paid)
        if tier:
            return tier, f"ml: {tier} {probs[tier]:.2f}", False
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
