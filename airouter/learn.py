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
_trained = {}


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
    def __init__(self, rows: list[dict] = (), encoder=None):
        self.encoder = encoder
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

    def predict(self, text: str) -> tuple[str | None, float]:
        if self.encoder and len({r["tier"] for r in self.rows}) > 1:
            probs = self._classifier(self.rows)(vectors([text], self.encoder))[0]
            best = int(probs.argmax())
            return TIERS[best], float(probs[best])
        return self._bayes(text)

    def _classifier(self, rows: list[dict]):
        import numpy as np
        key = (id(self.encoder), hashlib.sha1(json.dumps(
            [(r["text"], r["tier"], r.get("source")) for r in rows]).encode("utf-8")).hexdigest())
        if key not in _trained:
            X = vectors([r["text"] for r in rows], self.encoder)
            y = np.array([TIERS.index(r["tier"]) for r in rows])
            w = np.array([WEIGHTS.get(r.get("source"), 1.0) for r in rows])
            if len(_trained) > 8:
                _trained.clear()
            _trained[key] = fit_logreg(X, y, w)
        return _trained[key]

    def _bayes(self, text: str) -> tuple[str | None, float]:
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
        old = next((r for r in self.rows if _norm(r["text"]) == _norm(row["text"])), None)
        if old and _rank(row) < _rank(old):
            return
        if old:
            self._fit(old, -1)
            self.rows.remove(old)
        self._fit(row)

    def cross_val(self) -> list[tuple[float, str, str]]:
        out = []
        if not self.encoder:
            for row in list(self.rows):
                self._fit(row, -1)
                tier, prob = self._bayes(row["text"])
                self._fit(row, 1)
                self.rows.pop()
                out.append((prob, tier, row["tier"]))
            return out
        for k in range(FOLDS):
            train = [r for i, r in enumerate(self.rows) if i % FOLDS != k]
            test = [r for i, r in enumerate(self.rows) if i % FOLDS == k]
            if not test or len({r["tier"] for r in train}) < 2:
                continue
            probs = self._classifier(train)(vectors([r["text"] for r in test], self.encoder))
            out += [(float(p.max()), TIERS[int(p.argmax())], r["tier"]) for p, r in zip(probs, test)]
        return out

    def held_out(self) -> list[tuple[float, bool]]:
        return [(prob, guess == truth) for prob, guess, truth in self.cross_val()]

    def threshold(self, target: float, preds: list[tuple[float, bool]] | None = None) -> float:
        preds = self.held_out() if preds is None else preds
        for t in [x / 100 for x in range(60, 100, 2)]:
            sure = [ok for prob, ok in preds if prob >= t]
            if len(sure) >= MIN_SURE and sum(sure) / len(sure) >= target:
                return t
        return NEVER

    def cached_threshold(self, target: float) -> float:
        path = labels_path().with_name("ml_threshold.json")
        key = {"rows": len(self.rows), "target": target, "backend": self.backend}
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
        cv = self.cross_val()
        preds = [(prob, guess == truth) for prob, guess, truth in cv]
        t = self.threshold(target, preds)
        sure = [ok for prob, ok in preds if prob >= t]
        n = len(preds)
        under = sum(TIERS.index(guess) < TIERS.index(truth) for _, guess, truth in cv)
        result = {
            "backend": self.backend,
            "labels": len(self.rows),
            "by_source": dict(Counter(r.get("source", "?") for r in self.rows)),
            "accuracy_all": round(sum(ok for _, ok in preds) / n, 3) if n else None,
            "under_route": round(under / n, 3) if n else None,
            "threshold": t,
            "ml_decides": round(len(sure) / n, 3) if n else None,
            "accuracy_when_ml_decides": round(sum(sure) / len(sure), 3) if sure else None,
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


def teach(cfg: dict, info, batch: int = 25) -> int:
    model = Learner.load()
    known = {_norm(r["text"]) for r in model.rows}
    prompts = []
    for row in _jsonl(config.log_path()):
        text = str(row.get("prompt") or "").strip()
        if text and not text.startswith("/") and _norm(text) not in known:
            known.add(_norm(text))
            prompts.append(text)
    teacher = cfg["classifier"].get("teacher", "sonnet")
    added = 0
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        info(f"guru {teacher} melabeli {i + len(chunk)}/{len(prompts)} prompt...")
        for text, tier in zip(chunk, dispatch.llm_classify_batch(chunk, teacher)):
            if tier in TIERS:
                model.add(text, tier, "teacher")
                added += 1
    return added


def judge(prompt: str, cfg: dict, info, use_llm: bool = True, wait: bool = True) -> tuple[str | None, str | None, bool]:
    c = cfg["classifier"]
    model = Learner.load(get_encoder(wait) if c.get("embed", True) else None)
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
