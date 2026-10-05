import hashlib
import os
import threading
import urllib.request
from pathlib import Path

from . import config

REPO = "ibm-granite/granite-embedding-97m-multilingual-r2"
REVISION = "835ad14087e140460703cf0fae09f97d469d65c2"
FILES = {"model.onnx": "onnx/model_quint8_avx2.onnx", "tokenizer.json": "tokenizer.json"}
MAX_TOKENS = 512

_lock = threading.Lock()
_encoder = None
_failed = None
_vectors = {}
_CACHE_SAVE_BATCH = 64


def folder() -> Path:
    return Path(os.environ.get("AI_ROUTER_MODELS", config.REPO_ROOT / "logs" / "models")) / REPO.split("/")[1]


def present() -> bool:
    return all((folder() / name).exists() for name in FILES)


def download(info=lambda text: None):
    dest = folder()
    dest.mkdir(parents=True, exist_ok=True)
    for name, remote in FILES.items():
        target = dest / name
        if target.exists():
            continue
        info(f"mengunduh {REPO} {remote}...")
        part = target.with_suffix(".part")
        urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}", part)
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


def get(wait: bool = True):
    global _encoder, _failed
    if _encoder or _failed:
        return _encoder
    if not _lock.acquire(blocking=wait):
        return None
    try:
        if not (_encoder or _failed) and present():
            try:
                _encoder = Encoder(folder())
            except Exception as exc:
                _failed = str(exc)
    finally:
        _lock.release()
    return _encoder


def warm(info=lambda text: None):
    try:
        if not present():
            download(info)
    except OSError as exc:
        info(f"model klasifikasi tidak bisa diunduh, pakai Naive Bayes: {exc}")
        return None
    return get()


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cache_path() -> Path:
    return Path(os.environ.get("AI_ROUTER_LABELS", config.REPO_ROOT / "logs" / "labels.jsonl")).with_name("ml_vectors.npz")


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
    if disk and not cache and cache_path().exists():
        try:
            with np.load(cache_path()) as saved:
                cache.update(zip(saved["keys"].tolist(), saved["vecs"]))
        except (OSError, ValueError, KeyError):
            pass
    keys = [_key(t) for t in texts]
    missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in cache))
    before = len(cache)
    for i in range(0, len(missing), 16):
        batch = missing[i:i + 16]
        cache.update(zip(map(_key, batch), encoder(batch)))
    if missing and disk and (len(cache) - before >= _CACHE_SAVE_BATCH or not cache_path().exists()):
        _save_vectors(cache, cache_path())
    return np.stack([cache[k] for k in keys])
